# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
sys.path.insert(0, "/home/node/data/compsep_data/")
import torch
import numpy as np
import os
import json
import time

class NavierStokes2DSolver:
    """
    A GPU-accelerated 2D Navier-Stokes solver using the pseudo-spectral method.

    This class simulates the evolution of a 2D turbulent flow and the advection
    of passive tracers within it. The simulation is performed on a doubly-periodic
    square domain.

    Key features:
    - Pseudo-spectral method for spatial derivatives.
    - RK4 for time integration with an adaptive timestep (CFL-based).
    - Hyperviscosity for small-scale dissipation.
    - Stochastic white-in-time forcing in a specified wavenumber shell.
    - Advection of passive Lagrangian tracers using bilinear interpolation.
    - Two-phase simulation: spinup followed by a production run.
    - Regular snapshotting of tracer and Eulerian field data.
    - All computations are performed on a CUDA-enabled GPU using PyTorch.
    """

    def __init__(self, params):
        """
        Initializes the solver with specified parameters.

        Args:
            params (dict): A dictionary containing all simulation parameters.
        """
        self.params = params
        self.N = params['N']
        self.device = torch.device(params['device'])
        self.output_path = params['output_path']
        os.makedirs(self.output_path, exist_ok=True)

        self.L = 2 * np.pi
        self.dx = self.L / self.N
        self.CFL = params['CFL']
        self.nu_h = params['nu_h']
        self.p = params['p']
        self.epsilon_inj = params['epsilon_inj']
        self.k_force_min = params['k_force_min']
        self.k_force_max = params['k_force_max']
        self.N_tracers = params['N_tracers']
        self.dt_snap = params['dt_snap']
        self.dt_vel = params['dt_vel']
        self.T_spinup_fixed = params['T_spinup_fixed']
        self.coarsening_factor = self.N // params['N_coarse']

        self._setup_grid_and_wavenumbers()
        self._setup_operators()

        self.t = 0.0
        self.omega_hat = self._initialize_field()
        self.tracer_pos = None

    def _setup_grid_and_wavenumbers(self):
        """Sets up the computational grid and wavenumbers in spectral space."""
        k = torch.fft.fftfreq(self.N, d=self.dx) * self.L
        self.k_rfft = torch.fft.rfftfreq(self.N, d=self.dx) * self.L
        
        kx_grid, ky_grid = torch.meshgrid(self.k_rfft, k, indexing='ij')
        self.k_squared = (kx_grid**2 + ky_grid**2).to(self.device)
        self.k_magnitude = torch.sqrt(self.k_squared)
        
        self.kx_grid_full = kx_grid.to(self.device)
        self.ky_grid_full = ky_grid.to(self.device)

        self.k_squared_no_zero = self.k_squared.clone()
        self.k_squared_no_zero[0, 0] = 1.0

    def _setup_operators(self):
        """Pre-computes spectral operators for dissipation, de-aliasing, and forcing."""
        self.dissipation_op = -self.nu_h * (self.k_squared**self.p)

        k_max_dealias = self.N / 3.0
        self.dealias_mask = (torch.abs(self.kx_grid_full) < k_max_dealias) & \
                            (torch.abs(self.ky_grid_full) < k_max_dealias)
        self.dealias_mask = self.dealias_mask.to(self.device)

        self.force_mask = (self.k_magnitude >= self.k_force_min) & \
                          (self.k_magnitude <= self.k_force_max)
        self.force_mask = self.force_mask.to(self.device)
        self.num_force_modes = torch.sum(self.force_mask)
        if self.num_force_modes > 0:
            self.force_amplitude = torch.sqrt(2 * self.epsilon_inj * self.num_force_modes / (self.L**2))
        else:
            self.force_amplitude = 0

        k_bins_indices = torch.floor(self.k_magnitude).long()
        self.k_bins_indices_flat = k_bins_indices.flatten()
        self.k_bins_max = self.N // 2
        self.k_values = torch.arange(self.k_bins_max, device=self.device)
        
        k_counts = torch.bincount(self.k_bins_indices_flat, minlength=self.k_bins_max)
        self.k_counts_no_zero = torch.where(k_counts > 0, k_counts, 1).to(self.device)

    def _initialize_field(self):
        """Initializes the vorticity field with small random noise."""
        omega = torch.randn((self.N, self.N), device=self.device) * 1e-3
        return torch.fft.rfft2(omega)

    def _compute_velocity_hat(self, omega_hat):
        """Computes velocity components in spectral space from vorticity."""
        psi_hat = -omega_hat / self.k_squared_no_zero
        psi_hat[0, 0] = 0.0
        u_hat = 1j * self.ky_grid_full * psi_hat
        v_hat = -1j * self.kx_grid_full * psi_hat
        return u_hat, v_hat

    def _compute_advection_hat(self, omega_hat, u_hat, v_hat):
        """Computes the advection term in spectral space."""
        omega = torch.fft.irfft2(omega_hat, s=(self.N, self.N))
        u = torch.fft.irfft2(u_hat, s=(self.N, self.N))
        v = torch.fft.irfft2(v_hat, s=(self.N, self.N))
        
        u_omega = torch.fft.rfft2(u * omega)
        v_omega = torch.fft.rfft2(v * omega)
        
        advection_hat = -1j * self.kx_grid_full * u_omega - 1j * self.ky_grid_full * v_omega
        return advection_hat * self.dealias_mask

    def _compute_forcing_hat(self, dt):
        """Generates the stochastic forcing term in spectral space."""
        if self.force_amplitude == 0:
            return torch.zeros_like(self.omega_hat)
        
        rand_phases = torch.exp(2j * np.pi * torch.rand(self.omega_hat.shape, device=self.device))
        forcing_hat = self.force_amplitude * rand_phases * self.force_mask / torch.sqrt(torch.tensor(dt, device=self.device))
        return forcing_hat

    def _get_rhs_hat(self, omega_hat, u_hat, v_hat, dt):
        """Computes the right-hand side of the vorticity equation."""
        advection_hat = self._compute_advection_hat(omega_hat, u_hat, v_hat)
        dissipation_hat = self.dissipation_op * omega_hat
        forcing_hat = self._compute_forcing_hat(dt)
        return advection_hat + dissipation_hat + forcing_hat

    def _interpolate_velocity_at_tracers(self, tracer_pos, u, v):
        """Interpolates the velocity field at tracer positions."""
        pos_norm = (tracer_pos / self.L) * 2 - 1
        grid = pos_norm.view(1, 1, self.N_tracers, 2)

        u_field = u.view(1, 1, self.N, self.N)
        v_field = v.view(1, 1, self.N, self.N)

        u_interp = torch.nn.functional.grid_sample(u_field, grid, align_corners=True, mode='bilinear').squeeze()
        v_interp = torch.nn.functional.grid_sample(v_field, grid, align_corners=True, mode='bilinear').squeeze()

        return torch.stack([u_interp, v_interp], dim=1)

    def _get_tracer_rhs(self, u_hat, v_hat):
        """Computes the RHS for tracer advection (i.e., interpolated velocity)."""
        u = torch.fft.irfft2(u_hat, s=(self.N, self.N))
        v = torch.fft.irfft2(v_hat, s=(self.N, self.N))
        return self._interpolate_velocity_at_tracers(self.tracer_pos, u, v)

    def _step(self, dt):
        """Performs one full RK4 step for both fluid and tracers."""
        u_hat0, v_hat0 = self._compute_velocity_hat(self.omega_hat)
        k1_w = dt * self._get_rhs_hat(self.omega_hat, u_hat0, v_hat0, dt)
        if self.tracer_pos is not None:
            k1_p = dt * self._get_tracer_rhs(u_hat0, v_hat0)
        else:
            k1_p = 0

        w_temp = self.omega_hat + 0.5 * k1_w
        u_hat1, v_hat1 = self._compute_velocity_hat(w_temp)
        k2_w = dt * self._get_rhs_hat(w_temp, u_hat1, v_hat1, dt)
        if self.tracer_pos is not None:
            p_temp = self.tracer_pos + 0.5 * k1_p
            p_temp %= self.L
            k2_p = dt * self._interpolate_velocity_at_tracers(p_temp, torch.fft.irfft2(u_hat1, s=(self.N, self.N)), torch.fft.irfft2(v_hat1, s=(self.N, self.N)))
        else:
            k2_p = 0

        w_temp = self.omega_hat + 0.5 * k2_w
        u_hat2, v_hat2 = self._compute_velocity_hat(w_temp)
        k3_w = dt * self._get_rhs_hat(w_temp, u_hat2, v_hat2, dt)
        if self.tracer_pos is not None:
            p_temp = self.tracer_pos + 0.5 * k2_p
            p_temp %= self.L
            k3_p = dt * self._interpolate_velocity_at_tracers(p_temp, torch.fft.irfft2(u_hat2, s=(self.N, self.N)), torch.fft.irfft2(v_hat2, s=(self.N, self.N)))
        else:
            k3_p = 0

        w_temp = self.omega_hat + k3_w
        u_hat3, v_hat3 = self._compute_velocity_hat(w_temp)
        k4_w = dt * self._get_rhs_hat(w_temp, u_hat3, v_hat3, dt)
        if self.tracer_pos is not None:
            p_temp = self.tracer_pos + k3_p
            p_temp %= self.L
            k4_p = dt * self._interpolate_velocity_at_tracers(p_temp, torch.fft.irfft2(u_hat3, s=(self.N, self.N)), torch.fft.irfft2(v_hat3, s=(self.N, self.N)))
        else:
            k4_p = 0

        self.omega_hat += (k1_w + 2*k2_w + 2*k3_w + k4_w) / 6
        if self.tracer_pos is not None:
            self.tracer_pos += (k1_p + 2*k2_p + 2*k3_p + k4_p) / 6
            self.tracer_pos %= self.L
        
        self.t += dt

    def _compute_diagnostics(self):
        """Computes various diagnostic quantities for the current flow state."""
        u_hat, v_hat = self._compute_velocity_hat(self.omega_hat)
        
        energy_hat = 0.5 * (torch.abs(u_hat)**2 + torch.abs(v_hat)**2)
        E_total = torch.sum(energy_hat) / (self.N**2)
        U_rms = torch.sqrt(2 * E_total)
        
        omega_sq_hat = torch.abs(self.omega_hat)**2
        omega_rms = torch.sqrt(torch.sum(omega_sq_hat) / (self.N**2))
        
        T_L = self.L / U_rms if U_rms > 0 else float('inf')

        energy_hat_flat = energy_hat.flatten()
        E_k_sum = torch.bincount(self.k_bins_indices_flat, weights=energy_hat_flat, minlength=self.k_bins_max)
        E_k = E_k_sum / self.k_counts_no_zero[:self.k_bins_max]
        
        k_peak = self.k_values[torch.argmax(E_k)] if E_total > 0 else 0.0
        d_vv_estimate = 2 * np.pi / k_peak if k_peak > 0 else float('inf')

        diags = {
            'time': self.t,
            'E_total': E_total.item(),
            'U_rms': U_rms.item(),
            'omega_rms': omega_rms.item(),
            'T_L': T_L.item(),
            'k_peak': k_peak.item(),
            'd_vv_estimate': d_vv_estimate.item() if isinstance(d_vv_estimate, torch.Tensor) else d_vv_estimate
        }
        return diags, (self.k_values.cpu().numpy(), E_k.cpu().numpy())

    def run_simulation(self):
        """Executes the full simulation, including spinup and production phases."""
        print("--- Starting Simulation ---")
        
        print("--- Starting Spinup Phase ---")
        start_time_spinup = time.time()
        last_print_time = self.t
        while self.t < self.T_spinup_fixed:
            u_hat, _ = self._compute_velocity_hat(self.omega_hat)
            u = torch.fft.irfft2(u_hat, s=(self.N, self.N))
            U_max = torch.max(torch.abs(u))
            dt = self.CFL * self.dx / U_max if U_max > 0 else 0.1
            
            self._step(dt)
            
            if self.t - last_print_time >= 1.0:
                diags, _ = self._compute_diagnostics()
                elapsed = time.time() - start_time_spinup
                eta = (elapsed / (self.t / self.T_spinup_fixed)) * (1 - self.t / self.T_spinup_fixed) if self.t > 0 else 0
                print("Spinup: t={:.2f}/{:.2f}, U_rms={:.3f}, T_L={:.3f}, dt={:.2e}, ETA: {:.0f}s".format(
                    self.t, self.T_spinup_fixed, diags['U_rms'], diags['T_L'], dt, eta))
                last_print_time = self.t
        
        print("--- Spinup Phase Complete ---")

        diags, _ = self._compute_diagnostics()
        T_L_at_start_of_production = diags['T_L']
        T_prod = 50 * T_L_at_start_of_production
        self.params['T_L_at_start_of_production'] = T_L_at_start_of_production
        self.params['T_prod'] = T_prod
        
        print("T_L at end of spinup: {:.3f}".format(T_L_at_start_of_production))
        print("Production run duration: {:.3f}".format(T_prod))

        if diags['U_rms'] * self.dt_snap >= 0.1:
            print("WARNING: U_rms * dt_snap = {:.3f} >= 0.1. Snapshot interval may be too coarse.".format(
                diags['U_rms'] * self.dt_snap))
        
        self.tracer_pos = torch.rand((self.N_tracers, 2), device=self.device) * self.L
        
        print("--- Starting Production Phase ---")
        t_start_prod = self.t
        t_end_prod = t_start_prod + T_prod
        start_time_prod = time.time()
        last_print_time = self.t
        
        next_tracer_snap_time = self.t
        next_vel_snap_time = self.t
        
        tracer_pos_data = []
        tracer_vel_data = []
        tracer_times_data = []
        vel_snapshots_data = []
        vort_snapshots_data = []
        vel_times_data = []
        energy_spectrum_data = []
        diagnostics_list = []
        dt_list = []

        while self.t < t_end_prod:
            u_hat, v_hat = self._compute_velocity_hat(self.omega_hat)
            u = torch.fft.irfft2(u_hat, s=(self.N, self.N))
            v = torch.fft.irfft2(v_hat, s=(self.N, self.N))
            U_max = torch.max(torch.sqrt(u**2 + v**2))
            dt = self.CFL * self.dx / U_max if U_max > 0 else 0.1
            dt_list.append(dt.item())

            if self.t >= next_tracer_snap_time:
                diags, spec = self._compute_diagnostics()
                tracer_vel = self._interpolate_velocity_at_tracers(self.tracer_pos, u, v)
                
                tracer_pos_data.append(self.tracer_pos.cpu().numpy().astype(np.float32))
                tracer_vel_data.append(tracer_vel.cpu().numpy().astype(np.float32))
                tracer_times_data.append(self.t)
                energy_spectrum_data.append(spec[1])
                diagnostics_list.append(tuple(diags.values()))
                
                next_tracer_snap_time += self.dt_snap

            if self.t >= next_vel_snap_time:
                omega = torch.fft.irfft2(self.omega_hat, s=(self.N, self.N))
                
                coarse_pool = torch.nn.AvgPool2d(self.coarsening_factor)
                u_coarse = coarse_pool(u.unsqueeze(0).unsqueeze(0)).squeeze()
                v_coarse = coarse_pool(v.unsqueeze(0).unsqueeze(0)).squeeze()
                omega_coarse = coarse_pool(omega.unsqueeze(0).unsqueeze(0)).squeeze()
                
                vel_snapshots_data.append(torch.stack([u_coarse, v_coarse]).cpu().numpy().astype(np.float32))
                vort_snapshots_data.append(omega_coarse.cpu().numpy().astype(np.float32))
                vel_times_data.append(self.t)
                
                next_vel_snap_time += self.dt_vel

            self._step(dt)

            if self.t - last_print_time >= 5.0:
                progress = (self.t - t_start_prod) / T_prod
                elapsed = time.time() - start_time_prod
                eta = (elapsed / progress) * (1 - progress) if progress > 0 else 0
                print("Production: t={:.2f}/{:.2f} ({:.1f}%), U_rms={:.3f}, dt={:.2e}, ETA: {:.0f}s".format(
                    self.t - t_start_prod, T_prod, progress * 100, diags['U_rms'], dt, eta))
                last_print_time = self.t

        print("--- Production Phase Complete ---")
        
        print("--- Saving Data ---")
        self.params['dt_actual'] = np.mean(dt_list)
        self.params['N_tracer_snaps'] = len(tracer_times_data)
        self.params['N_vel_snaps'] = len(vel_times_data)

        np.save(os.path.join(self.output_path, 'tracer_positions.npy'), np.array(tracer_pos_data))
        np.save(os.path.join(self.output_path, 'tracer_velocities.npy'), np.array(tracer_vel_data))
        np.save(os.path.join(self.output_path, 'tracer_times.npy'), np.array(tracer_times_data, dtype=np.float64))
        np.save(os.path.join(self.output_path, 'velocity_snapshots.npy'), np.array(vel_snapshots_data))
        np.save(os.path.join(self.output_path, 'vorticity_snapshots.npy'), np.array(vort_snapshots_data))
        np.save(os.path.join(self.output_path, 'vel_times.npy'), np.array(vel_times_data, dtype=np.float64))
        np.save(os.path.join(self.output_path, 'energy_spectrum.npy'), np.array(energy_spectrum_data, dtype=np.float64))
        
        diag_dtype = [('time', 'f8'), ('E_total', 'f8'), ('U_rms', 'f8'), ('omega_rms', 'f8'), 
                      ('T_L', 'f8'), ('k_peak', 'f8'), ('d_vv_estimate', 'f8')]
        diagnostics_array = np.array(diagnostics_list, dtype=diag_dtype)
        np.save(os.path.join(self.output_path, 'diagnostics.npy'), diagnostics_array)

        with open(os.path.join(self.output_path, 'sim_params.json'), 'w') as f:
            json.dump(self.params, f, indent=4)
            
        print("Data saved to: " + self.output_path)
        print("--- Simulation Finished ---")


if __name__ == '__main__':
    simulation_parameters = {
        'N': 1024,
        'N_coarse': 256,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'output_path': '/home/node/work/projects/levy_flights_2dns_v2/data/',
        'CFL': 0.4,
        'nu_h': 1e-19,
        'p': 4,
        'epsilon_inj': 0.1,
        'k_force_min': 3.0,
        'k_force_max': 5.0,
        'N_tracers': 5000,
        'dt_snap': 0.05,
        'dt_vel': 2.0,
        'T_spinup_fixed': 60.0,
    }

    if simulation_parameters['device'] == 'cpu':
        print("WARNING: No CUDA device found. Running on CPU. This will be extremely slow.")

    solver = NavierStokes2DSolver(simulation_parameters)
    solver.run_simulation()