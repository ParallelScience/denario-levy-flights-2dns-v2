# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
sys.path.insert(0, "/home/node/data/compsep_data/")
sys.path.insert(0, "/home/node/data/compsep_data/")
import json
import os
import time
import sys

import torch
import numpy as np

class TwoDNS_Solver:
    """
    A GPU-accelerated 2D Navier-Stokes solver using pseudo-spectral methods.

    This class simulates the evolution of a 2D turbulent flow on a doubly
    periodic domain [0, 2*pi] x [0, 2*pi]. It uses the vorticity-streamfunction
    formulation and is accelerated using PyTorch on a CUDA-enabled GPU.

    The solver includes:
    - RK4 time integration with adaptive time stepping.
    - Stochastic white-in-time forcing in a specified wavenumber shell.
    - Hyperviscosity for dissipation at small scales.
    - De-aliasing using the 2/3 rule.
    - Advection of passive Lagrangian tracers.
    - Calculation and saving of various diagnostics and data snapshots.
    """

    def __init__(self, params):
        """
        Initializes the solver with simulation parameters.

        Args:
            params (dict): A dictionary of simulation parameters.
        """
        self.params = params
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device: " + str(self.device))

        self.N = params['N']
        self.L = 2 * np.pi
        self.dx = self.L / self.N
        
        k = torch.fft.fftfreq(self.N, d=self.dx) * self.L
        self.kx = k.view(-1, 1).to(self.device)
        self.ky = k.view(1, -1).to(self.device)
        
        # Use rfft for real-to-complex transform
        k_r = torch.fft.rfftfreq(self.N, d=self.dx) * self.L
        self.kx_r = k.view(-1, 1).to(self.device)
        self.ky_r = k_r.view(1, -1).to(self.device)
        self.k_sq_r = self.kx_r**2 + self.ky_r**2
        self.k_sq_r[0, 0] = 1.0

        self.dealias_mask = (torch.abs(self.kx_r) < (2/3) * (self.N/2)) & \
                            (torch.abs(self.ky_r) < (2/3) * (self.N/2))

        self.nu_h = params['nu_h']
        self.p = params['p']
        self.dissipation_op = -self.nu_h * (self.k_sq_r)**self.p

        self.epsilon_inj = params['epsilon_inj']
        self.kf_min = params['kf_min']
        self.kf_max = params['kf_max']
        self.forcing_mask = (self.k_sq_r >= self.kf_min**2) & (self.k_sq_r < self.kf_max**2)
        self.num_forcing_modes = self.forcing_mask.sum()
        self.forcing_scaling = torch.sqrt(2 * self.epsilon_inj * self.N**2 / self.num_forcing_modes)

        self.N_tracers = params['N_tracers']
        self.tracers_pos = None

        self.omega = None
        self.omega_hat = None
        self.t = 0.0
        self.dt = params['dt_initial']
        self.cfl = params['cfl']

    def _initialize_fields(self):
        """Initializes the vorticity field."""
        self.omega = torch.randn(self.N, self.N, device=self.device) * 1e-3
        self.omega_hat = torch.fft.rfft2(self.omega)
        self.omega_hat[0, 0] = 0.0
        self.omega_hat *= self.dealias_mask

    def _initialize_tracers(self):
        """Initializes tracer positions uniformly."""
        self.tracers_pos = torch.rand(self.N_tracers, 2, device=self.device) * self.L

    def _compute_velocity(self, omega_hat):
        """Computes velocity field (u, v) from vorticity spectrum."""
        psi_hat = -omega_hat / self.k_sq_r
        psi_hat[0, 0] = 0.0
        
        u_hat = 1j * self.ky_r * psi_hat
        v_hat = -1j * self.kx_r * psi_hat
        
        u = torch.fft.irfft2(u_hat, s=(self.N, self.N))
        v = torch.fft.irfft2(v_hat, s=(self.N, self.N))
        return u, v

    def _compute_rhs(self, omega_hat, dt_force):
        """Computes the right-hand side of the vorticity equation."""
        u, v = self._compute_velocity(omega_hat)
        self.omega = torch.fft.irfft2(omega_hat, s=(self.N, self.N))
        
        grad_omega_x = torch.fft.irfft2(1j * self.kx_r * omega_hat, s=(self.N, self.N))
        grad_omega_y = torch.fft.irfft2(1j * self.ky_r * omega_hat, s=(self.N, self.N))
        
        advection = u * grad_omega_x + v * grad_omega_y
        advection_hat = torch.fft.rfft2(advection)
        advection_hat *= self.dealias_mask

        force_hat = torch.zeros_like(omega_hat)
        phases = torch.exp(2j * np.pi * torch.rand(self.num_forcing_modes, device=self.device))
        force_hat[self.forcing_mask] = self.forcing_scaling * phases / torch.sqrt(torch.tensor(dt_force, device=self.device))
        
        dissipation_hat = self.dissipation_op * omega_hat
        
        return -advection_hat + dissipation_hat + force_hat

    def _get_velocity_at_pos(self, positions, u_grid, v_grid):
        """Interpolates velocity at tracer positions."""
        x_pos, y_pos = positions[:, 0], positions[:, 1]
        x_idx = (x_pos / self.dx)
        y_idx = (y_pos / self.dx)
        
        x_floor, y_floor = x_idx.long(), y_idx.long()
        x_frac, y_frac = x_idx - x_floor, y_idx - y_floor
        
        x0, x1 = x_floor % self.N, (x_floor + 1) % self.N
        y0, y1 = y_floor % self.N, (y_floor + 1) % self.N
        
        u11, u12 = u_grid[x0, y0], u_grid[x0, y1]
        u21, u22 = u_grid[x1, y0], u_grid[x1, y1]
        
        v11, v12 = v_grid[x0, y0], v_grid[x0, y1]
        v21, v22 = v_grid[x1, y0], v_grid[x1, y1]
        
        u_interp = (1-x_frac)*(1-y_frac)*u11 + x_frac*(1-y_frac)*u21 + (1-x_frac)*y_frac*u12 + x_frac*y_frac*u22
        v_interp = (1-x_frac)*(1-y_frac)*v11 + x_frac*(1-y_frac)*v21 + (1-x_frac)*y_frac*v12 + x_frac*y_frac*v22
                       
        return torch.stack([u_interp, v_interp], dim=1)

    def _update_tracers(self, dt):
        """Advects tracers using RK4 and bilinear interpolation."""
        u_grid, v_grid = self._compute_velocity(self.omega_hat)
        
        k1 = self._get_velocity_at_pos(self.tracers_pos, u_grid, v_grid)
        k2 = self._get_velocity_at_pos(self.tracers_pos + 0.5 * dt * k1, u_grid, v_grid)
        k3 = self._get_velocity_at_pos(self.tracers_pos + 0.5 * dt * k2, u_grid, v_grid)
        k4 = self._get_velocity_at_pos(self.tracers_pos + dt * k3, u_grid, v_grid)
        
        self.tracers_pos += (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        self.tracers_pos %= self.L

    def _calculate_diagnostics(self):
        """Calculates and returns various flow diagnostics."""
        u, v = self._compute_velocity(self.omega_hat)
        u_rms = torch.sqrt(torch.mean(u**2 + v**2)).item()
        
        k_magnitudes = torch.sqrt(self.k_sq_r)
        energy_spectrum_hat = 0.5 * (torch.abs(torch.fft.rfft2(u))**2 + torch.abs(torch.fft.rfft2(v))**2) / (self.N**4)
        
        k_bins = torch.arange(0.5, self.N // 2 + 1.5, 1.0, device=self.device)
        E_k = torch.zeros(len(k_bins) - 1, device=self.device)
        
        for i in range(len(k_bins) - 1):
            mask = (k_magnitudes >= k_bins[i]) & (k_magnitudes < k_bins[i+1])
            E_k[i] = energy_spectrum_hat[mask].sum()
            
        total_energy = E_k.sum().item() * 2 # Factor of 2 for rfft
        k_peak = (k_bins[:-1] + 0.5)[torch.argmax(E_k)].item() if total_energy > 0 else 0.0
        T_L = self.L / u_rms if u_rms > 0 else np.inf
        
        return {"T_L": T_L, "k_peak": k_peak, "E_k": E_k.cpu().numpy()}

    def run_simulation(self):
        """Main simulation loop for spinup and production."""
        output_path = self.params['output_path']
        os.makedirs(output_path, exist_ok=True)
        
        self._initialize_fields()
        
        print("--- Starting Spinup ---")
        T_L = np.inf
        spinup_start_time = time.time()
        while self.t / T_L < self.params['T_spinup_in_TL']:
            k1 = self._compute_rhs(self.omega_hat, self.dt)
            k2 = self._compute_rhs(self.omega_hat + 0.5 * self.dt * k1, self.dt)
            k3 = self._compute_rhs(self.omega_hat + 0.5 * self.dt * k2, self.dt)
            k4 = self._compute_rhs(self.omega_hat + self.dt * k3, self.dt)
            self.omega_hat += (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            self.t += self.dt
            
            if int(self.t / self.dt) % 20 == 0:
                u, v = self._compute_velocity(self.omega_hat)
                u_max = torch.max(torch.sqrt(u**2 + v**2))
                self.dt = min(self.params['dt_initial'], self.cfl * self.dx / u_max.item())
                diags = self._calculate_diagnostics()
                T_L = diags['T_L']
                print("Spinup t=" + "{:.2f}".format(self.t) + ", T_L=" + "{:.2f}".format(T_L) + ", t/T_L=" + "{:.2f}".format(self.t/T_L) + ", k_peak=" + "{:.2f}".format(diags['k_peak']))
        
        print("--- Spinup Complete ---")
        print("Time taken: " + "{:.2f}".format(time.time() - spinup_start_time) + "s")
        
        T_L_final_spinup = T_L
        T_prod = self.params['T_prod_in_TL'] * T_L_final_spinup
        self.params['T_L_at_start_of_production'] = T_L_final_spinup
        self.params['T_prod'] = T_prod
        
        self._initialize_tracers()
        
        # Data storage
        N_tracer_snaps = int(T_prod / self.params['dt_snap']) + 1
        N_vel_snaps = int(T_prod / self.params['dt_vel']) + 1
        
        tracer_pos_data = np.zeros((N_tracer_snaps, self.N_tracers, 2), dtype=np.float32)
        tracer_vel_data = np.zeros((N_tracer_snaps, self.N_tracers, 2), dtype=np.float32)
        tracer_times = np.zeros(N_tracer_snaps, dtype=np.float64)
        vel_times = np.zeros(N_vel_snaps, dtype=np.float64)
        
        coarsen_factor = self.N // 256
        vort_snaps = np.zeros((N_vel_snaps, 256, 256), dtype=np.float32)
        vel_snaps = np.zeros((N_vel_snaps, 2, 256, 256), dtype=np.float32)
        
        energy_spectra = np.zeros((N_tracer_snaps, self.N // 2), dtype=np.float64)
        diagnostics_dtype = [('time', 'f8'), ('E_total', 'f8'), ('E_rms', 'f8'), ('omega_rms', 'f8'), ('T_L', 'f8'), ('k_peak', 'f8'), ('d_vv_estimate', 'f8')]
        diagnostics_data = np.zeros(N_tracer_snaps, dtype=diagnostics_dtype)

        t_start_prod = self.t
        next_tracer_snap_time = self.t
        next_vel_snap_time = self.t
        tracer_snap_idx, vel_snap_idx = 0, 0
        
        print("--- Starting Production ---")
        prod_start_time = time.time()
        while self.t < t_start_prod + T_prod:
            if self.t >= next_tracer_snap_time:
                u_grid, v_grid = self._compute_velocity(self.omega_hat)
                tracer_vel = self._get_velocity_at_pos(self.tracers_pos, u_grid, v_grid)
                
                tracer_pos_data[tracer_snap_idx] = self.tracers_pos.cpu().numpy()
                tracer_vel_data[tracer_snap_idx] = tracer_vel.cpu().numpy()
                tracer_times[tracer_snap_idx] = self.t
                
                diags = self._calculate_diagnostics()
                energy_spectra[tracer_snap_idx] = diags['E_k']
                diagnostics_data[tracer_snap_idx] = (self.t, 0, 0, 0, diags['T_L'], diags['k_peak'], 0) # Some diags are placeholders
                
                print("Prod t=" + "{:.2f}".format(self.t) + ", Tracer snap " + str(tracer_snap_idx+1) + "/" + str(N_tracer_snaps))
                next_tracer_snap_time += self.params['dt_snap']
                tracer_snap_idx += 1

            if self.t >= next_vel_snap_time:
                u, v = self._compute_velocity(self.omega_hat)
                self.omega = torch.fft.irfft2(self.omega_hat, s=(self.N, self.N))
                
                vort_snaps[vel_snap_idx] = self.omega[::coarsen_factor, ::coarsen_factor].cpu().numpy()
                vel_snaps[vel_snap_idx, 0] = u[::coarsen_factor, ::coarsen_factor].cpu().numpy()
                vel_snaps[vel_snap_idx, 1] = v[::coarsen_factor, ::coarsen_factor].cpu().numpy()
                vel_times[vel_snap_idx] = self.t
                
                print("Prod t=" + "{:.2f}".format(self.t) + ", Velocity snap " + str(vel_snap_idx+1) + "/" + str(N_vel_snaps))
                next_vel_snap_time += self.params['dt_vel']
                vel_snap_idx += 1

            k1 = self._compute_rhs(self.omega_hat, self.dt)
            k2 = self._compute_rhs(self.omega_hat + 0.5 * self.dt * k1, self.dt)
            k3 = self._compute_rhs(self.omega_hat + 0.5 * self.dt * k2, self.dt)
            k4 = self._compute_rhs(self.omega_hat + self.dt * k3, self.dt)
            self.omega_hat += (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            
            self._update_tracers(self.dt)
            self.t += self.dt
            
            u, v = self._compute_velocity(self.omega_hat)
            u_max = torch.max(torch.sqrt(u**2 + v**2))
            self.dt = min(self.params['dt_initial'], self.cfl * self.dx / u_max.item())

        print("--- Production Complete ---")
        print("Time taken: " + "{:.2f}".format(time.time() - prod_start_time) + "s")

        # Save data
        print("--- Saving Data ---")
        np.save(os.path.join(output_path, 'tracer_positions.npy'), tracer_pos_data[:tracer_snap_idx])
        np.save(os.path.join(output_path, 'tracer_velocities.npy'), tracer_vel_data[:tracer_snap_idx])
        np.save(os.path.join(output_path, 'tracer_times.npy'), tracer_times[:tracer_snap_idx])
        np.save(os.path.join(output_path, 'vel_times.npy'), vel_times[:vel_snap_idx])
        np.save(os.path.join(output_path, 'vorticity_snapshots.npy'), vort_snaps[:vel_snap_idx])
        np.save(os.path.join(output_path, 'velocity_snapshots.npy'), vel_snaps[:vel_snap_idx])
        np.save(os.path.join(output_path, 'energy_spectrum.npy'), energy_spectra[:tracer_snap_idx])
        np.save(os.path.join(output_path, 'diagnostics.npy'), diagnostics_data[:tracer_snap_idx])
        
        self.params['N_tracer_snaps'] = tracer_snap_idx
        self.params['N_vel_snaps'] = vel_snap_idx
        with open(os.path.join(output_path, 'sim_params.json'), 'w') as f:
            json.dump(self.params, f, indent=4)
            
        print("All data saved to " + output_path)


if __name__ == '__main__':
    sim_params = {
        'N': 1024,
        'nu_h': 1e-19,
        'p': 4,
        'epsilon_inj': 0.1,
        'kf_min': 3,
        'kf_max': 5,
        'dt_initial': 1e-3,
        'cfl': 0.4,
        'N_tracers': 5000,
        'T_spinup_in_TL': 10.0,
        'T_prod_in_TL': 50.0,
        'dt_snap': 0.05,
        'dt_vel': 2.0,
        'output_path': '/home/node/work/projects/levy_flights_2dns_v2/data/'
    }

    if not torch.cuda.is_available():
        print("FATAL ERROR: This simulation requires a CUDA-enabled GPU.")
        sys.exit(1)

    solver = TwoDNS_Solver(sim_params)
    solver.run_simulation()