# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import json
import time
import numpy as np
import torch

class TwoDNS:
    """
    A GPU-accelerated pseudo-spectral solver for 2D Navier-Stokes equations
    with passive tracer advection.
    """
    def __init__(self, params):
        """
        Initializes the simulation environment.

        Args:
            params (dict): A dictionary of simulation parameters.
        """
        self.params = params
        self.N = params['N']
        self.device = torch.device(params['device'])
        self.dtype = torch.float32

        self.setup_grid()
        self.setup_operators()
        self.setup_output_paths()

        self.omega_h = self.initialize_field()
        self.tracer_pos = None

        self.t = 0.0
        self.step = 0
        self.next_tracer_snap = 0.0
        self.next_vel_snap = 0.0
        
        self.history = {
            'tracer_positions': [], 'tracer_velocities': [], 'tracer_times': [],
            'vorticity_snapshots': [], 'velocity_snapshots': [], 'vel_times': [],
            'energy_spectrum': [], 'diagnostics': []
        }

    def setup_grid(self):
        """Sets up the spatial and spectral grids."""
        self.L = 2.0 * np.pi
        self.dx = self.L / self.N
        
        k = torch.fft.fftfreq(self.N, d=self.dx) * self.L
        self.k_max_dealias = int(self.N * (2/3.) / 2)
        
        kx = k.reshape(self.N, 1)
        ky = k.reshape(1, self.N)
        self.k_squared = (kx**2 + ky**2).to(self.device)
        self.k_squared[0, 0] = 1e-16 

        self.k_abs = torch.sqrt(self.k_squared)
        
        self.dealias_mask = (torch.abs(kx) < self.k_max_dealias) * \
                            (torch.abs(ky) < self.k_max_dealias)
        self.dealias_mask = self.dealias_mask.to(self.device)

    def setup_operators(self):
        """Sets up spectral operators for dissipation and forcing."""
        p = self.params['p']
        nu_h = self.params['nu_h']
        self.dissipation = -nu_h * (self.k_squared**p)

        k_f_min = self.params['k_f_min']
        k_f_max = self.params['k_f_max']
        self.forcing_mask = ((self.k_abs >= k_f_min) & (self.k_abs <= k_f_max)).to(self.device)
        self.num_forcing_modes = self.forcing_mask.sum().item()

    def setup_output_paths(self):
        """Defines absolute paths for all output files."""
        base = self.params['output_base_path']
        if not os.path.exists(base):
            os.makedirs(base)
        self.paths = {
            "params": os.path.join(base, 'sim_params.json'),
            "positions": os.path.join(base, 'tracer_positions.npy'),
            "velocities": os.path.join(base, 'tracer_velocities.npy'),
            "vorticity": os.path.join(base, 'vorticity_snapshots.npy'),
            "velocity": os.path.join(base, 'velocity_snapshots.npy'),
            "tracer_times": os.path.join(base, 'tracer_times.npy'),
            "vel_times": os.path.join(base, 'vel_times.npy'),
            "spectrum": os.path.join(base, 'energy_spectrum.npy'),
            "diagnostics": os.path.join(base, 'diagnostics.npy'),
        }

    def initialize_field(self):
        """Initializes the vorticity field in Fourier space."""
        omega_h = torch.randn(self.N, self.N, dtype=torch.complex64, device=self.device)
        omega_h *= self.dealias_mask
        return omega_h

    def _compute_velocity_h(self, omega_h):
        """Computes velocity components in Fourier space."""
        psi_h = -omega_h / self.k_squared
        k = torch.fft.fftfreq(self.N, d=self.dx) * self.L
        kx = k.reshape(self.N, 1).to(self.device)
        ky = k.reshape(1, self.N).to(self.device)
        u_h = 1j * ky * psi_h
        v_h = -1j * kx * psi_h
        return u_h, v_h

    def _compute_advection_h(self, omega_h):
        """Computes the advection term in Fourier space."""
        u_h, v_h = self._compute_velocity_h(omega_h)
        u = torch.fft.ifft2(u_h).real
        v = torch.fft.ifft2(v_h).real
        omega = torch.fft.ifft2(omega_h).real
        
        adv_x = torch.fft.fft2(u * omega)
        adv_y = torch.fft.fft2(v * omega)
        
        k = torch.fft.fftfreq(self.N, d=self.dx) * self.L
        kx = k.reshape(self.N, 1).to(self.device)
        ky = k.reshape(1, self.N).to(self.device)
        
        advection_h = 1j * kx * adv_x + 1j * ky * adv_y
        return advection_h * self.dealias_mask

    def _compute_forcing_h(self, dt):
        """Computes the stochastic forcing term."""
        epsilon_inj = self.params['epsilon_inj']
        force_ampl = np.sqrt(2 * epsilon_inj * self.num_forcing_modes / dt)
        
        f_h = torch.randn(self.N, self.N, dtype=torch.complex64, device=self.device)
        f_h *= self.forcing_mask
        f_h *= force_ampl / torch.sqrt(torch.sum(torch.abs(f_h)**2)) * self.num_forcing_modes
        return f_h

    def _rhs(self, omega_h, f_h):
        """Computes the right-hand side of the vorticity equation."""
        return -self._compute_advection_h(omega_h) + self.dissipation * omega_h + f_h

    def _compute_diagnostics(self, omega_h):
        """Computes and returns key physical diagnostics."""
        u_h, v_h = self._compute_velocity_h(omega_h)
        E_h = 0.5 * (torch.abs(u_h)**2 + torch.abs(v_h)**2) / self.L**2
        E_total = E_h.sum().item()
        U_rms = np.sqrt(2 * E_total)
        
        k_bins = torch.arange(0.5, self.N//2 + 1.5, 1.0, device=self.device)
        E_k, _ = np.histogram(self.k_abs.cpu().numpy().flatten(), 
                              bins=k_bins.cpu().numpy(), 
                              weights=(E_h * self.dealias_mask).cpu().numpy().flatten())
        
        k_peak = k_bins[np.argmax(E_k)].item()
        T_L = self.L / U_rms if U_rms > 0 else np.inf
        
        omega_rms = torch.sqrt(torch.mean(torch.fft.ifft2(omega_h).real**2)).item()
        
        return {
            'E_total': E_total, 'U_rms': U_rms, 'omega_rms': omega_rms,
            'T_L': T_L, 'k_peak': k_peak, 'E_k': E_k
        }

    def _interpolate_velocity_at_tracers(self, u_h, v_h, pos):
        """Interpolates velocity at tracer positions using spectral method."""
        vel = torch.zeros_like(pos)
        k = torch.fft.fftfreq(self.N, d=1.0/self.L).to(self.device)
        kx, ky = torch.meshgrid(k, k, indexing='xy')
        
        exp_ikx = torch.exp(1j * kx.unsqueeze(0) * pos[:, 0].unsqueeze(1).unsqueeze(2))
        exp_iky = torch.exp(1j * ky.unsqueeze(0) * pos[:, 1].unsqueeze(1).unsqueeze(2))
        
        u_at_pos = torch.sum(u_h.unsqueeze(0) * exp_ikx * exp_iky, dim=(1,2)) / self.N**2
        v_at_pos = torch.sum(v_h.unsqueeze(0) * exp_ikx * exp_iky, dim=(1,2)) / self.N**2
        
        vel[:, 0] = u_at_pos.real
        vel[:, 1] = v_at_pos.real
        return vel

    def _advect_tracers_rk4(self, dt):
        """Advects tracers for one dt_snap using RK4."""
        def get_vel(pos):
            u_h, v_h = self._compute_velocity_h(self.omega_h)
            return self._interpolate_velocity_at_tracers(u_h, v_h, pos)

        k1 = get_vel(self.tracer_pos)
        k2 = get_vel(self.tracer_pos + 0.5 * dt * k1)
        k3 = get_vel(self.tracer_pos + 0.5 * dt * k2)
        k4 = get_vel(self.tracer_pos + dt * k3)
        
        self.tracer_pos += (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        self.tracer_pos %= self.L

    def _coarsen(self, field, factor=4):
        """Coarsens a 2D field by averaging."""
        return field.reshape(self.N//factor, factor, self.N//factor, factor).mean(dim=(1,3))

    def run_simulation(self):
        """Main simulation loop including spin-up and production."""
        print("--- Starting Simulation ---")
        
        # Spin-up
        print("--- Spin-up Phase ---")
        T_L_current = np.inf
        while self.t / T_L_current < self.params['T_spinup_TL']:
            dt = self.params['CFL'] * self.dx / self.params['U_max_est']
            f_h = self._compute_forcing_h(dt)
            
            k1 = self._rhs(self.omega_h, f_h)
            k2 = self._rhs(self.omega_h + 0.5 * dt * k1, f_h)
            k3 = self._rhs(self.omega_h + 0.5 * dt * k2, f_h)
            k4 = self._rhs(self.omega_h + dt * k3, f_h)
            self.omega_h += (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            
            self.t += dt
            self.step += 1
            
            if self.step % 100 == 0:
                diags = self._compute_diagnostics(self.omega_h)
                T_L_current = diags['T_L']
                print("Spin-up t=" + "{:.2f}".format(self.t) + ", T_L=" + "{:.2f}".format(T_L_current) + ", t/T_L=" + "{:.2f}".format(self.t / T_L_current))

        print("Spin-up complete at t=" + "{:.2f}".format(self.t))
        self.params['T_spinup'] = self.t
        diags = self._compute_diagnostics(self.omega_h)
        self.params['T_L_at_start_of_production'] = diags['T_L']
        T_prod = self.params['T_prod_TL'] * diags['T_L']
        self.params['T_prod'] = T_prod
        
        # Production
        print("--- Production Phase ---")
        self.tracer_pos = torch.rand(self.params['N_tracers'], 2, device=self.device) * self.L
        self.t = 0.0
        self.step = 0
        self.next_tracer_snap = 0.0
        self.next_vel_snap = 0.0
        
        t_start_prod = time.time()
        while self.t < T_prod:
            dt = self.params['CFL'] * self.dx / self.params['U_max_est']
            f_h = self._compute_forcing_h(dt)
            
            k1 = self._rhs(self.omega_h, f_h)
            k2 = self._rhs(self.omega_h + 0.5 * dt * k1, f_h)
            k3 = self._rhs(self.omega_h + 0.5 * dt * k2, f_h)
            k4 = self._rhs(self.omega_h + dt * k3, f_h)
            self.omega_h += (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            
            if self.t >= self.next_tracer_snap:
                self._advect_tracers_rk4(self.params['dt_snap'])
                
                diags = self._compute_diagnostics(self.omega_h)
                u_h, v_h = self._compute_velocity_h(self.omega_h)
                tracer_vels = self._interpolate_velocity_at_tracers(u_h, v_h, self.tracer_pos)
                
                self.history['tracer_positions'].append(self.tracer_pos.cpu().numpy().astype(np.float32))
                self.history['tracer_velocities'].append(tracer_vels.cpu().numpy().astype(np.float32))
                self.history['tracer_times'].append(self.t)
                self.history['energy_spectrum'].append(diags['E_k'])
                
                diag_tuple = (self.t, diags['E_total'], diags['U_rms'], diags['omega_rms'], diags['T_L'], diags['k_peak'], 0.0)
                self.history['diagnostics'].append(diag_tuple)
                
                if self.t >= self.next_vel_snap:
                    omega = torch.fft.ifft2(self.omega_h).real
                    u = torch.fft.ifft2(u_h).real
                    v = torch.fft.ifft2(v_h).real
                    self.history['vorticity_snapshots'].append(self._coarsen(omega).cpu().numpy().astype(np.float32))
                    vel_snapshot = torch.stack([self._coarsen(u), self._coarsen(v)], dim=0)
                    self.history['velocity_snapshots'].append(vel_snapshot.cpu().numpy().astype(np.float32))
                    self.history['vel_times'].append(self.t)
                    self.next_vel_snap += self.params['dt_vel']

                self.next_tracer_snap += self.params['dt_snap']
                
                if self.step % 100 == 0:
                    elapsed = time.time() - t_start_prod
                    eta = (T_prod - self.t) * (elapsed / (self.t + 1e-9))
                    print("Prod t=" + "{:.2f}".format(self.t) + "/" + "{:.2f}".format(T_prod) + " (ETA: " + "{:.0f}".format(eta) + "s)")

            self.t += dt
            self.step += 1
        
        print("Production complete.")
        self.save_results()

    def save_results(self):
        """Saves all collected data to disk."""
        print("--- Saving Results ---")
        
        self.params['N_tracer_snaps'] = len(self.history['tracer_times'])
        self.params['N_vel_snaps'] = len(self.history['vel_times'])
        
        with open(self.paths['params'], 'w') as f:
            json.dump(self.params, f, indent=4)
        
        np.save(self.paths['positions'], np.array(self.history['tracer_positions']))
        np.save(self.paths['velocities'], np.array(self.history['tracer_velocities']))
        np.save(self.paths['tracer_times'], np.array(self.history['tracer_times'], dtype=np.float64))
        np.save(self.paths['vorticity'], np.array(self.history['vorticity_snapshots']))
        np.save(self.paths['velocity'], np.array(self.history['velocity_snapshots']))
        np.save(self.paths['vel_times'], np.array(self.history['vel_times'], dtype=np.float64))
        np.save(self.paths['spectrum'], np.array(self.history['energy_spectrum'], dtype=np.float64))
        
        diag_dtype = np.dtype([('time', 'f8'), ('E_total', 'f8'), ('E_rms', 'f8'), 
                               ('omega_rms', 'f8'), ('T_L', 'f8'), ('k_peak', 'f8'), 
                               ('d_vv_estimate', 'f8')])
        diagnostics_array = np.array(self.history['diagnostics'], dtype=diag_dtype)
        np.save(self.paths['diagnostics'], diagnostics_array)
        
        print("All data saved successfully to " + self.params['output_base_path'])


if __name__ == '__main__':
    sim_params = {
        'N': 1024,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'nu_h': 1e-19,
        'p': 4,
        'epsilon_inj': 0.1,
        'k_f_min': 3,
        'k_f_max': 5,
        'CFL': 0.4,
        'U_max_est': 3.0, 
        'T_spinup_TL': 10.0,
        'T_prod_TL': 50.0,
        'N_tracers': 5000,
        'dt_snap': 0.05,
        'dt_vel': 2.0,
        'output_base_path': '/home/node/work/projects/levy_flights_2dns_v2/data/'
    }
    
    if sim_params['device'] == 'cpu':
        print("Warning: CUDA not available. Running on CPU, which will be very slow.")

    solver = TwoDNS(sim_params)
    solver.run_simulation()