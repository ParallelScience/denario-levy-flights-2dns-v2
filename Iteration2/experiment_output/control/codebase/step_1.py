# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import torch
import numpy as np
import json
import os
import time


def run_simulation():
    """
    Main function to run the 2D Navier-Stokes simulation.

    This function orchestrates the entire simulation process, including setting up
    parameters, calibrating the forcing, running the spinup and production phases,
    and saving the final data.
    """
    # --- Simulation Parameters ---
    N = 512
    L = 2.0 * np.pi
    dx = L / N
    N_coarse = 128
    nu_h = 1e-28
    p = 4
    epsilon_inj = 0.1
    k_force_min = 3
    k_force_max = 5
    CFL = 0.4
    T_spinup_fixed = 10.0
    N_tracers = 5000
    dt_snap = 0.01
    dt_vel = 0.5
    output_dir = "/home/node/work/projects/levy_flights_2dns_v2/data/"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # --- Device and Data Type Setup ---
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("CUDA is available. Using GPU.")
    else:
        device = torch.device('cpu')
        print("CUDA not available. Using CPU.")
    
    dtype_c = torch.cfloat
    dtype_f = torch.float

    # --- Wavenumber Grids ---
    k = torch.fft.fftfreq(N, d=1.0/N).to(device)
    kr = torch.fft.rfftfreq(N, d=1.0/N).to(device)
    kx, ky = torch.meshgrid(k, kr, indexing='ij')
    k2 = kx**2 + ky**2
    km = torch.sqrt(k2)
    k2_nz = k2.clone()
    k2_nz[0, 0] = 1.0

    # --- Dealiasing Mask ---
    k_max_dealias = N / 3.0
    dealias_mask = (torch.abs(kx) < k_max_dealias) & (torch.abs(ky) < k_max_dealias)

    # --- Forcing Mask ---
    force_mask = ((km >= k_force_min) & (km <= k_force_max)).to(dtype_c)
    force_mask *= dealias_mask

    # --- Dissipation Operator ---
    diss_op = -nu_h * (k2**p)

    # --- Helper Functions ---
    def get_energy(omega_hat):
        """Computes total kinetic energy from vorticity in Fourier space."""
        psi_hat = -omega_hat / k2_nz
        u_hat = 1j * ky * psi_hat
        v_hat = -1j * kx * psi_hat
        energy = 0.5 * (torch.sum(torch.abs(u_hat)**2) + torch.sum(torch.abs(v_hat)**2)) / (N**4)
        return energy.item()

    def get_velocity(omega_hat):
        """Computes velocity fields from vorticity in Fourier space."""
        psi_hat = -omega_hat / k2_nz
        u_hat = 1j * ky * psi_hat
        v_hat = -1j * kx * psi_hat
        u = torch.fft.irfft2(u_hat, s=(N, N))
        v = torch.fft.irfft2(v_hat, s=(N, N))
        return u, v

    def get_dt(u, v):
        """Computes adaptive time step based on CFL condition."""
        U_max = torch.sqrt(u**2 + v**2).max().item()
        return min(CFL * dx / U_max, 0.05) if U_max > 1e-8 else 0.05

    def nonlinear(omega_hat):
        """Computes the nonlinear advection term in Fourier space."""
        omega_hat_dealiased = omega_hat * dealias_mask
        u, v = get_velocity(omega_hat_dealiased)
        omega = torch.fft.irfft2(omega_hat_dealiased, s=(N, N))
        
        adv_x = u * torch.fft.irfft2(1j * kx * omega_hat_dealiased, s=(N, N))
        adv_y = v * torch.fft.irfft2(1j * ky * omega_hat_dealiased, s=(N, N))
        
        n_hat = torch.fft.rfft2(adv_x + adv_y)
        return -n_hat * dealias_mask

    def rk4_if_step(omega_hat, dt, IF, force_amp):
        """Performs one RK4 step with integrating factor for the fluid."""
        k1 = nonlinear(omega_hat)
        k2 = nonlinear(omega_hat + 0.5 * dt * k1)
        k3 = nonlinear(omega_hat + 0.5 * dt * k2)
        k4 = nonlinear(omega_hat + dt * k3)
        
        omega_hat_new = IF * (omega_hat + dt * (k1 + 2*k2 + 2*k3 + k4) / 6.0)
        
        phi = torch.exp(2j * np.pi * torch.rand(omega_hat.shape, device=device, dtype=dtype_f))
        forcing_term = (force_amp * torch.sqrt(torch.tensor(dt, device=device))) * phi * force_mask
        
        return omega_hat_new + forcing_term

    def get_tracer_velocity(tracer_pos, u, v):
        """Interpolates velocity at tracer positions."""
        pos_norm = (tracer_pos / L) * 2.0 - 1.0
        pos_norm = pos_norm.unsqueeze(0).unsqueeze(0)
        
        u_grid = u.unsqueeze(0).unsqueeze(0)
        v_grid = v.unsqueeze(0).unsqueeze(0)
        
        u_interp = torch.nn.functional.grid_sample(u_grid, pos_norm, align_corners=True).squeeze()
        v_interp = torch.nn.functional.grid_sample(v_grid, pos_norm, align_corners=True).squeeze()
        
        return torch.stack([u_interp, v_interp], dim=1)

    def rk4_tracer_step(tracer_pos, u, v, dt):
        """Performs one RK4 step for tracers."""
        v1 = get_tracer_velocity(tracer_pos, u, v)
        k1 = dt * v1
        
        v2 = get_tracer_velocity(tracer_pos + 0.5 * k1, u, v)
        k2 = dt * v2
        
        v3 = get_tracer_velocity(tracer_pos + 0.5 * k2, u, v)
        k3 = dt * v3
        
        v4 = get_tracer_velocity(tracer_pos + k3, u, v)
        k4 = dt * v4
        
        new_pos = tracer_pos + (k1 + 2*k2 + 2*k3 + k4) / 6.0
        return new_pos % L

    def get_energy_spectrum(omega_hat):
        """Computes the 1D azimuthally averaged energy spectrum."""
        psi_hat = -omega_hat / k2_nz
        u_hat = 1j * ky * psi_hat
        v_hat = -1j * kx * psi_hat
        energy_spec_2d = 0.5 * (torch.abs(u_hat)**2 + torch.abs(v_hat)**2) / (N**4)
        
        k_bins = torch.arange(0.5, N//2 + 1.5, 1.0, device=device)
        energy_spec_1d = torch.zeros(N//2 + 1, device=device)
        
        k_indices = torch.round(km).long()
        
        for k_val in range(1, N//2 + 1):
            mask = (k_indices == k_val)
            if mask.any():
                energy_spec_1d[k_val] = torch.sum(energy_spec_2d[mask])
                
        return energy_spec_1d[1:].cpu().numpy()

    # --- Force Calibration ---
    print("Calibrating forcing amplitude...")
    omega_calib = torch.zeros((N, N//2 + 1), dtype=dtype_c, device=device)
    dE_acc = 0.0
    dt_cal = 0.01
    n_cal_steps = 100
    for _ in range(n_cal_steps):
        E0 = get_energy(omega_calib)
        phi = torch.exp(2j * np.pi * torch.rand(omega_calib.shape, device=device, dtype=dtype_f))
        omega_calib += torch.sqrt(torch.tensor(dt_cal, device=device)) * phi * force_mask
        dE_acc += get_energy(omega_calib) - E0
    
    dE_per_amp2 = dE_acc / (n_cal_steps * dt_cal)
    force_amp = float(np.sqrt(epsilon_inj / dE_per_amp2))
    print("Calibration complete.")
    print("Calibrated force_amp = " + str(force_amp))
    print("Expected dE/dt = " + str(epsilon_inj) + ", Calibrated dE/dt = " + str(force_amp**2 * dE_per_amp2))

    # --- Spinup Phase ---
    print("Starting spinup phase (T_spinup = " + str(T_spinup_fixed) + ")...")
    t = 0.0
    omega_hat = torch.zeros((N, N//2 + 1), dtype=dtype_c, device=device)
    last_dt = -1.0
    IF = None
    
    start_time = time.time()
    while t < T_spinup_fixed:
        u, v = get_velocity(omega_hat)
        dt = get_dt(u, v)
        
        if abs(dt - last_dt) / dt > 0.01:
            IF = torch.exp(diss_op * dt)
            last_dt = dt
            
        omega_hat = rk4_if_step(omega_hat, dt, IF, force_amp)
        t += dt
        
        if int(t) > int(t-dt):
             print("Spinup time: " + str(round(t, 2)) + "/" + str(T_spinup_fixed))

    print("Spinup finished in " + str(round(time.time() - start_time, 2)) + " seconds.")

    # --- Post-Spinup / Pre-Production Setup ---
    u, v = get_velocity(omega_hat)
    E_total = get_energy(omega_hat)
    U_rms = np.sqrt(2.0 * E_total)
    T_L = L / U_rms if U_rms > 0 else float('inf')
    T_prod = 50.0 * T_L
    
    print("\n--- Post-Spinup Diagnostics ---")
    print("U_rms = " + str(U_rms))
    print("T_L = " + str(T_L))
    print("Production run duration T_prod = " + str(T_prod))
    
    displacement_check = U_rms * dt_snap
    print("Verification: U_rms * dt_snap = " + str(displacement_check))
    if displacement_check >= 0.1:
        print("WARNING: U_rms * dt_snap >= 0.1. Tracer displacement per step might be too large.")
    else:
        print("Verification successful: U_rms * dt_snap < 0.1.")

    # --- Production Phase ---
    print("\nStarting production phase (T_prod = " + str(round(T_prod, 2)) + ")...")
    tracer_pos = torch.rand((N_tracers, 2), device=device, dtype=dtype_f) * L
    
    tracer_positions_list = []
    tracer_velocities_list = []
    tracer_times_list = []
    vel_snapshots_list = []
    vort_snapshots_list = []
    vel_times_list = []
    energy_spectrum_list = []
    diagnostics_list = []

    t = 0.0
    t_last_snap = -dt_snap
    t_last_vel = -dt_vel
    start_time = time.time()
    
    while t < T_prod:
        u, v = get_velocity(omega_hat)
        dt = get_dt(u, v)
        
        if abs(dt - last_dt) / dt > 0.01:
            IF = torch.exp(diss_op * dt)
            last_dt = dt

        tracer_pos = rk4_tracer_step(tracer_pos, u, v, dt)
        omega_hat = rk4_if_step(omega_hat, dt, IF, force_amp)
        t += dt

        if torch.any(torch.isnan(omega_hat)) or torch.any(torch.isinf(omega_hat)):
            raise RuntimeError("NaN/Inf detected in vorticity field at t=" + str(t))

        if t >= t_last_snap + dt_snap:
            t_last_snap += dt_snap
            
            u_snap, v_snap = get_velocity(omega_hat)
            tracer_vel = get_tracer_velocity(tracer_pos, u_snap, v_snap)
            
            tracer_positions_list.append(tracer_pos.cpu().numpy().astype(np.float32))
            tracer_velocities_list.append(tracer_vel.cpu().numpy().astype(np.float32))
            tracer_times_list.append(t)
            
            E_total = get_energy(omega_hat)
            U_rms = np.sqrt(2.0 * E_total)
            omega = torch.fft.irfft2(omega_hat, s=(N, N))
            omega_rms = torch.sqrt(torch.mean(omega**2)).item()
            T_L_current = L / U_rms if U_rms > 0 else float('inf')
            
            E_spec = get_energy_spectrum(omega_hat)
            k_peak = np.argmax(E_spec) + 1
            energy_spectrum_list.append(E_spec)
            
            d_vv_estimate = 0.0 
            
            diagnostics_list.append((t, E_total, U_rms, omega_rms, T_L_current, k_peak, d_vv_estimate))
            
            if int(t / T_L) > int((t-dt) / T_L):
                print("Production time: " + str(round(t, 2)) + "/" + str(round(T_prod, 2)) + " (T/T_L_initial = " + str(round(t/T_L, 1)) + ")")

        if t >= t_last_vel + dt_vel:
            t_last_vel += dt_vel
            
            u_field, v_field = get_velocity(omega_hat)
            omega_field = torch.fft.irfft2(omega_hat, s=(N, N))
            
            pool_kernel = N // N_coarse
            u_coarse = torch.nn.functional.avg_pool2d(u_field.unsqueeze(0), pool_kernel).squeeze(0)
            v_coarse = torch.nn.functional.avg_pool2d(v_field.unsqueeze(0), pool_kernel).squeeze(0)
            omega_coarse = torch.nn.functional.avg_pool2d(omega_field.unsqueeze(0), pool_kernel).squeeze(0)
            
            vel_snapshots_list.append(torch.stack([u_coarse, v_coarse]).cpu().numpy().astype(np.float32))
            vort_snapshots_list.append(omega_coarse.cpu().numpy().astype(np.float32))
            vel_times_list.append(t)

    print("Production run finished in " + str(round(time.time() - start_time, 2)) + " seconds.")

    print("Saving data...")
    
    tracer_positions = np.array(tracer_positions_list)
    tracer_velocities = np.array(tracer_velocities_list)
    tracer_times = np.array(tracer_times_list, dtype=np.float64)
    velocity_snapshots = np.array(vel_snapshots_list)
    vorticity_snapshots = np.array(vort_snapshots_list)
    vel_times = np.array(vel_times_list, dtype=np.float64)
    energy_spectrum = np.array(energy_spectrum_list, dtype=np.float64)
    
    diag_dtype = np.dtype([('time', 'f8'), ('E_total', 'f8'), ('E_rms', 'f8'), 
                           ('omega_rms', 'f8'), ('T_L', 'f8'), ('k_peak', 'f8'), 
                           ('d_vv_estimate', 'f8')])
    diagnostics = np.array(diagnostics_list, dtype=diag_dtype)

    np.save(os.path.join(output_dir, 'tracer_positions.npy'), tracer_positions)
    np.save(os.path.join(output_dir, 'tracer_velocities.npy'), tracer_velocities)
    np.save(os.path.join(output_dir, 'tracer_times.npy'), tracer_times)
    np.save(os.path.join(output_dir, 'velocity_snapshots.npy'), velocity_snapshots)
    np.save(os.path.join(output_dir, 'vorticity_snapshots.npy'), vorticity_snapshots)
    np.save(os.path.join(output_dir, 'vel_times.npy'), vel_times)
    np.save(os.path.join(output_dir, 'energy_spectrum.npy'), energy_spectrum)
    np.save(os.path.join(output_dir, 'diagnostics.npy'), diagnostics)

    sim_params = {
        'N': N,
        'L': L,
        'nu_h': nu_h,
        'p': p,
        'epsilon_inj': epsilon_inj,
        'k_force_min': k_force_min,
        'k_force_max': k_force_max,
        'force_amp': force_amp,
        'CFL': CFL,
        'T_spinup_fixed': T_spinup_fixed,
        'N_tracers': N_tracers,
        'dt_snap': dt_snap,
        'dt_vel': dt_vel,
        'T_L_at_start_of_production': T_L,
        'T_prod_calculated': T_prod,
        'N_tracer_snaps': len(tracer_times),
        'N_vel_snaps': len(vel_times)
    }
    with open(os.path.join(output_dir, 'sim_params.json'), 'w') as f:
        json.dump(sim_params, f, indent=4)

    print("All data saved successfully to " + output_dir)
    print("Simulation finished.")

if __name__ == '__main__':
    run_simulation()