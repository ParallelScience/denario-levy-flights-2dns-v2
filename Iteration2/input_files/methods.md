0.  **Simulation Step — CRITICAL ENGINEERING NOTES (read every line before writing any code):**

    The engineer must implement the 2D NS simulation in step_1 with **exactly** these specifications. Deviations from any of these will cause either numerical blow-up or physically meaningless data.

    ### a. Grid and resolution
    - Use `N = 512` (not 1024). Grid: N×N, domain L = 2π.
    - `dx = L / N`

    ### b. Wavenumber convention
    Use integer wavenumbers (not physical wavenumbers divided by dx):
    ```python
    k  = torch.fft.fftfreq(N, d=1.0/N)   # integer wavenumbers: [0,1,...,N/2-1,-N/2,...,-1]
    kr = torch.fft.rfftfreq(N, d=1.0/N)  # [0, 1, ..., N/2]
    kx, ky = torch.meshgrid(k, kr, indexing='ij')
    k2 = kx**2 + ky**2
    km = torch.sqrt(k2)
    k2_nz = k2.clone(); k2_nz[0, 0] = 1.0  # avoid division by zero
    ```

    ### c. Hyperviscosity — MUST USE INTEGRATING FACTOR (IF) METHOD
    **DO NOT use explicit Euler or explicit RK4 for the hyperviscous term.** The dissipation operator
    `diss_op = -nu_h * k^(2p)` with `nu_h = 1e-28`, `p = 4` causes `nu_h * k_max^(2p) * dt >> 1`
    at the dealiasing cutoff `k_max = N/3 ≈ 170`, making any explicit scheme unconditionally unstable.

    The correct approach is the **integrating factor** (Lawson/exponential RK4):
    ```python
    IF = torch.exp(diss_op * dt)   # precompute; shape matches omega_hat

    def rk4_if(omega_hat, dt):
        # RK4 on nonlinear term only; dissipation handled exactly by IF
        k1 = nonlinear(omega_hat)
        k2 = nonlinear(omega_hat + 0.5*dt*k1)
        k3 = nonlinear(omega_hat + 0.5*dt*k2)
        k4 = nonlinear(omega_hat + dt*k3)
        omega_new = IF * (omega_hat + dt*(k1 + 2*k2 + 2*k3 + k4)/6)
        # Stochastic forcing: one kick per step (Ito convention)
        phi = torch.exp(2j * np.pi * torch.rand(omega_hat.shape, device=device))
        omega_new = omega_new + (force_amp * np.sqrt(dt)) * phi * force_mask
        return omega_new
    ```
    Cache `IF` and recompute only when `dt` changes by >1%.

    ### d. Forcing amplitude — MUST CALIBRATE EMPIRICALLY at startup
    The rfft2 convention (unnormalised) means the naive amplitude formula is wrong by ~700×.
    At startup, before the main loop, measure `dE_per_amp2` empirically and derive `force_amp`:
    ```python
    # Calibration: inject with amplitude=1 for 20 steps at dt=0.1, measure dE/dt
    omega_calib = torch.zeros(N, N//2+1, dtype=torch.cfloat, device=device)
    dE_acc = 0.0
    dt_cal = 0.1
    for _ in range(20):
        E0 = get_energy(omega_calib)
        phi = torch.exp(2j*np.pi*torch.rand(omega_calib.shape, device=device))
        omega_calib += np.sqrt(dt_cal) * phi * force_mask
        dE_acc += get_energy(omega_calib) - E0
    dE_per_amp2 = dE_acc / (20 * dt_cal)
    force_amp = float(np.sqrt(epsilon_inj / dE_per_amp2))
    print(f"force_amp = {force_amp:.2f}, calibrated dE/dt = {force_amp**2 * dE_per_amp2:.4f} (should be {epsilon_inj})")
    ```
    where `get_energy` computes `E = (0.5/N^4) * sum_k (|u_hat|^2 + |v_hat|^2)` using `u_hat = i*ky*(-omega_hat/k2_nz)`.

    For N=512 with epsilon_inj=0.1 and forcing at k∈[3,5], the calibrated force_amp ≈ 80,000–100,000.

    ### e. Forcing band
    - Force at integer wavenumber magnitude: `k_force_min = 3`, `k_force_max = 5`
    - `force_mask = (km >= 3) & (km <= 5)`
    - Typical number of forced modes: ~31

    ### f. Spinup — USE T_spinup = 10.0
    The inverse energy cascade completes (k_peak reaches 1) by t ≈ 8 at these parameters.
    **Use T_spinup = 10.0**, not 60.0. After spinup, record T_L = L/U_rms and set T_prod = 50*T_L.
    Expected after spinup: U_rms ≈ 1–3, T_L ≈ 2–8.
    ```python
    T_spinup_fixed = 10.0
    while t < T_spinup_fixed:
        dt = get_dt(omega_hat)   # CFL: dt = CFL * dx / U_max, CFL=0.4, capped at 0.5
        omega_hat = rk4_if(omega_hat, dt)
        t += dt
    ```

    ### g. CFL timestep
    ```python
    def get_dt(omega_hat):
        u_hat = 1j*ky*(-omega_hat/k2_nz); v_hat = -1j*kx*(-omega_hat/k2_nz)
        u = torch.fft.irfft2(u_hat, s=(N,N)); v = torch.fft.irfft2(v_hat, s=(N,N))
        U_max = torch.max(torch.sqrt(u**2+v**2)).item()
        return min(0.4*dx/U_max, 0.5) if U_max > 1e-8 else 0.05
    ```

    ### h. Snapshot interval — dt_snap = 0.01
    At U_rms ≈ 2–4, the condition `U_rms * dt_snap < 0.1` requires `dt_snap < 0.025–0.05`.
    **Use dt_snap = 0.01** (safe for any U_rms up to 10).
    The coarser Eulerian field snapshot: `dt_vel = 0.5` (was 2.0; reduce due to shorter T_prod).

    ### i. Number of tracers and coarsening
    - N_tracers = 5000 (unchanged)
    - Coarsen Eulerian fields from 512 → 128 (factor 4): `N_coarse = 128`

    ### j. Tracer advection
    Coupled RK4 for tracers within the same step as the fluid (correct temporal accuracy):
    ```python
    # Inside the main loop, before rk4_if:
    tracer_pos = rk4_tracer(tracer_pos, omega_hat, dt)   # uses current omega_hat
    omega_hat  = rk4_if(omega_hat, dt)
    ```
    `rk4_tracer` uses bilinear interpolation (`grid_sample`) at each substep.
    Map tracer positions to grid_sample input: `pos_norm = (pos / (2*pi))*2 - 1`

    ### k. Instability guard
    After each step, check for NaN/Inf:
    ```python
    if torch.any(torch.isnan(omega_hat)) or torch.any(torch.isinf(omega_hat)):
        raise RuntimeError(f"NaN/Inf at t={t:.3f}")
    ```

    ### l. Output files (all to `/home/node/work/projects/levy_flights_2dns_v2/data/`)
    Same as before: `tracer_positions.npy` (N_snaps, N_tracers, 2), `tracer_velocities.npy`,
    `tracer_times.npy`, `velocity_snapshots.npy` (N_vel_snaps, 2, 128, 128),
    `vorticity_snapshots.npy` (N_vel_snaps, 128, 128), `vel_times.npy`,
    `energy_spectrum.npy`, `diagnostics.npy`, `sim_params.json`.
    Also save `force_amp` in sim_params.json.

    ### m. Verification before production
    After spinup, print and verify:
    - `U_rms * dt_snap < 0.1` (should be ~0.02–0.04 with these parameters)
    - `T_prod = 50 * T_L` is finite and > 0
    - `k_peak` at end of spinup (expect 1–4)

1.  **Data Loading and Initial Verification:**

    a.  Load the simulation data from the specified absolute paths: `/home/node/work/projects/levy_flights_2dns_v2/data/tracer_positions.npy`, `/home/node/work/projects/levy_flights_2dns_v2/data/tracer_velocities.npy`, `/home/node/work/projects/levy_flights_2dns_v2/data/vorticity_snapshots.npy`, `/home/node/work/projects/levy_flights_2dns_v2/data/velocity_snapshots.npy`, `/home/node/work/projects/levy_flights_2dns_v2/data/tracer_times.npy`, `/home/node/work/projects/levy_flights_2dns_v2/data/vel_times.npy`, `/home/node/work/projects/levy_flights_2dns_v2/data/energy_spectrum.npy`, `/home/node/work/projects/levy_flights_2dns_v2/data/diagnostics.npy`, and `/home/node/work/projects/levy_flights_2dns_v2/data/sim_params.json`.
    b.  Extract relevant simulation parameters from `sim_params.json`.
    c.  Verify shapes: `tracer_positions.npy` → `(N_tracer_snaps, N_tracers, 2)`; `velocity_snapshots.npy` → `(N_vel_snaps, 2, 128, 128)`.
    d.  Calculate `k_box = 1` (fundamental wavenumber).
    e.  Confirm `T_prod > 0` and `T_L_at_start_of_production` is finite.

2.  **Snapshot Quality and Inverse Cascade Verification:**

    a.  **Tracer Displacement Verification:** Compute `⟨|Δr(1 step)|⟩` using periodic BCs. MUST be < 0.3 × 2π ≈ 1.88. Print diagnostic and stop if violated.
    b.  **VACF Verification:** Compute `C_vv(τ)` at lags τ = 1, 2, 4, 8, 16 steps. Report decorrelation time. At τ=1, `C_vv` should be > 0.3.
    c.  **Inverse Cascade Confirmation:** Plot `E(k)` at t = 0.1, 0.3, 0.6, 1.0 × T_prod. Plot `k_peak(t)` vs time from `diagnostics.npy`. Confirm k_peak drifts toward smaller k over the production run (even if starting from k≈1, it should remain at 1 and energy should grow at k=1).

3.  **Non-Stationarity Analysis:**

    a.  Define the window width `W = 8 * T_L_at_start_of_production` and the step size `ΔW = 2 * T_L_at_start_of_production`.
    b.  Create a series of overlapping temporal windows covering the full production run.
    c.  For each window `w`:
        i.   Extract tracer positions within the window.
        ii.  Compute displacement PDFs P(Δx, τ) at lags τ = {0.5, 1, 2} × T_L. Use KDE. Compute scalar displacements (x and y components separately).
        iii. Fit Lévy-stable distribution using McCulloch quantile method: compute five quantiles (p5, p25, p50, p75, p95) of scalar displacements, compute ν_α = (q95−q5)/(q75−q25) for each lag, report α(t) as the median across the three lags.
        iv.  Hill estimator cross-check: use top k=95th-percentile-index order statistics of |Δx|. Save a Hill plot (α_Hill vs k) for each window.
        v.   Compute MSD(τ) and fit power law to get anomalous exponent γ_w.
        vi.  Compute VACF C_vv(τ) and fit algebraic decay to get ν_w.
        vii. Extract mean d_vv(w) and k_peak(w) from `diagnostics.npy`.
    d.  Create time series α(t), γ(t), ν(t), d_vv(t), k_peak(t). Detrend before stationarity tests.
    e.  Run Augmented Dickey-Fuller and KPSS stationarity tests on each time series.
    f.  Plot α(t) vs k_peak(t)/k_box and fit α = a + b*(k_peak/k_box)^c.

4.  **Wavelet Decomposition and Eddy-Lifetime Scaling:**

    a.  Use Daubechies db4 or Symlets sym6 wavelet basis.
    b.  Apply 2D wavelet decomposition to velocity snapshots from `velocity_snapshots.npy` (shape: N_vel_snaps × 2 × 128 × 128), levels J=5 (adjusted from 6 due to 128-grid coarsening).
    c.  Reconstruct filtered fields `ũ_j` retaining levels j ≥ j_min for j_min ∈ {1, 2, 3, 4}.
    d.  For each filtered field:
        i.  Compute T_eddy(j) from temporal autocorrelation of wavelet energy C_E(τ,j) averaged over all 128×128 pixels. Integrate until C_E < 0.1, with max lag = min(T_prod/4, 50*dt_vel).
        ii. Re-advect 5000 tracers using filtered velocity fields (GPU, vectorised). Estimate wall-clock time before starting.
        iii. Compute α(j), γ_MSD(j), ν_VACF(j) from re-advected trajectories.
    e.  Plot α(j) vs T_eddy(j). Fit and report functional form.

5.  **Synthesis:**

    a.  Compare time-windowed α(t) with wavelet-scale α(j) at the dominant scale.
    b.  Create 2D map of α as a function of (T_eddy, k_peak/k_box).
    c.  State whether α is stationary or drifts with spectral condensation state.
