0.  **Simulation Step — CRITICAL ENGINEERING NOTES:**

    The engineer must implement the 2D NS simulation in step_1 with these exact specifications:

    a.  **Spinup criterion — USE FIXED TIME, NOT T/T_L.** The flow starts cold (U_rms ≈ 0), so T_L = L/U_rms → ∞ at t=0. The condition `while t/T_L < 10` NEVER EXITS. Instead use:
        ```python
        T_spinup_fixed = 60.0  # simulation time units >> 10*T_f ~ 29 units
        while self.t < T_spinup_fixed:
            # advance one step
        ```
        After spinup, record T_L_end = L/U_rms and set T_prod = 50 * T_L_end.

    b.  **Snapshot interval**: dt_snap = 0.05 for tracers. This gives (U_rms * dt_snap) ~ 0.05 * 0.5 ~ 0.025 domain units per step — well below the 0.3 threshold. Do NOT use dt_snap = 0.3.

    c.  **Verify before production**: After spinup, compute U_rms and check that U_rms * dt_snap < 0.1. If U_rms > 2.0, reduce dt_snap to 0.02.

    d.  **Data directory**: All output files must be written to `/home/node/work/projects/levy_flights_2dns_v2/data/`. Create this directory if it does not exist: `os.makedirs(output_path, exist_ok=True)`.

    e.  **Pre-allocate arrays**: For N_tracers=5000 and ~5000 tracer snapshots, tracer_positions.npy will be ~200 MB. Allocate a memmap or grow a list and save at the end.

    f.  **GPU memory**: 1024² float32 array = 4 MB; 4 arrays in GPU memory ~ 16 MB. Fine for 96 GB VRAM.

1.  **Data Loading and Initial Verification:**

    a.  Load the simulation data from the specified absolute paths: `/home/node/work/projects/levy_flights_2dns_v2/data/tracer_positions.npy`, `/home/node/work/projects/levy_flights_2dns_v2/data/tracer_velocities.npy`, `/home/node/work/projects/levy_flights_2dns_v2/data/vorticity_snapshots.npy`, `/home/node/work/projects/levy_flights_2dns_v2/data/velocity_snapshots.npy`, `/home/node/work/projects/levy_flights_2dns_v2/data/tracer_times.npy`, `/home/node/work/projects/levy_flights_2dns_v2/data/vel_times.npy`, `/home/node/work/projects/levy_flights_2dns_v2/data/energy_spectrum.npy`, `/home/node/work/projects/levy_flights_2dns_v2/data/diagnostics.npy`, and `/home/node/work/projects/levy_flights_2dns_v2/data/sim_params.json`.
    b.  Extract relevant simulation parameters from `sim_params.json`, including `N`, `dt_actual`, `nu_h`, `epsilon_inj`, `N_tracers`, `N_tracer_snaps`, `N_vel_snaps`, `dt_snap`, `dt_vel`, `T_spinup`, `T_prod`, and `T_L_at_start_of_production`.
    c.  Verify the shape and data types of each loaded array to ensure consistency with the data description. Specifically, ensure `tracer_positions.npy` has shape `(N_tracer_snaps, N_tracers, 2)` and `velocity_snapshots.npy` has shape `(N_vel_snaps, 2, 256, 256)`.
    d.  Calculate `k_box = 1` (fundamental wavenumber).
    e.  Ensure that `T_prod` is significantly larger than `T_L_at_start_of_production` (e.g., `T_prod > 50 * T_L_at_start_of_production`) to confirm that the production run duration meets the requirement.

2.  **Snapshot Quality and Inverse Cascade Verification:**

    a.  **Tracer Displacement Verification:** Compute the mean tracer displacement per snapshot: `⟨|Δr(1 step)|⟩ = ⟨|r(t+Δt_snap) - r(t)|⟩`. Calculate `r(t+Δt_snap) - r(t)` using periodic boundary conditions. This MUST be < 0.3 domain units (i.e., < 0.3 × 2π ≈ 1.88). If it exceeds this, report the issue and stop.
    b.  **VACF Verification:** Compute the Velocity Autocorrelation Function (VACF) `C_vv(τ)` at short lags (τ = 1, 2, 4, 8, 16 steps). The VACF is computed using the available `tracer_velocities.npy` data as follows: For each tracer, compute the dot product of its velocity at time t with its velocity at time t+τ, average over all tracers, and normalize by the variance of the velocities. At τ=1, `C_vv` should be > 0.3. Report the decorrelation time `τ_corr`.
    c.  **Inverse Cascade Confirmation:**
        i.   Load `energy_spectrum.npy`.
        ii.  Define target times for plotting the energy spectra: `t_targets = [0.1, 0.3, 0.6, 1.0] * T_prod`.
        iii. Find the indices in `tracer_times.npy` that are closest to the target times.
        iv.  Plot the energy spectra `E(k)` at the selected time indices.
        v.   Load `diagnostics.npy` and extract `k_peak(t)` at the same times as the energy spectra plots for better comparison. Plot `k_peak(t)` as a function of time to visualize the energy condensation process. Confirm that `k_peak(t)` shifts from ~4 towards ~1–2 over the production run.

3.  **Non-Stationarity Analysis:**

    a.  Define the window width `W = 8 * T_L_at_start_of_production` and the step size `ΔW = 2 * T_L_at_start_of_production`.
    b.  Create a series of overlapping temporal windows covering the full production run. The start times of the windows will be `t_start = np.arange(tracer_times[0], tracer_times[-1] - W, ΔW)`.
    c.  For each window `w`:
        i.   Find the indices in `tracer_times.npy` that fall within the current window.
        ii.  Extract the tracer positions within the window: `tracer_positions_w = tracer_positions[indices, :, :]`.
        iii. Compute displacement PDFs P(Δx, τ) at lags τ = {0.5, 1, 2} * `T_L_at_start_of_production`. The displacement Δx should be calculated using periodic boundary conditions. Use Kernel Density Estimation (KDE) to estimate the PDFs.
        iv.  Fit a Lévy-stable distribution to each P(Δx, τ) using the McCulloch quantile method to obtain the Lévy index α_w for each τ. The McCulloch estimator will be implemented using scipy.stats.levy_stable.fit. Also compute the Hill estimator as a cross-check.
        v.   Compute the Mean Squared Displacement (MSD) as a function of lag time τ within the window. Fit a power law to MSD(τ) to obtain the anomalous exponent γ_w.
        vi.  Compute the Velocity Autocorrelation Function (VACF) C_vv(τ) within the window. Fit an algebraic decay to C_vv(τ) to obtain the decay exponent ν_w.
        vii. Extract the mean inter-vortex distance `d_vv(w)` from the `diagnostics.npy` array for the current window by averaging the values within the window.
        viii. Extract the energy condensation wavenumber `k_peak(w)` from the `diagnostics.npy` array for the current window by averaging the values within the window.
    d.  Create time series of α(t), γ(t), ν(t), d_vv(t), and k_peak(t). Assign the window values to the center time of each window using a simple average. Detrend the time series of α(t), γ(t), ν(t), d_vv(t), and k_peak(t) before performing the stationarity tests.
    e.  Test the stationarity of each time series using both the Augmented Dickey-Fuller test and the Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test.
    f.  Plot α(t) vs k_peak(t)/k_box and fit the relationship α = a + b * (k_peak(t)/k_box)^c.

4.  **Wavelet Decomposition and Eddy-Lifetime Scaling:**

    a.  Choose a wavelet basis for 2D spatial decomposition. Recommended choices are Daubechies (db4, db6) or Symlets (sym6).
    b.  Apply 2D wavelet decomposition to each velocity snapshot u(x,y,t) and v(x,y,t) from `velocity_snapshots.npy` using the chosen wavelet basis and periodic boundary conditions. Obtain wavelet coefficient arrays `W_j(x,y,t)` for levels `j = 1, …, J` (J = 6).
    c.  Reconstruct filtered velocity fields `ũ_j(x,y,t)` retaining only levels `j >= j_min` for `j_min ∈ {1, 2, 3, 4}`.
    d.  For each filtered field `ũ_j`:
        i.   Compute the eddy lifetime `T_eddy(j)` from temporal autocorrelation of wavelet energy:
             `E_j(x,y,t) = sum_d |D_j^d(x,y,t)|²`, then `C_E(τ,j) = ⟨E_j(x,y,t) E_j(x,y,t+τ)⟩ / ⟨E_j²⟩`
             `T_eddy(j) = ∫_0^∞ C_E(τ,j) dτ` (trapezoidal, integrate until `C_E` drops below 0.1). Add a check to ensure that the eddy lifetime integral converges before reporting `T_eddy(j)`.
        ii.  Re-advect tracers in each filtered field `ũ_j` using saved tracer positions as seeds. Use the filtered velocity fields to interpolate tracer velocities using bilinear interpolation, then integrate with RK4.
        iii. Compute displacement statistics for each filtered field: α(j), γ_MSD(j), ν_VACF(j). These statistics are computed from the re-advected tracer trajectories in the same manner as in the non-stationarity analysis (step 3), but using the trajectories obtained from the filtered velocity fields.
    e.  Plot α(j) vs T_eddy(j) and test the CTRW prediction. Fit α(j) = f(T_eddy(j)) and report the functional form and exponent.

5.  **Synthesis:**

    a.  Compare the time-windowed α(t) values from Task 3 with the wavelet-scale analysis from Task 4 at the dominant scale. The dominant scale will be defined as the wavelet scale j corresponding to the wavenumber closest to `k_peak(t)`. This is determined by finding the j such that the central wavenumber of the wavelet level j is closest to `k_peak(t)`.
    b.  Create a 2D map of α as a function of (T_eddy, condensation state k_peak/k_box). Interpolate the α(j) values from Task 4 and the α(t) values from Task 3 onto a grid of (T_eddy, k_peak/k_box) values using bilinear interpolation.
    c.  State whether α is stationary or drifts with the spectral condensation state based on the results of the Augmented Dickey-Fuller test and the KPSS test, and the 2D map.
\