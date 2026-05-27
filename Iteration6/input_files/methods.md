0. **DNS Implementation — MANDATORY (copy template, GPU-only, do not deviate)**

   Copy `/home/node/work/projects/levy_flights_2dns_v2/dns_template_1024.py` to `codebase/step_1.py`.
   Modify ONLY the PARAMS dict and the production loop (output saving). Do NOT rewrite the solver.

   Key parameter changes from the template:
   ```python
   PARAMS = {
       'N': 1024,
       'device': 'cuda',          # MANDATORY - ALL ops on GPU
       'CFL': 0.4,
       'nu_h': 3.9e-31,           # scaled for 1024^2, p=4
       'p': 4,
       'epsilon_inj': 0.1,
       'k_force_min': 20.0,       # CHANGED: force at k in [20,40]
       'k_force_max': 40.0,       # This gives ~1.3 decades of cascade range
       'lambda_ekman': 0.05,      # NEW: linear Ekman drag to arrest condensation
       'N_tracers': 5000,
       'dt_snap': 0.005,          # Fine enough for high-U_rms flow
       'dt_vel': 0.2,             # Eulerian fields more frequent for FTLE
       'T_spinup_fixed': 5.0,     # SHORT spinup - seed tracers before cascade condenses
       'N_coarse': 256,           # Coarsen 1024 -> 256 for Eulerian snapshots
   }
   ```

   Ekman drag addition to the integrating factor:
   ```python
   # Combined dissipation + Ekman drag in one IF:
   diss_op = -nu_h * k2**p - lambda_ekman   # drag applies to all modes equally
   IF = torch.exp(diss_op * dt)
   # No separate forcing kick needed for drag - it's folded into IF
   ```

   Forcing amplitude calibration: same empirical procedure as template (inject with amp=1,
   measure dE/dt, set force_amp = sqrt(epsilon_inj / dE_per_amp2)).

   Production loop output (save to `/home/node/work/projects/levy_flights_2dns_v2/data/`):
   - `tracer_positions.npy` shape (N_snaps, 5000, 2) float32
   - `tracer_velocities.npy` shape (N_snaps, 5000, 2) float32
   - `tracer_times.npy` shape (N_snaps,) float64
   - `velocity_snapshots.npy` shape (N_vel, 2, 256, 256) float32 — coarsen with AvgPool2d(4)
   - `vorticity_snapshots.npy` shape (N_vel, 256, 256) float32
   - `vel_times.npy` shape (N_vel,) float64
   - `diagnostics.npy` structured array: fields time, E_total, U_rms, k_peak, T_L
   - `energy_spectrum.npy` shape (N_snaps, 512) float64
   - `sim_params.json`

   Verification after spinup (before seeding tracers): print U_rms, T_L, k_peak.
   k_peak should be ~20–30 (not yet condensed). If k_peak < 5 already, spinup was too long.

   Verification during production: every 10 T_L print k_peak — it should drift from ~25 toward
   ~2–5 over T_prod. With Ekman drag lambda=0.05, the cascade should arrest at k_peak ~ 3–8
   rather than collapsing to k=1.

1. **DNS Execution (1024² GPU, k∈[20,40], Ekman drag)**:
   - Use template with parameters above. EVERYTHING on device='cuda'.
   - T_prod = 100 * T_L (longer to capture full cascade evolution)
   - Expected: U_rms ~ 0.3–1.5, T_L ~ 5–20, T_prod ~ 500–2000 sim units
   - Expected k_peak trajectory: starts ~25, drifts to ~3–8 (arrested by Ekman drag)

2. **Data Quality Verification**:
   - Mean tracer displacement/snap < 0.3×2π ≈ 1.88 (mandatory)
   - VACF C_vv(τ=1) > 0.3
   - Confirm k_peak drifts over production (not stuck at 1 or at forcing scale)
   - Plot E(k,t) at t = 0.1, 0.3, 0.5, 0.7, 1.0 × T_prod showing cascade evolution

3. **Non-Stationarity Analysis (GPU-accelerated)**:
   - Overlapping windows W = 8 T_L, step ΔW = 2 T_L
   - McCulloch α per window (x and y separately, median across lags τ={0.5,1,2}×T_L)
   - CCDF log-log plot per window — check if slope is in range [−1, −3] (Lévy regime)
   - ADF and KPSS stationarity tests on α(t), γ(t)
   - Fit α = a + b×(k_peak/k_box)^c — this is the key result

4. **Gaussian Band-Pass Wavelet Analysis (GPU-accelerated)**:
   Use Gaussian filters in k-space (NOT hard wavelet thresholding):
   ```python
   G(k) = exp(-(|k| - k0)^2 / (2*sigma^2))
   ```
   Five bands spanning the cascade:
   - Band 1: k0=25, sigma=5   (near forcing scale)
   - Band 2: k0=15, sigma=4   (upper inertial range)
   - Band 3: k0=8,  sigma=3   (mid inertial range)
   - Band 4: k0=4,  sigma=2   (lower inertial range)
   - Band 5: k0=2,  sigma=1   (near condensation)

   For each band:
   a. Filter velocity snapshots using rfft2 on GPU
   b. Re-advect 2000 tracers using GPU grid_sample interpolation
   c. Compute MSD exponent γ(j) and CCDF slope
   d. Compute T_eddy(j) from temporal autocorrelation of band energy
   e. Plot α/γ vs T_eddy — test whether larger eddies → more anomalous transport

5. **FTLE-Based LCS Analysis (GPU)**:
   - Compute FTLE on 256×256 grid at 5 time points during cascade evolution
   - Integration time ΔT_FTLE = 2 T_L
   - Use GPU grid_sample for all tracer interpolation
   - Compute α conditional on tracer proximity to FTLE ridges (top 25% FTLE)
   - Test: α_near_ridge vs α_far_ridge — is anomalous diffusion localised to ridges?

6. **Synthesis**:
   - Main plot: α(t) and γ(t) vs k_peak(t)/k_box — show non-stationarity
   - Secondary: γ(j) vs T_eddy(j) per band — show scale-dependence
   - CCDF slopes per band — confirm power-law tails in at least some bands
   - Conclude: is it Lévy (CCDF slope ~ −1 to −2) or super-diffusive Gaussian (slope < −3)?
