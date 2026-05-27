0. **DNS Implementation — CRITICAL: USE THE PROVIDED TEMPLATE EXACTLY**

   A working, tested DNS solver is provided at:
   `/home/node/work/projects/levy_flights_2dns_v2/dns_template_1024.py`

   For Step 1 (DNS simulation), the engineer MUST:
   1. Copy this file to `codebase/step_1.py`
   2. Change only the production-run saving logic to write to the standard output paths
   3. Run it as-is — DO NOT rewrite the solver, change the integrating factor method, add explicit viscosity, or use a different timestep scheme

   The template already implements:
   - N=1024, domain L=2π, k_force_min=1, k_force_max=3, nu_h=3.9e-31, p=4
   - Integrating factor (IF) method for hyperviscosity (unconditionally stable)
   - Empirical force_amp calibration (correct rfft2 normalisation)
   - CFL adaptive timestep, dt cap at 0.5
   - Short production test (5 T_L) to verify snapshot quality
   - Tracer advection using GPU grid_sample

   Modifications needed to the template for the full production run:
   - Replace the "SHORT PRODUCTION TEST" block with the full production run (T_prod = 50 * T_L_end)
   - During production, save at dt_snap=0.01 (tracers) and dt_vel=0.5 (coarsened 1024→256 Eulerian fields)
   - Save all output files to `/home/node/work/projects/levy_flights_2dns_v2/data/`:
     * `tracer_positions.npy` — shape (N_snaps, 5000, 2)
     * `tracer_velocities.npy` — shape (N_snaps, 5000, 2)
     * `tracer_times.npy` — shape (N_snaps,)
     * `velocity_snapshots.npy` — shape (N_vel_snaps, 2, 256, 256)
     * `vorticity_snapshots.npy` — shape (N_vel_snaps, 256, 256)
     * `vel_times.npy` — shape (N_vel_snaps,)
     * `diagnostics.npy` — structured array with fields: time, E_total, U_rms, k_peak
     * `sim_params.json` — dict with N, nu_h, epsilon_inj, T_L, T_prod, force_amp, dt_snap, dt_vel
   - Coarsen Eulerian fields using `torch.nn.AvgPool2d(4)` before saving

1. **DNS Execution (1024² with k∈[1,3])**:
   - Copy `/home/node/work/projects/levy_flights_2dns_v2/dns_template_1024.py` to `codebase/step_1.py`
   - Modify only the production loop to save full output (see Section 0)
   - Run for T_prod = 50 × T_L. Expected: U_rms ≈ 0.5–2, T_L ≈ 3–15, T_prod ≈ 150–750 sim units

2. **Data Quality and Cascade Validation**:
   - Mean tracer displacement per snapshot < 0.3 × 2π ≈ 1.88 (must pass)
   - VACF C_vv(τ=1) > 0.3 (must pass)
   - Plot E(k,t) at t = 0.1, 0.3, 0.6, 1.0 × T_prod; confirm k_peak drifts toward k=1

3. **Non-Stationary Lévy Analysis**:
   - Overlapping windows W = 8 T_L, step ΔW = 2 T_L
   - McCulloch quantile method: x and y displacements separately, median α across lags τ = {0.5, 1, 2} × T_L
   - CCDF log-log plots per window; flag windows with < 1.5 decades as non-Lévy
   - ADF and KPSS stationarity tests on α(t), γ(t), ν(t)
   - Fit α = a + b × (k_peak/k_box)^c

4. **Lagrangian Coherent Structure (LCS) Identification**:
   - Compute Okubo-Weiss parameter Q = s² − ω² from velocity snapshots (on GPU)
   - Dynamic threshold Q_threshold(t) = σ_Q(t) (instantaneous std of Q)
   - Classify each tracer at each time as "vortex-trapped" (local Q < −Q_threshold) or "strain-dominated" (Q > +Q_threshold) or "neutral"
   - Compute α separately for the vortex-trapped and strain-dominated tracer populations

5. **FTLE Field Computation** (GPU-accelerated):
   - Integration time ΔT_FTLE = 2 × T_L
   - Use bilinear interpolation (grid_sample) between velocity snapshots at dt_vel=0.5
   - Compute FTLE on 256×256 grid (matching Eulerian snapshots)

6. **Residence Time and Flight Time Analysis**:
   - Categorize tracer trajectory segments as "trapped" or "flight" based on Q classification
   - Extract distributions of residence times and flight displacements
   - Test for power-law tails; compare α from CCDF slope to McCulloch α(t)
   - Distinguish "vortex death" vs "strain stripping" ejection events

7. **Causal Link Analysis**:
   - Time-lagged cross-correlation between dE(k=1)/dt and dα(t)/dt
   - Granger causality test (lag = 1–3 windows)
   - Correlation between hyperbolic point spatial density and α(t)

8. **Synthesis**:
   - 2D map of α vs (hyperbolic point density, k_peak/k_box)
   - Sensitivity analysis on Q_threshold (70th vs 80th percentile)
   - Conclude on whether non-stationarity of α is driven by spectral condensation + LCS reorganisation
