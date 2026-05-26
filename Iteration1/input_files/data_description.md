# Lévy Flights in 2D Turbulence: Wavelet-Filtered Tracer Advection and Non-Stationary Analysis (v2)

## Research Objectives

This project addresses two open questions from levy_flights_2dns_v1 (6 iterations):

1. **Wavelet-based eddy-lifetime scaling of Lévy flights**: Decompose the 2D velocity field into wavelet scales, re-advect tracers in each filtered field, vary eddy lifetime by retaining different scale ranges, and test whether α scales with T_eddy(j) as CTRW theory predicts.

2. **Non-stationarity analysis across multiple time windows**: Run a long-duration DNS (T >> 50 T_L), divide into overlapping temporal windows, and measure how α(t), γ(t), ν(t), d_vv(t), k_peak(t) evolve as the inverse cascade develops. Test whether α ≈ 1.4 (found in v1 Iter 4) is stationary or drifts with spectral condensation.

## Iteration 0 Failure — Fixed in This Iteration

Iteration 0 failed because:
1. Snapshots were saved at Δt_snap = 0.30 — too coarse. Tracers moved O(1) domain widths per step (σ_displacement ≈ 1.8 ≈ π/√3 at every lag), completely washing out Lévy tail statistics.
2. Energy was already condensed at k_peak ≈ 1 from t=0, so the evolving inverse cascade was not captured.
3. VACF was zero at all lags — no velocity memory resolvable.

Fixes applied in this iteration:
- Δt_snap ≤ 0.05 (6× finer), targeting ≥ 5000 tracer snapshots over T_prod = 50 T_L
- Forcing at k ∈ [3, 5] instead of [1, 3], so inverse cascade develops from k~4 toward k~1
- Proper spinup: run until T/T_L ≥ 10 before seeding tracers
- Eulerian velocity snapshots saved at coarser Δt_vel = 2.0 (manageable storage)

## Simulation Design

### Solver
GPU-accelerated pseudo-spectral 2D Navier-Stokes in vorticity-streamfunction form:
    ∂ω/∂t + (u · ∇)ω = ν_h (-1)^p ∇^(2p) ω + f
- Domain: [0, 2π] × [0, 2π], doubly periodic
- Resolution: N = 1024 × 1024 (de-aliased with 2/3 rule → 682 active modes per direction)
  **This resolution is mandatory — do not reduce to 512² or lower.**
- Time integration: RK4 with adaptive dt, CFL < 0.4
- Forcing: stochastic white-in-time at k ∈ [3, 5], constant energy injection rate ε_inj = 0.1
- Dissipation: hyperviscosity ν_h = 1e-19, p = 4 (8th-order). No large-scale drag.
- Use PyTorch (device='cuda') for all spectral transforms (torch.fft.rfft2 / irfft2)

### Run duration and snapshot strategy
- **Spinup**: run until T/T_L ≥ 10 (fully developed inverse cascade, k_peak visibly below k_f=4)
  - During spinup: save diagnostics only (no tracer/velocity snapshots)
  - Monitor T_L from the energy spectrum: T_L ≈ L / U_rms where U_rms = sqrt(2 E_total / N²)
- **Production**: run for T_prod = 50 × T_L_spinup_end after spinup
- **Tracer snapshot interval**: Δt_snap = 0.05 (targeting ~5000 snapshots over production run)
  - This must satisfy: (tracer_speed × Δt_snap) << L_domain, i.e., U_rms × 0.05 << 2π
  - At U_rms ~ 1.0–2.0, displacement per step ~ 0.05–0.10 domain units — adequate for tracking
- **Eulerian velocity snapshot interval**: Δt_vel = 2.0 (every 40th tracer snapshot)
  - Save at Δt_vel to keep velocity_snapshots.npy manageable (≤ 5 GB)
- Storage estimate at N=1024, 5000 tracer snapshots:
  - tracer_positions.npy: 5000 × 5000 × 2 × 4 bytes ≈ 200 MB ✓
  - velocity_snapshots.npy (at 256² coarsened): (T_prod/2.0) × 2 × 256 × 256 × 4 ≈ manageable
  - vorticity_snapshots.npy (at 256² coarsened): same order ✓

### Passive tracers
- N_tracers = 5000 tracers seeded uniformly at random at the START of the production run
- Advected by bilinear spectral interpolation at every tracer snapshot step
- Periodic boundary conditions
- Record tracer positions AND velocities at every tracer snapshot

## Output Files (ABSOLUTE PATHS — mandatory)

All output files written to `/home/node/work/projects/levy_flights_2dns_v2/data/`:

- `/home/node/work/projects/levy_flights_2dns_v2/data/tracer_positions.npy`
  Shape: (N_tracer_snaps, N_tracers, 2), dtype float32. Positions in [0, 2π].
- `/home/node/work/projects/levy_flights_2dns_v2/data/tracer_velocities.npy`
  Shape: (N_tracer_snaps, N_tracers, 2), dtype float32. Interpolated (u, v) at tracer positions.
- `/home/node/work/projects/levy_flights_2dns_v2/data/vorticity_snapshots.npy`
  Shape: (N_vel_snaps, 256, 256), dtype float32. Coarsened (1024→256) vorticity at Δt_vel.
- `/home/node/work/projects/levy_flights_2dns_v2/data/velocity_snapshots.npy`
  Shape: (N_vel_snaps, 2, 256, 256), dtype float32. Coarsened (u,v) at Δt_vel. For wavelet analysis.
- `/home/node/work/projects/levy_flights_2dns_v2/data/tracer_times.npy`
  Shape: (N_tracer_snaps,), dtype float64. Times of tracer snapshots (production run only).
- `/home/node/work/projects/levy_flights_2dns_v2/data/vel_times.npy`
  Shape: (N_vel_snaps,), dtype float64. Times of Eulerian velocity snapshots.
- `/home/node/work/projects/levy_flights_2dns_v2/data/energy_spectrum.npy`
  Shape: (N_tracer_snaps, N//2), dtype float64. Azimuthally-averaged E(k, t) at tracer snapshot times.
- `/home/node/work/projects/levy_flights_2dns_v2/data/diagnostics.npy`
  Structured array: fields = (time, E_total, E_rms, omega_rms, T_L, k_peak, d_vv_estimate).
  Shape: (N_tracer_snaps,).
- `/home/node/work/projects/levy_flights_2dns_v2/data/sim_params.json`
  JSON: N, dt_actual, nu_h, epsilon_inj, N_tracers, N_tracer_snaps, N_vel_snaps,
        dt_snap, dt_vel, T_spinup, T_prod, T_L_at_start_of_production.

## Analysis Tasks

### Task 1: Verify snapshot quality and inverse cascade

a. Compute the mean tracer displacement per snapshot: ⟨|Δr(1 step)|⟩ = ⟨|r(t+Δt_snap) - r(t)|⟩.
   This MUST be < 0.3 domain units (i.e., < 0.3 × 2π ≈ 1.88). If it exceeds this, the snapshot interval is too coarse for meaningful Lévy statistics — report the issue and stop.
b. Compute the VACF C_vv(τ) at short lags (τ = 1, 2, 4, 8, 16 steps). At τ=1, C_vv should be > 0.3 (non-negligible memory). Report the decorrelation time τ_corr.
c. Confirm the inverse cascade: plot E(k) at t = 0.1, 0.3, 0.6, 1.0 × T_prod. k_peak should drift from ~4 at t=0 toward ~1–2 by t=T_prod.

### Task 2: Non-stationarity analysis

Define overlapping windows of width W = 8 T_L, step ΔW = 2 T_L (~26 windows). For each window:
a. Fit Lévy-stable α via McCulloch quantile method (fast, O(N)) at lags τ = {0.5, 1, 2} T_L.
   The McCulloch estimator uses 5 sample quantiles (p5, p25, p50, p75, p95):
   ν_α = (q95 - q5) / (q75 - q25), then interpolate from the published McCulloch (1986) table.
   Also compute Hill estimator as a cross-check: α_Hill = 1 / mean(log(X_(i)/X_(k))) for top-k order stats.
b. Fit MSD(τ) power law → γ(t).
c. Fit VACF algebraic decay → ν(t).
d. Extract d_vv(t), k_peak(t) from diagnostics.

Output: α(t), γ(t), ν(t), d_vv(t), k_peak(t). Test stationarity (ADF + KPSS tests).
Plot α(t) vs k_peak(t)/k_box and fit α = a + b × (k_peak/k_box)^c.

### Task 3: Wavelet decomposition and eddy-lifetime scaling

Use `velocity_snapshots.npy` (coarsened to 256²). Engineer chooses wavelet basis — justify choice based on:
- Compact support (avoid ringing): Daubechies db4/db6 or Symlets sym6 recommended
- Orthogonality for energy decomposition
- Boundary treatment for periodic domain (periodic mode in PyWavelets)

Steps:
a. Apply 2D DWT decomposition level by level to each velocity snapshot. Extract approximation (A_j) and detail (D_j = {H,V,D}) coefficients at each level j = 1, …, J (J = floor(log2(256)) = 8, but use J=6 for stability).
b. Reconstruct filtered fields ũ_j for j_min ∈ {1, 2, 3, 4} by zeroing out detail coefficients below j_min.
c. For each scale j, estimate eddy lifetime T_eddy(j) from temporal autocorrelation of wavelet energy:
   E_j(x,y,t) = sum_d |D_j^d(x,y,t)|², then C_E(τ,j) = ⟨E_j(x,y,t) E_j(x,y,t+τ)⟩ / ⟨E_j²⟩
   T_eddy(j) = ∫_0^∞ C_E(τ,j) dτ  (trapezoidal, integrate until C_E drops below 0.1)
d. Re-advect tracers in each filtered field ũ_j using saved tracer positions as seeds.
   Use the filtered velocity fields to interpolate tracer velocities, then integrate with RK4.
e. For each filtered field: compute α(j), γ_MSD(j), ν_VACF(j).
f. Plot α(j) vs T_eddy(j). Fit α = f(T_eddy) and report functional form.
   Test CTRW prediction: if β ∝ T_eddy^δ, then α = 1 + β/(2−β).

### Task 4: Synthesis

a. Compare α(t) from Task 2 with α(j) from Task 3 at the dominant wavelet scale (j corresponding to k_peak(t)).
b. 2D map of α as function of (T_eddy, k_peak/k_box).
c. State whether α is stationary or condenses with the inverse cascade.

## Key Physical Predictions

1. α < 2 (non-Gaussian) observed at T >> T_L once hyperbolic manifolds are resolved
2. α decreases (heavier tails) as T_eddy increases — larger eddies produce more ballistic flights
3. α(t) drifts toward lower values as k_peak → k_box (flow becomes more anomalous as condensation progresses)
4. CTRW prediction α = 1 + β/(2−β) holds with β(j) ∝ T_eddy(j)^δ

## Hardware

- NVIDIA RTX PRO 6000 Blackwell Edition (96 GB VRAM), CUDA 13.0 — use for 1024² spectral solver
- 64 vCPUs, 128 GB RAM — use for post-processing (limit to 8–16 workers)
- PyWavelets (pywt) installed in /opt/denario-venv
- Storage: /home/node/work/projects/levy_flights_2dns_v2/data/ (several GB available)

## Prior Results Context

From levy_flights_2dns_v1 Iteration 4 (best):
- α ≈ 1.4 ± 0.1, VACF decay ν ≈ 0.61–0.72
- Two-channel ejection: vortex death (2.6%) + strain stripping (97.4%)
- Jump length scaling L ~ d_vv^0.63 (non-universal, χ² p < 0.001)
- MSD ratio ~3.9, kurtosis ratio ~0.4, S6 ratio ~5.5 (model inconsistency)

Key lesson: need T >> 50 T_L AND Δt_snap << τ_corr to resolve LCS and Lévy statistics.
