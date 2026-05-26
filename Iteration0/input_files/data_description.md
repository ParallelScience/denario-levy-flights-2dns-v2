# Lévy Flights in 2D Turbulence: Wavelet-Filtered Tracer Advection and Non-Stationary Analysis

## Research Objectives

This project extends a prior study (levy_flights_2dns_v1) on anomalous diffusion of passive tracers in 2D turbulence. Two specific open questions are addressed:

1. **Wavelet-based eddy-lifetime scaling of Lévy flights**: Decompose the 2D velocity field into wavelet scales, advect tracers in each filtered field, vary eddy lifetime by retaining different scale ranges, and test whether the Lévy index α scales predictably with the characteristic eddy lifetime T_eddy(j) as predicted by CTRW theory.

2. **Non-stationarity analysis across multiple time windows**: The prior project's results were derived from a single time slice or short integration window. This study runs a long-duration DNS (T >> 50 T_L), divides the run into overlapping temporal windows, and measures how α(t), the MSD scaling exponent γ(t), the VACF decay exponent ν(t), the mean inter-vortex distance d_vv(t), and the energy condensation wavenumber k_peak(t) evolve over time. This directly tests whether the previously reported α ≈ 1.4 is stationary or a transient.

## Simulation Design

### Solver
GPU-accelerated pseudo-spectral 2D Navier-Stokes in vorticity-streamfunction form:
    ∂ω/∂t + (u · ∇)ω = ν_h (-1)^p ∇^(2p) ω + f
Domain: [0, 2π] × [0, 2π], doubly periodic. Resolution: N = 512 × 512 (de-aliased with 2/3 rule).
Time integration: RK4 with adaptive dt, CFL < 0.4.
Forcing: stochastic white-in-time at k ∈ [1, 3], constant energy injection rate ε_inj = 0.1.
Dissipation: hyperviscosity ν_h = 1e-17, p = 4 (8th-order). No large-scale drag — essential for inverse cascade.
Use PyTorch (device='cuda') for spectral transforms.

### Run duration
- Spinup: T_spinup until T/T_L ≥ 8 (fully developed inverse cascade with large coherent vortices).
- Production: continue for at least T_prod = 60 × T_L additional time units after spinup.
- Save snapshots every Δt_save ≈ 0.05 T_L (adaptive to current T_L estimate), targeting ≥ 1000 snapshots over the production run.
- This temporal density is critical for the non-stationarity analysis and for wavelet coefficient autocorrelation estimates.

### Passive tracers
- N_tracers = 5000 tracers seeded uniformly after spinup.
- Advected by bilinear spectral interpolation at every saved timestep.
- Periodic boundary conditions.

## Output Files (ABSOLUTE PATHS — required by the engineer)

All files written to `/home/node/work/projects/levy_flights_2dns_v2/data/`:

### Primary simulation outputs:
- `/home/node/work/projects/levy_flights_2dns_v2/data/tracer_positions.npy`
  Shape: (N_snapshots, N_tracers, 2), dtype float32. tracer_positions[t, i, :] = (x, y) ∈ [0, 2π].
- `/home/node/work/projects/levy_flights_2dns_v2/data/tracer_velocities.npy`
  Shape: (N_snapshots, N_tracers, 2), dtype float32. Interpolated (u, v) at each tracer position.
- `/home/node/work/projects/levy_flights_2dns_v2/data/vorticity_snapshots.npy`
  Shape: (N_snapshots, 256, 256), dtype float32. Coarsened vorticity field at each snapshot.
- `/home/node/work/projects/levy_flights_2dns_v2/data/velocity_snapshots.npy`
  Shape: (N_snapshots, 2, 256, 256), dtype float32. Coarsened (u, v) velocity fields — required for wavelet decomposition.
- `/home/node/work/projects/levy_flights_2dns_v2/data/times.npy`
  Shape: (N_snapshots,), dtype float64. Physical times of each snapshot.
- `/home/node/work/projects/levy_flights_2dns_v2/data/energy_spectrum.npy`
  Shape: (N_snapshots, N//2), dtype float64. Azimuthally-averaged E(k, t).
- `/home/node/work/projects/levy_flights_2dns_v2/data/diagnostics.npy`
  Structured array: fields = (time, E_total, E_rms, omega_rms, T_L, k_peak, d_vv_estimate).
  Shape: (N_snapshots,).
- `/home/node/work/projects/levy_flights_2dns_v2/data/sim_params.json`
  JSON: all simulation parameters (N, dt, nu_h, epsilon_inj, N_tracers, N_snapshots, T_L_estimate, etc.).

## Analysis Tasks

### Task 1: Verify inverse cascade

Check E(k) ~ k^(-5/3) develops for k < k_f. Confirm energy condensation by tracking k_peak(t) shifting toward k=1 over the production run. Plot energy spectra at t = 0.1, 0.3, 0.6, 1.0 × T_prod.

### Task 2: Non-stationarity analysis

Define overlapping temporal windows of width W = 8 T_L, stepping by ΔW = 2 T_L, covering the full production run (yielding ~26 windows). For each window w:
a. Compute displacement PDFs P(Δx, τ) at lags τ = {0.5, 1, 2} T_L. Fit Lévy-stable distribution → α_w.
b. Compute MSD(τ) and fit power law → γ_w (anomalous exponent).
c. Compute VACF C_vv(τ) and fit algebraic decay → ν_w.
d. Compute mean inter-vortex distance d_vv(w) from vortex detection (Okubo-Weiss Λ < 0 criterion, connected-component labeling).
e. Extract k_peak(w) from the energy spectrum peak.

Output: time series α(t), γ(t), ν(t), d_vv(t), k_peak(t). Test stationarity via Augmented Dickey-Fuller test on each series. Plot α(t) vs k_peak(t)/k_box and fit α = a + b × (k_peak(t)/k_box)^c.

### Task 3: Wavelet decomposition and eddy-lifetime scaling

Use the saved `velocity_snapshots.npy` (shape N_snapshots × 2 × 256 × 256). The engineer should choose the wavelet basis appropriate for 2D spatial decomposition — recommended choices are Daubechies (db4, db6) or Symlets (sym6) for their compact support and orthogonality, or a 2D Morlet CWT if isotropic scale isolation is preferred. Justify the choice in the analysis output.

Steps:
a. Apply 2D wavelet decomposition to each velocity snapshot u(x,y,t) and v(x,y,t), obtaining wavelet coefficient arrays W_j(x,y,t) for levels j = 1, …, J (J ≈ 6–7 levels for 256² grid).
b. Reconstruct filtered velocity fields ũ_j(x,y,t) retaining only levels j ≥ j_min for j_min ∈ {1, 2, 3, 4} (low j_min = all scales, high j_min = large scales only). This effectively sets the minimum eddy size to ~2^j_min × dx.
c. For each filtered field ũ_j, compute the eddy lifetime T_eddy(j) as the integral timescale of the wavelet-coefficient temporal autocorrelation at scale j:
   T_eddy(j) = ∫_0^∞ C_W(τ, j) dτ / C_W(0, j)
   where C_W(τ, j) = ⟨W_j(x,y,t) W_j(x,y,t+τ)⟩_{x,y,t}.
d. Re-advect (or re-interpolate) the 5000 saved tracers in each filtered field ũ_j using the saved tracer positions as initial conditions. Compute displacement statistics for each filtered field: α(j), γ_MSD(j), ν_VACF(j).
e. Plot α(j) vs T_eddy(j) and test the CTRW prediction: if β ∝ T_eddy^δ, then α = 1 + β/(2-β) should give a monotonic curve. Fit α(j) = f(T_eddy(j)) and report the functional form and exponent.
f. Report whether removing small-scale eddies (increasing j_min) increases or decreases α (i.e., does filtering toward large-scale structures enhance or suppress Lévy flights?).

### Task 4: Synthesis

Combine Tasks 2 and 3:
- Do the time-windowed α(t) values from Task 2 agree with the wavelet-scale analysis from Task 3 at the dominant scale?
- Plot a 2D map of α as a function of (T_eddy, condensation state k_peak/k_box).
- State whether α is stationary or drifts with the spectral condensation state.

## Key Physical Predictions to Test

1. α decreases (heavier tails, more anomalous) as T_eddy increases — larger/longer-lived eddies produce more ballistic flights.
2. α(t) drifts toward lower values as the inverse cascade develops (k_peak → k_box), i.e., the flow becomes more anomalous as vortices grow to box scale.
3. The CTRW relation α = 1 + β/(2 − β) holds, with β(j) ∝ T_eddy(j)^δ.

## Hardware

- 64 vCPUs (AMD Ryzen Threadripper PRO 9995WX), 128 GB RAM
- NVIDIA RTX PRO 6000 Blackwell Edition (96 GB VRAM), CUDA 13.0
- PyTorch 2.12.0+cu130 — use device='cuda' for all spectral operations
- Use torch.fft.rfft2 / irfft2 for spectral transforms
- For wavelet decomposition: use PyWavelets (pywt) — already installed in the venv
- Multiprocessing: limit to 8 CPU workers for post-processing; all heavy compute on GPU

## Prior Results Context (for the engineer)

From levy_flights_2dns_v1 Iteration 4 (best iteration):
- α ≈ 1.4 ± 0.1, VACF decay ν ≈ 0.61–0.72
- Two-channel ejection: vortex death (~2.6%) + strain stripping (~97.4%)
- Jump length scaling L ~ d_vv^0.63 (but χ² test showed non-universality, p < 0.001)
- CTRW kurtosis ratio ~0.4 and S6 ratio ~5.5 — model inconsistency

From Iteration 5 (evaluator feedback):
- Short simulation window → α = 2.0 (Gaussian), FTLE ridge density R = 0
- Key lesson: need T >> 50 T_L and dense temporal sampling to resolve LCS and observe Lévy flights
- Non-stationarity hypothesis: α may depend on spectral condensation state k_peak(t)/k_box
