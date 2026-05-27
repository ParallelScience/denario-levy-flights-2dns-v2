0. **DNS Implementation — MANDATORY ENGINEERING NOTES:**

   Identical to Iteration 4 except for the forcing band. All implementation details (integrating factor method, empirical force_amp calibration, integer wavenumbers, GPU throughout) are unchanged.

   a. **Grid:** N = 1024. Domain L = 2π. dx = L/N.
   b. **Forcing band:** k_force_min = 1, k_force_max = 3 (changed from [10,12]). This forces at the *largest* scales, which at 1024² gives maximum inertial range for the inverse cascade to develop.
   c. **Hyperviscosity:** nu_h = 3.9e-31, p = 4 (same as Iteration 4).
   d. **Integrating factor (IF):** MANDATORY — same implementation as Iteration 4.
   e. **Force amplitude:** Calibrate empirically at startup (same procedure as Iteration 4). For k∈[1,3], n_force will be smaller (~5–10 modes), so force_amp will be larger. The calibration will handle this automatically.
   f. **Spinup:** T_spinup = 20.0 (fixed time, not T/T_L condition).
   g. **CFL:** dt = min(0.4 * dx / U_max, 0.5). Cold start: dt = 0.05.
   h. **Snapshot intervals:** dt_snap = 0.01 (tracers), dt_vel = 0.5 (coarsen 1024→256 by AvgPool2d(4)).
   i. **Tracers:** N_tracers = 5000, seeded uniformly at end of spinup.
   j. **Output:** /home/node/work/projects/levy_flights_2dns_v2/data/ (overwrite).
   k. **GPU:** ALL tensor operations on device='cuda'. Use torch.fft.rfft2/irfft2, torch.nn.functional.grid_sample for tracer interpolation.
   l. **NaN/Inf guard:** Check after each step, raise RuntimeError if triggered.
   m. **Verification:** After spinup print U_rms, T_L, k_peak. Require U_rms*dt_snap < 0.1.
   n. **NO wavelet re-advection step** in this iteration — replaced by LCS/Okubo-Weiss analysis.

1. **DNS Execution and Inverse Cascade Development**:
   - Execute the 2D Navier-Stokes solver on a $1024 \times 1024$ grid using the specified GPU-accelerated pseudo-spectral method.
   - Force at $k \in [1, 3]$ with $\epsilon_{inj} = 0.1$. This maximises scale separation on a 1024² grid.
   - Run spinup for T_spinup = 20.0, then seed tracers and run production for $T_{prod} = 50 \times T_L$.
   - Save tracer positions at $\Delta t_{snap} = 0.01$ and coarsened (1024→256) Eulerian fields at $\Delta t_{vel} = 0.5$.

2. **Verification of Snapshot Quality**:
   - Compute the mean tracer displacement $\langle|\Delta r|\rangle$ per snapshot; ensure it remains $< 0.3$ domain units.
   - Calculate the VACF $C_{vv}(\tau)$ at short lags to confirm $\tau_{corr}$ is well-resolved.
   - Verify the inverse cascade by plotting $E(k, t)$ at multiple intervals to confirm the drift of $k_{peak}$ from $k \approx 4$ toward $k \approx 1$.

3. **Non-Stationary Lévy Analysis**:
   - Divide the production run into overlapping temporal windows of width $W = 8 T_L$ with a step of $\Delta W = 2 T_L$.
   - Estimate $\alpha(t)$ using the McCulloch quantile method and Hill estimator at lags $\tau = \{0.5, 1, 2\} T_L$.
   - Perform ADF and KPSS tests to quantify non-stationarity, using Newey-West standard errors to account for temporal autocorrelation between overlapping windows.

4. **Lagrangian Coherent Structure (LCS) Identification**:
   - Compute the Okubo-Weiss parameter $Q = s^2 - \omega^2$ from Eulerian velocity snapshots.
   - Define "vortex" and "strain" regions using a dynamic threshold $Q_{threshold}(t) = \pm \sigma_Q(t)$ (the instantaneous standard deviation of $Q$) to maintain physical consistency as the flow coarsens.
   - Record $Q_{threshold}$ and the count of hyperbolic points in `diagnostics.npy`.

5. **FTLE Field Computation**:
   - Compute FTLE fields using a sliding window $\Delta t_{FTLE} \approx 2–4 \times T_L$.
   - Integrate the flow map $\Phi$ using linear interpolation between velocity snapshots at $\Delta t_{vel} = 0.5$.
   - Perform FTLE computation on GPU using PyTorch for speed.
   - Use a vectorized approach or compute in chunks to ensure memory usage remains within the 128 GB RAM limit.

6. **Residence Time and Flight Time Distribution**:
   - Categorize tracer segments as "trapped" ($Q > Q_{threshold}$) or "flight" ($Q < -Q_{threshold}$).
   - Apply a minimum displacement threshold to "flight" segments to filter out small-scale jitter.
   - Extract the probability distribution of residence and flight times; test for power-law consistency with $\alpha(t)$.
   - Tag tracers exiting vortices via "vortex death" ($Q$ becoming positive) vs. "strain stripping" (proximity to hyperbolic points) to quantify the two-channel ejection hypothesis.

7. **Causal Link Analysis**:
   - Perform a time-lagged cross-correlation and Granger Causality test between the rate of change of energy at the largest scales ($dE(k_{min})/dt$) and $d\alpha(t)/dt$.
   - Evaluate the correlation between the spatial density of hyperbolic points and $\alpha(t)$ to determine if spectral condensation drives the evolution of Lagrangian geometry and, consequently, the Lévy index.

8. **Synthesis of Results**:
   - Construct a 2D map of $\alpha$ as a function of hyperbolic point density and $k_{peak}/k_{box}$.
   - Perform sensitivity analysis on the $Q_{threshold}$ (e.g., 70th vs 80th percentile) to ensure robustness of power-law exponents.
   - Conclude whether the non-stationarity of $\alpha$ is a direct consequence of spectral condensation and the reorganization of the flow's Lagrangian coherent structures.