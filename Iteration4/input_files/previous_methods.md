0. **DNS Implementation — MANDATORY ENGINEERING NOTES (do not deviate):**

   All DNS implementation details from Iteration 2 apply here unchanged (integrating factor method, empirical force_amp calibration, integer wavenumbers, CFL timestep). The following parameters are modified:

   a. **Grid:** N = 512 (not 1024). Domain L = 2π.
   b. **Forcing band:** k_force_min = 10, k_force_max = 12 (NOT k∈[3,5]). This shifts the injection scale away from the box scale, giving the inverse cascade room to develop from k~11 down to k~1.
   c. **Hyperviscosity:** nu_h = 1e-28, p = 4 (unchanged).
   d. **Integrating factor method (IF):** MANDATORY. Do NOT use explicit RK4 for the hyperviscous term. See Iteration 2 methods for full implementation.
   e. **Force amplitude:** Calibrate empirically at startup (as in Iteration 2). Inject with force_amp=1 for 20 steps at dt=0.1, measure dE/dt per unit amp², then set force_amp = sqrt(epsilon_inj / dE_per_amp2). Expected force_amp ≈ 80,000–100,000 for N=512.
   f. **Spinup:** T_spinup = 20.0. Monitor k_peak; once it begins moving from ~11 toward lower k, proceed to production.
   g. **Snapshot intervals:** dt_snap = 0.01 (tracers), dt_vel = 0.5 (Eulerian fields coarsened 512→128).
   h. **CFL:** dt = min(0.4 * dx / U_max, 0.5). dt_init = 0.05 for cold start.
   i. **Tracers:** N_tracers = 5000, seeded uniformly at start of production.
   j. **Output directory:** /home/node/work/projects/levy_flights_2dns_v2/data/ (same as before — overwrite).
   k. **Verification before production:** U_rms * dt_snap < 0.1; T_L finite and > 0; k_peak ≥ 2 (not yet condensed).

1. **DNS Execution**:
   - Perform the 2D Navier-Stokes simulation on a $512 \times 512$ grid using the GPU-accelerated pseudo-spectral solver with integrating factor method.
   - Enforce the inverse cascade by forcing at $k \in [10, 12]$ with $\epsilon_{inj} = 0.1$.
   - Monitor $k_{peak}$ during spinup. Initiate the production phase once the spinup time T_spinup=20 is reached or k_peak first drops below 8.
   - Run the production phase for $T_{prod} = 50 \times T_L$, saving tracer positions at $\Delta t_{snap} = 0.01$ and Eulerian velocity/vorticity fields at $\Delta t_{vel} = 0.5$.

2. **Data Quality and Cascade Validation**:
   - Compute the mean tracer displacement $\langle|\Delta r|\rangle$ per snapshot; ensure it remains $< 0.3$ domain units.
   - Calculate the VACF $C_{vv}(\tau)$ to confirm non-zero memory at short lags ($\tau=1$ step).
   - Verify the inverse cascade by plotting $E(k, t)$ at multiple intervals, confirming the migration of $k_{peak}$ toward $k=1$.

3. **Non-Stationary Lévy Analysis**:
   - Divide the production run into overlapping windows of width $W = 8 T_L$ with step $\Delta W = 2 T_L$.
   - For each window, estimate $\alpha$ using the McCulloch quantile method.
   - Generate CCDF log-log plots for each window; if the distribution does not exhibit a linear region of at least 1.5 decades, flag the window as "non-power-law" and exclude it from the $\alpha(t)$ trend analysis.
   - Perform ADF and KPSS tests on the resulting $\alpha(t)$ time series to rigorously assess stationarity.

4. **Wavelet-Filtered Eddy Dynamics**:
   - Velocity snapshots are saved at 128×128 resolution (coarsened from 512×512 by factor 4 during the simulation). No additional downsampling needed.
   - Perform 2D DWT using the `db4` basis on the $128 \times 128$ fields for levels $j=1, \dots, 5$.
   - Compute the Okubo-Weiss parameter $Q = s^2 - \omega^2$ for each snapshot. When calculating $T_{eddy}(j)$, integrate the autocorrelation of wavelet energy only within regions where $Q < -Q_{threshold}$ to ensure $T_{eddy}$ reflects physical vortex lifetimes rather than background noise.

5. **Tracer Re-advection and Scaling Laws**:
   - Re-advect 5000 tracers through each filtered velocity field $\tilde{u}_j$ using RK4 integration.
   - Compute $\alpha(j)$, $\gamma_{MSD}(j)$, and $\nu_{VACF}(j)$ for each filtered field.
   - Plot $\alpha(j)$ vs $T_{eddy}(j)$ and test the CTRW prediction $\alpha = 1 + \beta/(2-\beta)$, where $\beta \propto T_{eddy}^\delta$.

6. **Synthesis and Correlation**:
   - Correlate $\alpha(t)$ with the instantaneous $k_{peak}(t)$ to determine if anomalous diffusion is a function of the flow's maturity (spectral condensation state).
   - Construct a 2D map of $\alpha$ as a function of $(T_{eddy}, k_{peak}/k_{box})$ to visualize how the condensation state dictates the anomalous diffusion regime.
   - Conclude on the stationarity of $\alpha$ by comparing the drift observed in the global analysis against the scale-dependent predictions derived from the wavelet decomposition.