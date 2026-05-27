0. **DNS Implementation — MANDATORY ENGINEERING NOTES:**

   All DNS implementation details from Iteration 2 apply (integrating factor method, empirical force_amp calibration, integer wavenumbers, CFL timestep, NaN/Inf guard). Key parameters for this iteration:

   a. **Grid:** N = 512. Domain L = 2π. dx = L/N.
   b. **Forcing band:** k_force_min = 10, k_force_max = 12.
   c. **Ekman drag (NEW):** Add a linear large-scale drag term to the vorticity equation:
      ```
      dω/dt = N(ω) - ν_h(-∇²)^p ω - λ*ω + f
      ```
      where λ = 0.05 (Ekman friction coefficient). In spectral space this is simply:
      ```python
      drag_op = -lambda_ekman   # scalar, applied to all modes
      # Combined dissipation + drag integrating factor:
      IF = torch.exp((diss_op + drag_op) * dt)  # diss_op = -nu_h*k^(2p)
      ```
      The drag prevents full condensation at k=1, creating a quasi-steady state where k_peak stabilises at an intermediate value (typically k ~ 2–5 for λ=0.05). This gives the production run a wide window over which k_peak evolves and α(t) can be measured.
   d. **Spinup:** T_spinup = 60.0 (longer to allow quasi-steady state to establish under Ekman drag).
   e. **Snapshot intervals:** dt_snap = 0.01 (tracers), dt_vel = 0.5 (Eulerian fields 512→128).
   f. **Force amplitude:** Calibrate empirically at startup as in Iteration 2.
   g. **Hyperviscosity:** nu_h = 1e-28, p = 4, integrating factor method (mandatory).
   h. **Output directory:** /home/node/work/projects/levy_flights_2dns_v2/data/ (overwrite).
   i. **Verification:** After spinup, print U_rms, T_L, k_peak. k_peak should be at some intermediate value (not 1), confirming Ekman drag is preventing full condensation.

1. **Simulation Setup and Calibration**:
   - Initialize the $512 \times 512$ pseudo-spectral solver with Ekman drag $\lambda = 0.05$. Set forcing at $k \in [10, 12]$ with $\epsilon_{inj} = 0.1$.
   - Run spinup for T_spinup = 60.0 simulation time units. Monitor k_peak: it should settle at some intermediate value (k~2--6) rather than collapsing to k=1, confirming Ekman drag is active.
   - Seed 5000 tracers uniformly at the end of spinup.

2. **Production Run and Data Acquisition**:
   - Execute the production phase, defined as the interval where $k_{peak}$ is actively migrating through the inertial range. If $k_{peak}$ reaches $k \approx 1$ before $T_{prod} = 50 T_L$, truncate the data to avoid saturated dipole artifacts.
   - Save tracer positions and velocities at $\Delta t_{snap} = 0.01$. Verify $\langle|\Delta r|\rangle < 0.3 \times 2\pi \approx 1.88$ domain units per snapshot.
   - Save coarsened Eulerian velocity/vorticity fields (512$\to$128) at $\Delta t_{vel} = 0.5$.

3. **Verification of Inverse Cascade and Memory**:
   - Monitor $E(k, t)$ to confirm the migration of $k_{peak}$ from $k \approx 30$ toward $k \approx 1$.
   - Calculate the VACF $C_{vv}(\tau)$ at short lags to confirm non-zero memory ($\tau_{corr} > 0$).
   - If the cascade rate is too fast, reduce $\epsilon_{inj}$ to ensure the non-stationary evolution is captured over the full production window.

4. **Non-Stationary Lévy Analysis**:
   - Divide the production run into overlapping windows of width $W = 8 T_L$ with a step of $\Delta W = 2 T_L$.
   - For each window, estimate the Lévy stability index $\alpha$ using the McCulloch quantile method and validate with the Hill estimator.
   - Extract $\gamma(t)$, $\nu(t)$, and $k_{peak}(t)$ for each window.
   - Apply ADF and KPSS tests to the time series of $\alpha(t)$ to quantify the trend and non-stationarity of the Lévy flight behavior relative to the evolving $k_{peak}(t)$.

5. **High-Resolution Wavelet Decomposition**:
   - Perform a 2D Discrete Wavelet Transform (DWT) on the coarsened $128 \times 128$ velocity snapshots using the `db4` wavelet basis.
   - Decompose each snapshot into approximation and detail coefficients up to level $J=5$.
   - Reconstruct filtered velocity fields $\tilde{u}_j$ for $j \in \{1, \dots, 6\}$ by retaining only the relevant detail coefficients to isolate specific eddy scales.

6. **Scale-Dependent Eddy-Lifetime Calculation**:
   - Compute the wavelet energy $E_j(x, y, t) = \sum_d |D_j^d(x, y, t)|^2$ at full resolution.
   - Estimate the eddy lifetime $T_{eddy}(j)$ by integrating the temporal autocorrelation $C_E(\tau, j)$ until it drops below 0.1.
   - Use these filtered fields to re-advect the tracers and compute scale-dependent statistics $\alpha(j)$ and $\gamma(j)$.

7. **Scaling Law Testing**:
   - Plot $\alpha(j)$ against $T_{eddy}(j)$ to determine the functional form of the relationship.
   - Test the CTRW prediction $\alpha = 1 + \beta/(2-\beta)$ by fitting $\beta \propto T_{eddy}^\delta$.
   - Compare the global $\alpha(t)$ from the non-stationary analysis with the scale-dependent $\alpha(j)$ corresponding to the instantaneous $k_{peak}(t)$ of the flow.

8. **Synthesis**:
   - Construct a 2D map of $\alpha$ as a function of $(T_{eddy}, k_{peak}/k_{box})$ to visualize the transition of Lévy flight characteristics.
   - Conclude on the stationarity of $\alpha$ by synthesizing the results from the temporal window analysis and the wavelet-based scale analysis, explicitly addressing how the Lévy index evolves as the inverse cascade condenses.