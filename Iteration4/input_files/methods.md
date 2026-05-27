1. **Simulation Setup and Calibration**:
   - Initialize the $1024 \times 1024$ pseudo-spectral solver. Set forcing at $k \in [30, 40]$ with an initial $\epsilon_{inj} = 0.1$.
   - Perform a calibration run to ensure the inverse cascade develops slowly: adjust $\epsilon_{inj}$ such that $k_{peak}$ migrates from $k \approx 20$ to $k \approx 2$ over a duration of at least $50 T_L$.
   - Run the simulation until the inverse cascade is fully established (spinup $T/T_L \geq 10$), then seed 5000 tracers uniformly at random.

2. **Production Run and Data Acquisition**:
   - Execute the production phase, defined as the interval where $k_{peak}$ is actively migrating through the inertial range. If $k_{peak}$ reaches $k \approx 1$ before $T_{prod} = 50 T_L$, truncate the data to avoid saturated dipole artifacts.
   - Save tracer positions and velocities at $\Delta t_{snap} = 0.05$. Verify $\langle|\Delta r|\rangle < 0.3$ domain units per snapshot to ensure temporal resolution is sufficient for Lévy statistics.
   - Save Eulerian velocity and vorticity fields at $\Delta t_{vel} = 2.0$ at the full $1024 \times 1024$ resolution to preserve small-scale strain field information.

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
   - Perform a 2D Discrete Wavelet Transform (DWT) on the full $1024 \times 1024$ velocity snapshots using the `db6` wavelet basis.
   - Decompose each snapshot into approximation and detail coefficients up to level $J=8$.
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