0. **DNS Implementation — MANDATORY ENGINEERING NOTES (do not deviate):**

   This iteration is identical to Iteration 3 with ONE change: N = 1024 (higher resolution). All other physics parameters, numerical schemes, and analysis steps are unchanged.

   a. **Grid:** N = 1024. Domain L = 2π. dx = L/N.
   b. **Forcing band:** k_force_min = 10, k_force_max = 12 (unchanged from Iteration 3).
   c. **Hyperviscosity:** nu_h = 3.9e-31, p = 4. This is nu_h_512 / 2^8 = 1e-28/256, scaled so dissipation at the dealiasing cutoff k_max~341 matches that of the 512² run.
   d. **Integrating factor method (IF):** MANDATORY. Do NOT use explicit RK4 for the hyperviscous term. The IF handles the stiff dissipation exactly:
      ```python
      diss_op = -nu_h * k2**p          # shape: (N, N//2+1)
      IF = torch.exp(diss_op * dt)     # recompute when dt changes by >1%
      # RK4 on nonlinear term only:
      k1 = nonlinear(omega_hat)
      k2_ = nonlinear(omega_hat + 0.5*dt*k1)
      k3 = nonlinear(omega_hat + 0.5*dt*k2_)
      k4 = nonlinear(omega_hat + dt*k3)
      omega_new = IF * (omega_hat + dt*(k1 + 2*k2_ + 2*k3 + k4)/6)
      # Stochastic forcing kick (one per step, Ito convention):
      phi = torch.exp(2j*np.pi*torch.rand(omega_hat.shape, device=device))
      omega_new = omega_new + (force_amp * np.sqrt(dt)) * phi * force_mask
      ```
   e. **Force amplitude calibration (MANDATORY):** Calibrate empirically at startup:
      ```python
      omega_calib = torch.zeros(N, N//2+1, dtype=torch.cfloat, device=device)
      dE_acc = 0.0; dt_cal = 0.1
      for _ in range(20):
          E0 = get_energy(omega_calib)
          phi = torch.exp(2j*np.pi*torch.rand(omega_calib.shape, device=device))
          omega_calib += np.sqrt(dt_cal) * phi * force_mask
          dE_acc += get_energy(omega_calib) - E0
      dE_per_amp2 = dE_acc / (20 * dt_cal)
      force_amp = float(np.sqrt(epsilon_inj / dE_per_amp2))
      ```
      where get_energy returns E = (0.5/N^4) * sum_k(|u_hat|^2 + |v_hat|^2). For N=1024 with epsilon_inj=0.1, expect force_amp ≈ 160,000–200,000 (scales as N^2 relative to 512²).
   f. **Wavenumber convention:** Integer wavenumbers (NOT physical):
      ```python
      k  = torch.fft.fftfreq(N, d=1.0/N)   # [0,1,...,N/2-1,-N/2,...,-1]
      kr = torch.fft.rfftfreq(N, d=1.0/N)  # [0,1,...,N/2]
      kx, ky = torch.meshgrid(k, kr, indexing='ij')
      k2 = kx**2 + ky**2;  k2_nz = k2.clone(); k2_nz[0,0] = 1.0
      ```
   g. **Spinup:** T_spinup = 20.0. Monitor k_peak; proceed to production when T_spinup reached.
   h. **CFL:** dt = min(0.4 * dx / U_max, 0.5). Use dt=0.05 for cold start before U builds up.
   i. **Snapshot intervals:** dt_snap = 0.01 (tracers), dt_vel = 0.5 (Eulerian fields coarsened 1024→256 by AvgPool2d(4)).
   j. **Tracers:** N_tracers = 5000, seeded uniformly at start of production. Tracer advection uses GPU grid_sample; map positions: pos_norm = (pos/(2*pi))*2 - 1.
   k. **Output directory:** /home/node/work/projects/levy_flights_2dns_v2/data/ (overwrite).
   l. **NaN/Inf guard:** After each step check torch.any(torch.isnan(omega_hat)) and raise RuntimeError if triggered.
   m. **Verification before production:** Print U_rms, T_L, k_peak. Require: U_rms*dt_snap < 0.1; T_L finite; k_peak ≥ 2.
   n. **GPU:** All tensors on device='cuda'. Use torch.fft.rfft2/irfft2 throughout.

1. **DNS Execution**:
   - Perform the 2D Navier-Stokes simulation on a $1024 \times 1024$ grid using the GPU-accelerated pseudo-spectral solver with integrating factor method.
   - Enforce the inverse cascade by forcing at $k \in [10, 12]$ with $\epsilon_{inj} = 0.1$.
   - Monitor $k_{peak}$ during spinup. Proceed to production after T_spinup = 20.0.
   - Run the production phase for $T_{prod} = 50 \times T_L$, saving tracer positions at $\Delta t_{snap} = 0.01$ and coarsened (1024→256) Eulerian fields at $\Delta t_{vel} = 0.5$.

2. **Data Quality and Cascade Validation**:
   - Compute mean tracer displacement $\langle|\Delta r|\rangle$ per snapshot; must be $< 0.3 \times 2\pi \approx 1.88$.
   - Calculate VACF $C_{vv}(\tau)$ to confirm velocity memory at short lags ($C_{vv}(\tau=1) > 0.3$).
   - Plot $E(k, t)$ at $t = 0.1, 0.3, 0.6, 1.0 \times T_{prod}$; confirm $k_{peak}$ drifts toward smaller $k$.

3. **Non-Stationary Lévy Analysis**:
   - Overlapping windows of width $W = 8 T_L$, step $\Delta W = 2 T_L$.
   - Estimate $\alpha$ per window using McCulloch quantile method (scalar x and y displacements separately; take median across lags $\tau = \{0.5, 1, 2\} \times T_L$).
   - CCDF log-log plots per window; flag windows with < 1.5 decades of power-law as non-Lévy.
   - ADF and KPSS stationarity tests on $\alpha(t)$, $\gamma(t)$, $\nu(t)$.
   - Fit $\alpha = a + b(k_{peak}/k_{box})^c$.

4. **Wavelet-Filtered Eddy Dynamics**:
   - Velocity snapshots are at 256×256 resolution. Apply 2D DWT using `db4` basis for levels $j=1,\dots,5$.
   - Compute Okubo-Weiss parameter $Q = s^2 - \omega^2$. When computing $T_{eddy}(j)$, use only pixels where $Q < -Q_{threshold}$ (set $Q_{threshold}$ to the 75th percentile of $|Q|$).
   - $T_{eddy}(j)$: integrate temporal autocorrelation of spatially-averaged wavelet energy until it drops below 0.1, max lag = min(T_prod/4, 50*dt_vel).

5. **Tracer Re-advection and Scaling Laws**:
   - Re-advect 5000 tracers through each filtered velocity field $\tilde{u}_j$ using GPU-accelerated RK4 with linear temporal interpolation between snapshots.
   - Compute $\alpha(j)$, $\gamma_{MSD}(j)$, $\nu_{VACF}(j)$.
   - Plot $\alpha(j)$ vs $T_{eddy}(j)$; fit and test CTRW prediction $\alpha = 1 + \beta/(2-\beta)$.

6. **Synthesis and Correlation**:
   - Correlate $\alpha(t)$ with $k_{peak}(t)$.
   - 2D map of $\alpha$ vs $(T_{eddy}, k_{peak}/k_{box})$.
   - Conclude on stationarity of $\alpha$ across the inverse cascade evolution.
