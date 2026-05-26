1.  **Data Loading and Preparation:**

    a.  Load the simulation data from the specified absolute paths: `tracer_positions.npy`, `tracer_velocities.npy`, `vorticity_snapshots.npy`, `velocity_snapshots.npy`, `times.npy`, `energy_spectrum.npy`, `diagnostics.npy`, and `sim_params.json`.
    b.  Extract relevant simulation parameters from `sim_params.json`, including `N`, `dt`, `nu_h`, `epsilon_inj`, `N_tracers`, `N_snapshots`, and the initial `T_L_estimate`.
    c.  Verify the shape and data types of each loaded array to ensure consistency with the data description.
    d.  Calculate `k_box = 1` (fundamental wavenumber).

2.  **Inverse Cascade Verification:**

    a.  Load `energy_spectrum.npy`.
    b.  Define target times for plotting the energy spectra: `t_targets = [0.1, 0.3, 0.6, 1.0] * T_prod`, where `T_prod` is the total production time (can be inferred from `times.npy`).
    c.  Find the indices in `times.npy` that are closest to the target times.
    d.  Plot the energy spectra E(k) at the selected time indices. Overlay a line representing E(k) ~ k^(-5/3) for comparison.
    e.  Load `diagnostics.npy` and extract `k_peak(t)`. Plot `k_peak(t)` as a function of time to visualize the energy condensation process. Confirm that `k_peak(t)` shifts towards `k=1` over the production run.

3.  **Non-Stationarity Analysis:**

    a.  Define the window width `W = 8 * T_L_estimate` and the step size `ΔW = 2 * T_L_estimate`. Justification: `W = 8 * T_L_estimate` is chosen to capture several eddy turnover times, providing sufficient data for statistical analysis within each window. `ΔW = 2 * T_L_estimate` provides sufficient overlap between windows to ensure temporal resolution while maintaining statistical independence between adjacent windows.
    b.  Create a series of overlapping temporal windows covering the full production run. The start times of the windows will be `t_start = np.arange(0, times[-1] - W, ΔW)`. The number of windows will be `len(t_start)`.
    c.  For each window `w`:
        i.   Find the indices in `times.npy` that fall within the current window.
        ii.  Extract the tracer positions within the window: `tracer_positions_w = tracer_positions[indices, :, :]`.
        iii. Compute displacement PDFs P(Δx, τ) at lags τ = {0.5, 1, 2} * `T_L_estimate`. Use Kernel Density Estimation (KDE) to estimate the PDFs.
        iv.  Fit a Lévy-stable distribution to each P(Δx, τ) to obtain the Lévy index α_w for each τ. Report confidence intervals or standard errors for the fitted α_w values. Analyze how α_w changes with τ within each window.
        v.   Compute the Mean Squared Displacement (MSD) as a function of lag time τ within the window. Fit a power law to MSD(τ) to obtain the anomalous exponent γ_w. Report confidence intervals or standard errors for the fitted γ_w values.
        vi.  Compute the Velocity Autocorrelation Function (VACF) C_vv(τ) within the window. Fit an algebraic decay to C_vv(τ) to obtain the decay exponent ν_w. Report confidence intervals or standard errors for the fitted ν_w values.
        vii. Extract the mean inter-vortex distance `d_vv(w)` from the `diagnostics.npy` array for the current window by averaging the values within the window.
        viii. Extract the energy condensation wavenumber `k_peak(w)` from the `diagnostics.npy` array for the current window by averaging the values within the window.
    d.  Create time series of α(t), γ(t), ν(t), d_vv(t), and k_peak(t). Assign the window values to the center time of each window using a simple average.
    e.  Test the stationarity of each time series using both the Augmented Dickey-Fuller test and the Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test.
    f.  Plot α(t) vs k_peak(t)/k_box and fit the relationship α = a + b * (k_peak(t)/k_box)^c. Report the fitted parameters a, b, and c, along with their confidence intervals or standard errors.

4.  **Wavelet Decomposition and Eddy-Lifetime Scaling:**

    a.  Choose a wavelet basis for 2D spatial decomposition. Recommended choices are Daubechies (db4, db6) or Symlets (sym6). Justification: The choice of wavelet basis will be justified based on factors such as computational cost, boundary effects, and the specific features of the velocity fields. Daubechies and Symlets families offer a good balance between compact support, orthogonality, and smoothness. Boundary effects will be considered when choosing the wavelet basis, and wavelets that handle boundaries well will be preferred.
    b.  Apply 2D wavelet decomposition to each velocity snapshot u(x,y,t) and v(x,y,t) from `velocity_snapshots.npy` using the chosen wavelet basis. Obtain wavelet coefficient arrays W_j(x,y,t) for levels j = 1, …, J (J ≈ 6–7 levels for 256² grid). Acknowledge and address potential boundary effects in the wavelet decomposition. Strategies for mitigating these effects, such as padding or tapering the velocity fields, will be explored.
    c.  Reconstruct filtered velocity fields ũ_j(x,y,t) retaining only levels j ≥ j_min for j_min ∈ {1, 2, 3, 4}.
    d.  For each filtered field ũ_j:
        i.   Compute the eddy lifetime T_eddy(j) as the integral timescale of the wavelet-coefficient temporal autocorrelation at scale j:
             T_eddy(j) = ∫_0^∞ C_W(τ, j) dτ / C_W(0, j)
             where C_W(τ, j) = ⟨W_j(x,y,t) W_j(x,y,t+τ)⟩_{x,y,t}. Estimate the integral using the trapezoidal rule. The upper limit of integration τ_max will be chosen by checking for convergence of the integral as τ_max increases. The impact of noise on the autocorrelation function will be considered, and potential methods for smoothing or filtering the autocorrelation function before integration will be explored.
        ii.  Re-interpolate the velocities at the saved tracer positions using the filtered velocity fields ũ_j. The same time step `dt` as in the original simulation will be used. Bilinear spectral interpolation will be used to interpolate the velocities at the tracer positions.
        iii. Compute displacement statistics for each filtered field: α(j), γ_MSD(j), ν_VACF(j).
    e.  Plot α(j) vs T_eddy(j) and test the CTRW prediction. Fit α(j) = f(T_eddy(j)) and report the functional form and exponent.
    f.  Report whether removing small-scale eddies (increasing j_min) increases or decreases α.

5.  **Synthesis:**

    a.  Compare the time-windowed α(t) values from Task 2 with the wavelet-scale analysis from Task 3 at the dominant scale. The dominant scale will be defined as the wavelet scale j corresponding to the wavenumber closest to `k_peak(t)`.
    b.  Create a 2D map of α as a function of (T_eddy, condensation state k_peak/k_box). Interpolate the α(j) values from Task 3 and the α(t) values from Task 2 onto a grid of (T_eddy, k_peak/k_box) values using bilinear interpolation.
    c.  State whether α is stationary or drifts with the spectral condensation state based on the results of the Augmented Dickey-Fuller test and the KPSS test, and the 2D map.

6. **Error Handling and Uncertainty Quantification:**

    a. Implement error handling strategies, such as checking for NaN or Inf values in the data and implementing appropriate handling mechanisms (e.g., skipping problematic data points or using imputation techniques).
    b. Estimate the uncertainties in the fitted parameters (α, γ, ν, a, b, c) and the eddy lifetimes using appropriate statistical methods (e.g., bootstrapping or error propagation).

7. **Limitations:**

    a. Discuss potential limitations of the study, such as the finite size of the domain and the limited number of tracers.
    b. Acknowledge that the results may be sensitive to the choice of simulation parameters and analysis methods.
\