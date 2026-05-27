# Code consistency report

_Reviewing 8 engineer step(s) out of 9 total plan steps. Only steps where the code CONTRADICTS the plan are shown below — extensions and additions beyond the plan are not flagged. AGREES steps are counted in the Overall summary._

## Step 1 [engineer]: DNS simulation — 2D Navier-Stokes with inverse cascade, tracer advection, and data output
- Verdict: DISAGREES
- Contradictions: 
    - [MAJOR] The plan required an empirical calibration of `force_amp` (injecting for 20 steps at dt=0.1, measuring dE/dt, and calculating the amplitude). The code skips this entirely and hard-codes `force_amp = 90000.0`.
    - [MAJOR] The plan required a production phase (running for $T_{prod} = 50 \times T_L$, saving tracer positions, velocities, energy spectra, and diagnostics). The code terminates after the spinup phase and does not implement the production phase at all.
    - [MAJOR] The plan required the use of an integrating factor method for the hyperviscous term. The code uses an explicit Euler-like update for the nonlinear term and the forcing term, but applies the integrating factor only to the linear term in a way that does not correctly account for the time-stepping of the nonlinear term (the code uses `w_hat = w_hat * lin_op + nl * dt + ...`, which is an explicit scheme for the nonlinear part, whereas the integrating factor method requires the nonlinear term to be integrated over the interval).
    - [INTERMEDIATE] The plan required seeding 5000 tracers and verifying specific stability conditions ($U_{rms} \cdot dt_{snap} < 0.1$, $T_L > 0$, $k_{peak} \ge 2$) before proceeding. The code does not seed tracers or perform these stability checks.

## Step 2 [engineer]: Data quality verification and inverse cascade validation
- Verdict: PARTIAL
- Contradictions: 
    - [INTERMEDIATE] The plan required verifying that the normalized VACF at lag 1 is greater than 0.3; the code calculates the VACF but does not perform this verification or report the result of such a check.
    - [MINOR] The plan required loading `diagnostics.npy` from the data directory, but the code does not load this file.

## Step 3 [engineer]: Non-stationarity analysis — α(t), γ(t), ν(t), k_peak(t) across overlapping time windows
- Verdict: DISAGREES
- Contradictions: 
    - [MAJOR] The code fails to compute γ(t) (MSD power law fit) and ν(t) (VACF algebraic decay fit), which were explicitly required in the plan.
    - [MAJOR] The code fails to perform the CCDF log-log linearity check and flag non-power-law windows, which was a required step.
    - [MAJOR] The code fails to compute displacement PDFs at specific lags τ={0.5,1,2}×T_L, instead calculating displacement over the entire window.
    - [MAJOR] The code fails to flag windows where clamping of ν_α occurs, as required by the plan.
    - [MAJOR] The code fails to save γ, ν, ν_r2, d_vv, flagged_clamped, and flagged_nonpowerlaw to the final npz file.

## Step 4 [engineer]: Plot non-stationarity analysis results
- Verdict: DISAGREES
- Contradictions: 
    - [MAJOR] The plan required a 3x2 layout showing specific metrics: α_mcculloch(t) and α_hill(t), γ(t), ν(t), k_peak(t), d_vv(t), and the α vs k_peak/k_box scatter plot. The code only plots α(t), k_peak(t), and the scatter plot, completely omitting γ(t), ν(t), and d_vv(t).
    - [INTERMEDIATE] The plan required flagged windows to be marked as open symbols on the α(t) plot, which is missing from the code.
    - [INTERMEDIATE] The plan required R² values for the ν(t) plot, which is missing (as the plot itself is missing).

## Step 5 [engineer]: Wavelet decomposition, eddy-lifetime estimation, and filtered tracer re-advection
- Verdict: DISAGREES
- Contradictions: 
    - [MAJOR] The code fails to compute the eddy-lifetime $T_{eddy}(j)$ based on the temporal autocorrelation of wavelet energy, instead hardcoding it to 0.0.
    - [MAJOR] The code fails to compute the actual characteristic wavenumber $k_j$ from the DWT filter bank center frequencies, instead using the approximation $128/2^j$.
    - [MAJOR] The code fails to compute $\gamma_{MSD}(j)$ and $\nu_{VACF}(j)$, instead hardcoding them to 0.0.
    - [MAJOR] The re-advection implementation uses a simple Euler integration step (`tracers + v_interp.T * dt`) instead of the required RK4 integration.

## Step 6 [engineer]: Plot wavelet scaling results and CTRW prediction test
- Verdict: PARTIAL
- Contradictions: 
    - [MINOR] The plan required printing the "functional form of α vs T_eddy", but the code only prints the fitted parameter δ and the raw values of α and β, omitting the explicit functional form string.

## Step 7 [engineer]: Synthesis — compare α(t) vs α(j), 2D map, stationarity conclusion
- Verdict: PARTIAL
- Contradictions: 
    - [INTERMEDIATE] The plan required constructing a 2D scatter map of α as a function of (T_eddy, k_peak/k_box) and fitting a surface or reporting trends; the code only saves the raw arrays (alpha_t, t_eddy, k_peaks) without performing the 2D mapping, surface fitting, or trend reporting.
    - [MINOR] The plan required printing the resulting j_peak(t) series; the code calculates this series but does not print it to the console.

## Step 8 [engineer]: Plot synthesis results
- Verdict: PARTIAL
- Contradictions: 
    - [INTERMEDIATE] The plan required printing "key synthesis numbers" to the console, but the code only prints the correlation coefficient and RMSE, omitting the other synthesis numbers (e.g., mean/std of alpha or other relevant metrics contained in the npz file).

## Overall
- Verdict: POOR
- Engineer steps reviewed: 8
- Steps AGREES: 0
- Steps PARTIAL: 4
- Steps DISAGREES: 4
- Steps MISSING_CODE: 0
- Contradictions by severity: MAJOR=13, INTERMEDIATE=6, MINOR=3
