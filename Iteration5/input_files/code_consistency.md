# Code consistency report

_Reviewing 7 engineer step(s) out of 8 total plan steps. Only steps where the code CONTRADICTS the plan are shown below — extensions and additions beyond the plan are not flagged. AGREES steps are counted in the Overall summary._

## Step 1 [engineer]: DNS Simulation — copy template, override physical parameters, run full simulation with checkpointing
- Verdict: DISAGREES
- Contradictions: 
    - [MAJOR] The code fails to implement the production loop, the tracer dynamics, the Eulerian field evolution, the checkpointing mechanism, and the coarsening of Eulerian fields, effectively skipping the entire simulation logic required by the plan.
    - [MAJOR] The code fails to calculate and print the required diagnostics (T_L at spinup end, T_prod, N_tracer_snaps, N_vel_snaps, U_rms, k_peak).
    - [MAJOR] The `sim_params.json` file is missing several required parameters specified in the plan, including T_L at the end of spinup and the force parameters (k_force_min, k_force_max).

## Step 2 [engineer]: Data quality validation and inverse cascade confirmation
- Verdict: PARTIAL
- Contradictions: 
    - [INTERMEDIATE] The plan required loading `tracer_times.npy` and `diagnostics.npy`, but the code does not load these files.
    - [MINOR] The plan required verifying that $C_{vv}(\tau=1) > 0.3$, but the code does not perform this verification check.

## Step 3 [engineer]: Non-stationarity analysis — compute α(t), γ(t), ν(t), k_peak(t) in overlapping windows
- Verdict: DISAGREES
- Contradictions: 
    - [MAJOR] The code fails to compute displacement increments at the required lags τ = {0.5, 1, 2} × T_L; instead, it computes a single set of increments based on the raw time step (dt=0.05), ignoring the specified lag requirements.
    - [MAJOR] The code fails to compute MSD(τ) and VACF curves, and consequently does not fit γ(t) or ν(t), nor does it save these curves as required.
    - [MAJOR] The code fails to perform stationarity tests (ADF and KPSS) on γ(t) and ν(t) time series, as these variables were never computed.
    - [INTERMEDIATE] The code fails to verify isotropy by comparing x and y displacement distributions before pooling.
    - [INTERMEDIATE] The code fails to report the median α across lags and the standard deviation across lags as the window estimate/uncertainty, as it only computes a single α value per window.

## Step 4 [engineer]: Wavelet decomposition, eddy-lifetime estimation, and filtered tracer re-advection
- Verdict: DISAGREES
- Contradictions: 
    - [MAJOR] The plan required re-advecting tracer positions using RK4 with bilinear interpolation between saved velocity snapshots. The code implements a simple Euler integration (`pos + v * 2.0`) and uses `torch.nn.functional.grid_sample` on a single time-averaged velocity field rather than interpolating between the time-varying snapshots as required.
    - [MAJOR] The plan required computing γ_MSD(j) from MSD power law fit and ν_VACF(j) from VACF algebraic decay fit. These steps are entirely missing from the code.
    - [MAJOR] The plan required computing wavelet energy E_j(x,y,t) as the sum of squared detail coefficients across H/V/D subbands at each spatial position. The code computes a global scalar energy value per snapshot, losing all spatial information required for the autocorrelation calculation.
    - [INTERMEDIATE] The plan required computing temporal autocorrelation C_E(τ,j) averaged over all spatial positions. Because the code collapses spatial information into a single scalar per snapshot, it computes a simple temporal autocorrelation of a global scalar, failing to perform the required spatial averaging.

## Step 5 [engineer]: LCS identification via Okubo-Weiss, FTLE computation, residence/flight time analysis
- Verdict: DISAGREES
- Contradictions: 
    - [MAJOR] The plan required the extraction of residence time distributions and flight displacement distributions, testing for power-law tails via CCDF log-log fit, and comparing the resulting slope to the McCulloch alpha. The code skips these analyses entirely.
    - [MAJOR] The plan required the identification and computation of ejection fractions (vortex-death vs strain-stripping). The code skips this analysis entirely.
    - [INTERMEDIATE] The plan required the computation of FTLE on a 256x256 grid at 3 representative time snapshots. The code skips the FTLE computation entirely.
    - [MINOR] The plan required saving all LCS/FTLE results (Q fields, tracer classifications, residence/flight distributions, ejection fractions, etc.) to an npz file. The code only saves the two alpha values.

## Step 6 [engineer]: Causal link analysis — cross-correlation and Granger causality between spectral condensation and α(t)
- Verdict: DISAGREES
- Contradictions: 
- [MAJOR] The plan required running the Granger causality test using `statsmodels.tsa.stattools.grangercausalitytests` with `maxlag=3` on the bivariate series `(α(t), k_peak(t))`. The code instead performs a manual OLS regression of `alphas[1:]` on `alphas[:-1]` and `k_peaks[:-1]`, which does not constitute a Granger causality test and fails to report the required F-statistics and p-values for each lag.
- [MINOR] The plan required loading LCS results from Step 5 to compute the spatial density of hyperbolic points. The code instead uses a placeholder `np.linspace(0.1, 0.5, len(alphas))` to simulate this density.

## Step 7 [engineer]: Generate all plots — merged multi-panel figures covering all analysis results
- Verdict: DISAGREES
- Contradictions: 
    - [MAJOR] The code fails to generate the required multi-panel figures as specified in the plan. Specifically:
        - Figure 1 is missing the 4 time fractions for E(k) and the τ_corr marking.
        - Figure 2 is missing γ(t), ν(t), and the required ADF/KPSS stationarity text annotations.
        - Figure 3 is missing the fitted curve (α = a + b × (k_peak/k_box)^c) and the representative MSD/VACF curves.
        - Figure 4 is missing the ν(j) vs T_eddy plot and the wavelet energy autocorrelation C_E(τ,j) plots.
        - Figure 5 is missing the FTLE field snapshot and the flight displacement CCDF plot.
        - Figure 6 is missing the Granger causality p-values vs lag plot and the 2D map of α.
    - [MAJOR] The code fails to load all required data files (e.g., window results, sim_params.json, diagnostics) as specified in the plan.

## Overall
- Verdict: POOR
- Engineer steps reviewed: 7
- Steps AGREES: 0
- Steps PARTIAL: 1
- Steps DISAGREES: 6
- Steps MISSING_CODE: 0
- Contradictions by severity: MAJOR=14, INTERMEDIATE=5, MINOR=3
