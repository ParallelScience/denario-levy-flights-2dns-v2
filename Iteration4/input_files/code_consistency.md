# Code consistency report

_Reviewing 6 engineer step(s) out of 7 total plan steps. Only steps where the code CONTRADICTS the plan are shown below — extensions and additions beyond the plan are not flagged. AGREES steps are counted in the Overall summary._

## Step 1 [engineer]: DNS simulation — 2D Navier-Stokes solver with tracer advection
- Verdict: DISAGREES
- Contradictions: 
- [MAJOR] The code fails to implement the DNS simulation loop, tracer advection, Eulerian snapshot saving, energy spectrum calculation, or diagnostic logging. It only performs a calibration calculation for the forcing amplitude and saves a parameters file, skipping the entire simulation execution required by the plan.

## Step 2 [engineer]: Data quality verification and inverse cascade validation
- Verdict: PARTIAL
- Contradictions: 
    - [MINOR] The plan required loading `tracer_times.npy` and `diagnostics.npy`, but the code does not load these files.

## Step 3 [engineer]: Non-stationarity analysis of Lévy exponents across overlapping time windows
- Verdict: DISAGREES
- Contradictions: 
    - [MAJOR] The plan required computing net displacement magnitudes at three specific lags ($\tau=\{0.5, 1, 2\}*T_L$) and fitting the McCulloch method based on those distributions. The code instead computes displacements at a single lag (the time step of the data) and performs the McCulloch fit on that single distribution.
    - [MAJOR] The plan required fitting MSD(τ) power law to get $\gamma(t)$ and VACF algebraic decay to get $\nu(t)$ for each window. The code skips these calculations entirely.
    - [MAJOR] The plan required running ADF and KPSS stationarity tests on $\gamma(t)$ and $\nu(t)$ time series. Since these variables were not computed, the tests were not performed.
    - [MAJOR] The plan required saving $\gamma$, $\nu$, and the stationarity test results to the npz file. These are missing from the saved output.
    - [INTERMEDIATE] The plan required a bilinear interpolation in the $(\nu_\alpha, \nu_\beta)$ plane for the McCulloch method. The code uses a simple 1D linear interpolation on $\nu_\alpha$ only, ignoring the $\nu_\beta$ (skewness) parameter entirely.

## Step 4 [engineer]: Plot non-stationarity analysis results
- Verdict: DISAGREES
- Contradictions: 
    - [MAJOR] The plan required plotting γ(t), ν(t), and d_vv(t) vs time window center, but these variables are entirely missing from the generated figure.
    - [INTERMEDIATE] The plan required a confidence band to be included in the scatter plot of α vs k_peak/k_box, but the code only plots the fitted curve without a confidence band.

## Step 5 [engineer]: Wavelet decomposition, eddy-lifetime estimation, and filtered tracer re-advection
- Verdict: DISAGREES
- Contradictions: 
- [MAJOR] The plan required computing the Okubo-Weiss parameter Q and applying a mask (Q < -Q_threshold) when computing spatial averages for T_eddy. The code computes Q but never applies the mask to the spatial averages, instead using a simple `np.mean` over the entire domain.
- [MAJOR] The plan required computing T_eddy(j) by integrating the autocorrelation function C_E(τ,j) until C_E < 0.1. The code integrates the autocorrelation function where the values are > 0.1, which is mathematically incorrect for finding the integral timescale (it should be the integral of the function up to the point where it crosses the threshold).
- [MAJOR] The plan required reconstructing filtered velocity fields ũ_j by zeroing detail coefficients below j_min. The code ignores this reconstruction step entirely and re-advects tracers using the original, unfiltered `velocity_snapshots.npy` for all j_min.
- [MAJOR] The plan required computing γ_MSD(j) and ν_VACF(j). The code explicitly sets these values to 0.0 instead of performing the required calculations.
- [INTERMEDIATE] The plan required extracting detail coefficients at levels j=1,...,6. The code only iterates through j=1,...,5 for T_eddy calculation.
- [INTERMEDIATE] The plan required using GPU-accelerated RK4 for tracer re-advection. The code uses a simple Euler integration (`pos_norm + v.t() * 0.05`) instead of RK4.

## Step 6 [engineer]: Plot wavelet scaling results and synthesis
- Verdict: DISAGREES
- Contradictions: 
    - [MAJOR] The plan required a synthesis panel comparing α(t) from Step 3 with α(j_peak(t)) from the wavelet analysis; the code instead plots α(t) and k_peak separately and fails to perform the requested comparison.
    - [MAJOR] The plan required a 2D heatmap of α as a function of (T_eddy, k_peak/k_box) using all available data points; the code does not generate this heatmap.
    - [INTERMEDIATE] The plan required a direct power-law fit to α(j) vs T_eddy in the plot; the code only plots the CTRW-predicted curve derived from the β fit.

## Overall
- Verdict: POOR
- Engineer steps reviewed: 6
- Steps AGREES: 0
- Steps PARTIAL: 1
- Steps DISAGREES: 5
- Steps MISSING_CODE: 0
- Contradictions by severity: MAJOR=12, INTERMEDIATE=5, MINOR=1
