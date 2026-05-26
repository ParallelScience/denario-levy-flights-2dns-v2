**Code Explanation:**

This script performs a non-stationarity analysis on tracer data from a 2D turbulence simulation. It loads simulation parameters, tracer positions, velocities, times, and diagnostic data. The core of the analysis involves iterating through overlapping temporal windows of a fixed width (8 times the initial large-eddy turnover time, `T_L`) and step size (2 * `T_L`).

For each window containing a sufficient number of snapshots (at least 50), the script calculates several key statistical metrics:
1.  **Lévy Stability Index (α):** This parameter characterizes the tails of the tracer displacement probability distribution. It is estimated using two methods for cross-validation:
    *   The **McCulloch quantile method**, which uses a ratio of sample quantiles to look up α from a pre-computed table.
    *   The **Hill estimator**, a classic method for tail index estimation based on the largest observed displacements.
2.  **Anomalous Diffusion Exponent (γ):** Calculated by fitting a power law to the Mean Squared Displacement (MSD) of the tracers.
3.  **VACF Decay Exponent (ν):** Calculated by fitting an algebraic decay model to the Velocity Autocorrelation Function (VACF).
4.  **Flow Diagnostics:** The average peak wavenumber (`k_peak`) and inter-vortex distance (`d_vv`) are computed from the diagnostics data within the window.

The script prints a formatted table of these calculated metrics for each valid time window. Finally, it aggregates all the computed time series (`α(t)`, `γ(t)`, `ν(t)`, `k_peak(t)`, `d_vv(t)`) and saves them into a single compressed NumPy file (`.npz`) for use in subsequent analysis steps. The code is designed to handle cases where the input data is too short to form any valid windows, in which case it will issue a warning and save an empty data file.

**Python Code:**
```python
import json
import os
import sys
import numpy as np
from scipy.stats import linregress

MCCULLOCH_NU_ALPHA = np.array([
    2.439, 2.483, 2.539, 2.615, 2.720, 2.879, 3.132, 3.575, 4.011, 4.590,
    5.185, 6.220, 7.573, 9.815, 12.89, 18.61, 29.03, 54.59, 105.9, 254.1
])
MCCULLOCH_ALPHA = np.array([
    2.0, 1.9, 1.8, 1.7, 1.6, 1.5, 1.4, 1.3, 1.2, 1.1,
    1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1
])

def load_required_data(base_path):
    """
    Loads the necessary data files for non-stationarity analysis.

    Args:
        base_path (str): The absolute path to the data directory.

    Returns:
        tuple: A tuple containing sim_params, tracer_positions, tracer_velocities,
               tracer_times, and diagnostics.
    """
    print("--- Loading Data for Step 2 ---")
    try:
        with open(os.path.join(base_path, 'sim_params.json'), 'r') as f:
            sim_params = json.load(f)
        tracer_positions = np.load(os.path.join(base_path, 'tracer_positions.npy'))
        tracer_velocities = np.load(os.path.join(base_path, 'tracer_velocities.npy'))
        tracer_times = np.load(os.path.join(base_path, 'tracer_times.npy'))
        diagnostics = np.load(os.path.join(base_path, 'diagnostics.npy'))
    except FileNotFoundError as e:
        print("Error: A required data file was not found.")
        print(e)
        sys.exit(1)
    print("Data loading complete.")
    return sim_params, tracer_positions, tracer_velocities, tracer_times, diagnostics

def calculate_displacements(positions, lag, domain_size=2*np.pi):
    """
    Calculates tracer displacements |r(t+lag) - r(t)| with periodic boundaries.

    Args:
        positions (np.ndarray): Tracer positions array of shape (n_times, n_tracers, 2).
        lag (int): The time lag in number of snapshot steps.
        domain_size (float): The size of the periodic domain.

    Returns:
        np.ndarray: A flattened array of displacement magnitudes.
    """
    delta_r = positions[lag:] - positions[:-lag]
    delta_r = (delta_r + 0.5 * domain_size) % domain_size - 0.5 * domain_size
    displacements = np.sqrt(np.sum(delta_r**2, axis=2))
    return displacements.flatten()

def estimate_alpha_mcculloch(displacements):
    """
    Estimates the Levy index alpha using the McCulloch quantile method.

    Args:
        displacements (np.ndarray): Array of displacement magnitudes.

    Returns:
        float: The estimated alpha value. Returns np.nan if calculation fails.
    """
    if len(displacements) < 10:
        return np.nan
    q = np.percentile(displacements, [5, 25, 75, 95])
    q5, q25, q75, q95 = q[0], q[1], q[2], q[3]
    
    denominator = q75 - q25
    if denominator <= 1e-9:
        return np.nan
        
    nu_alpha = (q95 - q5) / denominator
    alpha = np.interp(nu_alpha, MCCULLOCH_NU_ALPHA, MCCULLOCH_ALPHA)
    return alpha

def estimate_alpha_hill(displacements):
    """
    Estimates the Levy index alpha using the Hill estimator.

    Args:
        displacements (np.ndarray): Array of displacement magnitudes.

    Returns:
        float: The estimated alpha value. Returns np.nan if calculation fails.
    """
    if len(displacements) < 20:
        return np.nan
        
    k = max(10, int(0.05 * len(displacements)))
    
    sorted_displacements = np.sort(displacements)[::-1]
    top_k_displacements = sorted_displacements[:k]
    
    x_k = top_k_displacements[-1]
    
    if x_k <= 1e-9:
        return np.nan
        
    log_ratios = np.log(top_k_displacements / x_k)
    mean_log_ratio = np.mean(log_ratios)
    
    if mean_log_ratio <= 1e-9:
        return np.nan
        
    alpha = 1.0 / mean_log_ratio
    return alpha

def estimate_gamma_msd(positions, dt_snap, max_lag_frac=0.25):
    """
    Estimates the anomalous diffusion exponent gamma from the MSD.

    Args:
        positions (np.ndarray): Tracer positions for the window.
        dt_snap (float): Time step between snapshots.
        max_lag_frac (float): Maximum lag to use for fitting, as a fraction of window length.

    Returns:
        float: The estimated gamma value. Returns np.nan if fit fails.
    """
    n_times = positions.shape[0]
    max_lag = max(5, int(n_times * max_lag_frac))
    if n_times <= max_lag:
        return np.nan

    lags = np.arange(1, max_lag + 1)
    msd = np.zeros(len(lags))
    domain_size = 2 * np.pi

    for i, lag in enumerate(lags):
        delta_r = positions[lag:] - positions[:-lag]
        delta_r = (delta_r + 0.5 * domain_size) % domain_size - 0.5 * domain_size
        sq_disp = np.sum(delta_r**2, axis=2)
        msd[i] = np.mean(sq_disp)

    tau = lags * dt_snap
    
    valid_indices = msd > 1e-9
    if np.sum(valid_indices) < 2:
        return np.nan

    log_tau = np.log(tau[valid_indices])
    log_msd = np.log(msd[valid_indices])
    
    slope, _, _, _, _ = linregress(log_tau, log_msd)
    return slope

def estimate_nu_vacf(velocities, dt_snap, max_lag_frac=0.25):
    """
    Estimates the VACF decay exponent nu.

    Args:
        velocities (np.ndarray): Tracer velocities for the window.
        dt_snap (float): Time step between snapshots.
        max_lag_frac (float): Maximum lag to use for fitting, as a fraction of window length.

    Returns:
        float: The estimated nu value. Returns np.nan if fit fails.
    """
    n_times = velocities.shape[0]
    max_lag = max(5, int(n_times * max_lag_frac))
    if n_times <= max_lag:
        return np.nan

    lags = np.arange(1, max_lag + 1)
    vacf = np.zeros(len(lags))
    v_sq_mean = np.mean(np.sum(velocities**2, axis=2))
    if v_sq_mean < 1e-9:
        return np.nan

    for i, lag in enumerate(lags):
        dot_prod = np.sum(velocities[:-lag] * velocities[lag:], axis=2)
        vacf[i] = np.mean(dot_prod) / v_sq_mean

    tau = lags * dt_snap
    
    valid_indices = (vacf > 1e-9) & (vacf < 0.95)
    if np.sum(valid_indices) < 2:
        return np.nan

    log_tau = np.log(tau[valid_indices])
    log_vacf = np.log(vacf[valid_indices])
    
    slope, _, _, _, _ = linregress(log_tau, log_vacf)
    return -slope

def analyze_non_stationarity():
    """
    Main function to perform the non-stationarity analysis.
    """
    base_path = '/home/node/work/projects/levy_flights_2dns_v2/data/'
    params, positions, velocities, times, diagnostics = load_required_data(base_path)

    t_l_initial = params.get('T_L_at_start_of_production', 1.0)
    dt_snap = params.get('dt_snap', 0.05)
    
    window_width_time = 8 * t_l_initial
    window_step_time = 2 * t_l_initial

    print("\n--- Non-Stationarity Analysis Setup ---")
    print("Initial T_L: " + str(t_l_initial))
    print("Window width (W = 8*T_L): " + str(window_width_time))
    print("Window step (dW = 2*T_L): " + str(window_step_time))

    start_times = np.arange(times[0], times[-1] - window_width_time, window_step_time)
    
    results = []

    print("\n--- Processing Time Windows ---")
    header_str = "{:>15s} | {:>12s} | {:>12s} | {:>8s} | {:>8s} | {:>8s} | {:>8s}"
    header = header_str.format(
        "Win Center (t)", "alpha_McC", "alpha_Hill", "gamma", "nu", "k_peak", "d_vv"
    )
    print(header)
    print("-" * len(header))

    for t_start in start_times:
        t_end = t_start + window_width_time
        
        win_indices = np.where((times >= t_start) & (times < t_end))[0]
        
        if len(win_indices) < 50:
            msg = "Warning: Skipping window at t=" + str(t_start) + " due to insufficient snapshots (" + str(len(win_indices)) + " < 50)."
            print(msg)
            continue

        win_center_time = t_start + 0.5 * window_width_time
        
        win_pos = positions[win_indices]
        win_vel = velocities[win_indices]
        win_diag = diagnostics[win_indices]

        lags_time = np.array([0.5, 1.0, 2.0]) * t_l_initial
        lags_steps = np.round(lags_time / dt_snap).astype(int)
        lags_steps = np.unique(lags_steps[lags_steps > 0])
        
        alphas_mcc = []
        alphas_hill = []
        for lag in lags_steps:
            if lag >= len(win_indices):
                continue
            disps = calculate_displacements(win_pos, lag)
            alphas_mcc.append(estimate_alpha_mcculloch(disps))
            alphas_hill.append(estimate_alpha_hill(disps))
        
        avg_alpha_mcc = np.nanmean(alphas_mcc) if alphas_mcc else np.nan
        avg_alpha_hill = np.nanmean(alphas_hill) if alphas_hill else np.nan

        gamma = estimate_gamma_msd(win_pos, dt_snap)
        nu = estimate_nu_vacf(win_vel, dt_snap)

        avg_k_peak = np.mean(win_diag['k_peak'])
        avg_d_vv = np.mean(win_diag['d_vv_estimate'])

        results.append({
            'time': win_center_time,
            'alpha_mcc': avg_alpha_mcc,
            'alpha_hill': avg_alpha_hill,
            'gamma': gamma,
            'nu': nu,
            'k_peak': avg_k_peak,
            'd_vv': avg_d_vv
        })
        
        print("{:15.2f} | {:12.4f} | {:12.4f} | {:8.4f} | {:8.4f} | {:8.4f} | {:8.4f}".format(
            win_center_time, avg_alpha_mcc, avg_alpha_hill, gamma, nu, avg_k_peak, avg_d_vv
        ))

    if not results:
        print("\nNo valid windows found for analysis. The dataset might be too short.")
        time_series = {
            'window_center_times': np.array([]), 'alpha_mcculloch': np.array([]),
            'alpha_hill': np.array([]), 'gamma': np.array([]), 'nu': np.array([]),
            'k_peak': np.array([]), 'd_vv': np.array([])
        }
    else:
        time_series = {
            'window_center_times': np.array([r['time'] for r in results]),
            'alpha_mcculloch': np.array([r['alpha_mcc'] for r in results]),
            'alpha_hill': np.array([r['alpha_hill'] for r in results]),
            'gamma': np.array([r['gamma'] for r in results]),
            'nu': np.array([r['nu'] for r in results]),
            'k_peak': np.array([r['k_peak'] for r in results]),
            'd_vv': np.array([r['d_vv'] for r in results])
        }

    output_path = os.path.join("data", "non_stationary_analysis_timeseries.npz")
    np.savez(output_path, **time_series)
    print("\nTime series data saved to " + output_path)

if __name__ == '__main__':
    analyze_non_stationarity()
```