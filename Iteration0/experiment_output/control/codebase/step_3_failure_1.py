# filename: codebase/step_3.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
from scipy.stats import levy_stable
from scipy.optimize import curve_fit
from tqdm import tqdm

def calculate_displacements(positions, lag):
    """
    Calculates tracer displacements with periodic boundary conditions.

    Args:
        positions (np.ndarray): Tracer positions array of shape (T, N, 2).
        lag (int): The time lag in number of steps.

    Returns:
        np.ndarray: Displacement array of shape (T-lag, N, 2).
    """
    pos1 = positions[:-lag]
    pos2 = positions[lag:]
    delta = pos2 - pos1
    delta[delta > np.pi] -= 2 * np.pi
    delta[delta < -np.pi] += 2 * np.pi
    return delta

def fit_alpha_for_window(window_pos, lags_steps):
    """
    Fits the Lévy-stable alpha parameter for a given window and lags.

    Args:
        window_pos (np.ndarray): Tracer positions within the window.
        lags_steps (np.ndarray): Array of lags in integer steps.

    Returns:
        tuple: A tuple containing:
            - float: The mean alpha value.
            - float: The standard deviation of alpha values.
            - np.ndarray: The alpha values for each lag.
    """
    alphas = []
    for lag in lags_steps:
        if lag >= window_pos.shape[0] or lag == 0:
            alphas.append(np.nan)
            continue
        
        delta_pos = calculate_displacements(window_pos, lag)
        displacements_1d = delta_pos.flatten()
        
        # Filter out any potential NaNs or Infs from input
        displacements_1d = displacements_1d[np.isfinite(displacements_1d)]
        
        if displacements_1d.size < 100: # Not enough data to fit
            alphas.append(np.nan)
            continue

        try:
            # Use a smaller sample for fitting if the array is very large to speed up
            sample_size = min(len(displacements_1d), 200000)
            sample = np.random.choice(displacements_1d, sample_size, replace=False)
            alpha, _, _, _ = levy_stable.fit(sample)
            # Constrain alpha to its valid range
            if 0 < alpha <= 2:
                alphas.append(alpha)
            else:
                alphas.append(np.nan)
        except (RuntimeError, ValueError):
            alphas.append(np.nan)
            
    return np.nanmean(alphas), np.nanstd(alphas), np.array(alphas)

def fit_gamma_for_window(window_pos, dt):
    """
    Fits the MSD exponent gamma for a given window.

    Args:
        window_pos (np.ndarray): Tracer positions within the window.
        dt (float): The average time step.

    Returns:
        float: The fitted gamma exponent.
    """
    T_win = window_pos.shape[0]
    max_lag = T_win // 4
    if max_lag < 2:
        return np.nan
        
    lags = np.arange(1, max_lag)
    msd = []
    for lag in lags:
        delta_pos = calculate_displacements(window_pos, lag)
        msd.append(np.mean(np.sum(delta_pos**2, axis=-1)))
    
    lags_phys = lags * dt
    msd = np.array(msd)
    
    valid = msd > 0
    if np.sum(valid) < 2:
        return np.nan
        
    try:
        gamma, _ = np.polyfit(np.log(lags_phys[valid]), np.log(msd[valid]), 1)
    except (np.linalg.LinAlgError, ValueError):
        gamma = np.nan
        
    return gamma

def fit_nu_for_window(window_vel, dt, T_L):
    """
    Fits the VACF decay exponent nu for a given window.

    Args:
        window_vel (np.ndarray): Tracer velocities within the window.
        dt (float): The average time step.
        T_L (float): The large-eddy turnover time.

    Returns:
        float: The fitted nu exponent.
    """
    T_win = window_vel.shape[0]
    max_lag = T_win // 4
    if max_lag < 2:
        return np.nan

    lags = np.arange(1, max_lag)
    vacf = []
    v_sq_mean = np.mean(np.sum(window_vel**2, axis=-1))
    if v_sq_mean == 0:
        return np.nan

    for lag in lags:
        v1 = window_vel[:-lag]
        v2 = window_vel[lag:]
        vacf.append(np.mean(np.sum(v1 * v2, axis=-1)) / v_sq_mean)
    
    lags_phys = lags * dt
    vacf = np.array(vacf)
    
    fit_start_idx = np.argmin(np.abs(lags_phys - T_L))
    if fit_start_idx >= len(lags_phys) - 2:
        return np.nan
        
    valid = vacf[fit_start_idx:] > 0
    if np.sum(valid) < 2:
        return np.nan
        
    log_tau = np.log(lags_phys[fit_start_idx:][valid])
    log_vacf = np.log(vacf[fit_start_idx:][valid])
    
    try:
        neg_nu, _ = np.polyfit(log_tau, log_vacf, 1)
    except (np.linalg.LinAlgError, ValueError):
        neg_nu = np.nan
        
    return -neg_nu

def compute_non_stationarity_timeseries(data_path, output_path):
    """
    Computes time series of statistical properties over sliding windows.

    Args:
        data_path (str): Path to the prepared data .npz file.
        output_path (str): Path to save the resulting timeseries .npz file.
    """
    print("Loading data from " + data_path)
    data = np.load(data_path)
    tracer_pos = data['tracer_positions']
    tracer_vel = data['tracer_velocities']
    times = data['times']
    diagnostics = data['diagnostics']
    T_L_estimate = data['T_L_estimate'].item()

    W = 8 * T_L_estimate
    dW = 2 * T_L_estimate
    dt_avg = np.mean(np.diff(times))
    
    lags_alpha_phys = np.array([0.5, 1.0, 2.0]) * T_L_estimate
    lags_alpha_steps = np.round(lags_alpha_phys / dt_avg).astype(int)

    window_starts = np.arange(times[0], times[-1] - W, dW)
    results = []

    print("Starting non-stationarity analysis over " + str(len(window_starts)) + " windows...")
    for t_start in tqdm(window_starts):
        t_end = t_start + W
        idx_start = np.searchsorted(times, t_start)
        idx_end = np.searchsorted(times, t_end)
        
        if (idx_end - idx_start) < (max(lags_alpha_steps) + 5):
            continue

        window_pos = tracer_pos[idx_start:idx_end]
        window_vel = tracer_vel[idx_start:idx_end]
        
        diag_mask = (diagnostics['time'] >= t_start) & (diagnostics['time'] < t_end)
        window_diag = diagnostics[diag_mask]

        alpha_mean, alpha_std, alphas_per_lag = fit_alpha_for_window(window_pos, lags_alpha_steps)
        gamma = fit_gamma_for_window(window_pos, dt_avg)
        nu = fit_nu_for_window(window_vel, dt_avg, T_L_estimate)
        
        d_vv_mean = np.mean(window_diag['d_vv_estimate']) if window_diag.size > 0 else np.nan
        k_peak_mean = np.mean(window_diag['k_peak']) if window_diag.size > 0 else np.nan

        results.append({
            't_center': t_start + W / 2.0,
            'alpha': alpha_mean,
            'alpha_std': alpha_std,
            'alphas_per_lag': alphas_per_lag,
            'gamma': gamma,
            'nu': nu,
            'd_vv': d_vv_mean,
            'k_peak': k_peak_mean
        })

    if not results:
        print("No windows were processed. Check window parameters and data timespan.")
        return

    t_center = np.array([r['t_center'] for r in results])
    alpha = np.array([r['alpha'] for r in results])
    alpha_std = np.array([r['alpha_std'] for r in results])
    alphas_per_lag = np.array([r['alphas_per_lag'] for r in results])
    gamma = np.array([r['gamma'] for r in results])
    nu = np.array([r['nu'] for r in results])
    d_vv = np.array([r['d_vv'] for r in results])
    k_peak = np.array([r['k_peak'] for r in results])

    print("Saving time series data to " + output_path)
    np.savez_compressed(
        output_path,
        t_center=t_center,
        alpha=alpha,
        alpha_std=alpha_std,
        alphas_per_lag=alphas_per_lag,
        gamma=gamma,
        nu=nu,
        d_vv=d_vv,
        k_peak=k_peak
    )
    print("Analysis complete.")

if __name__ == '__main__':
    DATA_DIR = "data/"
    PREPARED_DATA_PATH = os.path.join(DATA_DIR, "prepared_data.npz")
    OUTPUT_PATH = os.path.join(DATA_DIR, "non_stationarity_timeseries.npz")
    
    compute_non_stationarity_timeseries(PREPARED_DATA_PATH, OUTPUT_PATH)