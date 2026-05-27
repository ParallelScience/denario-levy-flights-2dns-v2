# filename: codebase/step_3.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import json
import os
import time
import warnings
from statsmodels.tsa.stattools import adfuller, kpss
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Suppress RuntimeWarning from polyfit on noisy data
warnings.filterwarnings("ignore", category=np.RankWarning)
warnings.filterwarnings("ignore", message="p-value is smaller than the indicated p-value")
warnings.filterwarnings("ignore", message="p-value is greater than the indicated p-value")


def load_data(base_path):
    """
    Loads all simulation data files from the specified base path.

    Args:
        base_path (str): The directory containing the data files.

    Returns:
        dict: A dictionary containing all loaded data arrays and parameters.
    """
    data_files = {
        "sim_params": "sim_params.json",
        "tracer_positions": "tracer_positions.npy",
        "tracer_velocities": "tracer_velocities.npy",
        "tracer_times": "tracer_times.npy",
        "diagnostics": "diagnostics.npy"
    }
    data = {}
    print("Loading data...")
    for name, filename in data_files.items():
        path = os.path.join(base_path, filename)
        if not os.path.exists(path):
            raise FileNotFoundError("Required data file not found: " + path)
        if filename.endswith('.json'):
            with open(path, 'r') as f:
                data[name] = json.load(f)
        else:
            data[name] = np.load(path)
        print("Loaded " + filename)
    return data


def get_mcculloch_alpha(samples):
    """
    Estimates the Levy-stable alpha parameter using the McCulloch quantile method.

    Args:
        samples (np.ndarray): 1D array of data samples (e.g., displacements).

    Returns:
        float: The estimated alpha value.
    """
    # McCulloch (1986), JBES, Table 1, p. 118
    mcculloch_v = np.array([
        2.439, 2.468, 2.501, 2.540, 2.588, 2.648, 2.726, 2.828, 2.968, 3.167,
        3.467, 3.952, 4.814, 6.688, 13.32, 22.19, 45.19, 124.9, 768.9
    ])
    mcculloch_alpha = np.array([
        2.00, 1.95, 1.90, 1.85, 1.80, 1.75, 1.70, 1.65, 1.60, 1.55, 1.50,
        1.40, 1.30, 1.20, 1.00, 0.90, 0.80, 0.70, 0.60
    ])

    q = np.nanquantile(samples, [0.05, 0.25, 0.75, 0.95])
    q05, q25, q75, q95 = q[0], q[1], q[2], q[3]

    if np.isclose(q75 - q25, 0):
        return np.nan

    v_alpha = (q95 - q05) / (q75 - q25)

    if v_alpha < mcculloch_v[0]:
        warnings.warn("v_alpha " + str(v_alpha) + " is below table range. Clamping alpha to 2.0.")
        return 2.0
    if v_alpha > mcculloch_v[-1]:
        warnings.warn("v_alpha " + str(v_alpha) + " is above table range. Clamping alpha to " + str(mcculloch_alpha[-1]) + ".")
        return mcculloch_alpha[-1]

    # Interpolate, note that v is decreasing with alpha
    alpha = np.interp(v_alpha, mcculloch_v, mcculloch_alpha)
    return alpha


def get_hill_estimator(samples, k):
    """
    Computes the Hill estimator for a given number of order statistics k.

    Args:
        samples (np.ndarray): 1D array of absolute-valued samples.
        k (int): The number of top order statistics to use.

    Returns:
        float: The Hill estimate for alpha.
    """
    if k < 2 or k >= len(samples):
        return np.nan
    samples_sorted = np.sort(samples)[::-1]
    log_samples = np.log(samples_sorted[:k])
    hill_stat = np.mean(log_samples) - log_samples[-1]
    if hill_stat <= 0:
        return np.nan
    return 1.0 / hill_stat


def periodic_displacement(pos_t, pos_t_plus_tau, domain_size):
    """
    Calculates displacements on a periodic domain.

    Args:
        pos_t (np.ndarray): Positions at time t.
        pos_t_plus_tau (np.ndarray): Positions at time t + tau.
        domain_size (float): The size of the periodic domain.

    Returns:
        np.ndarray: The calculated displacements.
    """
    disp = pos_t_plus_tau - pos_t
    disp[disp > domain_size / 2] -= domain_size
    disp[disp < -domain_size / 2] += domain_size
    return disp


def analyze_windows(data):
    """
    Performs the main non-stationarity analysis over overlapping windows.

    Args:
        data (dict): The dictionary of loaded simulation data.

    Returns:
        tuple: A tuple containing the results dictionary and the figure for Hill plots.
    """
    params = data['sim_params']
    T_L = params['T_L_at_start_of_production']
    dt_snap = params['dt_snap']
    domain_size = 2 * np.pi

    W = 8 * T_L
    dW = 2 * T_L

    times = data['tracer_times']
    positions = data['tracer_positions']
    velocities = data['tracer_velocities']
    diagnostics = data['diagnostics']

    window_starts = np.arange(times[0], times[-1] - W, dW)
    n_windows = len(window_starts)
    
    results = {
        'window_times': np.zeros(n_windows),
        'alpha_series': np.zeros(n_windows),
        'gamma_series': np.zeros(n_windows),
        'nu_series': np.zeros(n_windows),
        'k_peak_series': np.zeros(n_windows),
        'd_vv_series': np.zeros(n_windows),
    }

    n_plots = min(n_windows, 6)
    fig_hill, axes_hill = plt.subplots(int(np.ceil(n_plots/2)), 2, figsize=(12, 4*int(np.ceil(n_plots/2))), sharex=True, sharey=True)
    axes_hill = axes_hill.flatten()
    plot_indices = np.linspace(0, n_windows - 1, n_plots, dtype=int)

    for i, t_start in enumerate(window_starts):
        t_end = t_start + W
        window_center = t_start + W / 2
        results['window_times'][i] = window_center
        print("Analyzing window " + str(i + 1) + "/" + str(n_windows) + " centered at t=" + "{:.2f}".format(window_center))

        idx_start = np.searchsorted(times, t_start)
        idx_end = np.searchsorted(times, t_end)
        
        pos_win = positions[idx_start:idx_end]
        vel_win = velocities[idx_start:idx_end]
        diag_win = diagnostics[(diagnostics['time'] >= t_start) & (diagnostics['time'] < t_end)]

        if len(pos_win) < 10:
            print("  Skipping window: not enough data.")
            for key in results: results[key][i] = np.nan
            continue

        # Alpha calculation
        lags_alpha_t = np.array([0.5, 1.0, 2.0]) * T_L
        lags_alpha_idx = np.round(lags_alpha_t / dt_snap).astype(int)
        
        alphas = []
        all_disps = []
        for lag_idx in lags_alpha_idx:
            if lag_idx >= len(pos_win): continue
            disp = periodic_displacement(pos_win[:-lag_idx], pos_win[lag_idx:], domain_size)
            all_disps.append(np.abs(disp.flatten()))
            alphas.append(get_mcculloch_alpha(disp[:, :, 0].flatten()))
            alphas.append(get_mcculloch_alpha(disp[:, :, 1].flatten()))
        
        results['alpha_series'][i] = np.nanmedian(alphas) if alphas else np.nan

        # Hill estimator
        if i in plot_indices:
            ax_idx = np.where(plot_indices == i)[0][0]
            ax = axes_hill[ax_idx]
            if all_disps:
                combined_disps = np.concatenate(all_disps)
                k_max = int(0.1 * len(combined_disps))
                k_vals = np.logspace(np.log10(10), np.log10(k_max), 50, dtype=int)
                hill_alphas = [get_hill_estimator(combined_disps, k) for k in k_vals]
                ax.plot(k_vals, hill_alphas, '.-')
                ax.set_title("Window at t=" + "{:.1f}".format(window_center))
                ax.set_xlabel("k (order statistics)")
                ax.set_ylabel("Hill alpha")
                ax.grid(True)
                ax.set_ylim(0.5, 2.5)
                ax.set_xscale('log')

        # MSD (gamma) calculation
        lags_msd_t = np.linspace(0.1 * T_L, 4 * T_L, 20)
        lags_msd_idx = np.unique(np.round(lags_msd_t / dt_snap).astype(int))
        lags_msd_idx = lags_msd_idx[lags_msd_idx > 0]
        
        msd = []
        valid_lags = []
        for lag_idx in lags_msd_idx:
            if lag_idx >= len(pos_win): continue
            disp = periodic_displacement(pos_win[:-lag_idx], pos_win[lag_idx:], domain_size)
            sq_disp = np.sum(disp**2, axis=2)
            msd.append(np.mean(sq_disp))
            valid_lags.append(lag_idx * dt_snap)
        
        if len(valid_lags) > 2:
            log_lags = np.log(valid_lags)
            log_msd = np.log(msd)
            p = np.polyfit(log_lags, log_msd, 1)
            results['gamma_series'][i] = p[0]
        else:
            results['gamma_series'][i] = np.nan

        # VACF (nu) calculation
        lags_vacf_idx = np.arange(1, int(4 * T_L / dt_snap))
        vacf = []
        valid_lags_vacf = []
        norm = np.mean(np.sum(vel_win**2, axis=2))
        for lag_idx in lags_vacf_idx:
            if lag_idx >= len(vel_win): break
            corr = np.mean(np.sum(vel_win[:-lag_idx] * vel_win[lag_idx:], axis=2))
            vacf.append(corr / norm)
            valid_lags_vacf.append(lag_idx * dt_snap)
        
        vacf = np.array(vacf)
        fit_mask = (vacf > 0.05) & (vacf < 0.5)
        if np.sum(fit_mask) > 2:
            log_lags_vacf = np.log(np.array(valid_lags_vacf)[fit_mask])
            log_vacf = np.log(vacf[fit_mask])
            p_nu = np.polyfit(log_lags_vacf, log_vacf, 1)
            results['nu_series'][i] = -p_nu[0]
        else:
            results['nu_series'][i] = np.nan

        # Diagnostics
        results['k_peak_series'][i] = np.mean(diag_win['k_peak']) if len(diag_win) > 0 else np.nan
        results['d_vv_series'][i] = np.mean(diag_win['d_vv_estimate']) if len(diag_win) > 0 else np.nan

    fig_hill.tight_layout()
    return results, fig_hill


def perform_stationarity_tests(results):
    """
    Performs and prints ADF and KPSS stationarity tests on the time series.

    Args:
        results (dict): The dictionary containing the computed time series.

    Returns:
        str: A formatted string with the test results.
    """
    report = []
    report.append("--- Stationarity Test Results ---")
    
    for name, series in [('alpha', results['alpha_series']),
                         ('gamma', results['gamma_series']),
                         ('nu', results['nu_series'])]:        
        series = series[~np.isnan(series)]
        if len(series) < 10:
            report.append("\nNot enough data for " + name + " series.")
            continue

        report.append("\n--- " + name.upper() + " Series ---")
        
        # ADF Test (H0: non-stationary)
        adf_result = adfuller(series)
        report.append("ADF Test:")
        report.append("  Test Statistic: " + "{:.4f}".format(adf_result[0]))
        report.append("  p-value: " + "{:.4f}".format(adf_result[1]))
        report.append("  Result: " + ("Stationary (reject H0)" if adf_result[1] < 0.05 else "Non-Stationary (fail to reject H0)"))

        # KPSS Test (H0: stationary)
        kpss_result = kpss(series, regression='c', nlags="auto")
        report.append("KPSS Test:")
        report.append("  Test Statistic: " + "{:.4f}".format(kpss_result[0]))
        report.append("  p-value: " + "{:.4f}".format(kpss_result[1]))
        report.append("  Result: " + ("Stationary (fail to reject H0)" if kpss_result[1] > 0.05 else "Non-Stationary (reject H0)"))

    return "\n".join(report)

if __name__ == '__main__':
    base_path = "/home/node/work/projects/levy_flights_2dns_v2/data/"
    output_dir = "data/"
    
    try:
        data = load_data(base_path)
        
        results, fig_hill = analyze_windows(data)
        
        # Save results
        results_path = os.path.join(output_dir, "non_stationarity_results.npz")
        np.savez(results_path, **results)
        print("Non-stationarity analysis results saved to " + results_path)
        
        # Save Hill plots
        timestamp = int(time.time())
        hill_plot_path = os.path.join(output_dir, "hill_plots_" + str(timestamp) + ".png")
        fig_hill.savefig(hill_plot_path, dpi=300)
        print("Hill plots saved to " + hill_plot_path)
        plt.close(fig_hill)
        
        # Perform and save stationarity tests
        stationarity_report = perform_stationarity_tests(results)
        print("\n" + stationarity_report)
        
        report_path = os.path.join(output_dir, "stationarity_tests.txt")
        with open(report_path, 'w') as f:
            f.write(stationarity_report)
        print("Stationarity test report saved to " + report_path)

    except Exception as e:
        print("An error occurred during the non-stationarity analysis.")
        import traceback
        traceback.print_exc()