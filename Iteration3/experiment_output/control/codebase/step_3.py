# filename: codebase/step_3.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import json
import numpy as np
from scipy import stats, optimize
from statsmodels.tsa.stattools import adfuller, kpss

def mcculloch_alpha(data):
    q = np.percentile(data, [5, 25, 50, 75, 95])
    nu = (q[4] - q[0]) / (q[3] - q[1])
    alpha = np.interp(nu, [1.0, 2.0], [2.0, 1.0])
    return np.clip(alpha, 1.0, 2.0), nu

def hill_estimator(data, k=100):
    sorted_data = np.sort(np.abs(data))
    tail = sorted_data[-k:]
    return 1.0 / np.mean(np.log(tail / tail[0]))

def analyze_nonstationarity():
    data_dir = '/home/node/work/projects/levy_flights_2dns_v2/data/'
    with open(os.path.join(data_dir, 'sim_params.json'), 'r') as f:
        params = json.load(f)
    pos = np.load(os.path.join(data_dir, 'tracer_positions.npy'))
    times = np.load(os.path.join(data_dir, 'tracer_times.npy'))
    diag = np.load(os.path.join(data_dir, 'diagnostics.npy'))
    t_l = params['T_L_at_start_of_production']
    w_size = 8 * t_l
    w_step = 2 * t_l
    num_windows = int((times[-1] - times[0] - w_size) / w_step) + 1
    alphas_mcc = []
    alphas_hill = []
    k_peaks = []
    window_centers = []
    for i in range(num_windows):
        t_start = times[0] + i * w_step
        t_end = t_start + w_size
        mask = (times >= t_start) & (times <= t_end)
        window_pos = pos[mask]
        disp = np.sqrt(np.sum(np.diff(window_pos, axis=0)**2, axis=-1))
        alpha, _ = mcculloch_alpha(disp.flatten())
        alphas_mcc.append(alpha)
        alphas_hill.append(hill_estimator(disp.flatten()))
        window_centers.append((t_start + t_end) / 2)
        k_peaks.append(np.mean(diag['k_peak'][mask]))
    alphas_mcc = np.array(alphas_mcc)
    if np.std(alphas_mcc) < 1e-9:
        adf_stat, adf_p = 0.0, 0.0
        kpss_stat, kpss_p = 0.0, 0.1
    else:
        adf_res = adfuller(alphas_mcc)
        kpss_res = kpss(alphas_mcc, policy='stationary')
        adf_stat, adf_p = adf_res[0], adf_res[1]
        kpss_stat, kpss_p = kpss_res[0], kpss_res[1]
    print('ADF Statistic: ' + str(adf_stat) + ', p-value: ' + str(adf_p))
    print('KPSS Statistic: ' + str(kpss_stat) + ', p-value: ' + str(kpss_p))
    def func(k, a, b, c):
        return a + b * (k / 1.0)**c
    popt, _ = optimize.curve_fit(func, k_peaks, alphas_mcc)
    print('Fit parameters: a=' + str(popt[0]) + ', b=' + str(popt[1]) + ', c=' + str(popt[2]))
    np.savez(os.path.join(data_dir, 'nonstationarity_results.npz'),
             alpha_mcculloch=alphas_mcc, alpha_hill=alphas_hill,
             window_centers=window_centers, k_peak=k_peaks,
             adf_stat=adf_stat, kpss_stat=kpss_stat,
             fit_params=popt)

if __name__ == '__main__':
    analyze_nonstationarity()