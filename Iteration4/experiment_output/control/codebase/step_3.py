# filename: codebase/step_3.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import os
from scipy.optimize import curve_fit
from statsmodels.tsa.stattools import adfuller, kpss

def mcculloch_alpha(q5, q25, q50, q75, q95):
    nu_alpha = (q95 - q5) / (q75 - q25)
    if nu_alpha < 2.439: return 2.0
    if nu_alpha > 6.0: return 0.6
    nu_vals = np.array([2.439, 2.5, 2.7, 3.0, 3.5, 4.0, 5.0, 6.0])
    alpha_vals = np.array([2.0, 1.9, 1.7, 1.5, 1.3, 1.1, 0.8, 0.6])
    return np.interp(nu_alpha, nu_vals, alpha_vals)

def run_analysis():
    data_dir = "/home/node/work/projects/levy_flights_2dns_v2/data/"
    pos = np.load(os.path.join(data_dir, "tracer_positions.npy"))
    times = np.load(os.path.join(data_dir, "tracer_times.npy"))
    diag = np.load(os.path.join(data_dir, "diagnostics.npy"))
    T_L = np.mean(diag['T_L'])
    W = int(8 * T_L / (times[1] - times[0]))
    dW = int(2 * T_L / (times[1] - times[0]))
    alphas, alphas_hill, gammas, nus, d_vvs, k_peaks, centers = [], [], [], [], [], [], []
    for start in range(0, len(times) - W, dW):
        end = start + W
        window_pos = pos[start:end]
        dx = (window_pos[1:, :, 0] - window_pos[:-1, :, 0] + np.pi) % (2 * np.pi) - np.pi
        dy = (window_pos[1:, :, 1] - window_pos[:-1, :, 1] + np.pi) % (2 * np.pi) - np.pi
        disps = np.concatenate([dx.flatten(), dy.flatten()])
        q = np.percentile(disps, [5, 25, 50, 75, 95])
        alphas.append(mcculloch_alpha(*q))
        sorted_disps = np.sort(np.abs(disps))
        k = int(0.05 * len(sorted_disps))
        alphas_hill.append(1.0 / np.mean(np.log(sorted_disps[-k:] / sorted_disps[-k])))
        centers.append(np.mean(times[start:end]))
        k_peaks.append(np.mean(diag['k_peak'][start:end]))
        d_vvs.append(np.mean(diag['d_vv_estimate'][start:end]))
    adf = adfuller(alphas)
    kpss_res = kpss(alphas, regression='c')
    def func(k, a, b, c): return a + b * (k)**c
    popt, _ = curve_fit(func, np.array(k_peaks)/1.0, alphas, p0=[1.0, 0.5, 1.0])
    np.savez(os.path.join(data_dir, "non_stationarity_results.npz"), alpha=alphas, alpha_hill=alphas_hill, k_peak=k_peaks, centers=centers)
    print("ADF p-value: " + str(adf[1]))
    print("KPSS p-value: " + str(kpss_res[1]))
    print("Fitted parameters: a=" + str(popt[0]) + ", b=" + str(popt[1]) + ", c=" + str(popt[2]))

if __name__ == '__main__':
    run_analysis()