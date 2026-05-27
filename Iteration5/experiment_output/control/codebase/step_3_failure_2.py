# filename: codebase/step_3.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import json
from scipy.optimize import curve_fit
from statsmodels.tsa.stattools import adfuller, kpss

def mcculloch_alpha(data):
    """Estimates alpha using McCulloch quantile method."""
    q = np.percentile(data, [5, 25, 50, 75, 95])
    nu_alpha = (q[4] - q[0]) / (q[3] - q[1])
    if nu_alpha >= 2.439: return 2.0
    if nu_alpha <= 1.0: return 0.5
    return 2.0 - (nu_alpha - 1.0) * 0.5

def hill_estimator(data):
    """Estimates alpha using Hill estimator."""
    sorted_data = np.sort(np.abs(data))
    k = int(len(data) * 0.1)
    tail = sorted_data[-k:]
    return 1.0 / np.mean(np.log(tail / tail[0]))

def run_non_stationarity_analysis():
    data_dir = "/home/node/work/projects/levy_flights_2dns_v2/data/"
    pos = np.load(os.path.join(data_dir, "tracer_positions.npy"))
    diag = np.load(os.path.join(data_dir, "diagnostics.npy"))
    with open(os.path.join(data_dir, "sim_params.json"), "r") as f:
        params = json.load(f)
    T_L = params.get("T_L_at_start_of_production", 10.0)
    dt = 0.05
    W = int(8 * T_L / dt)
    dW = int(2 * T_L / dt)
    alphas, hills, k_peaks, times = [], [], [], []
    for start in range(0, pos.shape[0] - W, dW):
        end = start + W
        window_pos = pos[start:end]
        window_diag = diag[start:end]
        diffs = (window_pos[1:] - window_pos[:-1] + np.pi) % (2 * np.pi) - np.pi
        pooled = diffs.flatten()
        alphas.append(mcculloch_alpha(pooled))
        hills.append(hill_estimator(pooled))
        k_peaks.append(np.mean(window_diag['k_peak']))
        times.append(np.mean(window_diag['time']))
    alphas, hills, k_peaks, times = np.array(alphas), np.array(hills), np.array(k_peaks), np.array(times)
    def model(x, a, b, c): return a + b * (x**c)
    popt, _ = curve_fit(model, k_peaks, alphas, p0=[1.0, 0.1, 1.0])
    print("Stationarity Tests:")
    adf = adfuller(alphas)
    kpss_stat = kpss(alphas, policy='stationary')
    print("Alpha ADF p-value: " + str(adf[1]))
    print("Alpha KPSS p-value: " + str(kpss_stat[1]))
    print("Fit parameters a, b, c: " + str(popt))
    np.savez(os.path.join(data_dir, "non_stationarity_results.npz"), alphas=alphas, hills=hills, k_peaks=k_peaks, times=times, popt=popt)

if __name__ == '__main__':
    run_non_stationarity_analysis()