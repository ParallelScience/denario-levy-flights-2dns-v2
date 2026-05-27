# filename: codebase/step_3.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import os
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
from scipy.optimize import curve_fit

def mcculloch_alpha(q5, q25, q50, q75, q95):
    nu = (q95 - q5) / (q75 - q25)
    if nu < 2.439: return 2.0
    if nu > 6.0: return 0.6
    return 2.0 - (nu - 2.439) * (2.0 - 0.6) / (6.0 - 2.439)

def analyze_non_stationarity():
    data_dir = "/home/node/work/projects/levy_flights_2dns_v2/data/"
    pos = np.load(os.path.join(data_dir, "tracer_positions.npy"))
    times = np.load(os.path.join(data_dir, "tracer_times.npy"))
    diag = np.load(os.path.join(data_dir, "diagnostics.npy"))
    t_l = np.mean(diag['T_L'])
    dt = times[1] - times[0]
    w = int(8 * t_l / dt)
    dw = int(2 * t_l / dt)
    windows = []
    start = 0
    while start + w < len(times):
        windows.append((start, start + w))
        start += dw
    if len(windows) < 5:
        print("Warning: Fewer than 5 windows. Falling back to W=4 T_L.")
        w = int(4 * t_l / dt)
        dw = int(1 * t_l / dt)
        windows = []
        start = 0
        while start + w < len(times):
            windows.append((start, start + w))
            start += dw
    alphas, kpeaks, centers = [], [], []
    for s, e in windows:
        centers.append(np.mean(times[s:e]))
        disp = np.sqrt(np.sum(np.diff(pos[s:e], axis=0)**2, axis=-1))
        q = np.percentile(disp.flatten(), [5, 25, 50, 75, 95])
        alphas.append(mcculloch_alpha(*q))
        kpeaks.append(np.mean(diag['k_peak'][s:e]))
    alphas = np.array(alphas)
    kpeaks = np.array(kpeaks)
    adf = adfuller(alphas)
    kpss_res = kpss(alphas, policy='fixed')
    print("ADF p-value: " + str(adf[1]))
    print("KPSS p-value: " + str(kpss_res[1]))
    def power_law(x, a, b, c): return a + b * (x**c)
    popt, _ = curve_fit(power_law, kpeaks, alphas)
    np.savez(os.path.join(data_dir, "window_stats.npz"), alphas=alphas, kpeaks=kpeaks, centers=centers)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0, 0].plot(centers, alphas)
    axes[0, 0].set_title("Alpha(t)")
    axes[0, 1].plot(centers, kpeaks)
    axes[0, 1].set_title("k_peak(t)")
    axes[1, 0].scatter(kpeaks, alphas)
    axes[1, 0].plot(kpeaks, power_law(kpeaks, *popt), 'r--')
    axes[1, 0].set_title("Alpha vs k_peak")
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "non_stationarity_3_1.png"), dpi=300)
    print("Plot saved to " + os.path.join(data_dir, "non_stationarity_3_1.png"))

if __name__ == '__main__':
    analyze_non_stationarity()