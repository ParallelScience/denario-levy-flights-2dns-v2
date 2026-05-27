# filename: codebase/step_5.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression

def run_synthesis():
    data_dir = "/home/node/work/projects/levy_flights_2dns_v2/data/"
    win_stats = np.load(os.path.join(data_dir, "window_stats.npz"))
    wav_stats = np.load(os.path.join(data_dir, "wavelet_stats.npz"))
    alphas_t = win_stats['alphas']
    kpeaks_t = win_stats['kpeaks']
    centers_t = win_stats['centers']
    t_eddies = wav_stats['t_eddies']
    alphas_j = wav_stats['alphas']
    j_vals = np.arange(1, 7)
    k_j = 256 / (2**j_vals)
    j_peak_t = np.array([np.argmin(np.abs(k_j - kp)) + 1 for kp in kpeaks_t])
    alphas_j_mapped = np.array([alphas_j[j-1] if j-1 < len(alphas_j) else alphas_j[-1] for j in j_peak_t])
    def alpha_model(x, a, b, c):
        return a + b * (x**c)
    popt, pcov = curve_fit(alpha_model, kpeaks_t, alphas_t)
    print("Fit parameters a, b, c: " + str(popt))
    beta_j = 2 * (alphas_j - 1) / alphas_j
    log_t = np.log(t_eddies[:4]).reshape(-1, 1)
    log_beta = np.log(beta_j[:4])
    reg = LinearRegression().fit(log_t, log_beta)
    delta = reg.coef_[0]
    r2 = reg.score(log_t, log_beta)
    print("CTRW delta: " + str(delta) + ", R^2: " + str(r2))
    np.savez(os.path.join(data_dir, "synthesis_results.npz"), delta=delta, r2=r2, fit_params=popt)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(centers_t, alphas_t, label="Alpha(t)")
    axes[0].plot(centers_t, alphas_j_mapped, '--', label="Alpha(j_peak(t))")
    axes[0].set_title("Alpha Comparison")
    axes[0].legend()
    axes[1].scatter(t_eddies[:4], alphas_j, c='red', label="Wavelet scales")
    axes[1].set_title("Alpha vs T_eddy")
    axes[2].plot(t_eddies[:4], beta_j[:4], 'o-')
    axes[2].set_title("Beta vs T_eddy")
    axes[2].set_yscale('log')
    axes[2].set_xscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "synthesis_5_1.png"), dpi=300)
    print("Synthesis plot saved to " + os.path.join(data_dir, "synthesis_5_1.png"))

if __name__ == '__main__':
    run_synthesis()