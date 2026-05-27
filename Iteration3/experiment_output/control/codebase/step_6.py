# filename: codebase/step_6.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def plot_wavelet_scaling():
    data_dir = "/home/node/work/projects/levy_flights_2dns_v2/data/"
    res = np.load(os.path.join(data_dir, "wavelet_results.npz"))
    t_eddy = res['t_eddy']
    alpha_j = res['alpha_j']
    gamma_j = res['gamma_j']
    nu_j = res['nu_j']
    min_len = min(len(t_eddy), len(alpha_j))
    t_eddy = t_eddy[:min_len]
    alpha_j = alpha_j[:min_len]
    levels = np.arange(1, min_len + 1)
    def beta_func(t, delta, c):
        return c * (t**delta)
    popt, _ = curve_fit(beta_func, t_eddy, alpha_j - 1.0)
    delta, c = popt
    beta_vals = beta_func(t_eddy, delta, c)
    alpha_pred = 1.0 + beta_vals / (2.0 - beta_vals)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0, 0].plot(t_eddy, alpha_j, 'bo', label='Observed')
    axes[0, 0].plot(t_eddy, alpha_pred, 'r--', label='CTRW Prediction')
    axes[0, 0].set_xlabel('T_eddy')
    axes[0, 0].set_ylabel('alpha')
    axes[0, 0].set_title('alpha vs T_eddy')
    axes[0, 0].legend()
    axes[0, 1].plot(levels, gamma_j[:min_len], 'go-')
    axes[0, 1].set_xlabel('Wavelet Level j')
    axes[0, 1].set_ylabel('gamma_MSD')
    axes[0, 1].set_title('MSD Scaling')
    axes[1, 0].plot(levels, nu_j[:min_len], 'mo-')
    axes[1, 0].set_xlabel('Wavelet Level j')
    axes[1, 0].set_ylabel('nu_VACF')
    axes[1, 0].set_title('VACF Decay')
    axes[1, 1].semilogy(levels, t_eddy, 'ko-')
    axes[1, 1].set_xlabel('Wavelet Level j')
    axes[1, 1].set_ylabel('T_eddy')
    axes[1, 1].set_title('Eddy Lifetime')
    for ax in axes.flatten():
        ax.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(data_dir, "wavelet_scaling_plot.png")
    plt.savefig(plot_path, dpi=300)
    print("Fitted delta: " + str(delta))
    print("Beta values: " + str(beta_vals))
    print("Predicted alpha: " + str(alpha_pred))
    print("Observed alpha: " + str(alpha_j))
    print("Plot saved to " + plot_path)

if __name__ == '__main__':
    plot_wavelet_scaling()