# filename: codebase/step_6.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import time

def plot_synthesis():
    data_dir = '/home/node/work/projects/levy_flights_2dns_v2/data/'
    wavelet_res = np.load(os.path.join(data_dir, 'wavelet_results.npz'))
    non_stat_res = np.load(os.path.join(data_dir, 'non_stationarity_results.npz'))
    t_eddies = np.array(wavelet_res['t_eddy'])
    alphas_j = np.array(wavelet_res['alpha'])
    t_eddies = t_eddies[:len(alphas_j)]
    beta = 2 * (alphas_j - 1) / alphas_j
    mask = (t_eddies > 0) & (beta > 0)
    log_t = np.log(t_eddies[mask])
    log_beta = np.log(beta[mask])
    slope, intercept, r_value, p_value, std_err = linregress(log_t, log_beta)
    delta = slope
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0, 0].scatter(t_eddies, alphas_j, label='Data')
    t_fit = np.linspace(min(t_eddies[mask]), max(t_eddies[mask]), 100)
    beta_fit = np.exp(intercept + slope * np.log(t_fit))
    alpha_fit = 1 + beta_fit / (2 - beta_fit)
    axes[0, 0].plot(t_fit, alpha_fit, color='red', label='CTRW Prediction')
    axes[0, 0].set_title('alpha vs T_eddy')
    axes[0, 0].set_xlabel('T_eddy')
    axes[0, 0].set_ylabel('alpha')
    axes[0, 0].text(0.05, 0.9, 'delta = ' + str(round(delta, 3)), transform=axes[0, 0].transAxes)
    axes[0, 0].legend()
    axes[0, 1].scatter(t_eddies, wavelet_res['gamma'][:len(t_eddies)], label='gamma')
    axes[0, 1].scatter(t_eddies, wavelet_res['nu'][:len(t_eddies)], label='nu')
    axes[0, 1].set_title('gamma and nu vs T_eddy')
    axes[0, 1].legend()
    axes[1, 0].plot(non_stat_res['centers'], non_stat_res['alpha'], label='alpha(t)')
    axes[1, 0].set_title('alpha(t) evolution')
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('alpha')
    h = axes[1, 1].hist2d(np.repeat(t_eddies, len(non_stat_res['alpha'])//len(t_eddies)), np.tile(non_stat_res['k_peak'], len(t_eddies)//len(non_stat_res['k_peak'])), bins=20, cmap='viridis')
    plt.colorbar(h[3], ax=axes[1, 1])
    axes[1, 1].set_title('alpha heatmap (T_eddy, k_peak)')
    for ax in axes.flatten():
        ax.grid(True)
    plt.tight_layout()
    timestamp = int(time.time())
    plot_path = os.path.join(data_dir, 'synthesis_plot_' + str(timestamp) + '.png')
    plt.savefig(plot_path, dpi=300)
    print('Synthesis plot saved to ' + plot_path)
    print('Calculated scaling exponent delta: ' + str(delta))

if __name__ == '__main__':
    plot_synthesis()