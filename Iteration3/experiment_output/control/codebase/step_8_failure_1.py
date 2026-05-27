# filename: codebase/step_8.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import matplotlib.pyplot as plt
import time

def plot_synthesis_results():
    data_dir = "data/"
    synthesis_res = np.load(os.path.join(data_dir, "synthesis_results.npz"))
    nonstat_res = np.load(os.path.join(data_dir, "nonstationarity_results.npz"))
    wavelet_res = np.load(os.path.join(data_dir, "wavelet_results.npz"))
    alpha_t = synthesis_res['alpha_t']
    alpha_j_peak = synthesis_res['alpha_j_peak']
    corr = synthesis_res['correlation']
    rmse = synthesis_res['rmse']
    centers = nonstat_res['window_centers']
    t_eddy = wavelet_res['t_eddy']
    k_peaks = nonstat_res['k_peak']
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].plot(centers, alpha_t, 'b-', label='alpha(t) [Windowed]')
    axes[0].plot(centers, alpha_j_peak, 'r--', label='alpha(j_peak(t)) [Wavelet]')
    axes[0].set_xlabel('Time [T_L]')
    axes[0].set_ylabel('alpha')
    axes[0].set_title('alpha(t) vs alpha(j_peak(t))')
    axes[0].text(0.05, 0.05, 'Corr: ' + str(round(float(corr), 3)) + '\nRMSE: ' + str(round(float(rmse), 3)), transform=axes[0].transAxes, bbox=dict(facecolor='white', alpha=0.5))
    axes[0].legend()
    axes[0].grid(True)
    sc = axes[1].scatter(np.tile(t_eddy, len(k_peaks)//len(t_eddy) + 1)[:len(alpha_t)], np.array(k_peaks)/1.0, c=alpha_t, cmap='viridis', s=50)
    axes[1].set_xlabel('T_eddy')
    axes[1].set_ylabel('k_peak/k_box')
    axes[1].set_title('alpha map: (T_eddy, k_peak)')
    plt.colorbar(sc, ax=axes[1], label='alpha')
    axes[1].grid(True)
    plt.tight_layout()
    plot_path = os.path.join(data_dir, "synthesis_plot_" + str(int(time.time())) + ".png")
    plt.savefig(plot_path, dpi=300)
    print("Synthesis Plot saved to " + plot_path)
    print("Pearson Correlation: " + str(corr))
    print("RMSE: " + str(rmse))

if __name__ == '__main__':
    plot_synthesis_results()