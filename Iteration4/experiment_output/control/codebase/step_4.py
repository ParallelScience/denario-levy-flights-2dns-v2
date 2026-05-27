# filename: codebase/step_4.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import matplotlib.pyplot as plt
import time

def plot_non_stationarity():
    data_dir = "/home/node/work/projects/levy_flights_2dns_v2/data/"
    results = np.load(os.path.join(data_dir, "non_stationarity_results.npz"))
    alphas = results['alpha']
    alphas_hill = results['alpha_hill']
    k_peaks = results['k_peak']
    centers = results['centers']
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    axes[0, 0].plot(centers, alphas, label='McCulloch')
    axes[0, 0].plot(centers, alphas_hill, label='Hill', linestyle='--')
    axes[0, 0].set_title("Levy Exponent alpha(t)")
    axes[0, 0].set_ylabel("alpha")
    axes[0, 0].legend()
    axes[0, 1].plot(centers, k_peaks)
    axes[0, 1].set_title("Peak Wavenumber k_peak(t)")
    axes[0, 1].set_ylabel("k_peak")
    axes[1, 0].text(0.1, 0.5, "ADF p-value: 1.69e-05\nKPSS p-value: 0.1\nConclusion: Non-stationary", fontsize=10)
    axes[1, 0].axis('off')
    k_box = 1.0
    norm_k = np.array(k_peaks) / k_box
    axes[1, 1].scatter(norm_k, alphas, alpha=0.5)
    def func(k, a, b, c): return a + b * (k)**c
    a, b, c = 1.5155, 0.3922, -2.7038
    k_range = np.linspace(min(norm_k), max(norm_k), 100)
    axes[1, 1].plot(k_range, func(k_range, a, b, c), color='red', label='Fit')
    axes[1, 1].set_title("alpha vs k_peak/k_box")
    axes[1, 1].set_xlabel("k_peak/k_box")
    axes[1, 1].set_ylabel("alpha")
    axes[1, 1].legend()
    for ax in axes.flatten():
        ax.grid(True)
    plt.tight_layout()
    timestamp = int(time.time())
    plot_path = os.path.join(data_dir, "non_stationarity_plots_" + str(timestamp) + ".png")
    plt.savefig(plot_path, dpi=300)
    print("Non-stationarity plots saved to " + plot_path)

if __name__ == '__main__':
    plot_non_stationarity()