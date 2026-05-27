# filename: codebase/step_4.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import matplotlib.pyplot as plt
import json
import time

def plot_nonstationarity_results():
    data_dir = 'data/'
    res = np.load(os.path.join(data_dir, 'nonstationarity_results.npz'))
    with open(os.path.join(data_dir, 'sim_params.json'), 'r') as f:
        params = json.load(f)
    
    alpha_mcc = res['alpha_mcculloch']
    alpha_hill = res['alpha_hill']
    centers = res['window_centers']
    k_peaks = res['k_peak']
    adf_stat = res['adf_stat']
    kpss_stat = res['kpss_stat']
    popt = res['fit_params']
    
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    
    axes[0, 0].plot(centers, alpha_mcc, 'b-', label='McCulloch')
    axes[0, 0].plot(centers, alpha_hill, 'r--', label='Hill')
    axes[0, 0].set_title('Levy Exponent alpha(t)')
    axes[0, 0].set_xlabel('Time [T_L]')
    axes[0, 0].set_ylabel('alpha')
    axes[0, 0].text(0.05, 0.95, 'ADF: ' + str(round(float(adf_stat), 2)) + '\nKPSS: ' + str(round(float(kpss_stat), 2)), 
                    transform=axes[0, 0].transAxes, verticalalignment='top')
    axes[0, 0].legend()
    
    axes[0, 1].plot(centers, k_peaks, 'g-')
    axes[0, 1].set_title('Peak Wavenumber k_peak(t)')
    axes[0, 1].set_xlabel('Time [T_L]')
    axes[0, 1].set_ylabel('k_peak')
    
    axes[1, 0].scatter(np.array(k_peaks)/1.0, alpha_mcc, alpha=0.5)
    k_range = np.linspace(min(k_peaks), max(k_peaks), 100)
    axes[1, 0].plot(k_range/1.0, popt[0] + popt[1] * (k_range/1.0)**popt[2], 'r-')
    axes[1, 0].set_title('alpha vs k_peak/k_box')
    axes[1, 0].set_xlabel('k_peak/k_box')
    axes[1, 0].set_ylabel('alpha')
    
    for ax in axes.flatten():
        ax.grid(True)
        
    plt.tight_layout()
    plot_path = os.path.join(data_dir, 'nonstationarity_plot_' + str(int(time.time())) + '.png')
    plt.savefig(plot_path, dpi=300)
    
    print('ADF Statistic: ' + str(adf_stat))
    print('KPSS Statistic: ' + str(kpss_stat))
    print('Fit parameters: a=' + str(popt[0]) + ', b=' + str(popt[1]) + ', c=' + str(popt[2]))
    print('Saved to ' + plot_path)

if __name__ == '__main__':
    plot_nonstationarity_results()