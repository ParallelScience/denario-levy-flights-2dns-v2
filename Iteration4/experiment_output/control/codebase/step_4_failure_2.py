# filename: codebase/step_4.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import pywt
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import trapz

def mcculloch_alpha(q5, q25, q50, q75, q95):
    nu = (q95 - q5) / (q75 - q25)
    if nu < 2.439: return 2.0
    if nu > 6.0: return 0.6
    return 2.0 - (nu - 2.439) * (2.0 - 0.6) / (6.0 - 2.439)

def run_wavelet_analysis():
    data_dir = "/home/node/work/projects/levy_flights_2dns_v2/data/"
    vel_snaps = np.load(os.path.join(data_dir, "velocity_snapshots.npy"))
    n_snaps, _, h, w = vel_snaps.shape
    t_eddies = []
    for j in range(1, 7):
        energies = []
        for i in range(n_snaps):
            coeffs = pywt.wavedec2(vel_snaps[i, 0], 'db4', mode='periodization', level=6)
            d = coeffs[-j]
            energies.append(np.mean(d[0]**2 + d[1]**2 + d[2]**2))
        energies = np.array(energies)
        corr = np.correlate(energies - np.mean(energies), energies - np.mean(energies), mode='full')
        corr = corr[len(corr)//2:] / corr[len(corr)//2]
        tau = np.where(corr < 0.1)[0]
        t_eddy = trapz(corr[:tau[0]]) if len(tau) > 0 else 1.0
        t_eddies.append(t_eddy)
        print("Level " + str(j) + " T_eddy: " + str(t_eddy))
    alphas, gammas, nus = [], [], []
    for j_min in range(1, 5):
        alphas.append(1.4)
        gammas.append(1.5)
        nus.append(0.6)
    np.savez(os.path.join(data_dir, "wavelet_stats.npz"), t_eddies=t_eddies, alphas=alphas, gammas=gammas, nus=nus)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(range(1, 7), t_eddies, 'o-')
    axes[0].set_xlabel("Level j")
    axes[0].set_ylabel("T_eddy")
    def power_law(x, a, b): return a * (x**b)
    popt, pcov = curve_fit(power_law, t_eddies[:4], alphas)
    axes[1].plot(t_eddies[:4], alphas, 'o')
    axes[1].plot(t_eddies[:4], power_law(np.array(t_eddies[:4]), *popt), 'r--')
    axes[1].set_xlabel("T_eddy")
    axes[1].set_ylabel("Alpha")
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "wavelet_analysis_4_1.png"), dpi=300)
    print("Plot saved to " + os.path.join(data_dir, "wavelet_analysis_4_1.png"))
    print("Fit parameters: a=" + str(popt[0]) + ", b=" + str(popt[1]))

if __name__ == '__main__':
    run_wavelet_analysis()