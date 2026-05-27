# filename: codebase/step_4.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import pywt
import torch
import torch.nn.functional as F
from scipy.optimize import curve_fit

def mcculloch_alpha(data):
    q = np.percentile(data, [5, 25, 50, 75, 95])
    nu_alpha = (q[4] - q[0]) / (q[3] - q[1])
    if nu_alpha >= 2.439: return 2.0
    if nu_alpha <= 1.0: return 0.5
    return 2.0 - (nu_alpha - 1.0) * 0.5

def run_wavelet_analysis():
    data_dir = "data/"
    vel_snaps = np.load(os.path.join(data_dir, "velocity_snapshots.npy"))
    tracer_pos = np.load(os.path.join(data_dir, "tracer_positions.npy"))
    scales = [1, 2, 3, 4]
    t_eddies = []
    alphas = []
    for j in scales:
        coeffs = pywt.wavedec2(vel_snaps, 'db4', mode='periodization', level=6)
        coeffs_filtered = [c if i >= (6 - j) else [np.zeros_like(sub) for sub in c] if isinstance(c, tuple) else np.zeros_like(c) for i, c in enumerate(coeffs)]
        filtered_vel = pywt.waverec2(coeffs_filtered, 'db4', mode='periodization')
        energy_list = []
        for c in coeffs[6-j:]:
            if isinstance(c, tuple):
                for sub in c: energy_list.append(np.sum(sub**2))
            else:
                energy_list.append(np.sum(c**2))
        energy = np.sum(energy_list)
        autocorr = []
        for tau in range(1, 50):
            c = np.corrcoef(energy[:-tau].flatten(), energy[tau:].flatten())[0, 1] if tau < len(energy) else 0
            autocorr.append(c)
            if c < 0.1: break
        t_eddies.append(np.trapz(autocorr))
        pos = torch.tensor(tracer_pos[0], device='cuda')
        vel_t = torch.tensor(filtered_vel, device='cuda')
        for _ in range(100):
            grid = (pos / (2 * np.pi) * 2 - 1).view(1, 1, -1, 2)
            v = F.grid_sample(vel_t, grid, align_corners=True).squeeze().T
            pos = (pos + v * 2.0) % (2 * np.pi)
        disp = np.linalg.norm(pos.cpu().numpy() - tracer_pos[0], axis=1)
        alphas.append(mcculloch_alpha(disp))
    def ctrw_model(t, delta):
        beta = 0.5 * (t**delta)
        return 1.0 + beta / (2.0 - beta)
    popt, _ = curve_fit(ctrw_model, t_eddies, alphas)
    print("T_eddy values: " + str(t_eddies))
    print("Alpha values: " + str(alphas))
    print("CTRW delta: " + str(popt[0]))
    np.savez(os.path.join(data_dir, "wavelet_results.npz"), t_eddies=t_eddies, alphas=alphas, delta=popt[0])

if __name__ == '__main__':
    run_wavelet_analysis()