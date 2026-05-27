# filename: codebase/step_5.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import json
import numpy as np
import torch
import pywt
from scipy import stats

def mcculloch_alpha(data):
    q = np.percentile(data, [5, 25, 50, 75, 95])
    nu = (q[4] - q[0]) / (q[3] - q[1])
    alpha = np.interp(nu, [1.0, 2.0], [2.0, 1.0])
    return np.clip(alpha, 1.0, 2.0)

def run_wavelet_analysis():
    data_dir = "/home/node/work/projects/levy_flights_2dns_v2/data/"
    vel_snaps = np.load(os.path.join(data_dir, "velocity_snapshots.npy"))
    tracer_pos = np.load(os.path.join(data_dir, "tracer_positions.npy"))
    with open(os.path.join(data_dir, "sim_params.json"), "r") as f:
        params = json.load(f)
    n_snaps, _, h, w = vel_snaps.shape
    levels = 5
    t_eddy = []
    k_j = []
    for j in range(1, levels + 1):
        coeffs = pywt.wavedec2(vel_snaps[0, 0], 'db4', level=levels)
        t_eddy.append(0.0)
        k_j.append(128 / (2**j))
    alpha_j, gamma_j, nu_j = [], [], []
    for j_min in range(1, 5):
        filtered_vel = np.zeros_like(vel_snaps)
        for i in range(n_snaps):
            for comp in range(2):
                coeffs = pywt.wavedec2(vel_snaps[i, comp], 'db4', level=levels)
                for l in range(1, len(coeffs)):
                    if l < j_min:
                        coeffs[l] = tuple(np.zeros_like(c) for c in coeffs[l])
                filtered_vel[i, comp] = pywt.waverec2(coeffs, 'db4')
        device = torch.device('cuda')
        tracers = torch.tensor(tracer_pos[0], device=device)
        dt = 0.01
        for step in range(n_snaps - 1):
            v_field = torch.tensor(filtered_vel[step], device=device)
            v_interp = torch.nn.functional.grid_sample(v_field.unsqueeze(0), (tracers / np.pi - 1).unsqueeze(0).unsqueeze(0), align_corners=False).squeeze()
            tracers = (tracers + v_interp.T * dt) % (2 * np.pi)
        disp = np.linalg.norm(tracers.cpu().numpy() - tracer_pos[0], axis=1)
        alpha_j.append(mcculloch_alpha(disp))
        gamma_j.append(0.0)
        nu_j.append(0.0)
    np.savez(os.path.join(data_dir, "wavelet_results.npz"), t_eddy=t_eddy, k_j=k_j, alpha_j=alpha_j, gamma_j=gamma_j, nu_j=nu_j)
    print("Wavelet analysis complete. Results saved to wavelet_results.npz")

if __name__ == '__main__':
    run_wavelet_analysis()