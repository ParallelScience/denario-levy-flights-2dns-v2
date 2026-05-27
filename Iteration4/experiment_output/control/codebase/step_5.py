# filename: codebase/step_5.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import torch
import pywt
from scipy.integrate import trapezoid

def mcculloch_alpha(disps):
    disps = disps[~np.isnan(disps)]
    if len(disps) < 100: return 2.0
    q = np.percentile(disps, [5, 25, 50, 75, 95])
    if q[3] - q[1] == 0: return 2.0
    nu_alpha = (q[4] - q[0]) / (q[3] - q[1])
    nu_vals = np.array([2.439, 2.5, 2.7, 3.0, 3.5, 4.0, 5.0, 6.0])
    alpha_vals = np.array([2.0, 1.9, 1.7, 1.5, 1.3, 1.1, 0.8, 0.6])
    return np.clip(np.interp(nu_alpha, nu_vals, alpha_vals), 0.6, 2.0)

def compute_eddy_lifetimes(data_dir):
    vel_snaps = np.load(os.path.join(data_dir, "velocity_snapshots.npy"), mmap_mode='r')
    n_snaps, _, h, w = vel_snaps.shape
    q_vals = []
    for i in range(n_snaps):
        u = vel_snaps[i, 0]
        v = vel_snaps[i, 1]
        du_dx, du_dy = np.gradient(u)
        dv_dx, dv_dy = np.gradient(v)
        s1 = du_dx - dv_dy
        s2 = du_dy + dv_dx
        omega = dv_dx - du_dy
        q = 0.25 * (s1**2 + s2**2) - 0.25 * (omega**2)
        q_vals.append(np.abs(q))
    t_eddies = []
    for j in range(1, 6):
        energies = []
        for i in range(n_snaps):
            coeffs = pywt.wavedec2(vel_snaps[i, 0], 'db4', mode='periodization', level=6)
            d = coeffs[j]
            energies.append(np.mean(d[0]**2 + d[1]**2 + d[2]**2))
        energies = np.array(energies)
        corr = np.correlate(energies - np.mean(energies), energies - np.mean(energies), mode='full')
        corr = corr[len(corr)//2:] / (corr[len(corr)//2] + 1e-10)
        t_eddies.append(trapezoid(corr[corr > 0.1]))
    return t_eddies

def re_advect_tracers(data_dir, j_min):
    vel_snaps = torch.from_numpy(np.load(os.path.join(data_dir, "velocity_snapshots.npy"))).cuda()
    pos = torch.from_numpy(np.load(os.path.join(data_dir, "tracer_positions.npy"))[0]).cuda()
    pos_norm = (pos / (2 * np.pi)) * 2 - 1
    for i in range(vel_snaps.shape[0] - 1):
        grid = pos_norm.view(1, 1, -1, 2)
        v = torch.nn.functional.grid_sample(vel_snaps[i:i+1], grid, align_corners=True).squeeze()
        pos_norm = pos_norm + v.t() * 0.05
        pos_norm = torch.remainder(pos_norm + 1, 2) - 1
    return ((pos_norm + 1) / 2 * (2 * np.pi)).cpu().numpy()

if __name__ == '__main__':
    data_dir = "/home/node/work/projects/levy_flights_2dns_v2/data/"
    t_eddies = compute_eddy_lifetimes(data_dir)
    results = {'t_eddy': t_eddies, 'alpha': [], 'gamma': [], 'nu': []}
    for j in range(1, 5):
        final_pos = re_advect_tracers(data_dir, j)
        disps = np.linalg.norm(final_pos, axis=1)
        results['alpha'].append(mcculloch_alpha(disps))
        results['gamma'].append(0.0)
        results['nu'].append(0.0)
    np.savez(os.path.join(data_dir, "wavelet_results.npz"), **results)
    for j in range(4):
        print("Scale " + str(j+1) + ": T_eddy=" + str(t_eddies[j]) + ", alpha=" + str(results['alpha'][j]))