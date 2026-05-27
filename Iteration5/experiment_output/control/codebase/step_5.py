# filename: codebase/step_5.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import torch
import torch.nn.functional as F

def mcculloch_alpha(data):
    if len(data) < 100: return 2.0
    q = np.percentile(data, [5, 25, 50, 75, 95])
    nu_alpha = (q[4] - q[0]) / (q[3] - q[1])
    if nu_alpha >= 2.439: return 2.0
    if nu_alpha <= 1.0: return 0.5
    return 2.0 - (nu_alpha - 1.0) * 0.5

def run_lcs_analysis():
    data_dir = "/home/node/work/projects/levy_flights_2dns_v2/data/"
    vel_snaps = torch.tensor(np.load(os.path.join(data_dir, "velocity_snapshots.npy")), device='cuda')
    tracer_pos = torch.tensor(np.load(os.path.join(data_dir, "tracer_positions.npy")), device='cuda')
    tracer_times = torch.tensor(np.load(os.path.join(data_dir, "tracer_times.npy")), device='cuda')
    vel_times = torch.tensor(np.load(os.path.join(data_dir, "vel_times.npy")), device='cuda')
    N_vel, _, H, W = vel_snaps.shape
    N_tracer_snaps, N_tracers, _ = tracer_pos.shape
    dx = 2.0 * np.pi / H
    u = vel_snaps[:, 0, :, :]
    v = vel_snaps[:, 1, :, :]
    du_dx = (torch.roll(u, -1, dims=2) - torch.roll(u, 1, dims=2)) / (2 * dx)
    du_dy = (torch.roll(u, -1, dims=1) - torch.roll(u, 1, dims=1)) / (2 * dx)
    dv_dx = (torch.roll(v, -1, dims=2) - torch.roll(v, 1, dims=2)) / (2 * dx)
    dv_dy = (torch.roll(v, -1, dims=1) - torch.roll(v, 1, dims=1)) / (2 * dx)
    omega = dv_dx - du_dy
    s2 = (du_dx - dv_dy)**2 + (du_dy + dv_dx)**2
    Q = s2 - omega**2
    tracer_class = torch.zeros((N_tracer_snaps, N_tracers), device='cuda')
    for t in range(N_tracer_snaps):
        vel_idx = torch.argmin(torch.abs(vel_times - tracer_times[t]))
        q_t = Q[vel_idx].unsqueeze(0).unsqueeze(0)
        q_std = torch.std(q_t)
        grid = (tracer_pos[t] / (2 * np.pi) * 2 - 1).view(1, 1, -1, 2)
        q_tracers = F.grid_sample(q_t, grid, align_corners=True).squeeze()
        tracer_class[t, q_tracers < -q_std] = 1
        tracer_class[t, q_tracers > q_std] = 2
    pos_diff = (tracer_pos[1:] - tracer_pos[:-1] + np.pi) % (2 * np.pi) - np.pi
    disp = torch.norm(pos_diff, dim=-1).cpu().numpy().flatten()
    vortex_mask = (tracer_class[:-1] == 1).cpu().numpy().flatten()
    strain_mask = (tracer_class[:-1] == 2).cpu().numpy().flatten()
    alpha_vortex = mcculloch_alpha(disp[vortex_mask])
    alpha_strain = mcculloch_alpha(disp[strain_mask])
    print("Alpha Vortex: " + str(alpha_vortex))
    print("Alpha Strain: " + str(alpha_strain))
    np.savez(os.path.join(data_dir, "lcs_results.npz"), alpha_vortex=alpha_vortex, alpha_strain=alpha_strain)

if __name__ == '__main__':
    run_lcs_analysis()