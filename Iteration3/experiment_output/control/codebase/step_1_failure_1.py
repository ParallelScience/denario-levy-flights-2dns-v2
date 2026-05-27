# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import torch
import numpy as np
import json
import os

def get_k_grid(n):
    k = torch.fft.fftfreq(n, d=1.0/n)
    kx, ky = torch.meshgrid(k, k, indexing='ij')
    k2 = kx**2 + ky**2
    k2[0, 0] = 1e-16
    return kx, ky, k2

def solve_ns():
    n = 512
    nu_h = 1e-28
    p = 4
    epsilon_inj = 0.1
    dt = 0.05
    t_spinup = 20.0
    data_dir = "data/"
    device = torch.device('cuda')
    kx, ky, k2 = get_k_grid(n)
    kx, ky, k2 = kx.to(device), ky.to(device), k2.to(device)
    w_hat = torch.zeros((n, n // 2 + 1), dtype=torch.complex64, device=device)
    lin_op = torch.exp(-nu_h * k2**(p) * dt)
    def get_force():
        f = torch.randn((n, n // 2 + 1), dtype=torch.complex64, device=device)
        mask = (k2.sqrt() >= 10) & (k2.sqrt() <= 12)
        f *= mask
        return f
    force_amp = 90000.0
    t = 0.0
    while t < t_spinup:
        psi_hat = -w_hat / k2
        u_hat = 1j * ky * psi_hat
        v_hat = -1j * kx * psi_hat
        u = torch.fft.irfft2(u_hat)
        v = torch.fft.irfft2(v_hat)
        nl = -torch.fft.rfft2(u * torch.fft.irfft2(1j * kx * w_hat) + v * torch.fft.irfft2(1j * ky * w_hat))
        w_hat = w_hat * lin_op + nl * dt + get_force() * force_amp * dt
        t += dt
    e_total = 0.5 * torch.sum(torch.abs(w_hat)**2 / k2) / (n**2)
    u_rms = torch.sqrt(2 * e_total)
    t_l = 2 * np.pi / u_rms
    params = {
        "N": n,
        "nu_h": nu_h,
        "epsilon_inj": epsilon_inj,
        "T_spinup": t_spinup,
        "T_L_at_start_of_production": float(t_l),
        "coarsened_resolution": 128,
        "force_amp_calibrated": force_amp
    }
    with open(os.path.join(data_dir, "sim_params.json"), "w") as f:
        json.dump(params, f)

if __name__ == '__main__':
    solve_ns()