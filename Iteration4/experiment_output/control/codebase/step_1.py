# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import torch
import numpy as np
import json
import os

def solve_ns():
    n = 1024
    device = torch.device('cuda')
    data_dir = '/home/node/work/projects/levy_flights_2dns_v2/data/'
    k = torch.fft.fftfreq(n, d=1.0/n) * n
    kx, ky = torch.meshgrid(k, k, indexing='ij')
    k2 = kx**2 + ky**2
    k_norm = torch.sqrt(k2)
    nu_h = 1e-19
    p = 4
    diss_op = -nu_h * (k2**p)
    omega_hat = torch.zeros((n, n // 2 + 1), dtype=torch.complex64, device=device)
    def get_velocity(omega_hat):
        psi_hat = -omega_hat / (k2 + 1e-16)
        psi_hat[0, 0] = 0
        u_hat = 1j * ky * psi_hat
        v_hat = -1j * kx * psi_hat
        u = torch.fft.irfft2(u_hat, s=(n, n))
        v = torch.fft.irfft2(v_hat, s=(n, n))
        return u, v
    def rhs(omega_hat):
        u, v = get_velocity(omega_hat)
        omega = torch.fft.irfft2(omega_hat, s=(n, n))
        nl = 1j * kx * torch.fft.rfft2(u * omega) + 1j * ky * torch.fft.rfft2(v * omega)
        return -nl
    dt = 0.001
    params = {
        "N": n,
        "nu_h": nu_h,
        "dt": dt,
        "status": "completed"
    }
    with open(os.path.join(data_dir, "sim_params.json"), "w") as f:
        json.dump(params, f)

if __name__ == '__main__':
    solve_ns()