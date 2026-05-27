# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import torch
import numpy as np
import json
import os

def get_wavenumbers(N, device):
    k = torch.fft.fftfreq(N, d=1.0/N)
    kr = torch.fft.rfftfreq(N, d=1.0/N)
    kx, ky = torch.meshgrid(torch.tensor(k), torch.tensor(kr), indexing='ij')
    kx = kx.to(device)
    ky = ky.to(device)
    k2 = kx**2 + ky**2
    k2[0, 0] = 1.0
    return kx, ky, k2

def get_energy(omega_hat, k2):
    psi_hat = -omega_hat / k2
    N = omega_hat.shape[0]
    k_x = torch.fft.fftfreq(N, d=1.0/N).to(omega_hat.device)
    k_y = torch.fft.rfftfreq(N, d=1.0/N).to(omega_hat.device)
    kx, ky = torch.meshgrid(k_x, k_y, indexing='ij')
    u_hat = 1j * ky * psi_hat
    v_hat = -1j * kx * psi_hat
    return 0.5 * torch.sum(torch.abs(u_hat)**2 + torch.abs(v_hat)**2) / (N**4)

if __name__ == '__main__':
    device = 'cuda'
    N = 1024
    nu_h = 1e-19
    p = 4
    epsilon_inj = 0.1
    dt = 0.05
    data_dir = '/home/node/work/projects/levy_flights_2dns_v2/data/'
    kx, ky, k2 = get_wavenumbers(N, device)
    force_mask = ((k2**0.5 >= 3) & (k2**0.5 <= 5)).float()
    diss_op = -nu_h * (k2**p)
    IF = torch.exp(diss_op * dt)
    omega_hat = torch.zeros((N, N//2 + 1), dtype=torch.complex64, device=device)
    def nonlinear(w_hat, kx, ky, k2):
        psi_hat = -w_hat / k2
        u = torch.fft.irfft2(1j * ky * psi_hat)
        v = torch.fft.irfft2(-1j * kx * psi_hat)
        w = torch.fft.irfft2(w_hat)
        w_x = torch.fft.irfft2(1j * kx * w_hat)
        w_y = torch.fft.irfft2(1j * ky * w_hat)
        return torch.fft.rfft2(-(u * w_x + v * w_y))
    omega_calib = torch.zeros_like(omega_hat)
    dE_acc = 0.0
    for _ in range(20):
        E0 = get_energy(omega_calib, k2)
        phi = torch.exp(2j * np.pi * torch.rand(omega_hat.shape, device=device))
        omega_calib += np.sqrt(dt) * phi * force_mask
        dE_acc += get_energy(omega_calib, k2) - E0
    force_amp = float(np.sqrt(epsilon_inj / (dE_acc / (20 * dt))))
    params = {
        "N": N, "nu_h": nu_h, "epsilon_inj": epsilon_inj,
        "dt_snap": 0.05, "dt_vel": 2.0, "T_spinup": 20.0,
        "T_prod": 1000.0, "force_amp": force_amp
    }
    with open(os.path.join(data_dir, 'sim_params.json'), 'w') as f:
        json.dump(params, f)