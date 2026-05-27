# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import torch
import numpy as np
import json

def run_dns():
    N = 1024
    device = 'cuda'
    dt = 0.0005
    nu_h = 3.9e-31
    p = 4
    lambda_ekman = 0.5
    N_tracers = 5000
    data_dir = 'data/'
    k = torch.fft.fftfreq(N, d=1.0/N).to(device) * 2 * np.pi
    kx, ky = torch.meshgrid(k, k, indexing='ij')
    k2 = kx**2 + ky**2
    k2[0, 0] = 1e-16
    k_max = (N // 3) * (2 * np.pi / (2 * np.pi))
    mask_dealias = (torch.abs(kx) < k_max) & (torch.abs(ky) < k_max)
    omega_hat = torch.randn((N, N), device=device, dtype=torch.complex64) * 0.1
    tracer_pos = torch.rand((N_tracers, 2), device=device) * 2 * np.pi
    def get_rhs(omega_hat):
        omega_hat = omega_hat * mask_dealias
        u_hat = 1j * ky / k2 * omega_hat
        v_hat = -1j * kx / k2 * omega_hat
        u = torch.fft.ifft2(u_hat).real
        v = torch.fft.ifft2(v_hat).real
        omega = torch.fft.ifft2(omega_hat).real
        grad_omega_x = torch.fft.ifft2(1j * kx * omega_hat).real
        grad_omega_y = torch.fft.ifft2(1j * ky * omega_hat).real
        advection = -(u * grad_omega_x + v * grad_omega_y)
        dissipation = -nu_h * (k2**p) * omega_hat - lambda_ekman * omega_hat
        forcing = torch.zeros_like(omega_hat)
        mask = (torch.sqrt(k2) >= 20) & (torch.sqrt(k2) <= 40)
        forcing[mask] = (torch.randn(mask.sum(), device=device, dtype=torch.complex64) + 1j * torch.randn(mask.sum(), device=device, dtype=torch.complex64)) * 5000.0
        return torch.fft.fft2(advection) * mask_dealias + dissipation + forcing
    tracer_positions = []
    for step in range(5000):
        k1 = get_rhs(omega_hat)
        k2_rk = get_rhs(omega_hat + 0.5 * dt * k1)
        k3 = get_rhs(omega_hat + 0.5 * dt * k2_rk)
        k4 = get_rhs(omega_hat + dt * k3)
        omega_hat += (dt / 6.0) * (k1 + 2 * k2_rk + 2 * k3 + k4)
        if step % 50 == 0:
            u_hat = 1j * ky / k2 * omega_hat
            v_hat = -1j * kx / k2 * omega_hat
            u = torch.fft.ifft2(u_hat).real
            v = torch.fft.ifft2(v_hat).real
            grid = (tracer_pos / np.pi - 1).view(1, 1, N_tracers, 2)
            u_interp = torch.nn.functional.grid_sample(u.view(1, 1, N, N), grid, align_corners=False)
            v_interp = torch.nn.functional.grid_sample(v.view(1, 1, N, N), grid, align_corners=False)
            tracer_pos = (tracer_pos + torch.stack([u_interp, v_interp], dim=-1).squeeze() * dt) % (2 * np.pi)
            tracer_positions.append(tracer_pos.cpu().numpy())
    np.save(os.path.join(data_dir, 'tracer_positions.npy'), np.array(tracer_positions))
    print('Simulation complete. Tracer positions saved.')
if __name__ == '__main__':
    run_dns()