# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import torch
import numpy as np

class DNS2D:
    def __init__(self, N=1024, nu_h=1e-19, nu=1e-4, p=4, device='cuda'):
        self.N = N
        self.L = 2 * np.pi
        self.device = device
        self.nu_h = nu_h
        self.nu = nu
        self.p = p
        k = torch.fft.fftfreq(N, d=self.L/N).to(device) * N
        kx, ky = torch.meshgrid(k, k[:N//2+1], indexing='ij')
        self.kx, self.ky = kx, ky
        self.k2 = kx**2 + ky**2
        self.k2[0, 0] = 1.0
        self.inv_k2 = 1.0 / self.k2
        self.inv_k2[0, 0] = 0.0
        self.mask = (torch.sqrt(self.k2) < (2/3 * N/2)).float()
        self.w = torch.randn((N, N), device=device) * 0.01

    def get_velocity(self, w_hat):
        psi_hat = -w_hat * self.inv_k2
        u = torch.fft.irfft2(1j * self.ky * psi_hat)
        v = torch.fft.irfft2(-1j * self.kx * psi_hat)
        return u, v

    def rhs(self, w_hat):
        u, v = self.get_velocity(w_hat)
        w_x = torch.fft.irfft2(1j * self.kx * w_hat)
        w_y = torch.fft.irfft2(1j * self.ky * w_hat)
        nl = -torch.fft.rfft2(u * w_x + v * w_y)
        lin = -(self.nu * self.k2 + self.nu_h * (self.k2**self.p)) * w_hat
        forcing = torch.zeros_like(w_hat)
        k_mod = torch.sqrt(self.k2)
        forcing[(k_mod >= 3) & (k_mod <= 5)] = torch.randn_like(forcing[(k_mod >= 3) & (k_mod <= 5)]) * 0.5
        return (nl + lin + forcing) * self.mask

    def step(self, w, dt):
        w_hat = torch.fft.rfft2(w)
        k1 = self.rhs(w_hat)
        k2 = self.rhs(w_hat + 0.5 * dt * k1)
        k3 = self.rhs(w_hat + 0.5 * dt * k2)
        k4 = self.rhs(w_hat + dt * k3)
        w_hat_new = w_hat + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        if torch.isnan(w_hat_new).any():
            raise RuntimeError('NaN detected')
        return torch.fft.irfft2(w_hat_new)

if __name__ == '__main__':
    data_dir = 'data/'
    solver = DNS2D()
    dt = 0.00005
    for _ in range(1000):
        solver.w = solver.step(solver.w, dt)
    n_tracers = 5000
    tracers = torch.rand((n_tracers, 2), device='cuda') * 2 * np.pi
    tracer_pos = []
    for _ in range(5000):
        solver.w = solver.step(solver.w, dt)
        u, v = solver.get_velocity(torch.fft.rfft2(solver.w))
        grid = (tracers / np.pi) - 1.0
        u_interp = torch.nn.functional.grid_sample(u.unsqueeze(0).unsqueeze(0), grid.view(1, 1, -1, 2), align_corners=False)
        v_interp = torch.nn.functional.grid_sample(v.unsqueeze(0).unsqueeze(0), grid.view(1, 1, -1, 2), align_corners=False)
        tracers += torch.stack([u_interp, v_interp], dim=-1).squeeze() * dt
        tracers %= (2 * np.pi)
        tracer_pos.append(tracers.cpu().numpy())
    np.save(os.path.join(data_dir, 'tracer_positions.npy'), np.array(tracer_pos))
    print('Simulation complete. Tracer positions saved to ' + data_dir + 'tracer_positions.npy')