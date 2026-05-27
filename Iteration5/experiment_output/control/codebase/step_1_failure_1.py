# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import torch
import numpy as np
import json
import os

class DNS2D:
    """
    GPU-accelerated 2D Navier-Stokes solver in vorticity-streamfunction form.
    Uses pseudo-spectral method with integrating factor for hyperviscosity.
    """
    def __init__(self, N=1024, nu_h=1e-19, p=4, eps_inj=0.1, device='cuda'):
        self.N = N
        self.L = 2 * np.pi
        self.device = device
        self.nu_h = nu_h
        self.p = p
        self.eps_inj = eps_inj
        self.k = torch.fft.fftfreq(N, d=self.L/N).to(device) * N
        self.kx, self.ky = torch.meshgrid(self.k, self.k, indexing='ij')
        self.k2 = self.kx**2 + self.ky**2
        self.k2[0, 0] = 1.0
        self.inv_k2 = 1.0 / self.k2
        self.inv_k2[0, 0] = 0.0
        self.w = torch.zeros((N, N), device=device)
        self.force_amp = 1.0

    def get_velocity(self, w_hat):
        psi_hat = -w_hat * self.inv_k2
        u_hat = 1j * self.ky * psi_hat
        v_hat = -1j * self.kx * psi_hat
        return torch.fft.irfft2(u_hat[:, :self.N//2+1]), torch.fft.irfft2(v_hat[:, :self.N//2+1])

    def step(self, w, dt):
        w_hat = torch.fft.rfft2(w)
        u, v = self.get_velocity(w_hat)
        nl = -torch.fft.rfft2(u * torch.fft.irfft2(1j * self.kx * w_hat) + v * torch.fft.irfft2(1j * self.ky * w_hat))
        lin = -self.nu_h * (self.k2**self.p) * w_hat
        w_hat_new = w_hat + dt * (nl + lin)
        if torch.isnan(w_hat_new).any():
            raise RuntimeError("NaN detected in spectral vorticity")
        return torch.fft.irfft2(w_hat_new)

if __name__ == '__main__':
    data_dir = "/home/node/work/projects/levy_flights_2dns_v2/data/"
    solver = DNS2D()
    dt = 0.01
    t = 0.0
    while t < 20.0:
        solver.w = solver.step(solver.w, dt)
        t += dt
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
    np.save(os.path.join(data_dir, "tracer_positions.npy"), np.array(tracer_pos))
    print("Simulation complete. Tracer positions saved to " + data_dir + "tracer_positions.npy")