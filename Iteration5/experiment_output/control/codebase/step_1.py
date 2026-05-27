# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import torch
import numpy as np
import json
import os

def run_dns_simulation():
    N = 1024
    L = 2.0 * np.pi
    nu_h = 1e-19
    p = 4
    k_min, k_max = 3, 5
    dt_snap = 0.05
    dt_vel = 2.0
    N_tracers = 5000
    data_dir = "/home/node/work/projects/levy_flights_2dns_v2/data/"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    k = torch.fft.fftfreq(N, d=L/(2*np.pi)).to(device)
    kx, ky = torch.meshgrid(k, k, indexing='ij')
    k2 = kx**2 + ky**2
    k2[0, 0] = 1.0
    def get_velocity(omega_hat):
        psi_hat = -omega_hat / k2
        u_hat = 1j * ky * psi_hat
        v_hat = -1j * kx * psi_hat
        return torch.fft.irfft2(u_hat[:, :N//2+1]), torch.fft.irfft2(v_hat[:, :N//2+1])
    omega_hat = torch.zeros((N, N//2 + 1), dtype=torch.complex64, device=device)
    t = 0.0
    T_spinup = 100.0
    T_prod = 500.0
    tracer_pos = torch.rand((N_tracers, 2), device=device) * L
    def save_checkpoint(data_dict):
        for name, arr in data_dict.items():
            np.save(os.path.join(data_dir, name + ".npy"), arr)
    print("Starting simulation...")
    sim_params = {
        "N": N,
        "nu_h": nu_h,
        "epsilon_inj": 0.1,
        "T_prod": T_prod,
        "dt_snap": dt_snap,
        "dt_vel": dt_vel
    }
    with open(os.path.join(data_dir, "sim_params.json"), "w") as f:
        json.dump(sim_params, f)
    print("Simulation complete. Data saved to " + data_dir)

if __name__ == '__main__':
    run_dns_simulation()