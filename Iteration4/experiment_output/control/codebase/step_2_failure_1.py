# filename: codebase/step_2.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import matplotlib.pyplot as plt
import time

def verify_data():
    data_dir = "data/"
    tracer_pos = np.load(os.path.join(data_dir, "tracer_positions.npy"))
    tracer_vel = np.load(os.path.join(data_dir, "tracer_velocities.npy"))
    energy_spec = np.load(os.path.join(data_dir, "energy_spectrum.npy"))
    N_snaps, N_tracers, _ = tracer_pos.shape
    diffs = np.diff(tracer_pos, axis=0)
    diffs = (diffs + np.pi) % (2 * np.pi) - np.pi
    mean_disp = np.mean(np.sqrt(np.sum(diffs**2, axis=-1)))
    print("Mean tracer displacement per snapshot: " + str(mean_disp))
    if mean_disp > 1.88:
        print("WARNING: Mean displacement exceeds threshold!")
    vacf = []
    lags = [1, 2, 4, 8, 16]
    for lag in lags:
        v_t = tracer_vel[:-lag]
        v_t_lag = tracer_vel[lag:]
        corr = np.mean(np.sum(v_t * v_t_lag, axis=-1)) / np.mean(np.sum(v_t**2, axis=-1))
        vacf.append(corr)
    print("VACF at lags " + str(lags) + ": " + str(vacf))
    tau_corr = 0
    for i, val in enumerate(vacf):
        if val < 1/np.e:
            tau_corr = lags[i]
            break
    print("Decorrelation time (tau_corr): " + str(tau_corr))
    plt.figure(figsize=(8, 6))
    fractions = [0.1, 0.3, 0.6, 1.0]
    for f in fractions:
        idx = int(f * (N_snaps - 1))
        plt.loglog(energy_spec[idx], label="t = " + str(f) + " * T_prod")
        k_peak = np.argmax(energy_spec[idx][1:]) + 1
        print("k_peak at t=" + str(f) + " * T_prod: " + str(k_peak))
    plt.xlabel("Wavenumber k")
    plt.ylabel("Energy E(k)")
    plt.title("Energy Spectrum Evolution")
    plt.legend()
    plt.grid(True)
    timestamp = int(time.time())
    plot_path = os.path.join(data_dir, "verification_plot_2_" + str(timestamp) + ".png")
    plt.savefig(plot_path, dpi=300)
    print("Verification plot saved to " + plot_path)

if __name__ == '__main__':
    verify_data()