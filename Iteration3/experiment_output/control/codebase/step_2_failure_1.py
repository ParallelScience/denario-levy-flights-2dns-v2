# filename: codebase/step_2.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import json
import numpy as np
import matplotlib.pyplot as plt
import time

def verify_data_quality():
    data_dir = "data/"
    with open(os.path.join(data_dir, "sim_params.json"), "r") as f:
        params = json.load(f)
    tracer_pos = np.load(os.path.join(data_dir, "tracer_positions.npy"))
    tracer_vel = np.load(os.path.join(data_dir, "tracer_velocities.npy"))
    tracer_times = np.load(os.path.join(data_dir, "tracer_times.npy"))
    energy_spec = np.load(os.path.join(data_dir, "energy_spectrum.npy"))
    dt_snap = tracer_times[1] - tracer_times[0]
    diffs = np.diff(tracer_pos, axis=0)
    diffs = (diffs + np.pi) % (2 * np.pi) - np.pi
    mean_disp = np.mean(np.sqrt(np.sum(diffs**2, axis=-1)), axis=-1)
    print("Mean tracer displacement per snapshot: " + str(np.mean(mean_disp)))
    if np.mean(mean_disp) > 1.88:
        print("FLAG: Mean displacement exceeds 0.3 domain units!")
    vacf = []
    lags = [1, 2, 4, 8, 16]
    for lag in lags:
        v0 = tracer_vel[:-lag]
        vt = tracer_vel[lag:]
        corr = np.mean(np.sum(v0 * vt, axis=-1)) / np.mean(np.sum(v0 * v0, axis=-1))
        vacf.append(corr)
    print("Normalized VACF at lags " + str(lags) + ": " + str(vacf))
    tau_corr_steps = 0
    for i, val in enumerate(vacf):
        if val < 1/np.e:
            tau_corr_steps = lags[i]
            break
    print("Decorrelation time: " + str(tau_corr_steps) + " steps (" + str(tau_corr_steps * dt_snap) + " time units)")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].plot(tracer_times[1:], mean_disp)
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Mean Displacement")
    axes[0].set_title("Tracer Displacement")
    axes[1].plot(lags, vacf, 'o-')
    axes[1].axhline(1/np.e, color='r', linestyle='--')
    axes[1].set_xlabel("Lag (steps)")
    axes[1].set_ylabel("Normalized VACF")
    axes[1].set_title("VACF")
    time_fractions = [0.1, 0.3, 0.6, 1.0]
    k = np.arange(1, energy_spec.shape[1] + 1)
    for frac in time_fractions:
        idx = int(frac * (len(tracer_times) - 1))
        spec = energy_spec[idx]
        axes[2].loglog(k, spec, label="t=" + str(round(tracer_times[idx], 2)))
        k_peak = k[np.argmax(spec)]
        print("k_peak at t=" + str(tracer_times[idx]) + ": " + str(k_peak))
    axes[2].set_xlabel("k")
    axes[2].set_ylabel("E(k)")
    axes[2].set_title("Energy Spectrum")
    axes[2].legend()
    plt.tight_layout()
    plot_path = os.path.join(data_dir, "data_quality_1_" + str(int(time.time())) + ".png")
    plt.savefig(plot_path, dpi=300)
    print("Saved to " + plot_path)

if __name__ == '__main__':
    verify_data_quality()