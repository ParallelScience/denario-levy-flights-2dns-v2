# filename: codebase/step_2.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import os
import json
import matplotlib.pyplot as plt

def verify_simulation():
    data_dir = "/home/node/work/projects/levy_flights_2dns_v2/data/"
    try:
        pos = np.load(os.path.join(data_dir, "tracer_positions.npy"))
        vel = np.load(os.path.join(data_dir, "tracer_velocities.npy"))
        times = np.load(os.path.join(data_dir, "tracer_times.npy"))
        spec = np.load(os.path.join(data_dir, "energy_spectrum.npy"))
        diag = np.load(os.path.join(data_dir, "diagnostics.npy"))
    except FileNotFoundError as e:
        print("Error: Required files not found. Ensure Step 1 completed successfully.")
        raise e
    n_snaps, n_tracers, _ = pos.shape
    if n_snaps < 1000:
        print("Error: N_tracer_snaps is " + str(n_snaps) + ", which is below the threshold of 1000.")
        return
    diffs = np.diff(pos, axis=0)
    diffs = (diffs + np.pi) % (2 * np.pi) - np.pi
    mean_disp = np.mean(np.sqrt(np.sum(diffs**2, axis=-1)))
    print("Mean tracer displacement per snapshot: " + str(mean_disp))
    if mean_disp > 1.88:
        print("WARNING: Mean displacement exceeds 1.88 domain units. Lévy statistics may be invalid.")
    vacf = []
    lags = [1, 2, 4, 8, 16]
    for lag in lags:
        v_t = vel[:-lag]
        v_t_lag = vel[lag:]
        corr = np.mean(np.sum(v_t * v_t_lag, axis=-1)) / np.mean(np.sum(v_t**2, axis=-1))
        vacf.append(corr)
    tau_corr = 0
    for i, val in enumerate(vacf):
        if val < 1/np.e:
            tau_corr = lags[i]
            break
    print("VACF at lag 1: " + str(vacf[0]))
    print("Decorrelation time (tau_corr): " + str(tau_corr))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    indices = [int(0.1 * n_snaps), int(0.3 * n_snaps), int(0.6 * n_snaps), n_snaps - 1]
    for idx in indices:
        axes[0].loglog(spec[idx], label="t=" + str(round(times[idx], 2)))
    axes[0].set_title("Energy Spectrum Evolution")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("E(k)")
    axes[0].legend()
    axes[1].plot(np.mean(np.sqrt(np.sum(diffs**2, axis=-1)), axis=1))
    axes[1].set_title("Mean Displacement per Step")
    axes[1].set_xlabel("Snapshot Index")
    axes[2].plot(lags, vacf, 'o-')
    axes[2].set_title("VACF vs Lag")
    axes[2].set_xlabel("Lag")
    axes[2].set_ylabel("C_vv")
    plt.tight_layout()
    plot_path = os.path.join(data_dir, "verification_plot_2_1.png")
    plt.savefig(plot_path, dpi=300)
    print("Plot saved to " + plot_path)
    stats = {
        "mean_displacement": float(mean_disp),
        "vacf_lags": lags,
        "vacf_values": [float(v) for v in vacf],
        "tau_corr": int(tau_corr),
        "n_tracer_snaps": int(n_snaps),
        "t_prod": float(times[-1] - times[0])
    }
    with open(os.path.join(data_dir, "verification_stats.json"), "w") as f:
        json.dump(stats, f)

if __name__ == '__main__':
    verify_simulation()