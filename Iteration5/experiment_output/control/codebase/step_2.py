# filename: codebase/step_2.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import matplotlib.pyplot as plt
import os
import json

def validate_data():
    data_dir = "/home/node/work/projects/levy_flights_2dns_v2/data/"
    
    tracer_pos = np.load(os.path.join(data_dir, "tracer_positions.npy"))
    tracer_vel = np.load(os.path.join(data_dir, "tracer_velocities.npy"))
    energy_spec = np.load(os.path.join(data_dir, "energy_spectrum.npy"))
    
    with open(os.path.join(data_dir, "sim_params.json"), "r") as f:
        params = json.load(f)
    
    L = 2.0 * np.pi
    
    diffs = np.diff(tracer_pos, axis=0)
    diffs = (diffs + np.pi) % (2 * np.pi) - np.pi
    mean_disp = np.mean(np.sqrt(np.sum(diffs**2, axis=-1)))
    
    print("Mean tracer displacement per snapshot: " + str(mean_disp))
    if mean_disp >= 0.3 * L:
        print("FAILURE: Mean displacement exceeds threshold.")
    
    lags = [1, 2, 4, 8, 16]
    vacf = []
    for lag in lags:
        v_t = tracer_vel[:-lag]
        v_t_lag = tracer_vel[lag:]
        corr = np.mean(np.sum(v_t * v_t_lag, axis=-1)) / np.mean(np.sum(v_t**2, axis=-1))
        vacf.append(corr)
    
    tau_corr = 0
    for i, val in enumerate(vacf):
        if val < 1.0 / np.e:
            tau_corr = lags[i]
            break
            
    print("VACF at lags " + str(lags) + ": " + str(vacf))
    print("Decorrelation time tau_corr: " + str(tau_corr))
    
    n_snaps = energy_spec.shape[0]
    time_indices = [int(0.1 * n_snaps), int(0.3 * n_snaps), int(0.6 * n_snaps), int(0.99 * n_snaps)]
    k = np.arange(1, energy_spec.shape[1] + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for idx in time_indices:
        spec = energy_spec[idx]
        k_peak = k[np.argmax(spec)]
        axes[0].loglog(k, spec, label="t/T_prod = " + str(idx/n_snaps))
        axes[0].plot(k_peak, np.max(spec), 'x')
        print("k_peak at t/T_prod=" + str(idx/n_snaps) + ": " + str(k_peak))
        
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("E(k)")
    axes[0].set_title("Energy Spectrum Evolution")
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(lags, vacf, 'o-')
    axes[1].axhline(1.0/np.e, color='r', linestyle='--')
    axes[1].set_xlabel("Lag (steps)")
    axes[1].set_ylabel("C_vv(tau)")
    axes[1].set_title("VACF")
    axes[1].grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(data_dir, "cascade_vacf_1_20231027.png")
    plt.savefig(plot_path, dpi=300)
    print("Plot saved to " + plot_path)

if __name__ == '__main__':
    validate_data()