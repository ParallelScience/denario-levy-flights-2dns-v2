# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import json
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

def load_data(base_path):
    """
    Loads all simulation data files from the specified base path.

    Args:
        base_path (str): The absolute path to the data directory.

    Returns:
        tuple: A tuple containing a dictionary of loaded numpy arrays and a
               dictionary of simulation parameters. Returns (None, None)
               if any file is not found.
    """
    files_to_load = {
        "tracer_positions": "tracer_positions.npy",
        "tracer_velocities": "tracer_velocities.npy",
        "vorticity_snapshots": "vorticity_snapshots.npy",
        "velocity_snapshots": "velocity_snapshots.npy",
        "tracer_times": "tracer_times.npy",
        "vel_times": "vel_times.npy",
        "energy_spectrum": "energy_spectrum.npy",
        "diagnostics": "diagnostics.npy",
    }
    data = {}
    try:
        for name, filename in files_to_load.items():
            path = os.path.join(base_path, filename)
            data[name] = np.load(path)

        params_path = os.path.join(base_path, "sim_params.json")
        with open(params_path, 'r') as f:
            params = json.load(f)

    except FileNotFoundError as e:
        print("Error: Could not find a required data file.")
        print(e)
        return None, None

    return data, params

def verify_data(data, params):
    """
    Verifies and prints the shapes, data types, and parameters of the loaded data.

    Args:
        data (dict): Dictionary of loaded numpy arrays.
        params (dict): Dictionary of simulation parameters.
    """
    print("--- Data Verification ---")
    for name, arr in data.items():
        print("Array: " + name + ", Shape: " + str(arr.shape) + ", Dtype: " + str(arr.dtype))
    
    print("\n--- Simulation Parameters ---")
    for key, value in params.items():
        print(key + ": " + str(value))
    print("-" * 27 + "\n")


def check_snapshot_quality(tracer_positions, domain_size=2 * np.pi):
    """
    Computes and checks the mean tracer displacement per snapshot.

    Args:
        tracer_positions (np.ndarray): Array of tracer positions with shape
                                       (time, tracers, 2).
        domain_size (float): The size of the periodic domain.

    Returns:
        bool: True if the check passes, False otherwise.
    """
    print("--- Snapshot Quality Check ---")
    delta_r = tracer_positions[1:, :, :] - tracer_positions[:-1, :, :]
    delta_r -= np.round(delta_r / domain_size) * domain_size
    magnitudes = np.linalg.norm(delta_r, axis=2)
    mean_displacement = np.mean(magnitudes)
    
    threshold = 0.3 * domain_size
    print("Mean tracer displacement per snapshot: " + str(mean_displacement))
    print("Threshold (0.3 * 2 * pi): " + str(threshold))

    if mean_displacement >= threshold:
        print("Error: Mean displacement exceeds threshold. Snapshot interval is too coarse.")
        return False
    else:
        print("Snapshot quality check passed.")
        print("-" * 28 + "\n")
        return True

def calculate_vacf(tracer_velocities, dt_snap):
    """
    Calculates and prints the Velocity Autocorrelation Function (VACF) and
    decorrelation time.

    Args:
        tracer_velocities (np.ndarray): Array of tracer velocities.
        dt_snap (float): Time step between tracer snapshots.
    """
    print("--- Velocity Autocorrelation Function (VACF) ---")
    lags_to_check = [1, 2, 4, 8, 16]
    v = tracer_velocities
    denominator = np.mean(np.sum(v**2, axis=2))

    if denominator == 0:
        print("Error: Velocity variance is zero. Cannot compute VACF.")
        return

    print("VACF at short lags:")
    for lag in lags_to_check:
        if lag >= v.shape[0]:
            continue
        numerator = np.mean(np.sum(v[:-lag] * v[lag:], axis=2))
        vacf = numerator / denominator
        print("  Lag " + str(lag) + " steps: " + str(vacf))

    max_lag_corr = min(200, v.shape[0] - 1)
    vacf_corr = np.zeros(max_lag_corr)
    for tau in range(1, max_lag_corr + 1):
        numerator = np.mean(np.sum(v[:-tau] * v[tau:], axis=2))
        vacf_corr[tau - 1] = numerator / denominator

    corr_lags = np.where(vacf_corr < 1 / np.e)[0]
    if len(corr_lags) > 0:
        tau_corr_steps = corr_lags[0] + 1
        tau_corr_time = tau_corr_steps * dt_snap
        print("\nDecorrelation time (tau_corr): " + str(tau_corr_time) + " (" + str(tau_corr_steps) + " steps)")
    else:
        print("\nVACF did not drop below 1/e within " + str(max_lag_corr) + " steps.")
    print("-" * 44 + "\n")


def plot_inverse_cascade(energy_spectrum, tracer_times, diagnostics, T_prod, output_dir):
    """
    Generates and saves plots confirming the inverse cascade.

    Args:
        energy_spectrum (np.ndarray): Array of energy spectra.
        tracer_times (np.ndarray): Array of times for tracer snapshots.
        diagnostics (np.ndarray): Structured array of diagnostic data.
        T_prod (float): Total production run time.
        output_dir (str): Directory to save plots.
    """
    print("--- Inverse Cascade Verification ---")
    rcParams['text.usetex'] = False
    timestamp = int(time.time())

    time_fractions = [0.1, 0.3, 0.6, 1.0]
    target_times = [f * T_prod for f in time_fractions]
    time_indices = [np.argmin(np.abs(tracer_times - t)) for t in target_times]

    fig1, ax1 = plt.subplots(figsize=(10, 7))
    wavenumbers = np.arange(energy_spectrum.shape[1])
    for i, idx in enumerate(time_indices):
        label_text = 't = ' + "{:.2f}".format(tracer_times[idx])
        ax1.loglog(wavenumbers[1:], energy_spectrum[idx, 1:], label=label_text)
    
    ax1.set_title('Energy Spectrum Evolution')
    ax1.set_xlabel('Wavenumber, k')
    ax1.set_ylabel('Energy, E(k)')
    ax1.grid(True, which="both", ls="--")
    ax1.legend()
    plt.tight_layout()
    
    plot1_filename = "energy_spectrum_evolution_1_" + str(timestamp) + ".png"
    plot1_path = os.path.join(output_dir, plot1_filename)
    plt.savefig(plot1_path, dpi=300)
    plt.close(fig1)
    print("Plot saved to " + plot1_path)

    fig2, ax2 = plt.subplots(figsize=(10, 7))
    diag_time = diagnostics['time']
    diag_k_peak = diagnostics['k_peak']
    ax2.plot(diag_time, diag_k_peak, label='k_peak(t)')
    
    k_peak_at_times = diag_k_peak[time_indices]
    ax2.plot(tracer_times[time_indices], k_peak_at_times, 'ro', label='Spectrum plot times')
    
    print("\nk_peak at selected times:")
    for i, idx in enumerate(time_indices):
        print("  t = " + "{:.2f}".format(tracer_times[idx]) + ": k_peak = " + "{:.4f}".format(diag_k_peak[idx]))

    ax2.set_title('Peak Wavenumber Evolution')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Peak Wavenumber, k_peak')
    ax2.grid(True)
    ax2.legend()
    plt.tight_layout()

    plot2_filename = "k_peak_evolution_1_" + str(timestamp) + ".png"
    plot2_path = os.path.join(output_dir, plot2_filename)
    plt.savefig(plot2_path, dpi=300)
    plt.close(fig2)
    print("Plot saved to " + plot2_path)
    print("-" * 34 + "\n")


if __name__ == '__main__':
    BASE_PATH = "/home/node/work/projects/levy_flights_2dns_v2/data/"
    OUTPUT_DIR = "data/"

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    data, params = load_data(BASE_PATH)

    if data is not None and params is not None:
        verify_data(data, params)

        if not check_snapshot_quality(data["tracer_positions"]):
            sys.exit(1)

        calculate_vacf(data["tracer_velocities"], params.get(