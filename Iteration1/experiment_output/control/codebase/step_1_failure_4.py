# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import json
import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def load_data(base_path):
    """
    Loads all simulation data files from the specified base path.

    Args:
        base_path (str): The absolute path to the directory containing the data files.

    Returns:
        tuple: A tuple containing the simulation parameters and loaded numpy arrays.
    """
    print("--- Loading Data ---")
    paths = {
        "params": os.path.join(base_path, 'sim_params.json'),
        "positions": os.path.join(base_path, 'tracer_positions.npy'),
        "velocities": os.path.join(base_path, 'tracer_velocities.npy'),
        "tracer_times": os.path.join(base_path, 'tracer_times.npy'),
        "energy_spectrum": os.path.join(base_path, 'energy_spectrum.npy'),
        "diagnostics": os.path.join(base_path, 'diagnostics.npy'),
    }

    try:
        with open(paths["params"], 'r') as f:
            sim_params = json.load(f)

        tracer_positions = np.load(paths["positions"])
        tracer_velocities = np.load(paths["velocities"])
        tracer_times = np.load(paths["tracer_times"])
        energy_spectrum = np.load(paths["energy_spectrum"])
        diagnostics = np.load(paths["diagnostics"])
    except FileNotFoundError as e:
        print("Error: A data file was not found.")
        print(e)
        sys.exit(1)

    print("Data loading complete.")
    return sim_params, tracer_positions, tracer_velocities, tracer_times, energy_spectrum, diagnostics

def verify_data(sim_params, tracer_positions, tracer_velocities, tracer_times, energy_spectrum, diagnostics):
    """
    Verifies and prints the shapes and data types of the loaded arrays.

    Args:
        sim_params (dict): Simulation parameters.
        tracer_positions (np.ndarray): Tracer positions.
        tracer_velocities (np.ndarray): Tracer velocities.
        tracer_times (np.ndarray): Tracer snapshot times.
        energy_spectrum (np.ndarray): Energy spectrum data.
        diagnostics (np.ndarray): Diagnostic data.
    """
    print("\n--- Verifying Data Shapes and Types ---")
    print("Simulation Parameters: " + str(sim_params.keys()))
    print("Tracer Positions: shape=" + str(tracer_positions.shape) + ", dtype=" + str(tracer_positions.dtype))
    print("Tracer Velocities: shape=" + str(tracer_velocities.shape) + ", dtype=" + str(tracer_velocities.dtype))
    print("Tracer Times: shape=" + str(tracer_times.shape) + ", dtype=" + str(tracer_times.dtype))
    print("Energy Spectrum: shape=" + str(energy_spectrum.shape) + ", dtype=" + str(energy_spectrum.dtype))
    print("Diagnostics: shape=" + str(diagnostics.shape) + ", dtype=" + str(diagnostics.dtype))

def check_snapshot_quality(tracer_positions):
    """
    Performs a snapshot quality check by computing the mean tracer displacement.

    Args:
        tracer_positions (np.ndarray): Tracer positions array.
    """
    print("\n--- Snapshot Quality Check ---")
    domain_size = 2 * np.pi
    delta_r = tracer_positions[1] - tracer_positions[0]
    delta_r = (delta_r + 0.5 * domain_size) % domain_size - 0.5 * domain_size
    
    displacements = np.sqrt(np.sum(delta_r**2, axis=1))
    mean_displacement = np.mean(displacements)
    
    threshold = 0.3 * domain_size
    print("Mean tracer displacement per snapshot: " + str(mean_displacement))
    print("Threshold (0.3 * 2pi): " + str(threshold))

    if mean_displacement > threshold:
        print("Error: Mean displacement exceeds threshold. Snapshot interval is too coarse.")
        sys.exit(1)
    else:
        print("Snapshot quality check passed.")

def calculate_vacf(tracer_velocities, dt_snap):
    """
    Calculates the Velocity Autocorrelation Function (VACF) and decorrelation time.

    Args:
        tracer_velocities (np.ndarray): Tracer velocities array.
        dt_snap (float): Time interval between tracer snapshots.
    """
    print("\n--- Velocity Autocorrelation Analysis ---")
    lags_to_check = [1, 2, 4, 8, 16]
    max_lag_for_corr = 200
    
    v_sq_mean = np.mean(np.sum(tracer_velocities**2, axis=2))
    if v_sq_mean == 0:
        print("Warning: Mean squared velocity is zero. Cannot compute VACF.")
        return

    vacf_values = []
    for lag in range(1, max_lag_for_corr + 1):
        if lag >= len(tracer_velocities):
            break
        dot_prod = np.sum(tracer_velocities[:-lag] * tracer_velocities[lag:], axis=2)
        vacf = np.mean(dot_prod) / v_sq_mean
        vacf_values.append(vacf)
        if lag in lags_to_check:
            print("VACF at lag " + str(lag) + " steps: " + str(vacf))

    vacf_values = np.array(vacf_values)
    corr_lag_idx = np.where(vacf_values < 1/np.e)[0]
    if len(corr_lag_idx) > 0:
        decorrelation_lag = corr_lag_idx[0] + 1
        tau_corr = decorrelation_lag * dt_snap
        print("Decorrelation time (tau_corr): " + str(tau_corr) + " (at lag " + str(decorrelation_lag) + ")")
    else:
        print("VACF did not drop below 1/e within " + str(max_lag_for_corr) + " lags.")

def verify_inverse_cascade(energy_spectrum, diagnostics, tracer_times, sim_params):
    """
    Confirms the inverse cascade by plotting energy spectra and k_peak evolution.

    Args:
        energy_spectrum (np.ndarray): Energy spectrum data.
        diagnostics (np.ndarray): Diagnostic data.
        tracer_times (np.ndarray): Tracer snapshot times.
        sim_params (dict): Simulation parameters.
    """
    print("\n--- Inverse Cascade Verification ---")
    plt.rcParams['text.usetex'] = False
    
    t_prod = sim_params.get('T_prod', tracer_times[-1] - tracer_times[0])
    time_fractions = [0.1, 0.3, 0.6, 1.0]
    target_times = [tracer_times[0] + frac * t_prod for frac in time_fractions]
    time_indices = [np.argmin(np.abs(tracer_times - t)) for t in target_times]

    # Plot 1: Energy Spectrum Evolution
    fig1, ax1 = plt.subplots(figsize=(10, 7))
    k = np.arange(1, energy_spectrum.shape[1] + 1)
    
    print("k_peak values at specified times:")
    for i, index in enumerate(time_indices):
        time_val = tracer_times[index]
        label_text = 't = ' + "{:.2f}".format(time_val)
        ax1.loglog(k, energy_spectrum[index, :], label=label_text)
        k_peak_val = diagnostics['k_peak'][index]
        print("  t = " + "{:.2f}".format(time_val) + ", k_peak = " + str(k_peak_val))

    ax1.set_xlabel('Wavenumber, k')
    ax1.set_ylabel('Energy Spectrum, E(k)')
    ax1.set_title('Energy Spectrum Evolution')
    ax1.grid(True, which="both", ls="--")
    ax1.legend()
    plt.tight_layout()
    
    timestamp = int(time.time())
    output_dir = "data/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    filepath1 = os.path.join(output_dir, 'energy_spectrum_evolution_1_' + str(timestamp) + '.png')
    plt.savefig(filepath1, dpi=300)
    plt.close(fig1)
    print("Plot saved to " + filepath1)

    # Plot 2: k_peak Evolution
    fig2, ax2 = plt.subplots(figsize=(10, 7))
    ax2.plot(diagnostics['time'], diagnostics['k_peak'], label='k_peak(t)')
    ax2.scatter(tracer_times[time_indices], diagnostics['k_peak'][time_indices], color='red', zorder=5, label='Spectrum plot times')
    
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Peak Wavenumber, k_peak')
    ax2.set_title('Peak Wavenumber Evolution')
    ax2.grid(True)
    ax2.legend()
    plt.tight_layout()
    
    filepath2 = os.path.join(output_dir, 'k_peak_evolution_1_' + str(timestamp) + '.png')
    plt.savefig(filepath2, dpi=300)
    plt.close(fig2)
    print("Plot saved to " + filepath2)

if __name__ == '__main__':
    BASE_DATA_PATH = '/home/node/work/projects/levy_flights_2dns_v2/data/'
    
    params, positions, velocities, times, spectrum, diags = load_data(BASE_DATA_PATH)
    
    verify_data(params, positions, velocities, times, spectrum, diags)
    
    check_snapshot_quality(positions)
    
    calculate_vacf(velocities, params.get('dt_snap', 0.05))
    
    verify_inverse_cascade(spectrum, diags, times, params)
    
    print("\nStep 1: Data loading and initial verification complete.")