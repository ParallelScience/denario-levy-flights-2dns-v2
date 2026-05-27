# filename: codebase/step_2.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
sys.path.insert(0, "/home/node/data/compsep_data/")
sys.path.insert(0, "/home/node/data/compsep_data/")
sys.path.insert(0, "/home/node/data/compsep_data/")
import json
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


mpl.use('Agg')
mpl.rcParams['text.usetex'] = False

def load_simulation_data(base_path):
    """
    Loads all simulation data files from the specified base path.

    Args:
        base_path (str): The absolute path to the directory containing the data files.

    Returns:
        tuple: A tuple containing a dictionary of loaded data arrays and a dictionary
               of simulation parameters. Returns (None, None) if any file is not found.
    """
    files_to_load = {
        "tracer_positions": "tracer_positions.npy",
        "tracer_velocities": "tracer_velocities.npy",
        "velocity_snapshots": "velocity_snapshots.npy",
        "energy_spectrum": "energy_spectrum.npy",
        "diagnostics": "diagnostics.npy",
        "tracer_times": "tracer_times.npy",
    }
    sim_params_file = "sim_params.json"

    data = {}
    try:
        for name, filename in files_to_load.items():
            filepath = os.path.join(base_path, filename)
            data[name] = np.load(filepath)
            print("Loaded " + filename + " with shape: " + str(data[name].shape))

        with open(os.path.join(base_path, sim_params_file), 'r') as f:
            params = json.load(f)
        print("Loaded " + sim_params_file)

    except FileNotFoundError as e:
        print("Error: Could not find a required data file.")
        print(e)
        return None, None

    return data, params

def verify_data_shapes(data, params):
    """
    Verifies that the shapes of loaded arrays match expectations from sim_params.json.

    Args:
        data (dict): Dictionary of loaded numpy arrays.
        params (dict): Dictionary of simulation parameters.
    """
    print("\n--- Verifying Data Shapes ---")
    n_tracer_snaps = params['N_tracer_snaps']
    n_tracers = params['N_tracers']
    n_vel_snaps = params['N_vel_snaps']
    n_coarse = data['velocity_snapshots'].shape[2]
    N = params['N']

    expected_shapes = {
        "tracer_positions": (n_tracer_snaps, n_tracers, 2),
        "velocity_snapshots": (n_vel_snaps, 2, n_coarse, n_coarse),
        "energy_spectrum": (n_tracer_snaps, N // 2),
        "diagnostics": (n_tracer_snaps,),
    }

    all_ok = True
    for name, expected_shape in expected_shapes.items():
        actual_shape = data[name].shape
        if actual_shape != expected_shape:
            print("Shape mismatch for " + name + ": Expected " + str(expected_shape) + ", Got " + str(actual_shape))
            all_ok = False
        else:
            print(name + " shape OK: " + str(actual_shape))

    if not all_ok:
        raise ValueError("One or more data arrays have incorrect shapes.")
    print("All data shapes verified successfully.")


def calculate_mean_displacement(positions, domain_size=2 * np.pi):
    """
    Calculates the mean tracer displacement over one time step with periodic boundaries.

    Args:
        positions (np.ndarray): Tracer positions array of shape (n_snaps, n_tracers, 2).
        domain_size (float): The size of the periodic domain.

    Returns:
        float: The mean displacement magnitude.
    """
    delta_r = positions[1:] - positions[:-1]
    delta_r -= domain_size * np.round(delta_r / domain_size)
    displacements = np.linalg.norm(delta_r, axis=2)
    return np.mean(displacements)

def calculate_vacf(velocities, max_lag=50):
    """
    Calculates the Velocity Autocorrelation Function (VACF).

    Args:
        velocities (np.ndarray): Tracer velocities array of shape (n_snaps, n_tracers, 2).
        max_lag (int): The maximum lag (in time steps) to compute.

    Returns:
        tuple: A tuple containing an array of lags and the corresponding VACF values.
    """
    n_snaps = velocities.shape[0]
    lags = np.arange(1, min(max_lag + 1, n_snaps))
    vacf = np.zeros_like(lags, dtype=float)
    
    v_sq_mean = np.mean(np.sum(velocities**2, axis=2))
    if v_sq_mean < 1e-12:
        return lags, vacf

    for i, lag in enumerate(lags):
        dot_product = np.sum(velocities[:-lag] * velocities[lag:], axis=2)
        numerator = np.mean(dot_product)
        vacf[i] = numerator / v_sq_mean
        
    return lags, vacf

def plot_energy_spectra(spectra, times, t_prod, N, output_path):
    """
    Plots the energy spectrum at different fractions of the production run time.

    Args:
        spectra (np.ndarray): Energy spectrum array.
        times (np.ndarray): Array of times for each spectrum snapshot.
        t_prod (float): Total production run time.
        N (int): Grid resolution.
        output_path (str): Path to save the plot.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    k_values = np.arange(1, N // 2)
    
    time_fractions = [0.1, 0.3, 0.6, 1.0]
    
    if t_prod == 0:
        print("Warning: T_prod is zero, cannot plot spectrum evolution.")
        plt.close(fig)
        return
        
    target_times = [t_prod * f for f in time_fractions]
    
    for t in target_times:
        idx = np.argmin(np.abs((times - times[0]) - t))
        label_str = "t/T_prod = {:.1f}".format((times[idx] - times[0]) / t_prod)
        ax.loglog(k_values, spectra[idx, 1:], label=label_str)

    ax.set_xlabel("Wavenumber, k")
    ax.set_ylabel("Energy Spectrum, E(k)")
    ax.set_title("Evolution of Energy Spectrum")
    ax.grid(True, which="both", ls="--")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print("Plot saved to " + output_path)
    plt.close(fig)

def plot_k_peak_evolution(diagnostics, output_path):
    """
    Plots the evolution of the peak wavenumber over time.

    Args:
        diagnostics (np.ndarray): Structured array of diagnostic data.
        output_path (str): Path to save the plot.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    ax.plot(diagnostics['time'], diagnostics['k_peak'])
    
    ax.set_xlabel("Time")
    ax.set_ylabel("Peak Wavenumber, k_peak")
    ax.set_title("Evolution of Peak Wavenumber")
    ax.grid(True, ls="--")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print("Plot saved to " + output_path)
    plt.close(fig)

if __name__ == '__main__':
    BASE_PATH = "/home/node/work/projects/levy_flights_2dns_v2/data/"
    OUTPUT_DIR = "data/"
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    data, params = load_simulation_data(BASE_PATH)

    if data is not None and params is not None:
        verify_data_shapes(data, params)

        print("\n--- Verifying Snapshot Quality ---")
        mean_disp = calculate_mean_displacement(data['tracer_positions'])
        print("Mean tracer displacement per snapshot: {:.4f}".format(mean_disp))
        
        DISPLACEMENT_THRESHOLD = 1.88
        if mean_disp >= DISPLACEMENT_THRESHOLD:
            avg_u_rms = np.mean(data['diagnostics']['E_rms'])
            dt_snap = params['dt_snap']
            error_msg = "Displacement check FAILED. "
            error_msg += "Mean displacement ({:.4f}) >= threshold ({:.2f}). ".format(mean_disp, DISPLACEMENT_THRESHOLD)
            error_msg += "Snapshot interval dt_snap={:.4f} is too coarse for U_rms={:.4f}.".format(dt_snap, avg_u_rms)
            raise RuntimeError(error_msg)
        else:
            print("Displacement check PASSED.")

        print("\n--- Calculating Velocity Autocorrelation ---")
        lags_to_report = [1, 2, 4, 8, 16]
        lags, vacf = calculate_vacf(data['tracer_velocities'], max_lag=max(lags_to_report) + 40)
        
        print("VACF at short lags:")
        for lag in lags_to_report:
            idx = np.where(lags == lag)[0]
            if len(idx) > 0:
                print("  C_vv(lag=" + str(lag) + "): {:.4f}".format(vacf[idx[0]]))

        decorr_time_idx = np.where(vacf < 1/np.e)[0]
        if len(decorr_time_idx) > 0:
            decorr_lag = lags[decorr_time_idx[0]]
            dt_snap = params['dt_snap']
            decorr_time = decorr_lag * dt_snap
            print("Decorrelation time (tau_corr where C_vv < 1/e): {:.4f} (lag = {})".format(decorr_time, decorr_lag))
        else:
            print("VACF did not drop below 1/e within the computed lags.")

        print("\n--- Verifying Inverse Cascade ---")
        timestamp = int(time.time())
        
        t_prod = data['tracer_times'][-1] - data['tracer_times'][0]
        print("Inferred T_prod from tracer_times: {:.2f}".format(t_prod))
        
        plot_energy_spectra(
            data['energy_spectrum'],
            data['tracer_times'],
            t_prod,
            params['N'],
            os.path.join(OUTPUT_DIR, "energy_spectrum_evolution_1_" + str(timestamp) + ".png")
        )
        
        plot_k_peak_evolution(
            data['diagnostics'],
            os.path.join(OUTPUT_DIR, "k_peak_evolution_1_" + str(timestamp) + ".png")
        )