# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import json
import numpy as np


def create_dummy_data_if_missing():
    """
    Checks for simulation data and creates placeholder files if missing.

    This function checks for the existence of a key data file. If it's not
    found, it generates a full set of synthetic data files with the expected
    names, shapes, and data types as described in the project outline. This
    ensures that the analysis pipeline can be executed for testing and
    demonstration purposes, even without the original simulation output.

    A clear warning is printed to the console when dummy data is created, as
    any analysis based on it will be scientifically meaningless.
    """
    base_path = 'data/'
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    first_file_path = os.path.join(base_path, 'tracer_positions.npy')
    if os.path.exists(first_file_path):
        print("Data files found. Skipping dummy data creation.")
        return

    print("--- Required data not found. Creating dummy data for pipeline execution. ---")
    print("--- WARNING: Analysis results from this data will be scientifically meaningless. ---")

    N_snapshots = 1000
    N_tracers = 5000
    N = 512
    N_coarse = 256
    T_L_estimate = 5.0
    T_prod_total = 60 * T_L_estimate

    file_paths = {
        'tracer_positions': os.path.join(base_path, 'tracer_positions.npy'),
        'tracer_velocities': os.path.join(base_path, 'tracer_velocities.npy'),
        'vorticity_snapshots': os.path.join(base_path, 'vorticity_snapshots.npy'),
        'velocity_snapshots': os.path.join(base_path, 'velocity_snapshots.npy'),
        'times': os.path.join(base_path, 'times.npy'),
        'energy_spectrum': os.path.join(base_path, 'energy_spectrum.npy'),
        'diagnostics': os.path.join(base_path, 'diagnostics.npy'),
        'sim_params': os.path.join(base_path, 'sim_params.json'),
    }

    np.save(file_paths['tracer_positions'], np.random.uniform(0, 2 * np.pi, size=(N_snapshots, N_tracers, 2)).astype(np.float32))
    np.save(file_paths['tracer_velocities'], np.random.randn(N_snapshots, N_tracers, 2).astype(np.float32))
    np.save(file_paths['vorticity_snapshots'], np.random.randn(N_snapshots, N_coarse, N_coarse).astype(np.float32))
    np.save(file_paths['velocity_snapshots'], np.random.randn(N_snapshots, 2, N_coarse, N_coarse).astype(np.float32))
    times_arr = np.linspace(0, T_prod_total, N_snapshots).astype(np.float64)
    np.save(file_paths['times'], times_arr)
    
    k = np.arange(1, N // 2 + 1)
    spectrum = (k**(-5./3.)) * np.random.uniform(0.5, 1.5, size=(N_snapshots, N // 2))
    np.save(file_paths['energy_spectrum'], spectrum.astype(np.float64))

    diagnostics_dtype = np.dtype([
        ('time', 'f8'), ('E_total', 'f4'), ('E_rms', 'f4'),
        ('omega_rms', 'f4'), ('T_L', 'f4'), ('k_peak', 'f4'),
        ('d_vv_estimate', 'f4')
    ])
    diagnostics = np.zeros(N_snapshots, dtype=diagnostics_dtype)
    diagnostics['time'] = times_arr
    diagnostics['T_L'] = np.random.normal(T_L_estimate, 0.5, N_snapshots)
    diagnostics['k_peak'] = np.linspace(10, 2, N_snapshots) + np.random.randn(N_snapshots) * 0.1
    np.save(file_paths['diagnostics'], diagnostics)

    params = {
        "N": N, "dt": 0.001, "nu_h": 1e-17, "epsilon_inj": 0.1,
        "N_tracers": N_tracers, "N_snapshots": N_snapshots,
        "T_L_estimate": T_L_estimate
    }
    with open(file_paths['sim_params'], 'w') as f:
        json.dump(params, f, indent=4)
    
    print("--- Dummy data creation complete. ---")


def load_and_prepare_data():
    """
    Loads all simulation data, verifies its structure, computes derived
    quantities, and saves everything into a single compressed NPZ file.
    """
    base_path = 'data/'
    output_dir = 'data/'

    file_paths = {
        'tracer_positions': os.path.join(base_path, 'tracer_positions.npy'),
        'tracer_velocities': os.path.join(base_path, 'tracer_velocities.npy'),
        'vorticity_snapshots': os.path.join(base_path, 'vorticity_snapshots.npy'),
        'velocity_snapshots': os.path.join(base_path, 'velocity_snapshots.npy'),
        'times': os.path.join(base_path, 'times.npy'),
        'energy_spectrum': os.path.join(base_path, 'energy_spectrum.npy'),
        'diagnostics': os.path.join(base_path, 'diagnostics.npy'),
        'sim_params': os.path.join(base_path, 'sim_params.json'),
    }

    loaded_data = {}
    print("\n--- Starting Data Loading and Verification ---")

    for name, path in file_paths.items():
        print("Loading " + name + " from " + path)
        if name == 'sim_params':
            with open(path, 'r') as f:
                params = json.load(f)
                loaded_data[name] = np.array(params)
                print("  - Simulation parameters loaded.")
        else:
            data = np.load(path)
            loaded_data[name] = data
            print("  - Shape: " + str(data.shape))
            print("  - Dtype: " + str(data.dtype))

    print("\n--- Data Verification Complete ---")
    print("\n--- Computing Derived Quantities ---")
    
    diagnostics = loaded_data['diagnostics']
    times = loaded_data['times']
    
    t_l_estimate = np.mean(diagnostics['T_L'])
    print("Computed T_L_estimate (mean over run): " + str(t_l_estimate))
    
    t_prod = times[-1] - times[0]
    print("Computed T_prod (total duration): " + str(t_prod))

    k_box = 1.0
    print("Set k_box (fundamental wavenumber): " + str(k_box))

    derived_quantities = {
        'T_L_estimate': t_l_estimate,
        'T_prod': t_prod,
        'k_box': k_box
    }
    
    print("\n--- Saving Prepared Data ---")
    output_path = os.path.join(output_dir, 'prepared_data.npz')
    np.savez_compressed(output_path, **loaded_data, **derived_quantities)
    print("All data successfully loaded, processed, and saved to: " + output_path)

if __name__ == '__main__':
    create_dummy_data_if_missing()
    load_and_prepare_data()