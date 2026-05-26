# filename: codebase/step_2.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import matplotlib.pyplot as plt
import os
import time

def plot_inverse_cascade(data_path, output_dir):
    """
    Loads simulation data and generates plots to verify the inverse cascade.

    This function creates a two-panel figure. The first panel shows the energy
    spectrum at different times, compared against the theoretical k^(-5/3) slope.
    The second panel shows the evolution of the peak energy wavenumber over time.

    Args:
        data_path (str): The path to the .npz file containing prepared data.
        output_dir (str): The directory where the output plot will be saved.
    """
    try:
        data = np.load(data_path)
        energy_spectrum = data['energy_spectrum']
        times = data['times']
        diagnostics = data['diagnostics']
        T_prod = data['T_prod']
    except FileNotFoundError:
        print("Error: Data file not found at " + data_path)
        return

    fig, axes = plt.subplots(2, 1, figsize=(10, 12))
    plt.rcParams['text.usetex'] = False

    # Panel 1: Energy Spectra
    ax1 = axes[0]
    time_fractions = [0.1, 0.3, 0.6, 1.0]
    target_times = [frac * T_prod for frac in time_fractions]
    
    num_k = energy_spectrum.shape[1]
    k = np.arange(1, num_k + 1)

    for i, t in enumerate(target_times):
        idx = np.argmin(np.abs(times - t))
        label_str = 't/T_prod = ' + str(time_fractions[i])
        ax1.loglog(k, energy_spectrum[idx, :], label=label_str)

    # Add k^(-5/3) reference line
    k_ref_range = k[k > 3]
    if len(k_ref_range) > 0:
        # Normalize the reference line to the final spectrum in the inertial range
        k_ref_point = k_ref_range[5]
        idx_ref_point = np.where(k == k_ref_point)[0][0]
        E_ref_point = energy_spectrum[-1, idx_ref_point]
        C = E_ref_point / (k_ref_point ** (-5.0/3.0))
        E_theory = C * (k_ref_range ** (-5.0/3.0))
        ax1.loglog(k_ref_range, E_theory, 'k--', label='k^(-5/3)')

    ax1.axvspan(1, 3, color='gray', alpha=0.3, label='Forcing range')
    ax1.set_xlabel('Wavenumber, k')
    ax1.set_ylabel('Energy Spectrum, E(k)')
    ax1.set_title('Energy Spectra at Different Times')
    ax1.legend()
    ax1.grid(True, which="both", ls="--")
    ax1.set_xlim([1, num_k])
    min_energy = np.min(energy_spectrum[energy_spectrum > 0])
    if min_energy > 0:
        ax1.set_ylim(bottom=min_energy)

    # Panel 2: Peak Wavenumber Evolution
    ax2 = axes[1]
    sim_times = diagnostics['time']
    k_peak = diagnostics['k_peak']
    
    ax2.plot(sim_times, k_peak, label='k_peak(t)')
    ax2.axvline(0, color='r', linestyle='--', label='Start of Production Run')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Peak Wavenumber, k_peak')
    ax2.set_title('Evolution of Peak Energy Wavenumber')
    ax2.grid(True)
    ax2.set_xlim([sim_times.min(), sim_times.max()])
    ax2.legend()

    plt.tight_layout()

    timestamp = int(time.time())
    filename = 'inverse_cascade_verification_1_' + str(timestamp) + '.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300)
    print("Plot saved to " + filepath)
    plt.close(fig)

if __name__ == '__main__':
    DATA_DIR = "data/"
    PREPARED_DATA_PATH = os.path.join(DATA_DIR, "prepared_data.npz")
    
    if not os.path.exists(DATA_DIR):
        # This case should not happen if step 1 ran correctly
        os.makedirs(DATA_DIR)
        print("Created directory: " + DATA_DIR)

    plot_inverse_cascade(PREPARED_DATA_PATH, DATA_DIR)