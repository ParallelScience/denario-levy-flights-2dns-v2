# filename: codebase/step_5.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import pywt
import os
import sys

def perform_wavelet_decomposition(velocity_data_path, output_dir):
    """
    Performs 2D wavelet decomposition on velocity field snapshots.

    This function loads velocity snapshots, applies a multi-level 2D discrete
    wavelet transform (DWT) to each velocity component of each snapshot,
    and saves the resulting coefficients in a compressed format for later analysis.

    Args:
        velocity_data_path (str): The path to the .npy file containing the
                                  velocity snapshots. The expected shape is
                                  (N_snapshots, 2, Nx, Ny).
        output_dir (str): The directory where the output .npz file will be saved.
    """
    print("Loading velocity snapshots from " + velocity_data_path)
    try:
        velocity_snapshots = np.load(velocity_data_path)
    except FileNotFoundError:
        print("Error: Velocity data file not found at " + velocity_data_path)
        sys.exit(1)

    n_snapshots, n_components, nx, ny = velocity_snapshots.shape
    print("Data loaded. Shape: " + str(velocity_snapshots.shape))

    wavelet = 'db4'
    print("Chosen wavelet basis: " + wavelet)
    justification = "Daubechies 4 ('db4') was chosen for its good balance of compact support and smoothness, making it efficient and effective for analyzing turbulent flow structures. It is orthogonal, which helps in decorrelating the signal across scales. The default 'symmetric' padding in PyWavelets is a reasonable choice for handling the boundaries of the periodic domain."
    print("Justification: " + justification)

    J = pywt.dwt_max_level(min(nx, ny), wavelet)
    print("Number of decomposition levels (J): " + str(J))

    print("Performing wavelet decomposition...")
    
    sample_coeffs = pywt.wavedec2(velocity_snapshots[0, 0, :, :], wavelet, level=J)
    coeffs_arr_sample, coeffs_slices = pywt.coeffs_to_array(sample_coeffs)
    
    n_coeffs_flat = coeffs_arr_sample.size
    original_coeffs_shape = coeffs_arr_sample.shape
    
    all_coeffs_arr = np.zeros((n_snapshots, n_components, n_coeffs_flat), dtype=np.float32)

    for t in range(n_snapshots):
        for c in range(n_components):
            coeffs = pywt.wavedec2(velocity_snapshots[t, c, :, :], wavelet, level=J)
            coeffs_arr, _ = pywt.coeffs_to_array(coeffs)
            all_coeffs_arr[t, c, :] = coeffs_arr.flatten().astype(np.float32)
        
        if (t + 1) % 100 == 0 or t == n_snapshots - 1:
            print("Processed snapshot " + str(t + 1) + "/" + str(n_snapshots))

    output_filename = "wavelet_coefficients.npz"
    output_path = os.path.join(output_dir, output_filename)
    
    np.savez_compressed(
        output_path,
        coeffs_array=all_coeffs_arr,
        coeffs_slices=np.array(coeffs_slices, dtype=object),
        original_coeffs_shape=original_coeffs_shape,
        wavelet=wavelet,
        J=J
    )
    print("Wavelet decomposition complete.")
    print("Wavelet coefficients saved to " + output_path)

if __name__ == '__main__':
    DATA_DIR = "data/"
    VELOCITY_PATH = os.path.join(DATA_DIR, "velocity_snapshots.npy")
    
    if not os.path.exists(VELOCITY_PATH):
        print("Error: Required input file not found: " + VELOCITY_PATH)
        sys.exit(1)
        
    perform_wavelet_decomposition(VELOCITY_PATH, DATA_DIR)