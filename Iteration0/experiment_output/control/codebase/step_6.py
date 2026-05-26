# filename: codebase/step_6.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import pywt
from scipy.stats import levy_stable
from scipy.linalg import LinAlgError
from scipy.integrate import trapezoid
import time

def calculate_eddy_lifetimes(coeffs_array, J, original_shape, coeffs_slices, times):
    """
    Calculates the eddy lifetime for each wavelet scale.

    The eddy lifetime is computed as the integral of the temporal
    autocorrelation of the wavelet coefficients, up to the first zero-crossing.

    Args:
        coeffs_array (np.ndarray): The flattened wavelet coefficients.
        J (int): The number of wavelet decomposition levels.
        original_shape (tuple): The original shape of the un-flattened coefficients.
        coeffs_slices (list): The list of slice dictionaries from PyWavelets.
        times (np.ndarray): Array of snapshot times.

    Returns:
        np.ndarray: An array of eddy lifetimes, one for each wavelet scale.
    """
    print("Calculating eddy lifetimes for each wavelet scale...")
    n_snapshots = coeffs_array.shape[0]
    dt = np.mean(np.diff(times))
    
    T_eddy = np.zeros(J)

    coeffs_array_reshaped = coeffs_array.reshape(
        n_snapshots, 2, original_shape[0], original_shape[1]
    )

    for slice_idx in range(1, J + 1):
        pywt_level = J - slice_idx + 1
        
        slice_dict = coeffs_slices[slice_idx]
        
        h_slices = slice_dict['da']
        v_slices = slice_dict['ad']
        d_slices = slice_dict['dd']

        h_coeffs = coeffs_array_reshaped[:, :, h_slices[0], h_slices[1]]
        v_coeffs = coeffs_array_reshaped[:, :, v_slices[0], v_slices[1]]
        d_coeffs = coeffs_array_reshaped[:, :, d_slices[0], d_slices[1]]

        level_coeffs_t = np.concatenate(
            [h_coeffs.reshape(n_snapshots, -1),
             v_coeffs.reshape(n_snapshots, -1),
             d_coeffs.reshape(n_snapshots, -1)],
            axis=1
        )
        
        max_lag = n_snapshots // 2
        autocorr = np.zeros(max_lag)
        
        level_coeffs_t_mean = np.mean(level_coeffs_t, axis=0)
        level_coeffs_t_centered = level_coeffs_t - level_coeffs_t_mean

        for lag in range(max_lag):
            if lag == 0:
                prod = level_coeffs_t_centered * level_coeffs_t_centered
            else:
                prod = level_coeffs_t_centered[:-lag] * level_coeffs_t_centered[lag:]
            autocorr[lag] = np.mean(prod)
        
        if autocorr[0] > 1e-10:
            autocorr /= autocorr[0]
        
        zero_cross_idx = np.where(autocorr <= 0)[0]
        if len(zero_cross_idx) > 0:
            integration_limit = zero_cross_idx[0]
        else:
            integration_limit = len(autocorr)
        
        t_eddy_idx = pywt_level - 1
        
        if integration_limit > 1:
            T_eddy[t_eddy_idx] = trapezoid(autocorr[:integration_limit], dx=dt)
        else:
            T_eddy[t_eddy_idx] = 0

        print("  Scale j=" + str(pywt_level) + ", T_eddy = " + str(T_eddy[t_eddy_idx]))
        
    return T_eddy

def bilinear_interpolate_periodic(field, positions):
    """
    Performs vectorized 2D bilinear interpolation on a periodic domain.

    Args:
        field (np.ndarray): The 2D data grid (Nx, Ny).
        positions (np.ndarray): Tracer positions (N_tracers, 2) in [0, 2*pi].

    Returns:
        np.ndarray: Interpolated values at tracer positions (N_tracers,).
    """
    nx, ny = field.shape
    
    pos_norm = positions / (2.0 * np.pi)
    x_idx = pos_norm[:, 0] * nx
    y_idx = pos_norm[:, 1] * ny

    ix = np.floor(x_idx).astype(int)
    iy = np.floor(y_idx).astype(int)
    fx = x_idx - ix
    fy = y_idx - iy

    ix1 = (ix + 1) % nx
    iy1 = (iy + 1) % ny
    ix = ix % nx
    iy = iy % ny

    v00 = field[ix, iy]
    v10 = field[ix1, iy]
    v01 = field[ix, iy1]
    v11 = field[ix1, iy1]

    interp_vals = (v00 * (1 - fx) * (1 - fy) +
                   v10 * fx * (1 - fy) +
                   v01 * (1 - fx) * fy +
                   v11 * fx * fy)
    
    return interp_vals

def fit_levy_stable_robust(data, **kwargs):
    """
    Robust fitting of a levy_stable distribution.
    """
    try:
        if len(data) > 100000:
            data = np.random.choice(data, 100000, replace=False)
        params = levy_stable.fit(data, **kwargs)
        return params[0]
    except (LinAlgError, ValueError):
        return np.nan

def calculate_dispersive_stats(positions, velocities, times, T_L_estimate):
    """
    Computes alpha, gamma, and nu for a set of trajectories.
    """
    n_snapshots = len(times)
    dt = np.mean(np.diff(times))
    
    lags_T = np.array([0.5, 1.0, 2.0]) * T_L_estimate
    lag_indices = np.round(lags_T / dt).astype(int)
    alphas = []
    for lag in lag_indices:
        if lag >= n_snapshots or lag == 0: continue
        displacements = positions[lag:] - positions[:-lag]
        dx = displacements[:, :, 0].flatten()
        dy = displacements[:, :, 1].flatten()
        alpha_x = fit_levy_stable_robust(dx)
        alpha_y = fit_levy_stable_robust(dy)
        if not np.isnan(alpha_x) and not np.isnan(alpha_y):
            alphas.append((alpha_x + alpha_y) / 2.0)
    alpha = np.nanmean(alphas) if alphas else np.nan

    max_lag_msd = n_snapshots // 4
    lags_msd = np.arange(1, max_lag_msd)
    msd = np.zeros(len(lags_msd))
    for i, lag in enumerate(lags_msd):
        disp_sq = (positions[lag:] - positions[:-lag])**2
        msd[i] = np.mean(disp_sq)
    
    valid_msd = msd > 0
    if np.any(valid_msd):
        tau_msd = lags_msd[valid_msd] * dt
        log_tau = np.log(tau_msd)
        log_msd = np.log(msd[valid_msd])
        gamma_fit = np.polyfit(log_tau, log_msd, 1)
        gamma = gamma_fit[0]
    else:
        gamma = np.nan

    max_lag_vacf = n_snapshots // 4
    lags_vacf = np.arange(1, max_lag_vacf)
    vacf = np.zeros(len(lags_vacf))
    for i, lag in enumerate(lags_vacf):
        vel_prod = velocities[:-lag] * velocities[lag:]
        vacf[i] = np.mean(vel_prod)
    
    norm = np.mean(velocities**2)
    if norm > 1e-10:
        vacf /= norm
        
    positive_vacf = vacf > 0
    if np.any(positive_vacf):
        tau_vacf = lags_vacf[positive_vacf] * dt
        log_tau_vacf = np.log(tau_vacf)
        log_vacf = np.log(vacf[positive_vacf])
        nu_fit = np.polyfit(log_tau_vacf, log_vacf, 1)
        nu = -nu_fit[0]
    else:
        nu = np.nan
        
    return alpha, gamma, nu

def main():
    """
    Main function to run the eddy lifetime and filtered statistics analysis.
    """
    DATA_DIR = "data/"
    COEFFS_PATH = os.path.join(DATA_DIR, "wavelet_coefficients.npz")
    PREPARED_DATA_PATH = os.path.join(DATA_DIR, "prepared_data.npz")
    OUTPUT_PATH = os.path.join(DATA_DIR, "wavelet_analysis_results.npz")

    print("Loading data...")
    try:
        coeffs_data = np.load(COEFFS_PATH, allow_pickle=True)
        prep_data = np.load(PREPARED_DATA_PATH)
    except FileNotFoundError as e:
        print("Error: Required data file not found. " + str(e))
        sys.exit(1)

    times = prep_data['times']
    tracer_positions = prep_data['tracer_positions']
    T_L_estimate = prep_data['T_L_estimate'].item()
    
    wavelet = str(coeffs_data['wavelet'])
    J = int(coeffs_data['J'])
    coeffs_array = coeffs_data['coeffs_array']
    original_coeffs_shape = coeffs_data['original_coeffs_shape']
    
    coeffs_slices_obj = coeffs_data['coeffs_slices']
    if isinstance(coeffs_slices_obj, np.ndarray):
        if coeffs_slices_obj.shape == ():
            coeffs_slices = coeffs_slices_obj.item()
        elif coeffs_slices_obj.shape == (1,) and isinstance(coeffs_slices_obj[0], list):
            coeffs_slices = coeffs_slices_obj[0]
        else:
            coeffs_slices = list(coeffs_slices_obj)
    else:
        coeffs_slices = coeffs_slices_obj

    T_eddy = calculate_eddy_lifetimes(coeffs_array, J, original_coeffs_shape, coeffs_slices, times)
    
    n_snapshots, n_tracers, _ = tracer_positions.shape

    j_min_levels = np.array([1, 2, 3, 4])
    alphas_j = np.zeros(len(j_min_levels))
    gammas_j = np.zeros(len(j_min_levels))
    nus_j = np.zeros(len(j_min_levels))

    for i, j_min in enumerate(j_min_levels):
        print("\nProcessing filtered field for j_min = " + str(j_min))
        filtered_velocities = np.zeros((n_snapshots, n_tracers, 2), dtype=np.float32)
        
        for t in range(n_snapshots):
            u_recon, v_recon = None, None
            for c in range(2):
                coeffs_flat = coeffs_array[t, c, :]
                coeffs_arr = coeffs_flat.reshape(original_coeffs_shape)
                coeffs = pywt.array_to_coeffs(coeffs_arr, coeffs_slices, output_format='wavedec2')
                
                for j_level in range(1, j_min):
                    if j_level > J: continue
                    level_idx = J - j_level + 1
                    coeffs[level_idx] = tuple([np.zeros_like(d) for d in coeffs[level_idx]])
                
                if c == 0:
                    u_recon = pywt.waverec2(coeffs, wavelet)
                else:
                    v_recon = pywt.waverec2(coeffs, wavelet)

            filtered_velocities[t, :, 0] = bilinear_interpolate_periodic(u_recon, tracer_positions[t])
            filtered_velocities[t, :, 1] = bilinear_interpolate_periodic(v_recon, tracer_positions[t])
            
            if (t + 1) % 200 == 0:
                print("  Reconstructed and interpolated snapshot " + str(t + 1) + "/" + str(n_snapshots))

        print("  Calculating statistics for filtered field...")
        alpha, gamma, nu = calculate_dispersive_stats(
            tracer_positions, filtered_velocities, times, T_L_estimate
        )
        alphas_j[i] = alpha
        gammas_j[i] = gamma
        nus_j[i] = nu
        print("  Results: alpha=" + str(alpha) + ", gamma=" + str(gamma) + ", nu=" + str(nu))

    print("\nSaving results to " + OUTPUT_PATH)
    np.savez_compressed(
        OUTPUT_PATH,
        T_eddy=T_eddy,
        j_levels=np.arange(1, J + 1),
        alpha_j=alphas_j,
        gamma_j=gammas_j,
        nu_j=nus_j,
        j_min_levels=j_min_levels
    )
    print("Analysis complete.")

if __name__ == '__main__':
    main()