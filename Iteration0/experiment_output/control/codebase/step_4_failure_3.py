# filename: codebase/step_4.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from scipy.optimize import curve_fit
from statsmodels.tsa.stattools import adfuller, kpss


def run_stationarity_tests(series, name):
    """
    Performs and prints the results of ADF and KPSS stationarity tests.

    Args:
        series (np.ndarray): The time series data to test.
        name (str): The name of the time series for printing.
    """
    series_clean = series[~np.isnan(series)]
    if len(series_clean) < 20:
        print("Skipping stationarity tests for " + name + " due to insufficient data points.")
        return

    print("\n--- Stationarity Tests for " + name + " ---")
    
    try:
        adf_result = adfuller(series_clean)
        print("ADF Test:")
        print("  Test Statistic: " + str(adf_result[0]))
        print("  p-value: " + str(adf_result[1]))
        if adf_result[1] <= 0.05:
            print("  Result: Stationary (reject H0)")
        else:
            print("  Result: Non-Stationary (fail to reject H0)")
    except Exception as e:
        print("ADF Test failed for " + name + ": " + str(e))

    try:
        kpss_result = kpss(series_clean, regression='c', nlags="auto")
        print("KPSS Test:")
        print("  Test Statistic: " + str(kpss_result[0]))
        print("  p-value: " + str(kpss_result[1]))
        if kpss_result[1] <= 0.05:
            print("  Result: Non-Stationary (reject H0)")
        else:
            print("  Result: Stationary (fail to reject H0)")
    except Exception as e:
        print("KPSS Test failed for " + name + ": " + str(e))
    print("--------------------------------------")


def power_law_model(x, a, b, c):
    """
    Defines the power-law model to be fitted: a + b * x^c.

    Args:
        x (np.ndarray): Independent variable.
        a (float): Offset parameter.
        b (float): Amplitude parameter.
        c (float): Exponent parameter.

    Returns:
        np.ndarray: The model's output.
    """
    return a + b * np.power(x, c)


def analyze_and_visualize_non_stationarity(data_path, prepared_data_path, output_dir):
    """
    Loads time series data, performs stationarity tests, fits a model to
    alpha vs. k_peak, and generates a summary plot.

    Args:
        data_path (str): Path to the .npz file with non-stationarity time series.
        prepared_data_path (str): Path to the .npz file with prepared data.
        output_dir (str): Directory to save the output plot.
    """
    try:
        ts_data = np.load(data_path)
        prep_data = np.load(prepared_data_path)
    except FileNotFoundError as e:
        print("Error: Data file not found. " + str(e))
        return

    window_times = ts_data['window_center_times']
    alpha_t = ts_data['alpha_t']
    alpha_std_t = ts_data['alpha_std_t']
    gamma_t = ts_data['gamma_t']
    nu_t = ts_data['nu_t']
    k_peak_t = ts_data['k_peak_t']
    k_box = prep_data['k_box'].item()

    run_stationarity_tests(alpha_t, "alpha(t)")
    run_stationarity_tests(gamma_t, "gamma(t)")
    run_stationarity_tests(nu_t, "nu(t)")

    plt.rcParams['text.usetex'] = False
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Non-Stationarity Analysis of Tracer Statistics', fontsize=16)

    ax1 = axes[0, 0]
    ax1.errorbar(window_times, alpha_t, yerr=alpha_std_t, fmt='-o', capsize=4, label='alpha(t)')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Lévy Index, alpha')
    ax1.set_title('Evolution of Lévy Index')
    ax1.grid(True)
    ax1.legend()

    ax2 = axes[0, 1]
    ax2.plot(window_times, gamma_t, '-o', label='gamma(t)')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Anomalous Exponent, gamma')
    ax2.set_title('Evolution of MSD Exponent')
    ax2.grid(True)
    ax2.legend()

    ax3 = axes[1, 0]
    ax3.plot(window_times, nu_t, '-o', label='nu(t)')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('VACF Decay Exponent, nu')
    ax3.set_title('Evolution of VACF Decay Exponent')
    ax3.grid(True)
    ax3.legend()

    ax4 = axes[1, 1]
    x_data = k_peak_t / k_box
    y_data = alpha_t
    
    valid_indices = ~np.isnan(x_data) & ~np.isnan(y_data)
    x_fit_data = x_data[valid_indices]
    y_fit_data = y_data[valid_indices]

    ax4.scatter(x_fit_data, y_fit_data, label='Data')

    try:
        p0_guess = [1.5, 0.1, -1.0]
        popt, pcov = curve_fit(power_law_model, x_fit_data, y_fit_data, p0=p0_guess, maxfev=5000)
        perr = np.sqrt(np.diag(pcov))

        print("\n--- Model Fit: alpha = a + b * (k_peak/k_box)^c ---")
        print("Fitted parameters:")
        print("  a = " + str(popt[0]) + " +/- " + str(perr[0]))
        print("  b = " + str(popt[1]) + " +/- " + str(perr[1]))
        print("  c = " + str(popt[2]) + " +/- " + str(perr[2]))
        print("----------------------------------------------------")

        x_curve = np.linspace(min(x_fit_data), max(x_fit_data), 200)
        y_curve = power_law_model(x_curve, *popt)
        fit_label = 'Fit: a + b*x^c'
        ax4.plot(x_curve, y_curve, 'r-', label=fit_label)
    except RuntimeError:
        print("\nCould not fit the model alpha = a + b * (k_peak/k_box)^c. Skipping plot overlay.")
    except Exception as e:
        print("\nAn error occurred during model fitting: " + str(e))

    ax4.set_xlabel('Normalized Peak Wavenumber, k_peak / k_box')
    ax4.set_ylabel('Lévy Index, alpha')
    ax4.set_title('Lévy Index vs. Spectral Condensation')
    ax4.grid(True)
    ax4.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    timestamp = int(time.time())
    filename = 'non_stationarity_analysis_1_' + str(timestamp) + '.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300)
    print("\nPlot saved to " + filepath)
    plt.close(fig)

if __name__ == '__main__':
    DATA_DIR = "data/"
    TIME_SERIES_PATH = os.path.join(DATA_DIR, "non_stationarity_timeseries.npz")
    PREPARED_DATA_PATH = os.path.join(DATA_DIR, "prepared_data.npz")
    
    analyze_and_visualize_non_stationarity(TIME_SERIES_PATH, PREPARED_DATA_PATH, DATA_DIR)