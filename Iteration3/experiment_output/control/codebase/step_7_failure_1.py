# filename: codebase/step_7.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

def perform_synthesis():
    data_dir = "/home/node/work/projects/levy_flights_2dns_v2/data/"
    
    nonstat_res = np.load(os.path.join(data_dir, "nonstationarity_results.npz"))
    wavelet_res = np.load(os.path.join(data_dir, "wavelet_results.npz"))
    
    alpha_t = nonstat_res['alpha_mcculloch']
    k_peaks = nonstat_res['k_peak']
    
    k_j = wavelet_res['k_j']
    alpha_j = wavelet_res['alpha_j']
    t_eddy = wavelet_res['t_eddy']
    
    j_peak_series = []
    alpha_j_peak = []
    
    for kp in k_peaks:
        idx = np.argmin(np.abs(k_j - kp))
        j_peak_series.append(idx)
        alpha_j_peak.append(alpha_j[idx])
        
    alpha_j_peak = np.array(alpha_j_peak)
    
    corr, _ = pearsonr(alpha_t, alpha_j_peak)
    rmse = np.sqrt(mean_squared_error(alpha_t, alpha_j_peak))
    
    print("Synthesis Results:")
    print("Pearson Correlation between alpha(t) and alpha(j_peak): " + str(corr))
    print("RMSE between alpha(t) and alpha(j_peak): " + str(rmse))
    
    t_eddy_grid = np.array(t_eddy)
    kpeak_grid = np.array(k_peaks)
    alpha_2d = np.array(alpha_t)
    
    np.savez(os.path.join(data_dir, "synthesis_results.npz"),
             alpha_t=alpha_t,
             alpha_j_peak=alpha_j_peak,
             j_peak_series=j_peak_series,
             correlation=corr,
             rmse=rmse,
             t_eddy_grid=t_eddy_grid,
             kpeak_grid=kpeak_grid,
             alpha_2d=alpha_2d)
    
    print("Synthesis results saved to synthesis_results.npz")

if __name__ == '__main__':
    perform_synthesis()