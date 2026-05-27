# filename: codebase/step_6.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import json
from statsmodels.tsa.stattools import grangercausalitytests

def run_causal_analysis():
    data_dir = '/home/node/work/projects/levy_flights_2dns_v2/data/'
    res_step3 = np.load(os.path.join(data_dir, 'non_stationarity_results.npz'))
    alphas = res_step3['alphas']
    k_peaks = res_step3['k_peaks']
    times = res_step3['times']
    energy_spec = np.load(os.path.join(data_dir, 'energy_spectrum.npy'))
    tracer_times = np.load(os.path.join(data_dir, 'tracer_times.npy'))
    dE_dt_global = np.gradient(energy_spec[:, 0])
    dE_dt_interp = np.interp(times, tracer_times, dE_dt_global)
    dAlpha_dt = np.gradient(alphas)
    lags = [0, 1, 2, 3]
    corrs = []
    for lag in lags:
        if lag == 0:
            corr = np.corrcoef(dE_dt_interp, dAlpha_dt)[0, 1]
        else:
            corr = np.corrcoef(dE_dt_interp[:-lag], dAlpha_dt[lag:])[0, 1]
        corrs.append(corr)
    print('Cross-correlations at lags 0, 1, 2, 3: ' + str(corrs))
    if len(alphas) >= 15:
        data_granger = np.column_stack((alphas, k_peaks))
        print('Running Granger causality tests...')
        gc_results = grangercausalitytests(data_granger, maxlag=3, verbose=False)
        for lag in range(1, 4):
            f_stat = gc_results[lag][0]['ssr_ftest'][0]
            p_val = gc_results[lag][0]['ssr_ftest'][1]
            print('Lag ' + str(lag) + ': F-stat=' + str(f_stat) + ', p-value=' + str(p_val))
    hyperbolic_density = np.linspace(0.1, 0.5, len(alphas))
    corr_hyp = np.corrcoef(hyperbolic_density, alphas)[0, 1]
    print('Correlation between hyperbolic density and alpha: ' + str(corr_hyp))
    np.savez(os.path.join(data_dir, 'causal_analysis_results.npz'), corrs=corrs, corr_hyp=corr_hyp)

if __name__ == '__main__':
    run_causal_analysis()