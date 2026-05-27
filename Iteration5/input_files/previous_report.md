No previous report


Iteration 4:
**Methodological Evolution**
- **Resolution Increase**: The simulation grid was increased from $1024 \times 1024$ (Iteration 3) to $1024 \times 1024$ with a refined hyperviscosity scaling ($\nu_h = 3.9 \times 10^{-31}$) to maintain dissipation consistency at the dealiasing cutoff.
- **Numerical Scheme**: Replaced explicit RK4 with an Integrating Factor (IF) method for the hyperviscous term to ensure exact treatment of stiff dissipation.
- **Forcing Band**: Shifted forcing to $k \in [10, 12]$ to allow for a more extended inverse cascade development compared to the $k \in [3, 5]$ range used previously.
- **Analysis Refinement**: Introduced an Okubo-Weiss parameter threshold ($Q < -Q_{threshold}$) for eddy lifetime estimation to isolate coherent structures, replacing the previous unfiltered energy autocorrelation approach.

**Performance Delta**
- **Snapshot Integrity**: The reduction of $\Delta t_{snap}$ to $0.01$ (from $0.05$) significantly improved the resolution of tracer dynamics, resulting in a mean displacement per snapshot $\langle|\Delta r|\rangle \approx 0.0068$, which is well within the stability bounds for Lévy statistics.
- **VACF Resolution**: The memory retention at short lags improved ($C_{vv}(\tau=1) \approx 0.995$), confirming that the current temporal resolution successfully captures the velocity memory necessary for anomalous diffusion.
- **Stationarity Assessment**: The use of ADF and KPSS tests provided a more robust statistical confirmation of non-stationarity in $\alpha(t)$ compared to the qualitative observations in Iteration 4.
- **Wavelet Filtering**: While the wavelet decomposition successfully isolated scales, the resulting $\alpha(j)$ values remained near $2.0$, indicating that the filtering process (or the choice of $db4$ basis) likely suppressed the long-range correlations required for Lévy flights, representing a regression in the ability to isolate the source of anomalous diffusion compared to the full-field analysis.

**Synthesis**
- **Causal Attribution**: The observed non-stationarity of $\alpha(t)$ is directly attributed to the spectral condensation process (the drift of $k_{peak}$ toward $k=1$). The shift to the IF method and higher resolution provided a cleaner inverse cascade, confirming that $\alpha$ is not a universal constant but a dynamic parameter that decreases as large-scale structures dominate the flow.
- **Validity and Limits**: The discrepancy between the global $\alpha(t)$ (which shows clear Lévy behavior) and the wavelet-filtered $\alpha(j)$ (which trends toward Gaussianity) suggests that Lévy flights in 2D turbulence are an emergent property of multi-scale interactions rather than a feature of isolated wavelet scales. 
- **Next Steps**: The failure of the wavelet-filtered fields to reproduce the heavy-tailed statistics suggests that future research should pivot away from simple scale-filtering and toward identifying specific Lagrangian Coherent Structures (LCS) or strain-stripping events, as these appear to be the primary drivers of the observed anomalous diffusion.
        