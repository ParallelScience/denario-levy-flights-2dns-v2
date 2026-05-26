Iteration 0 failed due to insufficient temporal resolution of tracer snapshots. Key issues:

1. **Snapshot interval too coarse**: Δt_snap = 0.3 → tracers moved O(1) domain width per step (σ_displacement ≈ 1.8 ≈ π/√3 at any lag), completely erasing Lévy tail statistics. All α estimates returned 2.0 (Gaussian).

2. **VACF = 0 at all lags**: No Lagrangian velocity memory resolvable. The Lagrangian correlation time is much shorter than Δt_snap.

3. **Energy already condensed at k=1 from t=0**: Forcing at k∈[1,3] did not allow the inverse cascade to develop from an intermediate wavenumber toward k=1. The non-stationarity study requires this evolution.

4. **d_vv = 0 everywhere**: Inter-vortex distance was not properly computed.

**Required fixes for Iteration 1:**
- N = 1024² (mandatory per supervisor)
- Δt_snap ≤ 0.05 (6× finer) so that (U_rms × Δt_snap) / (2π) < 0.05
- Forcing at k ∈ [3, 5] to allow inverse cascade to develop k_peak: 4 → 1
- Spinup T/T_L ≥ 10 before seeding tracers
- Save Eulerian velocity fields at Δt_vel = 2.0 (separate coarser grid for storage efficiency)
- Verify VACF > 0.3 at lag=1 step before proceeding to Lévy statistics
