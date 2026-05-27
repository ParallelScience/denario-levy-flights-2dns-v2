"""
Honest reanalysis of iter5 DNS data.

The previous pipeline reported alpha in [1.28, 2.0], but the actual displacement
distributions are near-Gaussian (McCulloch nu < 2.44). This script computes:

1. Proper displacement statistics at multiple lags
2. Displacement PDFs and CCDF log-log plots to check for power-law tails
3. MSD scaling exponent gamma
4. VACF and decorrelation time
5. Conditional statistics near/far from FTLE ridges
6. Wavelet band-pass filtered α at each scale

Output: honest numbers + publication-quality figures
"""
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import linregress, kstest, norm
import os, json, time

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

BASE = '/home/node/work/projects/levy_flights_2dns_v2/data/'
OUT  = '/home/node/work/projects/levy_flights_2dns_v2/data/reanalysis/'
os.makedirs(OUT, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
vel_snaps  = np.load(BASE+'velocity_snapshots.npy')  # (499,2,128,128)
vel_times  = np.load(BASE+'vel_times.npy')
tracer_pos = np.load(BASE+'tracer_positions.npy')    # (24927,5000,2)
tracer_t   = np.load(BASE+'tracer_times.npy')
tracer_vel = np.load(BASE+'tracer_velocities.npy')   # (24927,5000,2)
diag       = np.load(BASE+'diagnostics.npy')

N_field   = vel_snaps.shape[-1]   # 128
L         = 2 * np.pi
dt_snap   = float(tracer_t[1] - tracer_t[0])
dt_vel    = float(vel_times[1] - vel_times[0])
T_L_arr   = diag['T_L']
T_L_mean  = float(np.mean(T_L_arr))
k_peak    = diag['k_peak']
U_rms     = diag['E_rms']

print(f"T_L: {T_L_arr.min():.3f}–{T_L_arr.max():.3f}, mean={T_L_mean:.3f}")
print(f"k_peak: {k_peak.min():.2f}–{k_peak.max():.2f}")
print(f"U_rms: {U_rms.min():.3f}–{U_rms.max():.3f}")
print(f"Production run: t=[{tracer_t[0]:.2f}, {tracer_t[-1]:.2f}], N_snaps={len(tracer_t)}")

# ── 1. Displacement statistics at multiple lags ───────────────────────────────
print("\n--- Displacement statistics ---")
lag_fracs = [0.1, 0.5, 1.0, 2.0, 5.0]
disp_stats = {}

for lf in lag_fracs:
    lag = max(1, int(lf * T_L_mean / dt_snap))
    if lag >= len(tracer_t) // 4:
        continue
    # Non-overlapping displacements (periodic wrap)
    dp = (tracer_pos[lag::lag] - tracer_pos[:-lag:lag] + np.pi) % (2*np.pi) - np.pi
    dx = dp[:, :, 0].flatten()
    dy = dp[:, :, 1].flatten()
    r  = np.sqrt(dx**2 + dy**2)

    # CCDF slope (power-law exponent in tail)
    r_s  = np.sort(r[r > 0])
    ccdf = 1 - np.arange(1, len(r_s)+1) / len(r_s)
    # Fit only the top decade
    thresh = np.percentile(r_s, 90)
    mask   = (r_s > thresh) & (ccdf > 0.001)
    if mask.sum() > 10:
        slope, intercept, r2, _, _ = linregress(np.log(r_s[mask]), np.log(ccdf[mask]))
    else:
        slope, r2 = np.nan, np.nan

    # MSD at this lag
    msd = float(np.mean(r**2))

    # Kurtosis
    from scipy.stats import kurtosis as kurtosis_scipy
    kurt = kurtosis_scipy(dx, fisher=True)  # excess kurtosis; Gaussian=0

    disp_stats[lf] = {
        'lag_steps': lag,
        'dt': lag * dt_snap,
        'msd': msd,
        'ccdf_slope': slope,
        'ccdf_r2': r2,
        'kurtosis': kurt,
        'r_array': r_s,
        'ccdf_array': ccdf,
        'dx': dx,
    }
    print(f"  lag={lf:.1f}T_L (dt={lag*dt_snap:.2f}): MSD={msd:.4f}, "
          f"CCDF slope={slope:.3f} (R²={r2:.3f}), kurtosis={kurt:.3f}")

# ── 2. MSD scaling exponent ───────────────────────────────────────────────────
print("\n--- MSD scaling ---")
lags_steps = np.unique(np.logspace(0, np.log10(len(tracer_t)//8), 40).astype(int))
msd_vals   = []
t_lags     = []
for lag in lags_steps:
    dp  = (tracer_pos[lag:] - tracer_pos[:-lag] + np.pi) % (2*np.pi) - np.pi
    msd = float(np.mean(dp[:, :, 0]**2 + dp[:, :, 1]**2))
    msd_vals.append(msd)
    t_lags.append(lag * dt_snap)

msd_vals = np.array(msd_vals)
t_lags   = np.array(t_lags)

# Fit in intermediate range (0.5 – 5 T_L)
fit_mask = (t_lags > 0.3 * T_L_mean) & (t_lags < 5 * T_L_mean) & (msd_vals > 0)
if fit_mask.sum() > 3:
    gamma, log_D, _, _, _ = linregress(np.log(t_lags[fit_mask]), np.log(msd_vals[fit_mask]))
    D_eff = np.exp(log_D)
else:
    gamma, D_eff = np.nan, np.nan
print(f"  MSD scaling exponent γ = {gamma:.4f}  (normal=1, ballistic=2)")
print(f"  Effective diffusion coefficient D = {D_eff:.4e}")

# ── 3. VACF ───────────────────────────────────────────────────────────────────
print("\n--- VACF ---")
max_lag_vacf = min(500, len(tracer_t)//10)
v_all = tracer_vel[:max_lag_vacf+1]  # (max_lag+1, 5000, 2)
C_vv  = np.zeros(max_lag_vacf)
v0    = v_all[0]  # (5000,2)
v0_var = float(np.mean(v0**2)) + 1e-30
for lag in range(1, max_lag_vacf):
    C_vv[lag-1] = float(np.mean(v_all[lag] * v0)) / v0_var
tau_corr_steps = np.argmax(C_vv < 0) if np.any(C_vv < 0) else max_lag_vacf
tau_corr = tau_corr_steps * dt_snap
print(f"  C_vv(τ=1) = {C_vv[0]:.4f}  (>0.3 means memory resolved)")
print(f"  Decorrelation time τ_corr ≈ {tau_corr:.3f} ({tau_corr/T_L_mean:.2f} T_L)")

# ── 4. FTLE-based conditional analysis ───────────────────────────────────────
print("\n--- FTLE computation (GPU) ---")
k1d  = torch.fft.fftfreq(N_field, d=1.0/N_field).to(device)
k1dr = torch.fft.rfftfreq(N_field, d=1.0/N_field).to(device)
kx, ky = torch.meshgrid(k1d, k1dr, indexing='ij')
k2    = kx**2 + ky**2; k2[0,0] = 1.0

def interp_vel_gpu(pos_gpu, u_gpu, v_gpu, N):
    pos_norm = (pos_gpu / L) * 2 - 1
    grid = pos_norm.view(1, 1, -1, 2)
    ui = torch.nn.functional.grid_sample(u_gpu.view(1,1,N,N), grid,
                                          align_corners=True, mode='bilinear').squeeze()
    vi = torch.nn.functional.grid_sample(v_gpu.view(1,1,N,N), grid,
                                          align_corners=True, mode='bilinear').squeeze()
    return torch.stack([ui, vi], dim=1)

def compute_ftle_gpu(vel_np, t_start_idx, dt_ftle):
    Ng = N_field
    x  = torch.linspace(0, L*(1-1/Ng), Ng, device=device)
    y  = torch.linspace(0, L*(1-1/Ng), Ng, device=device)
    gx, gy = torch.meshgrid(x, y, indexing='ij')
    pos  = torch.stack([gx.flatten(), gy.flatten()], dim=1)
    pos0 = pos.clone()
    t_start = vel_times[t_start_idx]
    t_end   = t_start + dt_ftle
    filt_t  = torch.from_numpy(vel_np).to(device)
    for i in range(t_start_idx, len(vel_times)-1):
        t0, t1 = vel_times[i], vel_times[i+1]
        if t0 >= t_end: break
        u0,v0 = filt_t[i,0], filt_t[i,1]
        u1,v1 = filt_t[min(i+1,len(vel_times)-1),0], filt_t[min(i+1,len(vel_times)-1),1]
        t_cur, dt_s = t0, dt_vel/4
        while t_cur < min(t1, t_end) - 1e-8:
            dt_s2 = min(dt_s, min(t1,t_end)-t_cur)
            at = (t_cur-t0)/(t1-t0+1e-30)
            u_t = (1-at)*u0+at*u1; v_t = (1-at)*v0+at*v1
            vel = interp_vel_gpu(pos, u_t, v_t, Ng)
            pos = (pos + dt_s2*vel) % L
            t_cur += dt_s2
    dp_np = ((pos - pos0 + np.pi) % (2*np.pi) - np.pi).cpu().numpy().reshape(Ng,Ng,2)
    dxdx = np.gradient(dp_np[:,:,0], axis=0)/(L/Ng)
    dxdy = np.gradient(dp_np[:,:,0], axis=1)/(L/Ng)
    dydx = np.gradient(dp_np[:,:,1], axis=0)/(L/Ng)
    dydy = np.gradient(dp_np[:,:,1], axis=1)/(L/Ng)
    C11=dxdx**2+dydx**2; C12=dxdx*dxdy+dydx*dydy; C22=dxdy**2+dydy**2
    tr=C11+C22; det=C11*C22-C12**2
    lam = tr/2 + np.sqrt(np.maximum((tr/2)**2-det,0))
    dt_act = min(t_end, vel_times[-1]) - t_start
    return np.log(np.sqrt(np.maximum(lam,1e-30))) / (dt_act+1e-30)

# Compute FTLE at 5 times
dt_ftle = 4 * T_L_mean
ftle_times_idx = [20, 100, 200, 300, 420]
ftle_fields, ftle_t_vals = [], []
for idx in ftle_times_idx:
    if idx >= len(vel_times)-1: continue
    f = compute_ftle_gpu(vel_snaps, idx, dt_ftle)
    ftle_fields.append(f)
    ftle_t_vals.append(vel_times[idx])
    print(f"  FTLE t={vel_times[idx]:.1f}: max={f.max():.3f}, p75={np.percentile(f,75):.3f}")
ftle_fields = np.array(ftle_fields)

# Conditional α: near vs far FTLE ridges
# Use FTLE field at t~middle of production
ftle_mid  = ftle_fields[len(ftle_fields)//2]
ftle_p75  = np.percentile(ftle_mid, 75)
t_mid_val = ftle_t_vals[len(ftle_t_vals)//2]
t_idx_tr  = np.argmin(np.abs(tracer_t - t_mid_val))
window_half = min(200, t_idx_tr//2, (len(tracer_t)-t_idx_tr)//2)

alpha_results = {}
if window_half > 20:
    pos_ref = tracer_pos[t_idx_tr]  # (5000,2)
    ix = (pos_ref[:,0] / L * N_field).astype(int) % N_field
    iy = (pos_ref[:,1] / L * N_field).astype(int) % N_field
    ftle_at_tr = ftle_mid[ix, iy]
    near = ftle_at_tr >= ftle_p75
    far  = ftle_at_tr <  ftle_p75

    lag = max(1, int(1.0 * T_L_mean / dt_snap))
    dp_win = (tracer_pos[t_idx_tr+lag] - tracer_pos[t_idx_tr] + np.pi) % (2*np.pi) - np.pi

    for mask, lbl in [(near,'near'), (far,'far')]:
        if mask.sum() < 50: continue
        dx_g = dp_win[mask, 0]
        dy_g = dp_win[mask, 1]
        r_g  = np.sqrt(dx_g**2 + dy_g**2)
        r_s  = np.sort(r_g[r_g > 0])
        ccdf_g = 1 - np.arange(1,len(r_s)+1)/len(r_s)
        thresh = np.percentile(r_s, 85)
        mask2  = (r_s > thresh) & (ccdf_g > 0.005)
        if mask2.sum() > 5:
            slope_g, _, r2_g, _, _ = linregress(np.log(r_s[mask2]), np.log(ccdf_g[mask2]))
        else:
            slope_g, r2_g = np.nan, np.nan
        kurt_g = float(np.mean(dx_g**4)/(np.mean(dx_g**2)+1e-30)**2) - 3
        alpha_results[lbl] = {
            'n': int(mask.sum()), 'ccdf_slope': slope_g, 'r2': r2_g,
            'kurtosis': kurt_g, 'r': r_s, 'ccdf': ccdf_g
        }
        print(f"  FTLE {lbl} (n={mask.sum()}): CCDF slope={slope_g:.3f}, kurtosis={kurt_g:.3f}")

# ── 5. Gaussian band-pass filtered fields ─────────────────────────────────────
print("\n--- Band-pass filtered tracer re-advection ---")
km_np = torch.sqrt(kx**2+ky**2).cpu().numpy()

# Focus on bands with non-negligible energy
bands = [
    {'name': 'k=1 (condensate)', 'k0': 1.0, 'sigma': 0.3},
    {'name': 'k=2-3',            'k0': 2.0, 'sigma': 0.6},
    {'name': 'k=3-6',            'k0': 4.0, 'sigma': 1.2},
]

N_re_adv = 2000  # tracers for re-advection
rng = np.random.default_rng(0)
band_alpha = {}

for band in bands:
    k0, sig = band['k0'], band['sigma']
    G = np.exp(-((km_np - k0)**2) / (2*sig**2)).astype(np.float32)
    E_frac = 0.0

    # Filter all velocity snapshots
    fv = np.zeros_like(vel_snaps)
    for i in range(len(vel_snaps)):
        for c in range(2):
            fhat = np.fft.rfft2(vel_snaps[i,c])
            fv[i,c] = np.fft.irfft2(fhat * G[:, :G.shape[1]], s=(N_field,N_field)).real
    E_frac = float(np.mean(fv**2) / (np.mean(vel_snaps**2)+1e-30))

    # Short re-advection: only first 20 T_L
    pos_re = torch.from_numpy(
        rng.uniform(0, L, (N_re_adv, 2)).astype(np.float32)).to(device)
    fv_t = torch.from_numpy(fv).to(device)
    T_re  = 20 * T_L_mean
    t_end_idx = np.searchsorted(vel_times, T_re)
    pos_hist = [pos_re.cpu().numpy().copy()]
    for i in range(min(t_end_idx-1, len(vel_times)-2)):
        u0 = fv_t[i,0]; v0 = fv_t[i,1]
        u1 = fv_t[i+1,0]; v1 = fv_t[i+1,1]
        for _ in range(4):   # 4 sub-steps per dt_vel
            dt_s = dt_vel / 4
            vel_mid = interp_vel_gpu(pos_re, (u0+u1)/2, (v0+v1)/2, N_field)
            pos_re = (pos_re + dt_s * vel_mid) % L
        pos_hist.append(pos_re.cpu().numpy().copy())
    pos_hist = np.array(pos_hist)

    # Estimate gamma from MSD
    lags_r = np.unique(np.logspace(0, np.log10(max(2,len(pos_hist)//8)),20).astype(int))
    msd_r  = np.array([np.mean(((pos_hist[l:]-pos_hist[:-l]+np.pi)%(2*np.pi)-np.pi)**2 *2)
                       for l in lags_r if l < len(pos_hist)//2])
    t_r    = np.array([l * dt_vel for l in lags_r if l < len(pos_hist)//2])
    if len(t_r) > 3 and np.all(msd_r > 0):
        gam, _, _, _, _ = linregress(np.log(t_r[msd_r>0]), np.log(msd_r[msd_r>0]))
    else:
        gam = np.nan

    # CCDF at 1 T_L
    lag_disp = max(1, int(1.0 * T_L_mean / dt_vel))
    if lag_disp < len(pos_hist)-1:
        dp_r = (pos_hist[lag_disp] - pos_hist[0] + np.pi) % (2*np.pi) - np.pi
        r_r  = np.linalg.norm(dp_r, axis=-1)
        r_rs = np.sort(r_r[r_r>0])
        ccdf_r = 1 - np.arange(1,len(r_rs)+1)/len(r_rs)
        t85 = np.percentile(r_rs, 85)
        mk  = (r_rs > t85) & (ccdf_r > 0.005)
        if mk.sum() > 5:
            slope_r, _, _, _, _ = linregress(np.log(r_rs[mk]), np.log(ccdf_r[mk]))
        else:
            slope_r = np.nan
    else:
        slope_r = np.nan

    band_alpha[band['name']] = {
        'k0': k0, 'sigma': sig, 'E_frac': E_frac,
        'gamma': gam, 'ccdf_slope': slope_r,
    }
    print(f"  {band['name']}: E_frac={E_frac:.3f}, γ={gam:.3f}, CCDF slope={slope_r:.3f}")

del fv_t
torch.cuda.empty_cache()

# ── 6. Publication-quality figure ─────────────────────────────────────────────
print("\nGenerating figures...")
plt.rcParams.update({'font.size': 11, 'axes.labelsize': 12})
fig = plt.figure(figsize=(16, 12))
gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.38)

colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(lag_fracs)))

# 1. CCDF at multiple lags
ax1 = fig.add_subplot(gs[0, 0:2])
for ci, lf in enumerate(lag_fracs):
    if lf not in disp_stats: continue
    s = disp_stats[lf]
    ax1.loglog(s['r_array'], s['ccdf_array'], color=colors[ci],
               label=f"τ={lf:.1f}T_L (γ={s['ccdf_slope']:.2f})", lw=1.5)
ax1.set_xlabel('Displacement |Δr|')
ax1.set_ylabel('CCDF P(|Δr|>r)')
ax1.set_title('Tracer Displacement CCDF at Multiple Lags')
ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

# 2. MSD vs time lag
ax2 = fig.add_subplot(gs[0, 2:4])
ax2.loglog(t_lags, msd_vals, 'b.-', lw=1.5, ms=3)
if not np.isnan(gamma):
    t_fit = t_lags[fit_mask]
    ax2.loglog(t_fit, D_eff * t_fit**gamma, 'r--', lw=2,
               label=f'γ = {gamma:.3f}', zorder=5)
ax2.axvline(T_L_mean, ls=':', color='k', alpha=0.5, label=f'T_L={T_L_mean:.2f}')
ax2.set_xlabel('Time lag τ'); ax2.set_ylabel('MSD ⟨|Δr|²⟩')
ax2.set_title('Mean Squared Displacement Scaling')
ax2.legend(fontsize=9); ax2.grid(alpha=0.3)

# 3. VACF
ax3 = fig.add_subplot(gs[1, 0:2])
tau_ax = np.arange(len(C_vv)) * dt_snap
ax3.plot(tau_ax / T_L_mean, C_vv, 'b-', lw=1.5)
ax3.axhline(0, color='k', lw=0.8)
ax3.axhline(0.3, ls='--', color='red', alpha=0.7, label='C=0.3 threshold')
ax3.axvline(tau_corr/T_L_mean, ls=':', color='orange', alpha=0.8,
            label=f'τ_corr≈{tau_corr:.2f} ({tau_corr/T_L_mean:.1f}T_L)')
ax3.set_xlabel('Lag τ / T_L'); ax3.set_ylabel('C_vv(τ)')
ax3.set_xlim(0, min(10, tau_ax[-1]/T_L_mean)); ax3.set_ylim(-0.2, 1.05)
ax3.set_title('Velocity Autocorrelation Function')
ax3.legend(fontsize=8); ax3.grid(alpha=0.3)

# 4. FTLE field
ax4 = fig.add_subplot(gs[1, 2:4])
ftle_mid_plot = ftle_fields[len(ftle_fields)//2]
im = ax4.imshow(ftle_mid_plot.T, origin='lower',
                extent=[0,2*np.pi,0,2*np.pi], cmap='inferno', aspect='equal')
plt.colorbar(im, ax=ax4, label='FTLE (σ)')
ax4.set_title(f'FTLE field at t={t_mid_val:.1f}')
ax4.set_xlabel('x'); ax4.set_ylabel('y')
# Overlay tracer positions coloured by their FTLE value
if window_half > 20:
    pos_ref_plt = tracer_pos[t_idx_tr]
    sc = ax4.scatter(pos_ref_plt[::10,0], pos_ref_plt[::10,1],
                     c=ftle_at_tr[::10], cmap='cool', s=1, alpha=0.4, vmin=0, vmax=ftle_p75*2)

# 5. CCDF near vs far FTLE
ax5 = fig.add_subplot(gs[2, 0:2])
for lbl, col in [('near','blue'), ('far','orange')]:
    if lbl not in alpha_results: continue
    r = alpha_results[lbl]
    ax5.loglog(r['r'], r['ccdf'], color=col, lw=1.5,
               label=f"{lbl} FTLE ridge (n={r['n']}, slope={r['ccdf_slope']:.2f})")
ax5.set_xlabel('Displacement |Δr|'); ax5.set_ylabel('CCDF')
ax5.set_title('Conditional CCDF Near/Far FTLE Ridges')
ax5.legend(fontsize=8); ax5.grid(alpha=0.3)
ax5.text(0.05,0.05, f"τ = 1 T_L = {T_L_mean:.2f}", transform=ax5.transAxes, fontsize=9)

# 6. Band-filtered MSD exponents
ax6 = fig.add_subplot(gs[2, 2:4])
bnames = list(band_alpha.keys())
bgammas = [band_alpha[b]['gamma'] for b in bnames]
bslopes = [band_alpha[b]['ccdf_slope'] for b in bnames]
bk0s    = [band_alpha[b]['k0'] for b in bnames]
bar_w = 0.35
x = np.arange(len(bnames))
bars1 = ax6.bar(x - bar_w/2, bgammas, bar_w, label='MSD exponent γ', color='steelblue', edgecolor='k')
bars2 = ax6.bar(x + bar_w/2, [-s if not np.isnan(s) else np.nan for s in bslopes],
                bar_w, label='|CCDF slope|', color='coral', edgecolor='k')
ax6.axhline(1.0, ls='--', color='k', alpha=0.5, label='Normal diffusion (γ=1)')
ax6.axhline(2.0, ls=':', color='k', alpha=0.5, label='Ballistic (γ=2)')
ax6.set_xticks(x); ax6.set_xticklabels([b.split('(')[0].strip() for b in bnames], fontsize=9)
ax6.set_ylabel('Exponent'); ax6.set_title('Band-Pass Filtered Transport Exponents')
ax6.legend(fontsize=8); ax6.set_ylim(0, 3)
for bar, val in zip(bars1, bgammas):
    if not np.isnan(val):
        ax6.text(bar.get_x()+bar.get_width()/2, val+0.05, f'{val:.2f}', ha='center', fontsize=8)

fig.suptitle('Honest Reanalysis: Lagrangian Transport in 2D Turbulent Inverse Cascade\n'
             '1024² DNS, forcing k∈[1,3], T_prod≈250 time units',
             fontsize=13, fontweight='bold')

out_fig = OUT + 'honest_reanalysis.png'
plt.savefig(out_fig, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {out_fig}")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("REANALYSIS SUMMARY")
print("="*60)
print(f"Production run: T_prod≈{tracer_t[-1]:.1f}, T_L_mean={T_L_mean:.3f}")
print(f"MSD scaling exponent γ = {gamma:.4f}  (1=normal, 2=ballistic)")
print(f"VACF C_vv(τ=1) = {C_vv[0]:.4f}, τ_corr ≈ {tau_corr:.3f}")
print()
print("CCDF tail slopes (log-log):")
for lf, s in disp_stats.items():
    print(f"  τ={lf:.1f}T_L: slope={s['ccdf_slope']:.3f} (R²={s['ccdf_r2']:.3f}), kurtosis={s['kurtosis']:.3f}")
print()
print("Conditional FTLE analysis:")
for lbl, r in alpha_results.items():
    print(f"  {lbl}: CCDF slope={r['ccdf_slope']:.3f}, kurtosis={r['kurtosis']:.3f}")
print()
print("Band-pass filtered:")
for bname, r in band_alpha.items():
    print(f"  {bname}: γ={r['gamma']:.3f}, CCDF slope={r['ccdf_slope']:.3f}")
print()
print("KEY FINDING:")
print(f"  γ = {gamma:.3f} — {'super-diffusive' if gamma > 1.2 else 'near-normal diffusion'}")
print(f"  CCDF slopes range {min(s['ccdf_slope'] for s in disp_stats.values() if not np.isnan(s['ccdf_slope'])):.2f} to "
      f"{max(s['ccdf_slope'] for s in disp_stats.values() if not np.isnan(s['ccdf_slope'])):.2f}")
print("  (Gaussian tails: slope < -3; Lévy: slope in [-1,-2])")
print(f"  Output: {OUT}")

# Save summary
summary = {
    'gamma': float(gamma),
    'D_eff': float(D_eff),
    'C_vv_lag1': float(C_vv[0]),
    'tau_corr': float(tau_corr),
    'tau_corr_over_TL': float(tau_corr/T_L_mean),
    'CCDF_slopes': {str(lf): float(s['ccdf_slope']) for lf, s in disp_stats.items()},
    'CCDF_R2':     {str(lf): float(s['ccdf_r2']) for lf, s in disp_stats.items()},
    'kurtosis':    {str(lf): float(s['kurtosis']) for lf, s in disp_stats.items()},
    'FTLE_conditional': {lbl: {'ccdf_slope': float(r['ccdf_slope']),
                                'kurtosis': float(r['kurtosis'])} for lbl,r in alpha_results.items()},
    'band_gamma': {b: float(r['gamma']) for b,r in band_alpha.items()},
    'T_L_mean': float(T_L_mean),
    'T_prod': float(tracer_t[-1]),
}
with open(OUT+'reanalysis_summary.json','w') as f:
    json.dump(summary, f, indent=2)
print(f"Summary saved to {OUT}reanalysis_summary.json")
