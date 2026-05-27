"""
Proper wavelet reanalysis of iter5 DNS data.

Instead of hard-thresholding wavelet coefficients (which destroys the Lévy mechanism),
we use:
1. Smooth Gaussian band-pass filters in k-space to isolate velocity at each scale
2. GPU-accelerated tracer re-advection in each filtered field
3. α estimated via McCulloch quantile method on re-advected trajectories
4. FTLE-based transport barrier analysis: test whether heavy tails are localised
   near FTLE ridges

Scale bands (in integer wavenumbers, matching the 128x128 field):
  Band 1: k in [1,  4]   — box-scale / condensate
  Band 2: k in [3,  8]   — large eddies
  Band 3: k in [6, 16]   — intermediate eddies
  Band 4: k in [12, 32]  — small eddies / forcing scale vicinity
  Band 5: k in [20, 64]  — small scales

Each band uses a Gaussian envelope in k-space: G(k) = exp(-(k-k0)^2 / (2*sigma^2))
with k0 = geometric mean of band, sigma = band half-width.
"""

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
import os, time, json

torch.backends.cudnn.benchmark = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

BASE = '/home/node/work/projects/levy_flights_2dns_v2/data/'
OUT  = '/home/node/work/projects/levy_flights_2dns_v2/data/reanalysis/'
os.makedirs(OUT, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
vel_snaps  = np.load(BASE+'velocity_snapshots.npy')   # (499, 2, 128, 128) float32
vel_times  = np.load(BASE+'vel_times.npy')             # (499,)
tracer_pos = np.load(BASE+'tracer_positions.npy')      # (24927, 5000, 2)
tracer_t   = np.load(BASE+'tracer_times.npy')          # (24927,)
diag       = np.load(BASE+'diagnostics.npy')
sp         = json.load(open(BASE+'sim_params.json'))

N_field = vel_snaps.shape[-1]   # 128
N_snaps = vel_snaps.shape[0]    # 499
N_tr    = tracer_pos.shape[1]   # 5000
L       = 2 * np.pi
dt_vel  = float(vel_times[1] - vel_times[0])   # ~0.5
dt_snap = float(tracer_t[1] - tracer_t[0])     # ~0.01
T_L_mean = float(np.mean(diag['T_L']))
T_L_min  = float(np.min(diag['T_L']))
print(f"N_field={N_field}, N_snaps={N_snaps}, N_tracers={N_tr}")
print(f"T_L: {T_L_min:.3f} – {float(np.max(diag['T_L'])):.3f}, mean={T_L_mean:.3f}")
print(f"k_peak: {diag['k_peak'].min():.2f} – {diag['k_peak'].max():.2f}")

# ── Wavenumbers for 128x128 rfft ──────────────────────────────────────────────
k1d   = torch.fft.fftfreq(N_field, d=1.0/N_field).to(device)   # integer
k1dr  = torch.fft.rfftfreq(N_field, d=1.0/N_field).to(device)
kx, ky = torch.meshgrid(k1d, k1dr, indexing='ij')
km = torch.sqrt(kx**2 + ky**2)

# ── Scale band definitions ────────────────────────────────────────────────────
# Gaussian envelope G(k) = exp(-(k-k0)^2 / (2*sigma^2))
# Energy is almost entirely at k=1-3 (condensate). Use tight bands there.
bands = [
    {'name': 'k=1',      'k0': 1.0, 'sigma': 0.4},   # fundamental mode
    {'name': 'k=1-2',    'k0': 1.5, 'sigma': 0.6},   # box-scale condensate
    {'name': 'k=2-4',    'k0': 3.0, 'sigma': 0.8},   # large eddies
    {'name': 'k=3-8',    'k0': 5.0, 'sigma': 2.0},   # intermediate
    {'name': 'broadband','k0': 0.0, 'sigma': 999.0},  # control: full field
]

def gaussian_filter(k0, sigma):
    G = torch.exp(-((km - k0)**2) / (2*sigma**2))
    # Normalise so filtered field has same total energy as original (approximately)
    return G.to(device)

# ── Compute filtered velocity fields ─────────────────────────────────────────
print("\nComputing Gaussian band-pass filtered velocity fields...")
vel_t = torch.from_numpy(vel_snaps).to(device)  # (499, 2, 128, 128)

filtered_vel = {}   # band_name -> (499, 2, 128, 128) on CPU
for band in bands:
    G = gaussian_filter(band['k0'], band['sigma'])
    filt_u = torch.zeros_like(vel_t)
    for i in range(N_snaps):
        for c in range(2):
            fhat = torch.fft.rfft2(vel_t[i, c])
            filt_u[i, c] = torch.fft.irfft2(fhat * G, s=(N_field, N_field))
    filtered_vel[band['name']] = filt_u.cpu().numpy()
    # Check energy fraction
    E_orig = float(vel_t.pow(2).mean())
    E_filt = float(filt_u.pow(2).mean())
    print(f"  {band['name']}: energy fraction = {E_filt/E_orig:.3f}")

del vel_t
torch.cuda.empty_cache()

# ── McCulloch α estimator ─────────────────────────────────────────────────────
def mcculloch_alpha(displacements):
    """Estimate Lévy α from 1D scalar displacements using McCulloch quantiles."""
    if len(displacements) < 50:
        return np.nan
    q = np.nanpercentile(displacements, [5, 25, 50, 75, 95])
    nu = (q[4] - q[0]) / (q[3] - q[1] + 1e-30)
    # McCulloch lookup table
    v_alpha = np.array([2.439,2.468,2.505,2.554,2.618,2.703,2.819,2.981,
                        3.214,3.569,4.151,5.195,7.434,13.08,32.58])
    alpha_tab = np.array([2.00,1.90,1.80,1.70,1.60,1.50,1.40,1.30,
                          1.20,1.10,1.00,0.90,0.80,0.70,0.60])
    if nu <= v_alpha[-1]:
        return 0.6
    if nu >= v_alpha[0]:
        return 2.0
    return float(np.interp(nu, v_alpha[::-1], alpha_tab[::-1]))

def estimate_alpha_from_trajectories(pos_array, times, T_L, lags_frac=(0.5, 1.0, 2.0)):
    """
    Estimate α from tracer trajectories.
    pos_array: (N_t, N_tracers, 2)
    Returns median α across lags.
    """
    dt = times[1] - times[0]
    alphas = []
    for lag_frac in lags_frac:
        lag_steps = max(1, int(lag_frac * T_L / dt))
        if lag_steps >= len(times) - 1:
            continue
        dx = (pos_array[lag_steps:] - pos_array[:-lag_steps])
        # periodic wrap
        dx = (dx + np.pi) % (2*np.pi) - np.pi
        dx_flat = dx[:, :, 0].flatten()
        dy_flat = dx[:, :, 1].flatten()
        a = (mcculloch_alpha(dx_flat) + mcculloch_alpha(dy_flat)) / 2
        alphas.append(a)
    return float(np.nanmedian(alphas)) if alphas else np.nan

# ── Re-advect tracers in each filtered field ──────────────────────────────────
print("\nRe-advecting tracers in each filtered field (GPU)...")

def interp_vel_gpu(pos_gpu, u_gpu, v_gpu, N, L):
    """Bilinear interpolation of (u,v) at tracer positions."""
    pos_norm = (pos_gpu / L) * 2 - 1  # [-1,1]
    grid = pos_norm.view(1, 1, -1, 2)
    ui = torch.nn.functional.grid_sample(
        u_gpu.view(1,1,N,N), grid, align_corners=True, mode='bilinear').squeeze()
    vi = torch.nn.functional.grid_sample(
        v_gpu.view(1,1,N,N), grid, align_corners=True, mode='bilinear').squeeze()
    return torch.stack([ui, vi], dim=1)

def advect_tracers_in_filtered(filt_vel_np, vel_times, T_prod_frac=1.0):
    """
    Advect N_re tracers through filtered velocity field.
    Returns (pos_history, times) arrays.
    """
    N_re = 2000   # fewer tracers for speed
    rng  = np.random.default_rng(42)
    pos  = torch.from_numpy(rng.uniform(0, L, (N_re, 2)).astype(np.float32)).to(device)

    t_end   = vel_times[-1] * T_prod_frac
    dt_adv  = dt_vel / 10   # sub-step within each vel interval

    pos_hist = [pos.cpu().numpy().copy()]
    t_hist   = [vel_times[0]]

    filt_t = torch.from_numpy(filt_vel_np).to(device)  # (499,2,128,128)

    t_wall0 = time.time()
    for i in range(len(vel_times) - 1):
        t0, t1 = vel_times[i], vel_times[i+1]
        u0 = filt_t[i, 0]; v0 = filt_t[i, 1]
        u1 = filt_t[i+1, 0]; v1 = filt_t[i+1, 1]

        t_cur = t0
        while t_cur < t1 - 1e-8:
            dt = min(dt_adv, t1 - t_cur)
            alpha_t = (t_cur - t0) / (t1 - t0 + 1e-30)
            # Linear temporal interpolation
            u_t = (1-alpha_t)*u0 + alpha_t*u1
            v_t = (1-alpha_t)*v0 + alpha_t*v1
            vel = interp_vel_gpu(pos, u_t, v_t, N_field, L)
            pos = (pos + dt * vel) % L
            t_cur += dt

        pos_hist.append(pos.cpu().numpy().copy())
        t_hist.append(t1)

        if (i+1) % 50 == 0:
            elapsed = time.time() - t_wall0
            eta = elapsed / (i+1) * (len(vel_times)-1 - i - 1)
            print(f"    step {i+1}/{len(vel_times)-1}, wall={elapsed:.0f}s, ETA={eta:.0f}s")

        if t1 > t_end:
            break

    return np.array(pos_hist), np.array(t_hist)

band_results = {}
for band in bands:
    bname = band['name']
    print(f"\n  Band: {bname}")
    t0w = time.time()
    fv  = filtered_vel[bname]
    pos_h, t_h = advect_tracers_in_filtered(fv, vel_times)
    elapsed = time.time() - t0w
    print(f"  Done in {elapsed:.0f}s. Estimating α...")

    a = estimate_alpha_from_trajectories(pos_h, t_h, T_L_mean)
    print(f"  α = {a:.4f}")

    # MSD
    lags  = np.arange(1, min(200, len(t_h)//4))
    msd   = np.array([np.mean(np.sum(
                ((pos_h[l:] - pos_h[:-l] + np.pi) % (2*np.pi) - np.pi)**2,
                axis=-1)) for l in lags])
    dt_h  = t_h[1] - t_h[0]
    tlag  = lags * dt_h
    fit_mask = (tlag > 0.5) & (tlag < 5*T_L_mean) & (msd > 0)
    if fit_mask.sum() > 3:
        from scipy.stats import linregress
        gamma, _, _, _, _ = linregress(np.log(tlag[fit_mask]), np.log(msd[fit_mask]))
    else:
        gamma = np.nan

    band_results[bname] = {
        'alpha': a,
        'gamma': gamma,
        'k0': band['k0'],
        'sigma': band['sigma'],
        'T_eddy': None,  # computed below
        'pos_hist': pos_h,
        't_hist': t_h,
        'msd': msd,
        'tlag': tlag,
    }
    print(f"  γ (MSD exponent) = {gamma:.4f}")

# ── Compute T_eddy from temporal autocorrelation of filtered energy ───────────
print("\nComputing T_eddy for each band...")
for band in bands:
    bname = band['name']
    fv = filtered_vel[bname]  # (499, 2, 128, 128)
    # Energy at each time: mean(u^2 + v^2)
    E_t = np.mean(fv[:,0]**2 + fv[:,1]**2, axis=(-1,-2))  # (499,)
    # Temporal autocorrelation
    E_mean = E_t.mean()
    E_cent = E_t - E_mean
    n = len(E_cent)
    autocorr = np.array([np.mean(E_cent[:n-lag]*E_cent[lag:]) for lag in range(min(200,n//2))])
    if autocorr[0] > 0:
        autocorr /= autocorr[0]
    # Integrate until < 0.1
    below = np.where(autocorr < 0.1)[0]
    if len(below) > 0:
        tau_int = int(below[0])
    else:
        tau_int = len(autocorr)
    T_eddy = np.trapezoid(autocorr[:tau_int+1]) * dt_vel
    band_results[bname]['T_eddy'] = T_eddy
    print(f"  {bname}: T_eddy = {T_eddy:.3f}")

# ── FTLE-based transport barrier analysis ────────────────────────────────────
print("\nComputing FTLE fields (GPU)...")

def compute_ftle(vel_np, vel_times, t_start_idx, dt_ftle_target):
    """
    Compute forward FTLE on a grid using GPU.
    Returns FTLE field (N_field, N_field).
    """
    Ng = N_field
    x = torch.linspace(0, L, Ng+1)[:-1].to(device)
    y = torch.linspace(0, L, Ng+1)[:-1].to(device)
    gx, gy = torch.meshgrid(x, y, indexing='ij')
    pos0 = torch.stack([gx.flatten(), gy.flatten()], dim=1)  # (Ng^2, 2)
    pos  = pos0.clone()

    filt_t = torch.from_numpy(vel_np).to(device)

    t_start = vel_times[t_start_idx]
    t_end   = t_start + dt_ftle_target

    for i in range(t_start_idx, len(vel_times)-1):
        t0, t1 = vel_times[i], vel_times[i+1]
        if t0 >= t_end:
            break
        dt_step = min(dt_vel/5, t1-t0, t_end-t0)
        u0 = filt_t[i,0]; v0 = filt_t[i,1]
        u1 = filt_t[min(i+1,len(vel_times)-1),0]
        v1 = filt_t[min(i+1,len(vel_times)-1),1]

        t_cur = t0
        while t_cur < min(t1, t_end) - 1e-8:
            dt = min(dt_step, min(t1,t_end)-t_cur)
            alpha_t = (t_cur-t0)/(t1-t0+1e-30)
            u_t = (1-alpha_t)*u0 + alpha_t*u1
            v_t = (1-alpha_t)*v0 + alpha_t*v1
            vel = interp_vel_gpu(pos, u_t, v_t, Ng, L)
            pos = (pos + dt*vel) % L
            t_cur += dt

    # Compute Cauchy-Green deformation tensor via finite differences
    pos_np  = pos.cpu().numpy().reshape(Ng, Ng, 2)
    pos0_np = pos0.cpu().numpy().reshape(Ng, Ng, 2)
    dp = (pos_np - pos0_np + np.pi) % (2*np.pi) - np.pi

    dx = dp[:,:,0]
    dy = dp[:,:,1]

    # Jacobian via central differences
    dxdx = np.gradient(dx, axis=0) / (L/Ng)
    dxdy = np.gradient(dx, axis=1) / (L/Ng)
    dydx = np.gradient(dy, axis=0) / (L/Ng)
    dydy = np.gradient(dy, axis=1) / (L/Ng)

    # Cauchy-Green = J^T J
    C11 = dxdx**2 + dydx**2
    C12 = dxdx*dxdy + dydx*dydy
    C22 = dxdy**2 + dydy**2

    # Max eigenvalue
    trace  = C11 + C22
    det    = C11*C22 - C12**2
    disc   = np.sqrt(np.maximum((trace/2)**2 - det, 0))
    lam_max = trace/2 + disc
    lam_max = np.maximum(lam_max, 1e-30)

    dt_actual = abs(min(t_end, vel_times[-1]) - t_start)
    ftle = np.log(np.sqrt(lam_max)) / (dt_actual + 1e-30)
    return ftle

# Compute FTLE at a few time points
dt_ftle = 4 * T_L_mean
ftle_times_idx = [10, 100, 200, 350, 450]
ftle_fields = []
ftle_t      = []
print(f"  Integration time: {dt_ftle:.2f} (4*T_L_mean)")
for idx in ftle_times_idx:
    if idx >= len(vel_times) - 1:
        continue
    ftle = compute_ftle(vel_snaps, vel_times, idx, dt_ftle)
    ftle_fields.append(ftle)
    ftle_t.append(vel_times[idx])
    print(f"  FTLE at t={vel_times[idx]:.1f}: max={ftle.max():.3f}, mean={ftle.mean():.3f}")

ftle_fields = np.array(ftle_fields)

# ── Conditional α near/far from FTLE ridges ───────────────────────────────────
print("\nConditional α analysis near/far from FTLE ridges...")
# Use one mid-point FTLE field
ftle_ref = ftle_fields[len(ftle_fields)//2]  # pick middle time
ftle_thresh = np.percentile(ftle_ref, 75)     # top 25% = ridges

# Find tracers near/far at the corresponding time
ftle_t_ref = ftle_t[len(ftle_t)//2]
# Find nearest tracer time index
t_idx = np.argmin(np.abs(tracer_t - ftle_t_ref))
t_window = 100  # steps around that time

if t_idx > t_window and t_idx + t_window < len(tracer_t):
    pos_window = tracer_pos[t_idx-t_window:t_idx+t_window]  # (200, 5000, 2)
    # Grid-lookup FTLE value at each tracer position (nearest-grid)
    pos_mid = tracer_pos[t_idx]  # (5000, 2)
    ix = (pos_mid[:,0] / L * N_field).astype(int) % N_field
    iy = (pos_mid[:,1] / L * N_field).astype(int) % N_field
    ftle_at_tracer = ftle_ref[ix, iy]

    near_ridge = ftle_at_tracer >= ftle_thresh
    far_ridge  = ftle_at_tracer <  ftle_thresh
    print(f"  Tracers near ridges: {near_ridge.sum()} / far: {far_ridge.sum()}")

    # Displacement over the window for each group
    dx_window = (pos_window[-1] - pos_window[0] + np.pi) % (2*np.pi) - np.pi  # (5000,2)
    alpha_near = (mcculloch_alpha(dx_window[near_ridge, 0]) +
                  mcculloch_alpha(dx_window[near_ridge, 1])) / 2
    alpha_far  = (mcculloch_alpha(dx_window[far_ridge,  0]) +
                  mcculloch_alpha(dx_window[far_ridge,  1])) / 2
    print(f"  α near FTLE ridges: {alpha_near:.4f}")
    print(f"  α far  FTLE ridges: {alpha_far:.4f}")

    # KS test to check if distributions differ
    ks_stat, ks_p = ks_2samp(np.linalg.norm(dx_window[near_ridge],axis=-1),
                              np.linalg.norm(dx_window[far_ridge], axis=-1))
    print(f"  KS test stat={ks_stat:.4f}, p={ks_p:.4e}")
else:
    alpha_near = alpha_far = np.nan
    ks_stat = ks_p = np.nan
    print("  Insufficient data for conditional analysis")

# ── Plots ─────────────────────────────────────────────────────────────────────
print("\nGenerating plots...")
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle('Wavelet Reanalysis — Gaussian Band-Pass Filtering', fontsize=13, fontweight='bold')

# 1. Band-filtered energy spectra (show what each band captures)
ax = axes[0,0]
k_1d = np.arange(1, N_field//2+1)
colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(bands)))
for bi, band in enumerate(bands):
    G = np.exp(-((k_1d - band['k0'])**2) / (2*band['sigma']**2))
    ax.plot(k_1d, G, color=colors[bi], label=f"k0={band['k0']:.0f}", lw=2)
ax.set_xlabel('Wavenumber k'); ax.set_ylabel('Filter amplitude G(k)')
ax.set_title('Gaussian band-pass filters'); ax.legend(fontsize=8)
ax.set_xlim(0, 64); ax.set_xscale('log')

# 2. α vs T_eddy for each band
ax = axes[0,1]
t_eddies = [band_results[b['name']]['T_eddy'] for b in bands]
alphas   = [band_results[b['name']]['alpha']   for b in bands]
k0s      = [b['k0'] for b in bands]
sc = ax.scatter(t_eddies, alphas, c=k0s, cmap='viridis', s=120, zorder=3)
for i, b in enumerate(bands):
    ax.annotate(b['name'].split('(')[0].strip(), (t_eddies[i], alphas[i]),
                textcoords='offset points', xytext=(5,5), fontsize=7)
plt.colorbar(sc, ax=ax, label='k0 (band centre)')
ax.set_xlabel('T_eddy (time units)'); ax.set_ylabel('α (McCulloch)')
ax.set_title('α vs Eddy Lifetime per Scale Band')
ax.axhline(2.0, ls='--', color='red', alpha=0.5, label='Gaussian (α=2)')
ax.set_ylim(0.8, 2.2); ax.legend(fontsize=8)

# 3. MSD for each band
ax = axes[0,2]
for bi, band in enumerate(bands):
    r = band_results[band['name']]
    mask = r['tlag'] < 4*T_L_mean
    ax.loglog(r['tlag'][mask], r['msd'][mask], color=colors[bi],
              label=f"k0={band['k0']:.0f}, γ={r['gamma']:.2f}", lw=1.5)
ax.set_xlabel('Time lag τ'); ax.set_ylabel('MSD')
ax.set_title('Mean Squared Displacement per Band')
ax.legend(fontsize=8)

# 4. FTLE field (mid-time)
ax = axes[1,0]
im = ax.imshow(ftle_fields[len(ftle_fields)//2].T, origin='lower',
               extent=[0,2*np.pi,0,2*np.pi], cmap='hot', aspect='equal')
plt.colorbar(im, ax=ax, label='FTLE')
ax.set_title(f'FTLE field at t={ftle_t[len(ftle_t)//2]:.1f}')
ax.set_xlabel('x'); ax.set_ylabel('y')
# Overlay tracer positions near/far ridges
if not np.isnan(alpha_near):
    pos_mid_np = tracer_pos[t_idx]
    ax.scatter(pos_mid_np[near_ridge,0], pos_mid_np[near_ridge,1],
               s=0.5, c='cyan', alpha=0.3, label=f'near (α={alpha_near:.2f})')
    ax.scatter(pos_mid_np[far_ridge,0],  pos_mid_np[far_ridge,1],
               s=0.5, c='yellow', alpha=0.3, label=f'far (α={alpha_far:.2f})')
    ax.legend(fontsize=7, markerscale=5)

# 5. Displacement PDFs near vs far from FTLE ridges (CCDF log-log)
ax = axes[1,1]
if not np.isnan(alpha_near) and t_idx > t_window:
    disp_near = np.linalg.norm(dx_window[near_ridge], axis=-1)
    disp_far  = np.linalg.norm(dx_window[far_ridge],  axis=-1)
    for disp, label, col, a in [
        (disp_near, f'Near FTLE ridge (α={alpha_near:.2f})', 'blue',  alpha_near),
        (disp_far,  f'Far FTLE ridge (α={alpha_far:.2f})',   'orange',alpha_far),
    ]:
        disp_s = np.sort(disp)
        ccdf   = 1 - np.arange(1, len(disp_s)+1) / len(disp_s)
        mask   = (disp_s > np.percentile(disp_s, 10)) & (ccdf > 0)
        ax.loglog(disp_s[mask], ccdf[mask], label=label, color=col, lw=1.5)
ax.set_xlabel('Displacement |Δr|'); ax.set_ylabel('CCDF P(|Δr|>x)')
ax.set_title('CCDF near vs far from FTLE ridges')
ax.legend(fontsize=8)
if not np.isnan(ks_p):
    ax.text(0.05, 0.05, f'KS p={ks_p:.2e}', transform=ax.transAxes, fontsize=9)

# 6. α per band (bar chart) + comparison with global α
ax = axes[1,2]
band_names_short = [f"k0={b['k0']:.0f}" for b in bands]
bar_alphas = [band_results[b['name']]['alpha'] for b in bands]
global_alpha_mean = float(np.load(BASE+'non_stationarity_results.npz')['alphas'].mean())
bars = ax.bar(band_names_short, bar_alphas, color=colors, edgecolor='k', linewidth=0.8)
ax.axhline(global_alpha_mean, ls='--', color='red', lw=2, label=f'Global α mean = {global_alpha_mean:.2f}')
ax.axhline(2.0, ls=':', color='grey', lw=1, label='Gaussian limit (α=2)')
ax.set_ylabel('α (McCulloch)'); ax.set_ylim(0.8, 2.2)
ax.set_title('α per Scale Band vs Global Result')
ax.legend(fontsize=8)
for bar, val in zip(bars, bar_alphas):
    ax.text(bar.get_x()+bar.get_width()/2, val+0.02, f'{val:.2f}', ha='center', fontsize=9)

plt.tight_layout()
out_path = OUT + 'wavelet_reanalysis.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {out_path}")

# ── Save numerical results ────────────────────────────────────────────────────
results = {
    'band_names':   [b['name'] for b in bands],
    'k0_values':    [b['k0'] for b in bands],
    'alphas':       [band_results[b['name']]['alpha'] for b in bands],
    'gammas':       [band_results[b['name']]['gamma'] for b in bands],
    'T_eddies':     [band_results[b['name']]['T_eddy'] for b in bands],
    'alpha_near_ftle':  float(alpha_near),
    'alpha_far_ftle':   float(alpha_far),
    'ks_stat':      float(ks_stat),
    'ks_pvalue':    float(ks_p),
    'global_alpha_mean': global_alpha_mean,
    'ftle_times':   ftle_t,
}
np.savez(OUT+'wavelet_reanalysis_results.npz', **{k: v for k,v in results.items()
                                                    if not isinstance(v, list) or
                                                    all(isinstance(x,(int,float)) for x in v)})
with open(OUT+'wavelet_reanalysis_summary.json', 'w') as f:
    json.dump({k: (v if not isinstance(v, float) or np.isfinite(v) else 'nan')
               for k,v in results.items()}, f, indent=2, default=str)

print("\n=== WAVELET REANALYSIS SUMMARY ===")
for i, band in enumerate(bands):
    bname = band['name']
    r = band_results[bname]
    print(f"  {bname}:")
    print(f"    α = {r['alpha']:.4f},  γ = {r['gamma']:.4f},  T_eddy = {r['T_eddy']:.3f}")
print(f"\n  Global α mean: {global_alpha_mean:.4f}")
print(f"  α near FTLE ridges: {alpha_near:.4f}")
print(f"  α far  FTLE ridges: {alpha_far:.4f}")
print(f"  KS test p-value: {ks_p:.4e}")
print(f"\nDone. Output in {OUT}")
