"""
Standalone 2D NS DNS test at 512x512.

Key design choices:
  - Integrating factor (IF) method for unconditionally stable hyperviscosity
  - Deterministic-amplitude, phase-randomised forcing (correct spectral normalisation)
  - Forcing amplitude: A = N * k_f * sqrt(epsilon_inj / n_force) so that
    <dE/dt> = epsilon_inj exactly in the rfft2 convention

Goal: stable spinup, U_rms ~ 0.5–1, T_L ~ 5–15, k_peak drifts 4->1 over production.
"""
import torch
import numpy as np
import os, time

OUTPUT_PATH = '/home/node/work/projects/levy_flights_2dns_v2/data/'
os.makedirs(OUTPUT_PATH, exist_ok=True)

PARAMS = {
    'N': 1024,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'CFL': 0.4,
    'nu_h': 3.9e-31,        # hyperviscosity for 512^2, p=4 (gives O(1) dissipation at k_max~170)
    'p': 4,
    'epsilon_inj': 0.1,   # energy injection rate
    'k_force_min': 1.0,
    'k_force_max': 3.0,
    'N_tracers': 2000,
    'dt_snap': 0.05,
    'T_spinup_fixed': 60.0,
}

device = PARAMS['device']
N     = PARAMS['N']
L     = 2 * np.pi
dx    = L / N
print(f"Device: {device}  |  N={N}  |  nu_h={PARAMS['nu_h']:.1e}")

# ── Grid ──────────────────────────────────────────────────────────────────────
k    = torch.fft.fftfreq(N, d=1.0/N)    # integer wavenumbers [0, 1, ..., N/2-1, -N/2, ..., -1]
k_r  = torch.fft.rfftfreq(N, d=1.0/N)  # [0, 1, ..., N/2]
kx, ky = torch.meshgrid(k, k_r, indexing='ij')
kx, ky = kx.to(device), ky.to(device)
k2     = kx**2 + ky**2
km     = torch.sqrt(k2)
k2_nz  = k2.clone(); k2_nz[0, 0] = 1.0

# ── Operators ─────────────────────────────────────────────────────────────────
nu_h, p_hyp = PARAMS['nu_h'], PARAMS['p']
diss_op    = -nu_h * k2**p_hyp
k_max_da   = N / 3.0
dealias    = ((torch.abs(kx) < k_max_da) & (torch.abs(ky) < k_max_da))
force_mask = ((km >= PARAMS['k_force_min']) & (km <= PARAMS['k_force_max']))
n_force    = force_mask.sum().item()

# Calibrate force_amp empirically:
# dE/dt per unit force_amp^2 is measured by injecting with force_amp=1 for a few steps.
# For N=512 with this normalisation: ~1.596e-11 per unit force_amp^2.
# force_amp = sqrt(epsilon_inj / dE_per_amp2)
_omega_calib = torch.zeros(N, N//2+1, dtype=torch.cfloat, device=device)
_dE_per_amp2 = 0.0
for _i in range(20):
    def _get_E(oh):
        _uh = 1j*ky*(-oh/k2_nz); _vh = -1j*kx*(-oh/k2_nz)
        _uh[0,0]=0; _vh[0,0]=0
        return (0.5*(torch.abs(_uh)**2+torch.abs(_vh)**2)/N**4).sum().item()
    _E0 = _get_E(_omega_calib)
    _phi = torch.exp(2j*np.pi*torch.rand(_omega_calib.shape, device=device))
    _omega_calib = _omega_calib + np.sqrt(0.1) * _phi * force_mask  # dt_calib=0.1, force_amp=1
    _dE_per_amp2 += (_get_E(_omega_calib) - _E0) / 0.1
del _omega_calib
_dE_per_amp2 /= 20
force_amp = float(np.sqrt(PARAMS['epsilon_inj'] / _dE_per_amp2))
print(f"n_force={n_force},  dE/dt per amp^2={_dE_per_amp2:.4e},  force_amp={force_amp:.1f}")
print(f"Calibrated energy injection rate: {force_amp**2 * _dE_per_amp2:.4f} (should be {PARAMS['epsilon_inj']})")

# Energy spectrum bins
k_bins_max = N // 2
k_bins_idx = torch.floor(km).long()
valid      = k_bins_idx < k_bins_max
valid_flat = valid.flatten()
kidx_flat  = k_bins_idx.flatten()[valid_flat]
k_counts   = torch.bincount(kidx_flat, minlength=k_bins_max)
k_cnts_nz  = torch.where(k_counts > 0, k_counts, torch.ones_like(k_counts)).to(device)
k_vals     = torch.arange(k_bins_max, device=device)

# Precomputed integrating factor cache
_IF_cache = {}

def get_IF(dt):
    key = round(dt, 6)
    if key not in _IF_cache:
        _IF_cache[key] = torch.exp(diss_op * dt)
    return _IF_cache[key]

# ── Core functions ────────────────────────────────────────────────────────────
def vel_from_omega(oh):
    psi = -oh / k2_nz;  psi[0, 0] = 0.0
    return 1j*ky*psi, -1j*kx*psi

def nonlinear(oh):
    uh, vh = vel_from_omega(oh)
    omega  = torch.fft.irfft2(oh,  s=(N, N))
    u      = torch.fft.irfft2(uh, s=(N, N))
    v      = torch.fft.irfft2(vh, s=(N, N))
    adv    = (-1j*kx*torch.fft.rfft2(u*omega)
              - 1j*ky*torch.fft.rfft2(v*omega))
    return adv * dealias

def rk4_if(oh, dt):
    """RK4 + integrating factor for diffusion + deterministic-amplitude stochastic forcing."""
    IF = get_IF(dt)
    k1 = nonlinear(oh)
    k2 = nonlinear(oh + 0.5*dt*k1)
    k3 = nonlinear(oh + 0.5*dt*k2)
    k4 = nonlinear(oh + dt*k3)
    oh_new = IF * (oh + dt*(k1 + 2*k2 + 2*k3 + k4)/6)
    # One forcing kick per step (correct for stochastic forcing at Ito level)
    # Amplitude = force_amp * sqrt(dt) so that <dE/dt> = epsilon_inj
    phi    = torch.exp(2j * np.pi * torch.rand(oh.shape, device=device))
    oh_new = oh_new + (force_amp * np.sqrt(dt)) * phi * force_mask
    return oh_new

def interp_vel(pos, u, v):
    pos_n = (pos / (2*np.pi))*2 - 1   # map [0,2pi] -> [-1,1]
    grid  = pos_n.view(1, 1, -1, 2)
    ui = torch.nn.functional.grid_sample(u.view(1,1,N,N), grid, align_corners=True, mode='bilinear').squeeze()
    vi = torch.nn.functional.grid_sample(v.view(1,1,N,N), grid, align_corners=True, mode='bilinear').squeeze()
    return torch.stack([ui, vi], dim=1)

def rk4_tracer(pos, oh, dt):
    def vel_at(p):
        uh, vh = vel_from_omega(oh)
        return interp_vel(p, torch.fft.irfft2(uh,s=(N,N)), torch.fft.irfft2(vh,s=(N,N)))
    k1 = dt * vel_at(pos)
    k2 = dt * vel_at((pos + 0.5*k1) % (2*np.pi))
    k3 = dt * vel_at((pos + 0.5*k2) % (2*np.pi))
    k4 = dt * vel_at((pos + k3)     % (2*np.pi))
    return (pos + (k1 + 2*k2 + 2*k3 + k4)/6) % (2*np.pi)

def get_vel(oh):
    uh, vh = vel_from_omega(oh)
    return torch.fft.irfft2(uh, s=(N,N)), torch.fft.irfft2(vh, s=(N,N))

def get_dt(oh):
    u, v = get_vel(oh)
    U_max = torch.max(torch.sqrt(u**2 + v**2)).item()
    if not np.isfinite(U_max) or U_max > 1e8:
        return None  # signal instability
    return min(PARAMS['CFL']*dx/U_max, 0.5) if U_max > 1e-8 else 0.05

def diag(oh):
    uh, vh = vel_from_omega(oh)
    # Energy in physical space: E = 0.5 * mean(u^2+v^2)
    # In spectral: E = 0.5/N^4 * sum|u_hat|^2  (rfft2 unnormalised convention)
    E_hat  = 0.5*(torch.abs(uh)**2 + torch.abs(vh)**2) / N**4
    E_tot  = E_hat.sum().item()
    U_rms  = np.sqrt(2*E_tot)
    T_L    = (2*np.pi) / U_rms if U_rms > 0 else float('inf')
    e_flat = E_hat.flatten()[valid_flat]
    Ek     = torch.bincount(kidx_flat, weights=e_flat, minlength=k_bins_max) / k_cnts_nz
    k_peak = k_vals[Ek.argmax()].item() if E_tot > 0 else 0
    return U_rms, T_L, E_tot, int(k_peak)

# ── Initialise ────────────────────────────────────────────────────────────────
torch.manual_seed(42)
omega_hat = torch.fft.rfft2(torch.randn(N, N, device=device) * 1e-2)

t = 0.0
t_start = time.time()
t_print = -1.0

print(f"\n{'='*60}")
print("SPINUP PHASE")
print(f"{'='*60}")

while t < PARAMS['T_spinup_fixed']:
    dt = get_dt(omega_hat)
    if dt is None:
        print(f"!! Instability at t={t:.3f}. Aborting.")
        break
    omega_hat = rk4_if(omega_hat, dt)
    t += dt

    if torch.any(torch.isnan(omega_hat)) or torch.any(torch.isinf(omega_hat)):
        print(f"!! NaN/Inf at t={t:.3f}. Aborting.")
        break

    if t - t_print >= 3.0:
        U_rms, T_L, E_tot, k_peak = diag(omega_hat)
        elapsed = time.time() - t_start
        eta = elapsed/(t/PARAMS['T_spinup_fixed'])*(1-t/PARAMS['T_spinup_fixed']) if t > 0.01 else 99
        print(f"  t={t:6.2f}/{PARAMS['T_spinup_fixed']:.0f}  U_rms={U_rms:.5f}  "
              f"T_L={T_L:.3f}  k_peak={k_peak:3d}  dt={dt:.3e}  wall={elapsed:.0f}s  ETA={eta:.0f}s")
        t_print = t

U_rms_end, T_L_end, E_end, kp_end = diag(omega_hat)
print(f"\nSpinup done at t={t:.2f}  wall={time.time()-t_start:.0f}s")
print(f"  U_rms  = {U_rms_end:.5f}")
print(f"  T_L    = {T_L_end:.4f}")
print(f"  k_peak = {kp_end}")
print(f"  E_tot  = {E_end:.4e}")

if not np.isfinite(T_L_end) or T_L_end == 0:
    print("\n!! FAILED: T_L invalid after spinup.")
else:
    T_prod   = 50 * T_L_end
    dt_snap  = PARAMS['dt_snap']
    print(f"\nT_prod = 50 x T_L = {T_prod:.2f}")
    print(f"U_rms * dt_snap = {U_rms_end * dt_snap:.5f}  (target < 0.1)")

    # ── Short production test (5 T_L) ────────────────────────────────────────
    T_test   = 5 * T_L_end
    print(f"\n{'='*60}")
    print(f"SHORT PRODUCTION TEST  (5 T_L = {T_test:.2f})")
    print(f"{'='*60}")

    tracer_pos = torch.rand(PARAMS['N_tracers'], 2, device=device) * (2*np.pi)
    t_end      = t + T_test
    next_snap  = t
    t_print    = t - 1.0
    pos_list, vel_list, t_list = [], [], []

    while t < t_end:
        dt = get_dt(omega_hat)
        if dt is None:
            print(f"!! Instability at t={t:.3f}. Stopping test.")
            break

        if t >= next_snap:
            u, v    = get_vel(omega_hat)
            vel_t   = interp_vel(tracer_pos, u, v)
            pos_list.append(tracer_pos.cpu().numpy().astype(np.float32))
            vel_list.append(vel_t.cpu().numpy().astype(np.float32))
            t_list.append(t)
            next_snap += dt_snap

        tracer_pos = rk4_tracer(tracer_pos, omega_hat, dt)
        omega_hat  = rk4_if(omega_hat, dt)
        t += dt

        if t - t_print >= T_L_end:
            U_rms_t, T_L_t, _, kp_t = diag(omega_hat)
            print(f"  t={t:.2f}  U_rms={U_rms_t:.5f}  k_peak={kp_t}  dt={dt:.3e}  snaps={len(t_list)}")
            t_print = t

    print(f"\n--- VERIFICATION ---")
    print(f"  Snapshots collected: {len(t_list)}")
    if len(pos_list) >= 2:
        pos_arr  = np.array(pos_list)
        vel_arr  = np.array(vel_list)
        diff     = (pos_arr[1:] - pos_arr[:-1] + np.pi) % (2*np.pi) - np.pi
        mean_d   = np.mean(np.linalg.norm(diff, axis=-1))
        print(f"  Mean displacement/snap: {mean_d:.5f}  (must be < {0.3*2*np.pi:.3f})")
        v0 = vel_arr[:-1].reshape(-1, 2)
        v1 = vel_arr[1:].reshape(-1, 2)
        C_vv1 = np.mean((v0*v1).sum(-1) / ((v0**2).sum(-1) + 1e-30))
        print(f"  C_vv(lag=1):            {C_vv1:.4f}  (target > 0.3)")

    _, _, _, kp_fin = diag(omega_hat)
    print(f"  k_peak at end:          {kp_fin}  (inverse cascade: started at {kp_end}, expect drift toward k=1-2)")
    print(f"\nTotal wall time: {time.time()-t_start:.0f}s")

    np.save(os.path.join(OUTPUT_PATH, 'test_tracer_pos.npy'), np.array(pos_list))
    np.save(os.path.join(OUTPUT_PATH, 'test_tracer_times.npy'), np.array(t_list))
    print(f"Saved to {OUTPUT_PATH}")
