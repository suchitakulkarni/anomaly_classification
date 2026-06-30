"""
src/theoretical_analysis.py

Oracle-approximation verification of anomaly separability in (ΔMSE, Δphysics)
space, as described in Section 4 of the paper.

Approach
--------
No trained model is required. The oracle approximation assumes a model trained
on clean SHO data reconstructs x_clean when given x_anomalous, so:
    MSE ≈ |x_clean − x_anomalous|²

For each of N_SAMPLES signal realisations:
  1. Generate a clean noisy SHO signal (random amplitude, phase, noise seed).
  2. For each anomaly type, inject NUM_ANOMALIES anomalies and compute:
       ΔMSE  = mean_{all windows} |x_clean − x_anom|²        (oracle MSE)
       Δphys = mean_{all windows} (ẍ+ω²x)²|_anom
               − mean_{all windows} (ẍ+ω²x)²|_clean         (physics delta)
  3. Each realisation yields one (ΔMSE, Δphys) point per anomaly type.

Averaging over all windows (not just anomalous ones) is intentional: it gives
the mean elevation above the clean baseline across the whole signal, which is
what the theoretical scaling laws predict for a fixed anomaly density.

The physics residual is NOT normalised by ω² here — we use the raw
    L_phys = mean_t[(ẍ_t + ω²x_t)²]
so that the magnitude directly reflects the physical violation.

Outputs
-------
results/theoretical/theoretical_scatter.png
    Log-log scatter of (ΔMSE, Δphys) across NUM_ANOMALIES realisations per class.
results/theoretical/scaling_laws.png
    Empirical verification of scaling laws: Δphys ~ δφ², Δphys ~ (δω/ω)², etc.
stdout
    Per-class summary statistics and pairwise Mahalanobis distances.

Usage
-----
    python -m src.theoretical_analysis
"""

import logging
import os

import numpy as np
import matplotlib.pyplot as plt

from src.config import Config
from src.dataset import (
    inject_amplitude_modulation,
    inject_amplitude_spikes,
    inject_damping_violations,
    inject_dc_offset_shifts,
    inject_frequency_violations,
    inject_harmonic_contamination,
    inject_missing_data,
    inject_phase_discontinuities,
    inject_white_noise_bursts,
    simulate_harmonic_oscillator,
)

logger = logging.getLogger(__name__)

RESULTS_DIR = os.path.join(Config.RESULTS_DIR, "theoretical")
os.makedirs(RESULTS_DIR, exist_ok=True)

NUM_ANOMALIES = 5   # anomalies injected per signal per class
N_SAMPLES     = 500 # independent signal realisations


# ── Loss computations ───────────────────────────────────────────────────────

def _physics_loss_signal(x, omega, dt, window_size):
    """
    Mean physics loss (ẍ + ω²x)² averaged over all sliding windows.
    Returns a single scalar.
    No ω² normalisation — magnitude directly reflects physical violation.
    """
    omega2 = omega ** 2
    dt2    = dt ** 2
    total  = 0.0
    count  = 0
    for i in range(len(x) - window_size + 1):
        w    = x[i : i + window_size]
        d2x  = (w[2:] - 2.0 * w[1:-1] + w[:-2]) / dt2
        total += np.mean((d2x + omega2 * w[1:-1]) ** 2)
        count += 1
    return total / count if count > 0 else 0.0


def _mse_signal(x_clean, x_anom, window_size):
    """
    Oracle MSE |x_clean − x_anom|² averaged over all sliding windows.
    Returns a single scalar.
    """
    total = 0.0
    count = 0
    for i in range(len(x_clean) - window_size + 1):
        total += np.mean(
            (x_clean[i : i + window_size] - x_anom[i : i + window_size]) ** 2
        )
        count += 1
    return total / count if count > 0 else 0.0


# ── Anomaly configurations ──────────────────────────────────────────────────

def _make_configs(omega, dt, seed):
    """
    Anomaly injection configs, one entry per class.
    seed is varied per sample so anomaly locations differ across realisations.
    mod_freq=0.5 matches dataset.py and test_suite_runner.py.
    """
    return [
        ("amplitude_spikes",
         inject_amplitude_spikes,
         {"num_anomalies": NUM_ANOMALIES, "severity": 5.0, "seed": seed}),

        ("frequency_violations",
         inject_frequency_violations,
         {"num_anomalies": NUM_ANOMALIES, "duration": 20,
          "omega_base": omega, "dt": dt, "seed": seed + 1}),

        ("phase_discontinuities",
         inject_phase_discontinuities,
         {"num_anomalies": NUM_ANOMALIES, "phase_shift_range": (np.pi / 4, np.pi),
          "seed": seed + 2}),

        ("damping_violations",
         inject_damping_violations,
         {"num_anomalies": NUM_ANOMALIES, "duration": 30, "damping_factor": 0.95,
          "dt": dt, "seed": seed + 3}),

        ("missing_data",
         inject_missing_data,
         {"num_anomalies": NUM_ANOMALIES, "gap_duration": 15, "seed": seed + 4}),

        ("white_noise_bursts",
         inject_white_noise_bursts,
         {"num_anomalies": NUM_ANOMALIES, "duration": 20, "noise_amplitude": 1.0,
          "seed": seed + 5}),

        ("harmonic_contamination",
         inject_harmonic_contamination,
         {"num_anomalies": NUM_ANOMALIES, "duration": 30, "omega_base": omega,
          "harmonic_order": 2, "dt": dt, "seed": seed + 6}),

        ("dc_offset_shifts",
         inject_dc_offset_shifts,
         {"num_anomalies": NUM_ANOMALIES, "offset_range": (-1.5, 1.5),
          "seed": seed + 7}),

        ("amplitude_modulation",
         inject_amplitude_modulation,
         {"num_anomalies": NUM_ANOMALIES, "duration": 40, "mod_freq": 2,
          "dt": dt, "seed": seed + 8}),
    ]


# ── Main sampling loop ──────────────────────────────────────────────────────

def compute_delta_distributions(omega, dt, window_size, timesteps, noise_std,
                                n_samples, rng):
    """
    Compute (ΔMSE, Δphys) for each anomaly class over n_samples realisations.

    Each realisation produces ONE point per class: the signal-level average loss
    of the anomalous signal minus the signal-level average of the clean baseline.
    Amplitude and phase are randomised per realisation; anomaly seeds are varied
    so injection locations differ across samples.

    Returns
    -------
    dict : anomaly_type -> {'delta_mse': array(n_samples,),
                            'delta_phys': array(n_samples,)}
    """
    configs_template = _make_configs(omega, dt, seed=0)  # names only
    names = [name for name, *_ in configs_template]
    results = {name: {"delta_mse": [], "delta_phys": []} for name in names}

    for s in range(n_samples):
        amplitude = rng.uniform(2.0, 4.0)
        phase     = rng.uniform(0.0, 2 * np.pi)
        noise_seed = int(rng.integers(0, 2 ** 31))

        x_clean = simulate_harmonic_oscillator(
            timesteps=timesteps, dt=dt, omega=omega,
            amplitude=amplitude, phase=phase,
            noise_std=noise_std, seed=noise_seed,
        )

        # clean baseline losses
        phys_clean = _physics_loss_signal(x_clean, omega, dt, window_size)
        # oracle MSE of clean vs clean is 0 by definition

        seed_s = Config.RANDOM_STATE + s * 10
        for name, inject_fn, kwargs in _make_configs(omega, dt, seed=seed_s):
            x_anom, _ = inject_fn(x_clean, **kwargs)

            delta_mse  = _mse_signal(x_clean, x_anom, window_size)
            delta_phys = _physics_loss_signal(x_anom, omega, dt, window_size) - phys_clean

            results[name]["delta_mse"].append(delta_mse)
            results[name]["delta_phys"].append(delta_phys)

    for name in results:
        results[name]["delta_mse"]  = np.array(results[name]["delta_mse"])
        results[name]["delta_phys"] = np.array(results[name]["delta_phys"])

    return results


# ── Mahalanobis distances ───────────────────────────────────────────────────

def compute_mahalanobis_matrix(results):
    """Pairwise Mahalanobis distance in log (ΔMSE, Δphys) space."""
    # keep only positive deltas (anomaly must raise at least one loss)
    def _log_valid(r):
        m = r["delta_mse"]
        p = r["delta_phys"]
        mask = (m > 0) & (p > 0)
        return np.log10(m[mask] + 1e-12), np.log10(p[mask] + 1e-12)

    keys = list(results.keys())
    n    = len(keys)
    mat  = np.zeros((n, n))

    for i, ki in enumerate(keys):
        lm_i, lp_i = _log_valid(results[ki])
        if len(lm_i) < 3:
            continue
        Xi   = np.column_stack([lm_i, lp_i])
        mu_i = Xi.mean(axis=0)
        ci   = np.cov(Xi.T)
        for j, kj in enumerate(keys):
            if i == j:
                continue
            lm_j, lp_j = _log_valid(results[kj])
            if len(lm_j) < 3:
                continue
            Xj   = np.column_stack([lm_j, lp_j])
            mu_j = Xj.mean(axis=0)
            cj   = np.cov(Xj.T)
            pooled = (ci + cj) / 2
            try:
                inv_p = np.linalg.inv(pooled)
                diff  = mu_i - mu_j
                mat[i, j] = float(np.sqrt(diff @ inv_p @ diff))
            except np.linalg.LinAlgError:
                mat[i, j] = float(np.linalg.norm(mu_i - mu_j))

    return mat, keys


def print_summary(results):
    print("\n── Per-class statistics ──")
    for name, r in results.items():
        m = r["delta_mse"]
        p = r["delta_phys"]
        mask = (m > 0) & (p > 0)
        print(f"  {name:28s}  "
              f"ΔMSE={m[mask].mean():.4e} ± {m[mask].std():.4e}  "
              f"Δphys={p[mask].mean():.4e} ± {p[mask].std():.4e}  "
              f"(n={mask.sum()})")

    mat, keys = compute_mahalanobis_matrix(results)
    n     = len(keys)
    upper = mat[np.triu_indices(n, k=1)]
    print(f"\nOracle avg pairwise Mahalanobis (log space): {upper.mean():.3f}  "
          f"std {upper.std():.3f}")


# ── Scaling law verification ────────────────────────────────────────────────

def _sho_euler(amplitude, phase, omega, dt, timesteps):
    """Noiseless SHO by Euler integration (matches simulate_harmonic_oscillator)."""
    x = np.zeros(timesteps)
    v = np.zeros(timesteps)
    x[0] = amplitude * np.cos(phase)
    v[0] = -amplitude * omega * np.sin(phase)
    for t in range(1, timesteps):
        a    = -omega ** 2 * x[t - 1]
        v[t] = v[t - 1] + a * dt
        x[t] = x[t - 1] + v[t] * dt
    return x


def verify_scaling_laws(omega, dt, window_size, n_samples, rng):
    """
    Sweep δφ and δω/ω and check predicted scaling of Δphys and ΔMSE.

    Predicted (Section 4 of paper):
      Phase jump:   Δphys ~ δφ²,          ΔMSE ~ δφ²
      Freq drift:   Δphys ~ (δω/ω)²,      ΔMSE ~ (δω·N·dt)²

    Uses noiseless SHO so small-anomaly scaling is not masked by noise floor.
    Anomaly signals are constructed analytically (exact control over δφ, δω).
    """
    timesteps       = 500
    delta_phi_vals  = np.linspace(np.pi / 8, np.pi, 14)
    delta_omega_rel = np.linspace(0.05, 0.80, 14)  # δω / ω

    x_ref        = _sho_euler(3.0, 0.5, omega, dt, timesteps)
    baseline_phy = _physics_loss_signal(x_ref, omega, dt, window_size)

    phys_phase, mse_phase = [], []
    for dphi in delta_phi_vals:
        pp, mp = [], []
        for _ in range(n_samples):
            A   = rng.uniform(2.0, 4.0)
            phi = rng.uniform(0, 2 * np.pi)
            t0  = int(rng.integers(window_size, timesteps - window_size))

            xc = _sho_euler(A, phi, omega, dt, timesteps)
            xa = xc.copy()
            t_arr = np.arange(timesteps) * dt
            xa[t0:] = A * np.cos(omega * t_arr[t0:] + phi + dphi)

            pp.append(_physics_loss_signal(xa, omega, dt, window_size) - baseline_phy)
            mp.append(_mse_signal(xc, xa, window_size))
        phys_phase.append(np.mean(pp))
        mse_phase.append(np.mean(mp))

    phys_freq, mse_freq = [], []
    for dw_rel in delta_omega_rel:
        domega = dw_rel * omega
        pp, mp = [], []
        for _ in range(n_samples):
            A   = rng.uniform(2.0, 4.0)
            phi = rng.uniform(0, 2 * np.pi)
            t0  = int(rng.integers(window_size, timesteps // 2))
            dur = window_size

            xc = _sho_euler(A, phi, omega, dt, timesteps)
            xa = xc.copy()
            t_arr = np.arange(timesteps) * dt
            xa[t0 : t0 + dur] = A * np.cos((omega + domega) * t_arr[t0 : t0 + dur] + phi)

            pp.append(_physics_loss_signal(xa, omega, dt, window_size) - baseline_phy)
            mp.append(_mse_signal(xc, xa, window_size))
        phys_freq.append(np.mean(pp))
        mse_freq.append(np.mean(mp))

    return (
        delta_phi_vals, np.array(phys_phase), np.array(mse_phase),
        delta_omega_rel, np.array(phys_freq), np.array(mse_freq),
        omega, window_size, dt,
    )


# ── Plotting ────────────────────────────────────────────────────────────────

STYLE = {
    "amplitude_spikes":       {"color": "#e74c3c", "label": "Amplitude Spikes"},
    "frequency_violations":   {"color": "#f39c12", "label": "Frequency Violations"},
    "phase_discontinuities":  {"color": "#2ecc71", "label": "Phase Discontinuities"},
    "damping_violations":     {"color": "#9b59b6", "label": "Damping Violations"},
    "missing_data":           {"color": "#95a5a6", "label": "Missing Data"},
    "white_noise_bursts":     {"color": "#1abc9c", "label": "White Noise Bursts"},
    "harmonic_contamination": {"color": "#e91e63", "label": "Harmonic Contamination"},
    "dc_offset_shifts":       {"color": "#00bcd4", "label": "DC Offset Shifts"},
    "amplitude_modulation":   {"color": "#f1c40f", "label": "Amplitude Modulation"},
}


def _log_ellipse(ax, log_x, log_y, n_std, color, ls):
    """
    1σ/2σ contour drawn as a parametric curve in log space, then mapped to
    linear axes. Correct for log-log plots (unlike matplotlib Ellipse patches).
    """
    if len(log_x) < 4:
        return
    X        = np.column_stack([log_x, log_y])
    mu       = X.mean(axis=0)
    eigvals, eigvecs = np.linalg.eigh(np.cov(X.T))
    eigvals  = np.maximum(eigvals, 1e-12)

    theta    = np.linspace(0, 2 * np.pi, 200)
    unit     = np.column_stack([np.cos(theta), np.sin(theta)])
    boundary = mu + (unit * (n_std * np.sqrt(eigvals))) @ eigvecs.T
    ax.plot(10 ** boundary[:, 0], 10 ** boundary[:, 1],
            color=color, linestyle=ls, linewidth=1.5, zorder=3)


def plot_scatter(results, omega, noise_std, window_size, save_path):
    fig, ax = plt.subplots(figsize=(9, 7))

    for name, style in STYLE.items():
        if name not in results:
            continue
        m = results[name]["delta_mse"]
        p = results[name]["delta_phys"]
        mask = (m > 0) & (p > 0)
        if mask.sum() < 2:
            continue

        lm = np.log10(m[mask] + 1e-12)
        lp = np.log10(p[mask] + 1e-12)

        ax.scatter(10 ** lm, 10 ** lp,
                   color=style["color"], label=style["label"],
                   alpha=0.5, s=20, linewidths=0, zorder=2)
        _log_ellipse(ax, lm, lp, n_std=1, color=style["color"], ls="-")
        _log_ellipse(ax, lm, lp, n_std=2, color=style["color"], ls="--")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\Delta$MSE  (oracle: $\|x_{\rm clean} - x_{\rm anom}\|^2$, "
                  r"all windows)", fontsize=11)
    ax.set_ylabel(r"$\Delta$Physics Loss  "
                  r"$\langle(\ddot{x}+\omega^2 x)^2\rangle_{\rm anom}"
                  r" - \langle(\ddot{x}+\omega^2 x)^2\rangle_{\rm clean}$",
                  fontsize=11)
    ax.set_title(
        rf"Theoretical anomaly separation (oracle, $\omega={omega}$, "
        rf"noise ${int(noise_std*100)}\%$, $N={window_size}$, "
        rf"{NUM_ANOMALIES} anomalies/signal)",
        fontsize=11,
    )
    ax.legend(fontsize=8, loc="upper left", ncol=2, framealpha=0.85)
    ax.grid(True, alpha=0.2, which="both")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved {save_path}")


def plot_scaling(delta_phi, phys_phase, mse_phase,
                 delta_omega_rel, phys_freq, mse_freq,
                 omega, window_size, dt, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    def _panel(ax, x, y, xlabel, ylabel, title, color, pred_label):
        ax.plot(x, y, "o", color=color, markersize=6, zorder=3, label="observed")
        slope = float(np.dot(x, y) / np.dot(x, x))
        x_fit = np.linspace(0, x.max(), 100)
        ax.plot(x_fit, slope * x_fit, "k--", lw=1.2,
                label=f"linear fit (slope={slope:.3g})")
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f"{title}\n[predicted: {pred_label}]", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    _panel(axes[0, 0], delta_phi**2, phys_phase,
           r"$\delta\phi^2$", r"$\Delta\mathcal{L}_{\rm phys}$",
           "Phase jump — physics loss", "#2ecc71", r"$\propto\delta\phi^2$")

    _panel(axes[0, 1], delta_phi**2, mse_phase,
           r"$\delta\phi^2$", r"$\Delta\mathcal{L}_{\rm MSE}$",
           "Phase jump — MSE", "#2ecc71", r"$\propto\delta\phi^2$")

    _panel(axes[1, 0], delta_omega_rel**2, phys_freq,
           r"$(\delta\omega/\omega)^2$", r"$\Delta\mathcal{L}_{\rm phys}$",
           "Frequency drift — physics loss", "#f39c12",
           r"$\propto(\delta\omega/\omega)^2$")

    _panel(axes[1, 1], (delta_omega_rel * window_size * dt)**2, mse_freq,
           r"$(\delta\omega\cdot N\cdot\Delta t)^2$",
           r"$\Delta\mathcal{L}_{\rm MSE}$",
           "Frequency drift — MSE", "#f39c12",
           r"$\propto(\delta\omega\cdot N\cdot\Delta t)^2$")

    fig.suptitle(
        "Scaling law verification (oracle, noiseless SHO, exact δφ and δω control)",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved {save_path}")


# ── Window-size / anomaly-duration sensitivity ──────────────────────────────

def _physics_per_window(x, omega, dt, window_size):
    """
    Return an array of per-window physics losses (ẍ + ω²x)² — one scalar
    per sliding window.  The mean of this array equals _physics_loss_signal,
    but the per-window distribution is what the detector actually thresholds.
    """
    omega2 = omega ** 2
    dt2    = dt ** 2
    losses = []
    for i in range(len(x) - window_size + 1):
        w = x[i : i + window_size]
        d2x = (w[2:] - 2.0 * w[1:-1] + w[:-2]) / dt2
        losses.append(np.mean((d2x + omega2 * w[1:-1]) ** 2))
    return np.array(losses)


def window_size_sensitivity(omega, dt, noise_std, n_samples, rng, save_path):
    """
    Demonstrates why window size N and anomaly duration d interact, using
    the correct quantity for detection: the per-window physics loss.

    Key insight
    -----------
    The MEAN of L_phys over all windows equals the time-average of R²
    regardless of N (the N's cancel in the double average).  Window size
    only matters for DETECTION, which thresholds a single window's physics
    loss.  The relevant quantity is therefore the peak (or 99th-percentile)
    per-window physics, not the signal-level mean.

    For a boundary spike (phase jump, effective duration ~ 1 timestep):
      peak_window_phys ∝ spike_magnitude / (N − 2)  ∝  1/N
      because the spike sits in one interior point, diluted over N − 2.

    For a persistent anomaly (freq. drift, duration d ≫ N):
      every window is fully anomalous → peak_window_phys ≈ L_bulk (const in N).

    Panel (a) — peak per-window physics vs N
        Both anomaly types decrease ∝ 1/N.  This is the correct result:
        inject_frequency_violations creates hard discontinuities at its
        segment endpoints; the resulting boundary spike is diluted over
        N − 2 interior points in the containing window, so peak ∝ 1/N.
        The same mechanism operates for the phase-jump spike.
        A truly boundary-free persistent anomaly (e.g. wrong-frequency
        signal with no abrupt onset) would show a flat peak — but that is
        not what the injection functions produce.

    Panel (b) — peak per-window physics vs anomaly duration d (N fixed)
        All anomalies have boundary spikes whose contribution to the peak
        is approximately independent of d (the spike size at the onset
        doesn't change with how long the anomaly lasts).  The peak is
        therefore roughly constant across d, dominated by the onset spike.

    Panel (c) — detection margin (peak / threshold) vs N
        Because both signal and threshold decrease with N, the margin
        (peak / threshold) follows the ratio: signal ∝ 1/N, threshold
        also falls (cleaner estimate with more averaging), so the margin
        decreases but less steeply than 1/N.

    Panel (d) — why the signal-level MEAN is flat (contrast with peak):
        Mean L_phys = time-average of R²; the N's cancel in the double
        sum.  This panel juxtaposes the flat mean with the declining peak
        to make the mean-vs-peak distinction concrete.
    """
    timesteps     = 1000
    win_sizes     = np.array([10, 15, 20, 25, 30, 40, 50, 70, 100])
    N_fixed       = 30
    dur_sweep     = np.array([5, 10, 15, 20, 30, 45, 60, 90, 120])
    dur_fixed_val = 120   # a clearly persistent anomaly

    # ── Sweep N ─────────────────────────────────────────────────────────────
    # NOTE: inject_frequency_violations creates hard discontinuities at both
    # endpoints of the injected segment.  These boundary spikes dominate the
    # per-window maximum and scale as 1/N (spike diluted over N interior
    # points) — exactly like a phase jump.  This is NOT a code artefact; it
    # is the correct physics: any signal discontinuity of magnitude Δx
    # contributes Δx/dt² to the finite-difference residual at the boundary
    # window, and that residual is averaged over N − 2 interior points →
    # peak per-window phys ∝ 1/N.  The flat-vs-1/N contrast would only
    # appear for a truly boundary-free persistent anomaly (e.g. wrong
    # frequency for the entire signal with no abrupt onset).
    spike_peak_mean, spike_peak_std = [], []
    spike_mean_mean = []            # signal-level mean (should be flat)
    persist_peak_mean, persist_peak_std = [], []
    persist_mean_mean = []
    threshold_mean = []             # 99th pct of clean per-window physics

    for N in win_sizes:
        sp_peaks, sp_means = [], []
        pv_peaks, pv_means = [], []
        thresh_vals = []

        for _ in range(n_samples):
            A    = rng.uniform(2.0, 4.0)
            phi  = rng.uniform(0, 2 * np.pi)
            seed = int(rng.integers(0, 2 ** 31))

            x_clean = simulate_harmonic_oscillator(
                timesteps=timesteps, dt=dt, omega=omega,
                amplitude=A, phase=phi, noise_std=noise_std, seed=seed,
            )

            # Clean: per-window distribution → threshold
            clean_pw = _physics_per_window(x_clean, omega, dt, N)
            thresh_vals.append(np.percentile(clean_pw, 99))

            # Spike: single phase jump, fixed δφ = π/4
            x_sp, _ = inject_phase_discontinuities(
                x_clean, num_anomalies=1,
                phase_shift_range=(np.pi / 4, np.pi / 4), seed=seed + 1,
            )
            sp_pw = _physics_per_window(x_sp, omega, dt, N)
            sp_peaks.append(np.max(sp_pw))
            sp_means.append(np.mean(sp_pw))

            # Persistent: frequency violation (duration = dur_fixed_val).
            # Also has boundary spikes → peak also ∝ 1/N (see note above).
            x_pv, _ = inject_frequency_violations(
                x_clean, num_anomalies=1, duration=dur_fixed_val,
                omega_base=omega, dt=dt, seed=seed + 2,
            )
            pv_pw = _physics_per_window(x_pv, omega, dt, N)
            pv_peaks.append(np.max(pv_pw))
            pv_means.append(np.mean(pv_pw))

        spike_peak_mean.append(np.mean(sp_peaks))
        spike_peak_std.append(np.std(sp_peaks))
        spike_mean_mean.append(np.mean(sp_means))

        persist_peak_mean.append(np.mean(pv_peaks))
        persist_peak_std.append(np.std(pv_peaks))
        persist_mean_mean.append(np.mean(pv_means))

        threshold_mean.append(np.mean(thresh_vals))

    spike_peak_mean   = np.array(spike_peak_mean)
    spike_peak_std    = np.array(spike_peak_std)
    spike_mean_mean   = np.array(spike_mean_mean)
    persist_peak_mean = np.array(persist_peak_mean)
    persist_peak_std  = np.array(persist_peak_std)
    persist_mean_mean = np.array(persist_mean_mean)
    threshold_mean    = np.array(threshold_mean)

    # ── Sweep d (N = N_fixed): boundary-injected anomaly of varying length ──
    # Here we deliberately use inject_frequency_violations so the duration
    # sweep makes sense. The boundary spikes will dominate for d < N, but
    # for d >> N the bulk physics takes over — that's the point of this panel.
    dur_peak_mean, dur_peak_std, dur_thresh = [], [], []
    for d in dur_sweep:
        pk_vals, th_vals = [], []
        for _ in range(n_samples):
            A    = rng.uniform(2.0, 4.0)
            phi  = rng.uniform(0, 2 * np.pi)
            seed = int(rng.integers(0, 2 ** 31))
            x_clean = simulate_harmonic_oscillator(
                timesteps=timesteps, dt=dt, omega=omega,
                amplitude=A, phase=phi, noise_std=noise_std, seed=seed,
            )
            clean_pw = _physics_per_window(x_clean, omega, dt, N_fixed)
            th_vals.append(np.percentile(clean_pw, 99))
            x_pv, _ = inject_frequency_violations(
                x_clean, num_anomalies=1, duration=int(d),
                omega_base=omega, dt=dt, seed=seed + 1,
            )
            pv_pw = _physics_per_window(x_pv, omega, dt, N_fixed)
            pk_vals.append(np.max(pv_pw))
        dur_peak_mean.append(np.mean(pk_vals))
        dur_peak_std.append(np.std(pk_vals))
        dur_thresh.append(np.mean(th_vals))

    dur_peak_mean = np.array(dur_peak_mean)
    dur_thresh    = np.array(dur_thresh)

    # ── Plotting ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    ref_N = win_sizes.astype(float)
    idx30 = np.argmin(np.abs(win_sizes - N_fixed))

    # Panel a: peak per-window physics vs N + threshold
    ax = axes[0, 0]
    ax.semilogy(win_sizes, spike_peak_mean, "s-", color="crimson",
                label=f"phase jump — peak (d≈1)")
    ax.semilogy(win_sizes, persist_peak_mean, "o-", color="steelblue",
                label=f"freq. drift — peak (d={dur_fixed_val})")
    ax.semilogy(win_sizes, threshold_mean, "k--", lw=1.5,
                label="clean 99th-pct threshold")
    # 1/N reference anchored at N=30
    ax.semilogy(ref_N, spike_peak_mean[idx30] * N_fixed / ref_N, "r:",
                alpha=0.6, lw=1.5, label=r"$\propto 1/N$")
    ax.axvline(N_fixed, color="gray", lw=0.8, ls=":")
    ax.set_xlabel("Window size $N$", fontsize=11)
    ax.set_ylabel("Peak per-window $\\mathcal{L}_{\\rm phys}$", fontsize=11)
    ax.set_title("(a) Peak physics signal vs $N$\n"
                 r"(spike $\propto 1/N$; persistent $\approx$ const)",
                 fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    # Panel b: peak per-window physics vs anomaly duration d
    ax = axes[0, 1]
    ax.errorbar(dur_sweep, dur_peak_mean, yerr=dur_peak_std,
                fmt="o-", color="steelblue", capsize=3,
                label=f"freq. drift peak (N={N_fixed})")
    ax.axhline(np.mean(dur_thresh), color="k", lw=1.5, ls="--",
               label="clean threshold (99th pct)")
    ax.axvline(N_fixed, color="gray", lw=0.8, ls=":",
               label=f"N={N_fixed}")
    ax.set_xlabel("Anomaly duration $d$ (timesteps)", fontsize=11)
    ax.set_ylabel("Peak per-window $\\mathcal{L}_{\\rm phys}$", fontsize=11)
    ax.set_title(f"(b) Peak physics signal vs $d$  ($N$={N_fixed})\n"
                 r"Detection requires peak $>$ threshold",
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Panel c: detection margin (peak / threshold) vs N
    ax = axes[1, 0]
    spike_margin   = spike_peak_mean   / threshold_mean
    persist_margin = persist_peak_mean / threshold_mean
    ax.semilogy(win_sizes, spike_margin,   "s-", color="crimson",
                label="phase jump (spike)")
    ax.semilogy(win_sizes, persist_margin, "o-", color="steelblue",
                label=f"freq. drift (d={dur_fixed_val})")
    ax.axhline(1.0, color="k", lw=1, ls="--", alpha=0.5,
               label="detection threshold = 1")
    ax.semilogy(ref_N, spike_margin[idx30] * N_fixed / ref_N, "r:",
                alpha=0.6, lw=1.5, label=r"$\propto 1/N$")
    ax.axvline(N_fixed, color="gray", lw=0.8, ls=":")
    ax.set_xlabel("Window size $N$", fontsize=11)
    ax.set_ylabel("Detection margin  (peak / threshold)", fontsize=11)
    ax.set_title("(c) Detection margin vs $N$\n"
                 r"Spike: $\propto 1/N$; persistent: $\approx$ const",
                 fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    # Panel d: signal-level MEAN vs N (flat — contrasts with peak)
    ax = axes[1, 1]
    ax.semilogy(win_sizes, spike_mean_mean, "s--", color="crimson", alpha=0.7,
                label="phase jump — mean (flat)")
    ax.semilogy(win_sizes, persist_mean_mean, "o--", color="steelblue", alpha=0.7,
                label=f"freq. drift — mean (flat)")
    ax.semilogy(win_sizes, spike_peak_mean, "s-", color="crimson",
                label="phase jump — peak (∝ 1/N)")
    ax.semilogy(win_sizes, persist_peak_mean, "o-", color="steelblue",
                label=f"freq. drift — peak")
    ax.axvline(N_fixed, color="gray", lw=0.8, ls=":")
    ax.set_xlabel("Window size $N$", fontsize=11)
    ax.set_ylabel("$\\mathcal{L}_{\\rm phys}$ (per-signal mean or peak)", fontsize=11)
    ax.set_title("(d) Mean vs peak: why mean is flat\n"
                 r"Mean = time-avg of $R^2$ (N cancels); peak $\propto 1/N$",
                 fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    fig.suptitle(
        rf"Window size $N$ vs anomaly duration $d$  "
        rf"($\omega={omega}$, noise={int(noise_std*100)}%, $\Delta t={dt}$)",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved {save_path}")


# ── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)

    rng = np.random.default_rng(Config.RANDOM_STATE)

    OMEGA     = Config.TEST_OMEGA   # 2.86
    DT        = Config.DT           # 0.01
    WINDOW    = Config.WINDOW_SIZE  # 30
    TIMESTEPS = Config.TIMESTEPS    # 1000
    NOISE     = Config.TEST_NOISE   # 0.1

    print(f"Step 1/4  Computing delta distributions "
          f"({N_SAMPLES} samples × 9 classes, "
          f"{NUM_ANOMALIES} anomalies/signal)...")
    results = compute_delta_distributions(
        omega=OMEGA, dt=DT, window_size=WINDOW,
        timesteps=TIMESTEPS, noise_std=NOISE,
        n_samples=N_SAMPLES, rng=rng,
    )

    print("Step 2/4  Plotting scatter...")
    plot_scatter(results, omega=OMEGA, noise_std=NOISE, window_size=WINDOW,
                 save_path=os.path.join(RESULTS_DIR, "theoretical_scatter.png"))

    print("Step 3/4  Summary statistics and Mahalanobis distances...")
    print_summary(results)

    print("\nStep 4/4  Verifying scaling laws (100 samples, noiseless)...")
    scaling = verify_scaling_laws(
        omega=OMEGA, dt=DT, window_size=WINDOW, n_samples=100, rng=rng,
    )
    plot_scaling(*scaling,
                 save_path=os.path.join(RESULTS_DIR, "scaling_laws.png"))

    print("\nStep 5/5  Window-size / anomaly-duration sensitivity (100 samples)...")
    window_size_sensitivity(
        omega=OMEGA, dt=DT, noise_std=NOISE,
        n_samples=100, rng=rng,
        save_path=os.path.join(RESULTS_DIR, "window_size_sensitivity.png"),
    )

    print(f"\nDone. Results saved to {RESULTS_DIR}/")
