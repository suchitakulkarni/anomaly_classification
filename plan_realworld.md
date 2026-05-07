# Real-World Extension: CWRU Bearing Fault Dataset

Goal: validate the physics-informed loss geometry approach on real industrial vibration data,
closing the simulated-data gap and the known-frequency assumption.

Each step below is designed to be independently implementable and testable.

---

## Step 0 — Understand the dataset

**What:** Download and explore the CWRU (Case Western Reserve University) bearing fault dataset.

**Source:** https://engineering.case.edu/bearingdatacenter/download-data-file  
The raw files are `.mat` files containing accelerometer time-series at 12 kHz or 48 kHz.

**Fault classes (maps to your 9 anomaly types):**
- Normal (baseline)
- Inner race fault (3 severities: 0.007", 0.014", 0.021" diameter)
- Outer race fault (3 severities)
- Ball fault (3 severities)
- Combined fault (inner + outer)

**Deliverable:** A notebook or script that loads one `.mat` file, plots the raw signal,
and prints basic statistics (sampling rate, signal length, SNR estimate).  
No model code needed yet.

---

## Step 1 — CWRU data loader

**What:** Write `src/cwru_dataset.py` — a self-contained data loader that:

1. Reads `.mat` files (use `scipy.io.loadmat`)
2. Slices the continuous signal into non-overlapping windows of configurable length
3. Returns `(windows, labels)` where label is one of `{normal, inner, outer, ball}`
4. Splits into train (normal only, for unsupervised LSTM training) / val (normal) / test (all classes)

**Key difference from current `dataset.py`:** no simulation, no anomaly injection — the
faults are baked into the raw recordings.

**Interface to match:**
```python
train_loader, val_loader, test_windows, test_labels, scaler = load_cwru(
    data_dir, window_size=256, batch_size=64
)
```

**Deliverable:** Loader works end-to-end. Unit test: `assert train_labels.unique() == {'normal'}`.

---

## Step 2 — FFT-based dominant frequency estimator

**What:** Write `src/frequency_estimator.py` — replaces the hardcoded `omega` assumption.

**Why this step is independent:** The current physics loss requires knowing `ω`. In the toy
project this was given. On CWRU data, `ω` must be estimated per window from the signal itself.

**Approach:**
```python
def estimate_omega(window: np.ndarray, dt: float) -> float:
    freqs = np.fft.rfftfreq(len(window), d=dt)
    power = np.abs(np.fft.rfft(window)) ** 2
    dominant_freq_hz = freqs[np.argmax(power)]
    return 2 * np.pi * dominant_freq_hz  # radians/s
```

For bearing vibration the dominant frequency is the shaft rotation rate (typically 29.2 Hz
in CWRU at 1750 RPM). Fault frequencies appear as sidebands — the physics loss penalises
deviation from the clean carrier, which is exactly what you want.

**Deliverable:** Function returns `ω` within 5% of the known shaft frequency on a clean
CWRU normal window. Plot FFT spectrum to verify.

---

## Step 3 — Adapt the physics loss for damped oscillation

**What:** Extend `calculate_physics_loss` in `src/utils.py` to support a damping term.

**Why:** Bearing vibration is better modelled as a damped harmonic oscillator:
```
d²x/dt² + 2ζω dx/dt + ω²x = 0
```
The current physics loss uses `d²x/dt² + ω²x = 0` (zero damping). For clean bearings
this is close enough; adding damping makes the residual more sensitive to fault-induced
decay patterns.

**Change:** Add optional `zeta` parameter (default `0.0` for backwards compatibility):
```python
def calculate_physics_loss(reconstruction, omega, dt, scaler, zeta=0.0):
    ...
    dx_dt = (x_phys[:, 1:] - x_phys[:, :-1]) / dt          # first derivative
    residual = d2x_dt2 + 2*zeta*omega * dx_dt[:, :-1] + omega**2 * x_phys[:, 1:-1]
    residual = residual / omega**2
    return torch.mean(residual ** 2)
```

**Deliverable:** Existing toy-project tests still pass with `zeta=0.0`. CWRU physics loss
is lower on normal windows than on fault windows (verify on a handful of windows before
any model training).

---

## Step 4 — Config for CWRU

**What:** Add a `CWRUConfig` class to `src/config.py` (or a new `src/cwru_config.py`)
with CWRU-appropriate hyperparameters.

Key values that will need tuning relative to the current `Config`:
- `WINDOW_SIZE`: 256 or 512 (CWRU sampled at 12 kHz; 256 samples ≈ 21 ms, ~0.6 shaft cycles)
- `DT`: `1 / 12000` (or `1 / 48000` for the DE channel)
- `OMEGA_RANGE`: fixed or narrow range around 2π × 29.2 rad/s
- `NUM_EPOCHS`: likely fewer — real data trains faster
- `PHYSICS_LOSS_WEIGHT`: will need re-tuning; start at `1e-3`

**Deliverable:** `CWRUConfig` importable and used by Steps 5 onward. No model changes yet.

---

## Step 5 — Train LSTM on CWRU normal data

**What:** Run existing `train.py` (with minimal changes) on the CWRU normal-only training
split using the loader from Step 1 and config from Step 4.

**What changes in `train.py`:**
- Accept `omega` per window from the frequency estimator (Step 2), not a global constant
- Pass `zeta` through to the physics loss (Step 3)

**Deliverable:** Two saved models (`results/cwru/model_standard.pt` and
`results/cwru/model_pinn.pt`) trained on normal bearing data. Validation reconstruction
loss should decrease and plateau. Plot training curves.

---

## Step 6 — Threshold calibration on CWRU

**What:** Run existing `threshold.py` on the CWRU validation split (normal windows only)
to fit the Mahalanobis threshold in (log MSE, log physics loss) space.

No code changes needed — this step validates that the existing threshold machinery works
on real data. The main thing to check: are the clean CWRU loss distributions roughly
Gaussian in log space (the assumption underlying Mahalanobis distance)?

**Deliverable:** Plot the 2D scatter of (log MSE, log physics loss) for clean validation
windows with the Mahalanobis ellipse overlaid. If the distribution is clearly non-Gaussian,
flag it — HDBSCAN (already listed as future work) would be the fix.

---

## Step 7 — Evaluate detection on CWRU test set

**What:** Run the detection pipeline on the CWRU test set (all fault classes + normal)
using the thresholds from Step 6.

Compute the same metrics as the toy project: detection rate per fault class, ROC/AUC,
precision/recall. Use existing `detection_metrics.py` — no changes needed.

**Expected behaviour:** Inner race faults are typically the easiest to detect (strong
periodic impulse pattern); ball faults are the hardest (weakest periodic component).

**Deliverable:** Detection evaluation table analogous to `results/detection_evaluation.csv`,
saved to `results/cwru/detection_evaluation.csv`.

---

## Step 8 — Evaluate classification on CWRU

**What:** Run kNN and GMM classifiers in (log MSE, log physics loss) space on detected
anomalous windows, using fault type as the class label.

This is the key generalisation test: does the loss-space geometry separate bearing fault
types the same way it separated anomaly types on the toy data?

Use existing `detection_metrics.py` classification evaluation. The only change: class
labels are `{inner, outer, ball}` instead of the nine toy anomaly types.

**Deliverable:**
- 2D scatter plot of fault windows coloured by fault type (analogous to `2d_scatter_comparison.png`)
- Confusion matrices for kNN and GMM (analogous to `e2e_confusion.png`)
- ARI / NMI table for GMM

---

## Step 9 — Severity sensitivity analysis

**What:** CWRU has three severity levels per fault type (0.007", 0.014", 0.021").
Check whether detection rate and classification confidence increase monotonically with
severity.

**Why this matters for real-world generalisability:** a system that only detects severe
faults is operationally much less useful than one that catches incipient faults.

**Deliverable:** Bar chart of detection rate vs. severity level per fault class.
Expected result: detection rate increases with severity; the physics-informed model
should outperform standard at low severities (where the reconstruction error alone is
marginal but the physics residual is already elevated).

---

## Step 10 — Update README with CWRU results

**What:** Add a "Real-world validation" section to the README documenting:

- Dataset source and setup
- Key differences from the toy setup (FFT frequency estimation, damping term)
- Detection and classification results table
- The severity sensitivity plot
- Honest discussion of where CWRU differs from full production deployment
  (single sensor axis, lab conditions, known fault types)

This step is what makes the project a portfolio artifact rather than a toy — the narrative
arc is now: concept proved on simulation → validated on real benchmark → limitations
clearly stated.

---

## Dependency map

```
Step 0  (explore data)
Step 1  (data loader)        ← independent of Steps 2–4
Step 2  (FFT estimator)      ← independent of Steps 1, 3–4
Step 3  (physics loss)       ← independent of Steps 1–2, 4
Step 4  (config)             ← independent of Steps 1–3

Step 5  (train)              ← needs Steps 1, 2, 3, 4
Step 6  (threshold)          ← needs Step 5
Step 7  (detection eval)     ← needs Step 6
Step 8  (classification)     ← needs Step 7
Step 9  (severity analysis)  ← needs Step 7
Step 10 (README)             ← needs Steps 7, 8, 9
```

Steps 1–4 can all be done in parallel. Steps 5–10 are sequential.
