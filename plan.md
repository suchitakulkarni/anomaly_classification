# Implementation Plan: Unsupervised Threshold-Based Anomaly Classification

## Goal
Transform the current clustering-only evaluation into a fully unsupervised detection +
classification pipeline with confidence scores. Anomaly indices are only used for
post-hoc evaluation metrics (ROC/AUC etc.), never by the system when making decisions.

---

## Step 1 — Restructure anomaly injection to single-anomaly signals
**File: `src/test_suite_runner.py`**

Change `num_anomalies` from 5 to 1 in every `anomaly_configs` entry.

Rationale: multiple anomalies per signal create overlapping windows with ambiguous
labels, making detection metrics meaningless. One anomaly per signal gives a clean
ground truth for evaluation.

Expected result: each test signal has exactly one anomaly event. The `anomaly_indices`
array returned by each injector will have length 1 (or a small contiguous block for
duration-based anomalies).

---

## Step 2 — Fit normal distribution from pretrained models
**New file: `src/threshold.py`**

Use the already-trained models (loaded from `results/saved_models/`) to score the
clean validation windows. This avoids retraining.

### Functions to write:

**`fit_normal_from_clean(model, val_loader, omega, dt, scaler, device) -> (mean, cov, cov_inv)`**
- Run clean validation windows through the model (use `evaluate_model_with_physics`
  from `src/evaluate.py`)
- Collect all (MSE, physics_loss) pairs
- Work in log10 space: `X = [log10(mse + 1e-10), log10(physics + 1e-10)]`
- Fit a 2D Gaussian: compute `mean` (shape [2]) and `cov` (shape [2,2])
- Return mean, cov, and `cov_inv = np.linalg.inv(cov)`

**`mahalanobis_distance(point, mean, cov_inv) -> float`**
- Standard formula: `sqrt((x - mu).T @ cov_inv @ (x - mu))`
- Used as the anomaly score for each window

**`set_threshold(model, val_loader, omega, dt, scaler, device, percentile=99) -> dict`**
- Calls `fit_normal_from_clean` to get mean, cov, cov_inv
- Computes Mahalanobis distance for every clean validation window
- Sets `threshold = np.percentile(distances, percentile)`
- Returns a `threshold_bundle` dict with keys:
  `{mean, cov, cov_inv, threshold, percentile, all_clean_distances}`
- Save this bundle to `results/threshold_bundle_pinn.npy` and `_standard.npy`
  using `np.save(..., allow_pickle=True)`

---

## Step 3 — Detection: flag anomalous windows
**Add to `src/threshold.py`**

**`detect(mse_array, physics_array, threshold_bundle) -> (bool_mask, score_array)`**
- Convert inputs to log10 space
- Compute Mahalanobis distance for each window
- `bool_mask = distances > threshold_bundle['threshold']`
- Return both the mask and the raw distance array (the distance is the anomaly score
  used for ROC/AUC)

---

## Step 4 — Classification: assign anomaly type to flagged windows
**Add to `src/threshold.py`**

Classification is purely geometry-based in log(MSE) × log(physics) space.
No anomaly labels are used. The zones are defined by physics knowledge:

The key quantity is the **direction vector** from the normal cluster center to the
flagged window: `delta = log_point - mean`.

| delta[0] (ΔMSE) | delta[1] (Δphysics) | Anomaly type group |
|---|---|---|
| high | low | MSE-dominant: amplitude spikes, missing data |
| low | high | Physics-dominant: frequency violations, harmonic contamination, phase jumps, DC offset |
| high | high | Both fail: white noise bursts |
| low–medium | medium | Subtle: damping violations, amplitude modulation |

In practice, use the **angle** of `delta` (arctan2) and the **magnitude** (Mahalanobis
distance) to assign the label and compute confidence.

**`classify_flagged_window(mse, physics, threshold_bundle) -> (label, confidence)`**
- Compute `delta = log_point - mean`
- Compute angle `theta = arctan2(delta[1], delta[0])` in degrees
- Map angle to label using fixed boundaries (tune these after inspecting the scatter):
  - `theta < 20°`: MSE-dominant → "amplitude_spike / missing_data"
  - `20° <= theta < 70°`: Both → "white_noise / severe"
  - `theta >= 70°`: Physics-dominant → "frequency / phase / harmonic / dc_offset"
  - Magnitude below threshold: "normal"
- Confidence = `min(1.0, mahalanobis_distance / (2 * threshold))` clipped to [0, 1]
  (scales distance to a 0–1 confidence score; windows far from the boundary get ~1.0)

Note: the angle boundaries above are a starting point. Adjust them by overlaying
radial lines on the 2D scatter plot after running Step 6 visualizations.

---

## Step 5 — Evaluation metrics (anomaly_indices used here only)
**New file: `src/detection_metrics.py`**

These functions compare the system's output to ground truth. The system never sees
`anomaly_indices`; these functions are called externally after the fact.

**`windows_from_indices(anomaly_indices, duration, window_size, total_windows) -> bool_mask`**
- Given the start index/indices of the injected anomaly and its duration, compute
  which windows overlap with the anomaly region
- A window at position `i` covers timesteps `[i, i + window_size)`
- Return a boolean mask of shape `[total_windows]`

**`compute_roc_auc(score_array, true_mask) -> (fpr, tpr, auc)`**
- Use `sklearn.metrics.roc_curve` and `roc_auc_score`
- `score_array` is the Mahalanobis distance per window (higher = more anomalous)
- `true_mask` comes from `windows_from_indices`

**`compute_precision_recall(score_array, true_mask, threshold) -> (precision, recall, f1)`**
- Use `sklearn.metrics.precision_recall_fscore_support` at the chosen threshold

**`compute_detection_latency(bool_mask_predicted, bool_mask_true) -> int`**
- Find the first timestep where `true_mask` is True (anomaly onset)
- Find the first timestep where `predicted_mask` is True after onset
- Latency = difference in window indices
- Return `None` if the anomaly is not detected at all

**`run_detection_evaluation(pinn_results, standard_results, threshold_bundle_pinn,`**
**`    threshold_bundle_std, config) -> DataFrame`**
- For each anomaly type and each model:
  - Compute true window mask from anomaly_indices
  - Compute ROC AUC, precision, recall, F1, detection latency
- Return a tidy DataFrame with columns:
  `[model, anomaly_type, auc, precision, recall, f1, latency_windows]`
- Log the DataFrame and save to `results/detection_evaluation.csv`

---

## Step 6 — Visualizations
**File: `src/visualise.py`**

Add three new plot functions:

**`plot_2d_scatter_with_threshold(pinn_results, std_results, threshold_bundle_pinn,`**
**`    threshold_bundle_std, save_dir)`**
- Extend the existing 2D scatter
- Overlay the 1σ, 2σ, 3σ Mahalanobis ellipses from the fitted normal distribution
- Draw radial lines at the angle boundaries from Step 4 to show classification zones
- Label each zone with its anomaly group name

**`plot_roc_curves(eval_df, save_dir)`**
- One subplot per anomaly type, two lines per plot (PINN vs Standard)
- Title each subplot with the anomaly type and AUC values

**`plot_confusion_matrix(classifications, true_labels, save_dir)`**
- `classifications`: list of predicted labels from `classify_flagged_window`
- `true_labels`: list of true anomaly types (from test suite, known externally)
- Standard seaborn heatmap, normalized by row

---

## Step 7 — Wire everything into main.py

Order of operations in `run_full_test_suite`:

```
1. [existing] Generate clean training data
2. [existing] Load or train both models
3. [NEW] Fit threshold bundles from validation data using pretrained models
         → threshold_bundle_pinn, threshold_bundle_std
4. [existing, modified] Run test suite (single anomaly per signal)
5. [NEW] Run detection + classification on each test result
         using threshold.detect() and threshold.classify_flagged_window()
6. [NEW] Run detection_metrics.run_detection_evaluation()
7. [existing + extended] Generate all visualizations including new plots
```

---

## Step 8 — Update README.md

Add a section describing:
- The unsupervised detection pipeline (threshold set from clean data)
- The physics-knowledge classification zones (with the angle diagram)
- New metrics: AUC per anomaly type, detection latency, classification confidence
- Honest limitation: classification zones are defined by physics intuition, not data,
  so the system cannot classify anomaly types it has no prior knowledge about

---

## Implementation order

Do steps in this order; each is independently testable:

```
Step 1  → run main.py, confirm single-anomaly signals, check scatter looks less crowded
Step 2  → print threshold_bundle, check distances on clean val data look reasonable
Step 3  → print detection results for one anomaly type, sanity check flagged windows
Step 4  → print (label, confidence) for a few flagged windows, inspect against scatter
Step 5  → print the evaluation DataFrame, check AUC > 0.5 for most anomaly types
Step 6  → inspect new plots
Step 7  → full end-to-end run
Step 8  → update README
```
