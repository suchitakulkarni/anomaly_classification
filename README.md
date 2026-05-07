# Anomaly Classification via Physics-Informed Loss Geometry

## Summary

This project studies how physics-informed losses shape representation space in time-series models.
In a noisy and variable setting, anomaly types become separable in a 2D loss space defined by
reconstruction error and physics violation, enabling simple classification after unsupervised detection.

---

![Anomaly classification in 2D MSE -- physics loss plane](https://github.com/suchitakulkarni/anomaly_classification/blob/main/results/2d_scatter_comparison.png)

---

## What this demonstrates

* Unsupervised anomaly detection without labeled anomalies
* Physics-informed training induces **structure in representation space**
* This structure enables **anomaly type classification with simple models**
* Separation persists under:
  * noisy signals (~10% noise)
  * variable anomaly strength and duration
  * varying signal parameters (frequency, amplitude, phase)

---

## Problem setting

Most anomaly detection systems answer:

> "Is this anomalous?"

This project explores:

> "What *kind* of anomaly is this?"

This is a harder question. Anomalies are not binary events — a frequency drift behaves differently from
an amplitude spike in how it violates the underlying physics, and those differences leave a geometric
signature in the model's loss space. Exploiting that signature enables type-level classification without
any labeled anomalies at training time.

The evaluation uses simulated physical systems, where anomaly type, strength, and duration are all
controlled, allowing precise analysis of how different faults manifest in model behavior.

---

## Core idea

Two complementary signals are extracted from each time window:

* **Reconstruction loss (MSE):** measures data fidelity — how well the model reconstructs the signal
* **Physics loss:** measures deviation from known system dynamics — how much the signal violates the governing equation

Each window is mapped to a 2D feature vector:

```
z = (log MSE, log Physics Loss)
```

Different anomaly types occupy distinct regions of this space. A frequency drift raises physics loss
strongly; an amplitude spike raises MSE; damping affects both in a characteristic ratio. This geometry
is the classifier — the downstream model is intentionally simple to make that clear.

---

## Pipeline

### Stage 1 — Unsupervised detection

* Train LSTM on clean signals (no anomaly labels)
* Compute (log MSE, log physics loss) for each window
* Fit Mahalanobis distance in this 2D loss space on clean training data
* Flag windows exceeding the 99th-percentile threshold as anomalous

### Stage 2 — Anomaly type classification

Two classifiers are evaluated on detected anomalies, using the same 2D loss features:

**Supervised:** k-Nearest Neighbours (k=5) trained on labeled anomaly examples

**Unsupervised:** Gaussian Mixture Model (GMM) fitted without any anomaly labels, evaluated
via precision-coverage tradeoff as confidence threshold is swept

---

## Experimental setup

### Data

* Simulated simple harmonic oscillator
* ~10% additive noise
* Training data varies across frequency, amplitude, and phase to avoid trivial overfitting to a single signal

### Anomalies

Nine anomaly types, each with **variable strength and duration**:

| Class | Description |
|---|---|
| Amplitude spikes | Sudden jump in signal magnitude |
| Amplitude modulation | Gradual envelope variation |
| Damping violations | Unphysical decay or growth |
| Frequency violations | Drift from nominal frequency |
| Phase discontinuities | Instantaneous phase jumps |
| Harmonic contamination | Added harmonic components |
| White noise bursts | Localized broadband noise |
| DC offset shifts | Baseline level changes |
| Missing data | Signal dropout |

Variable strength and duration mean anomalies span a range of severities — weak, short anomalies
approach the noise floor, making them genuinely hard to detect and separate from clean variation.

---

## Baseline

To isolate the contribution of physics constraints, two LSTM variants are compared:

**Standard LSTM** — trained with reconstruction loss only (MSE)

**Physics-informed LSTM (PINN)** — trained with combined reconstruction + physics loss

Both use the same downstream pipeline. Any performance difference is attributable to the loss geometry
induced by the physics term.

---

## Results

### Detection

Mahalanobis thresholding achieves high recall without any labeled anomalies. The physics-informed model
significantly outperforms the standard model on detection rate across most anomaly classes.

| Model | Detection rate (e2e) |
|---|---|
| Physics-informed kNN | 0.77 |
| Standard kNN | 0.56 |
| Physics-informed GMM | 0.68 |
| Standard GMM | 0.49 |

![End-to-end detect → classify confusion matrix (kNN)](https://github.com/suchitakulkarni/anomaly_classification/blob/main/results/e2e_confusion.png)

### Classification (supervised kNN)

kNN in 2D loss space separates anomaly types with meaningful accuracy. Physics-informed features
improve both detection and fine-grained classification over the standard model.

* Physics-informed micro-class accuracy: **61%**
* Standard micro-class accuracy: **52%**
* 57% improvement in geometric cluster separation (Silhouette / Davies-Bouldin)

![Micro-class confusion matrix (kNN)](https://github.com/suchitakulkarni/anomaly_classification/blob/main/results/micro_class_confusion.png)

### Classification (unsupervised GMM)

As an unsupervised alternative, a GMM is fitted in the same 2D space without any anomaly labels.
Performance is measured via precision-coverage tradeoff (precision on classified windows as confidence
threshold is swept) and compared to the kNN flat-line baseline.

| Model | ARI | NMI | V-measure |
|---|---|---|---|
| Physics-informed GMM | 0.42 | 0.59 | 0.59 |
| Standard GMM | 0.31 | 0.51 | 0.51 |

The physics-informed GMM approaches kNN precision at high confidence thresholds (top ~30% of windows
by GMM posterior confidence), at the cost of leaving ~30% of detected anomalies unclassified.
Standard GMM degrades more sharply, reflecting weaker cluster geometry without physics constraints.

The gap relative to kNN is largest for spectrally similar anomalies (harmonic contamination, white
noise bursts) whose loss-space signatures overlap under Gaussian assumptions.

![GMM precision-coverage vs. kNN baseline](https://github.com/suchitakulkarni/anomaly_classification/blob/main/results/gmm_precision_coverage.png)
![End-to-end confusion matrix (GMM)](https://github.com/suchitakulkarni/anomaly_classification/blob/main/results/e2e_confusion_gmm.png)

### Representation quality

The 2D scatter below shows anomalous windows overlaid with Mahalanobis ellipses and angle-classifier zones.
Cluster coherence is substantially clearer for the physics-informed model.

![2D scatter with threshold ellipses](https://github.com/suchitakulkarni/anomaly_classification/blob/main/results/2d_scatter_with_threshold.png)

---

## Key insight

The model is not trained to classify anomaly types.

Instead:

> Physics-informed loss shapes the geometry of the loss space, making anomaly types separable —
> both for supervised and fully unsupervised classifiers.

This allows simple downstream models to succeed without complex feature engineering. The GMM
result is particularly telling: meaningful structure is recoverable with zero labels, and the gap
to supervised kNN is largest precisely where the 2D space is insufficient (spectrally similar classes),
not where the clustering approach itself fails.

---

## Design choices

**Why LSTM?**
Captures temporal dependencies; the physics loss is computed on the reconstructed sequence, so
temporal coherence matters.

**Why Mahalanobis distance?**
Models correlation between the two loss components and gives a single anomaly score without
requiring labeled anomalies at training time.

**Why kNN?**
Used intentionally to demonstrate that performance comes from representation structure, not
classifier complexity. A more complex classifier would obscure that point.

**Why GMM as unsupervised classifier?**
Tests whether the geometric structure induced by physics-informed training is strong enough for
purely unsupervised type discovery — no labels anywhere in the pipeline.

---

## Assumptions and limitations

* Physics loss assumes access to system frequency (can be estimated via FFT in practice)
* kNN classifier generalizes only to anomaly types seen during training
* GMM assumes roughly Gaussian cluster shapes; spectrally similar classes share loss geometry
  and are not cleanly separable under this assumption
* Evaluation is on simulated data; real sensor signals would introduce additional non-stationarity

---

## Future work

* **HDBSCAN for unsupervised classification** — handles non-Gaussian cluster shapes and naturally
  produces an uncertain class for ambiguous windows, giving better-calibrated abstentions than GMM
* **Additional features for spectral anomalies** — a frequency-domain feature (e.g. spectral entropy
  of the residual) as a third axis would likely break the harmonic/noise degeneracy
* **Calibrated GMM confidence** — temperature scaling on GMM posteriors to improve precision-coverage
  curve shape and make abstention decisions more trustworthy
* **Robustness to frequency estimation error** — evaluate effect of using FFT-estimated vs. ground-truth frequency in the physics loss
* **Real-world sensor datasets** — industrial vibration or ECG data as next validation target

---

## File structure

```
project/
  main.py
  requirements.txt
  src/
    config.py
    model.py
    dataset.py
    train.py
    evaluate.py
    threshold.py          — Mahalanobis threshold fitting and detection
    detection_metrics.py  — ROC/AUC, precision/recall, e2e classification eval
    quantitative_metrics.py
    test_suite_runner.py
    visualise.py
    utils.py
  results/
    saved_models/
    *.png                 — all output plots
    *.csv                 — numeric evaluation tables
```

---

## How to run

```bash
python main.py
```

Set `train_again=True` in `main.py` to retrain the model from scratch.

---

## Dependencies

```
numpy
pandas
scikit-learn
matplotlib
torch
```
