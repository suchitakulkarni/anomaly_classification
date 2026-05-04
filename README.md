# Anomaly classification using physics inspired LSTM

Most anomaly detection methods aim to detect the presence of an anomaly.
This project goes one step beyond and explores how to detect the cause of the anomaly.
We assume that a clean signal is available at the time of training.
The project is executed in the domain of time-series analysis and uses the LSTM architecture.

---
![Anomaly classification in 2D MSE -- physics loss plane](https://github.com/suchitakulkarni/anomaly_classification/blob/main/results/2d_scatter_comparison.png)


## Core idea

The central idea is as follows. An anomaly originating from a physical cause,
e.g., a glitch in the amplitude of oscillation, breaks the underlying physics
model differently than a damped amplitude. If this change in the amount and the
type of physics loss can be captured, one can identify the anomaly type
e.g. amplitude damping vs. glitch.

# Dataset
The framework uses simulated simple harmonic oscillator data as a controlled testbed for testing the formalism.

---

## Pipeline

The system runs a two-stage pipeline on each test signal:

**Stage 1 — Unsupervised anomaly detection**

A 2D Gaussian is fitted to the (log MSE, log physics loss) pairs of clean validation windows.
Each incoming window is scored by its Mahalanobis distance from this normal cluster.
A window is flagged as anomalous when its distance exceeds the 99th-percentile threshold
derived from clean data — no anomaly labels are used at this stage.

**Stage 2 — Supervised anomaly type classification**

Flagged windows are passed to a k-Nearest Neighbours (kNN, k=5) classifier trained in the
(log MSE, log physics loss) feature space. The kNN assigns each flagged window a fine-grained
anomaly type label (9 classes: amplitude spike, damping, frequency shift, phase jump, etc.).

The end-to-end result is shown below: rows are true anomaly types, columns are predicted labels,
and the "normal" column represents missed detections.

![End-to-end detect → classify confusion matrix](https://github.com/suchitakulkarni/anomaly_classification/blob/main/results/e2e_confusion.png)

The scatter below shows only the anomalous windows, overlaid with the Mahalanobis ellipses
fitted to the clean distribution and radial lines marking the angle-classifier zones.

![2D scatter of anomalous windows with threshold ellipses](https://github.com/suchitakulkarni/anomaly_classification/blob/main/results/2d_scatter_with_threshold.png)

---

## Key findings

- The physics informed approach performs better than the data driven approach in presence of noise.
- Adding physics constraints includes a tradeoff between reconstruction accuracy and physics compliance.
- The physics informed approach obtained 57% improvements over the data driven approach in terms of anomaly separation.
- Lower physics weight leads to better detection of anomalies.
- Unsupervised detection (Mahalanobis threshold) achieves high recall; classification accuracy varies by anomaly type and is higher for the physics-informed model.
- The kNN classifier trained on (log MSE, log physics loss) separates fine-grained anomaly types that are geometrically clustered in the 2D feature space.

---

## File structure

```
project
  main.py
  requirements.txt            -- Python dependencies
  src/
  |__ config.py
  |__ model.py
  |__ dataset.py
  |__ visualise.py
  |__ utils.py
  |__ evaluate.py
  |__ train.py
  |__ test_suite_runner.py
  |__ quantitative_metrics.py
  |__ threshold.py            -- Mahalanobis threshold fitting and detection
  |__ detection_metrics.py    -- ROC/AUC, precision/recall, e2e classification eval
  results/                    -- all output CSVs and PNGs
  |__ saved_models
  |__ rest of the plots
```

---

## Run order

```bash
python main.py
```
* Currently ```main.py``` will not retrain the model; to do so set the ```train_again``` flag to ```True``` in ```main.py```.

---
## Dependencies

```
pandas
scikit-learn
numpy
matplotlib
torch
```

---

## Design decisions

**Why LSTM?**
Very good for keeping track of long sequences.

**Why Mahalanobis distance for detection?**
It accounts for the correlation between MSE and physics loss in log space,
giving a single scalar anomaly score without requiring labelled anomalies at training time.

**Why kNN for classification?**
The anomaly types form geometrically coherent clusters in the 2D (log MSE, log physics loss) space.
kNN is a natural, interpretable choice that requires no additional hyperparameter tuning beyond k.

**Known limitation**
The kNN classifier generalises only to anomaly types seen during training; it cannot classify
anomaly types with no prior examples in the (log MSE, log physics loss) feature space.

---
