import logging
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_fscore_support

logger = logging.getLogger(__name__)


def windows_from_indices(anomaly_indices, duration, window_size, total_windows):
    """
    Build a boolean window mask marking every window that overlaps an anomaly region.

    Window i covers signal timesteps [i, i + window_size).
    Each anomaly region covers [s, s + duration) for s in anomaly_indices.
    """
    mask = np.zeros(total_windows, dtype=bool)
    for s in np.atleast_1d(anomaly_indices):
        s = int(s)
        lo = max(0, s - window_size + 1)
        hi = min(total_windows, s + duration)
        if lo < hi:
            mask[lo:hi] = True
    return mask


def compute_roc_auc(score_array, true_mask):
    """
    Compute ROC curve and AUC.

    Returns (fpr, tpr, auc) or (None, None, None) when only one class is present.
    """
    if true_mask.sum() == 0 or (~true_mask).sum() == 0:
        logger.warning("compute_roc_auc: true_mask is constant; AUC is undefined.")
        return None, None, None
    fpr, tpr, _ = roc_curve(true_mask.astype(int), score_array)
    auc = roc_auc_score(true_mask.astype(int), score_array)
    return fpr, tpr, auc


def compute_precision_recall(score_array, true_mask, threshold):
    """
    Threshold score_array and return precision, recall, and F1.

    Returns (precision, recall, f1) or (0, 0, 0) when the mask is constant.
    """
    if true_mask.sum() == 0:
        return 0.0, 0.0, 0.0
    predicted = (score_array > threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(
        true_mask.astype(int), predicted, average="binary", zero_division=0
    )
    return float(p), float(r), float(f1)


def compute_detection_latency(bool_mask_predicted, bool_mask_true):
    """
    Windows between anomaly onset and first detection.

    Returns None if the anomaly is never detected.
    """
    onset_indices = np.where(bool_mask_true)[0]
    if len(onset_indices) == 0:
        return None
    onset = int(onset_indices[0])

    detection_indices = np.where(bool_mask_predicted[onset:])[0]
    if len(detection_indices) == 0:
        return None
    return int(detection_indices[0])


def run_detection_evaluation(
    pinn_results,
    standard_results,
    threshold_bundle_pinn,
    threshold_bundle_std,
    config,
):
    """
    Compute detection metrics for every anomaly type and both models.

    anomaly_indices in each AnomalyTestResult are used here *only* as ground-truth
    labels — the models never saw them.

    Returns a tidy DataFrame saved to results/detection_evaluation.csv.
    """
    from src.threshold import detect

    rows = []

    anomaly_types = [k for k in pinn_results if k != "baseline"]

    for anomaly_type in anomaly_types:
        for model_tag, results, bundle in (
            ("physics_informed", pinn_results, threshold_bundle_pinn),
            ("standard", standard_results, threshold_bundle_std),
        ):
            result = results[anomaly_type]
            mse_vals = result.mse_values
            phy_vals = result.physics_values

            total_windows = len(mse_vals)

            true_mask = windows_from_indices(
                result.anomaly_indices,
                result.anomaly_duration,
                config.WINDOW_SIZE,
                total_windows,
            )

            bool_mask, score_array = detect(mse_vals, phy_vals, bundle)

            _, _, auc = compute_roc_auc(score_array, true_mask)
            precision, recall, f1 = compute_precision_recall(
                score_array, true_mask, bundle["threshold"]
            )
            latency = compute_detection_latency(bool_mask, true_mask)

            rows.append(
                {
                    "model": model_tag,
                    "anomaly_type": anomaly_type,
                    "auc": auc,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "latency_windows": latency,
                }
            )

            logger.info(
                "%s | %s — AUC=%.3f  P=%.3f  R=%.3f  F1=%.3f  latency=%s",
                model_tag, anomaly_type,
                auc if auc is not None else float("nan"),
                precision, recall, f1,
                str(latency),
            )

    df = pd.DataFrame(rows, columns=["model", "anomaly_type", "auc", "precision",
                                      "recall", "f1", "latency_windows"])

    save_path = os.path.join(config.RESULTS_DIR, "detection_evaluation.csv")
    df.to_csv(save_path, index=False)
    logger.info("Saved detection evaluation to %s", save_path)

    return df


def run_e2e_classification(pinn_results, standard_results,
                           threshold_bundle_pinn, threshold_bundle_std, config,
                           classifier_pinn=None, classifier_std=None, micro_order=None):
    """
    End-to-end pipeline evaluation: detect anomalous windows unsupervised, then
    classify each flagged window.

    When classifier_pinn / classifier_std (fitted sklearn classifiers) and
    micro_order (list of class name strings) are supplied, kNN classification
    is used.  Otherwise falls back to the angle-based classify_flagged_window.

    For every ground-truth anomalous window:
      - if detect() flags it  → predicted label from classifier or angle heuristic
      - if detect() misses it → predicted label = "normal"  (missed detection)

    Returns a dict keyed by model tag, each with lists of true and predicted labels.
    """
    from src.threshold import detect, classify_flagged_window

    out = {}
    for model_tag, results, bundle, clf in [
        ("physics_informed", pinn_results, threshold_bundle_pinn, classifier_pinn),
        ("standard",         standard_results, threshold_bundle_std, classifier_std),
    ]:
        true_labels, pred_labels = [], []
        for anomaly_type, result in results.items():
            if anomaly_type == "baseline":
                continue
            total_windows = len(result.mse_values)
            true_mask = windows_from_indices(
                result.anomaly_indices, result.anomaly_duration,
                config.WINDOW_SIZE, total_windows,
            )
            bool_mask, _ = detect(result.mse_values, result.physics_values, bundle)
            for i in np.where(true_mask)[0]:
                true_labels.append(anomaly_type)
                if bool_mask[i]:
                    if clf is not None and micro_order is not None:
                        log_point = np.array([[
                            np.log10(float(result.mse_values[i]) + 1e-10),
                            np.log10(float(result.physics_values[i]) + 1e-10),
                        ]])
                        pred = micro_order[int(clf.predict(log_point)[0])]
                    else:
                        pred, _ = classify_flagged_window(
                            result.mse_values[i], result.physics_values[i], bundle
                        )
                else:
                    pred = "normal"
                pred_labels.append(pred)

        out[model_tag] = {"true_labels": true_labels, "pred_labels": pred_labels}
        detected = sum(p != "normal" for p in pred_labels)
        classifier_name = "kNN" if clf is not None else "angle heuristic"
        logger.info(
            "e2e [%s] (%s): %d / %d true-anomalous windows detected (%.1f%%)",
            model_tag, classifier_name, detected, len(pred_labels),
            100 * detected / len(pred_labels) if pred_labels else 0,
        )

    return out
