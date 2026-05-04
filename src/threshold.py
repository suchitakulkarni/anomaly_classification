import numpy as np
import os
import logging

logger = logging.getLogger(__name__)


def fit_normal_from_clean(model, val_loader, omega, dt, scaler, device, window_size):
    """Fit a 2D Gaussian over (log10 MSE, log10 physics loss) from clean validation windows."""
    from src.evaluate import evaluate_model_with_physics

    mse_vals, phys_vals, _, _ = evaluate_model_with_physics(
        model, val_loader, omega, dt, scaler, device, window_size
    )

    log_mse = np.log10(mse_vals + 1e-10)
    log_phys = np.log10(phys_vals + 1e-10)
    X = np.stack([log_mse, log_phys], axis=1)  # shape [N, 2]

    mean = X.mean(axis=0)
    cov = np.cov(X, rowvar=False)
    cov_inv = np.linalg.inv(cov)

    return mean, cov, cov_inv


def mahalanobis_distance(point, mean, cov_inv):
    """Mahalanobis distance between a 2D point and the fitted normal distribution."""
    delta = np.asarray(point) - mean
    return float(np.sqrt(delta @ cov_inv @ delta))


def set_threshold(model, val_loader, omega, dt, scaler, device, window_size,
                  percentile=99, save_path=None):
    """
    Fit normal distribution from clean validation data, compute per-window distances,
    and set threshold at the given percentile.

    Returns a threshold_bundle dict and optionally saves it to save_path.
    """
    mean, cov, cov_inv = fit_normal_from_clean(
        model, val_loader, omega, dt, scaler, device, window_size
    )

    from src.evaluate import evaluate_model_with_physics
    mse_vals, phys_vals, _, _ = evaluate_model_with_physics(
        model, val_loader, omega, dt, scaler, device, window_size
    )

    log_mse = np.log10(mse_vals + 1e-10)
    log_phys = np.log10(phys_vals + 1e-10)
    points = np.stack([log_mse, log_phys], axis=1)

    distances = np.array([mahalanobis_distance(p, mean, cov_inv) for p in points])
    threshold = np.percentile(distances, percentile)

    bundle = {
        "mean": mean,
        "cov": cov,
        "cov_inv": cov_inv,
        "threshold": threshold,
        "percentile": percentile,
        "all_clean_distances": distances,
    }

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, bundle, allow_pickle=True)
        logger.info("Saved threshold bundle to %s", save_path)

    logger.info(
        "Threshold (p%d) = %.4f  |  mean=[%.4f, %.4f]  |  clean distances: "
        "median=%.4f  max=%.4f",
        percentile, threshold, mean[0], mean[1],
        np.median(distances), distances.max(),
    )

    return bundle


def detect(mse_array, physics_array, threshold_bundle):
    """
    Flag anomalous windows using Mahalanobis distance in log10 space.

    Returns:
        bool_mask   – True where the window is flagged as anomalous
        score_array – raw Mahalanobis distance per window (used for ROC/AUC)
    """
    mean = threshold_bundle["mean"]
    cov_inv = threshold_bundle["cov_inv"]
    threshold = threshold_bundle["threshold"]

    log_mse = np.log10(np.asarray(mse_array) + 1e-10)
    log_phys = np.log10(np.asarray(physics_array) + 1e-10)
    points = np.stack([log_mse, log_phys], axis=1)

    score_array = np.array([mahalanobis_distance(p, mean, cov_inv) for p in points])
    bool_mask = score_array > threshold

    return bool_mask, score_array


def classify_flagged_window(mse, physics, threshold_bundle):
    """
    Classify a single window into an anomaly type using geometry in log10 space.

    The angle of the direction vector from the normal cluster center determines
    the anomaly group; confidence scales linearly with Mahalanobis distance.

    Returns:
        label      – string anomaly type (or "normal" if below threshold)
        confidence – float in [0, 1]
    """
    mean = threshold_bundle["mean"]
    cov_inv = threshold_bundle["cov_inv"]
    threshold = threshold_bundle["threshold"]

    log_point = np.array([
        np.log10(float(mse) + 1e-10),
        np.log10(float(physics) + 1e-10),
    ])

    dist = mahalanobis_distance(log_point, mean, cov_inv)

    if dist <= threshold:
        return "normal", 0.0

    delta = log_point - mean
    theta = np.degrees(np.arctan2(delta[1], delta[0]))

    if theta < 20.0:
        label = "amplitude_spike/missing_data"
    elif theta < 70.0:
        label = "white_noise/severe"
    else:
        label = "frequency/phase/harmonic/dc_offset"

    confidence = float(np.clip(dist / (2.0 * threshold), 0.0, 1.0))

    return label, confidence
