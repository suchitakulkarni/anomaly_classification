import numpy as np
import os
import logging

logger = logging.getLogger(__name__)


def set_threshold(model, val_loader, omega, dt, scaler, device, window_size,
                  physics_loss_weight=0.004, percentile=99, save_path=None):
    """
    Compute per-window total scores (MSE + lambda*phys) on clean validation data
    and set a detection threshold at the given percentile.

    Also stores per-axis log-space percentile thresholds used for classification.
    """
    from src.evaluate import evaluate_model_with_physics

    mse_vals, phys_vals, _, _ = evaluate_model_with_physics(
        model, val_loader, omega, dt, scaler, device, window_size
    )

    total_scores = mse_vals + physics_loss_weight * phys_vals
    threshold = np.percentile(total_scores, percentile)

    log_mse = np.log10(mse_vals + 1e-10)
    log_phys = np.log10(phys_vals + 1e-10)

    bundle = {
        "threshold": threshold,
        "mse_threshold": np.percentile(log_mse, percentile),
        "phys_threshold": np.percentile(log_phys, percentile),
        "physics_loss_weight": physics_loss_weight,
        "percentile": percentile,
        "all_clean_scores": total_scores,
    }

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, bundle, allow_pickle=True)
        logger.info("Saved threshold bundle to %s", save_path)

    logger.info(
        "Threshold (p%d) = %.6f  |  clean scores: median=%.6f  max=%.6f",
        percentile, threshold, np.median(total_scores), total_scores.max(),
    )

    return bundle


def detect(mse_array, physics_array, threshold_bundle):
    """
    Flag anomalous windows using total score (MSE + lambda*phys).

    Returns:
        bool_mask   – True where the window is flagged as anomalous
        score_array – raw total score per window (used for ROC/AUC)
    """
    lam = threshold_bundle["physics_loss_weight"]
    threshold = threshold_bundle["threshold"]

    score_array = np.asarray(mse_array) + lam * np.asarray(physics_array)
    bool_mask = score_array > threshold

    return bool_mask, score_array


