import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os, sys, logging

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
from torch.utils.data import DataLoader, TensorDataset

from src.utils import setup_logging
from src.config import Config
from src.model import LSTMAutoencoder
from src.train import train_model_with_validation
from src.dataset import simulate_harmonic_oscillator, prepare_data, build_multi_segment_dataset
from src.utils import set_all_seeds, seed_worker
from src.test_suite_runner import compare_models_on_anomalies, run_anomaly_test_suite
from src.visualise import (create_comprehensive_report, plot_2d_scatter_with_threshold,
                           plot_roc_curves, plot_macro_class_confusion,
                           plot_micro_class_confusion, plot_e2e_confusion,
                           plot_score_distributions, plot_training_diagnostics,
                           plot_gmm_precision_coverage, plot_multi_freq_auc)
from src.quantitative_metrics import run_full_quantitative_analysis
from src.threshold import set_threshold
from src.detection_metrics import run_detection_evaluation, windows_from_indices, run_e2e_classification
from torch.serialization import add_safe_globals
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

def load_model(checkpoint_path, device):

    add_safe_globals([StandardScaler])

    ckpt = torch.load(
    checkpoint_path,
    map_location=device,
    weights_only=False
    )
    assert "model_state_dict" in ckpt
    assert "scaler" in ckpt
    assert "config" in ckpt

    cfg = ckpt["config"]
    model = LSTMAutoencoder(
        seq_len=cfg["window_size"],
        hidden_dim=cfg["hidden_dim"]
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, ckpt["scaler"], cfg

def train_both_models(config, train_loader, val_loader, device, scaler, mlflow_logging=False):
    """
    Trains both physics-informed and standard models.
    
    Returns:
        Tuple of (model_pinn, model_standard)
    """
    logger.info("=== Training Physics-Informed Model ===")
    set_all_seeds(config.RANDOM_STATE)
    
    model_pinn = LSTMAutoencoder(
        seq_len=config.WINDOW_SIZE,
        hidden_dim=config.HIDDEN_DIM
    ).to(device)
    
    optimizer_pinn = torch.optim.Adam(model_pinn.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.MSELoss(reduction='none')
    
    model_pinn, history_pinn, best_epoch_pinn = train_model_with_validation(
        model=model_pinn,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer_pinn,
        num_epochs=config.NUM_EPOCHS,
        physics_loss_weight=config.PHYSICS_LOSS_WEIGHT,
        dt=config.DT,
        device=device,
        scaler=scaler,
        stop_early=True,
        mlflow_prefix="pinn" if mlflow_logging else None,
    )
    
    logger.info(f"Physics-Informed trained. Best epoch: {best_epoch_pinn}")
    
    logger.info("=== Training Standard Model ===")
    set_all_seeds(config.RANDOM_STATE)
    
    model_standard = LSTMAutoencoder(
        seq_len=config.WINDOW_SIZE,
        hidden_dim=config.HIDDEN_DIM
    ).to(device)
    
    optimizer_standard = torch.optim.Adam(model_standard.parameters(), 
                                         lr=config.LEARNING_RATE)
    
    model_standard, history_standard, best_epoch_standard = train_model_with_validation(
        model=model_standard,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer_standard,
        num_epochs=config.NUM_EPOCHS,
        physics_loss_weight=0.0,
        dt=config.DT,
        device=device,
        scaler=scaler,
        stop_early=True,
        mlflow_prefix="std" if mlflow_logging else None,
    )
    
    logger.info(f"Standard trained. Best epoch: {best_epoch_standard}")

    model_dir = os.path.join(config.RESULTS_DIR, "saved_models")
    os.makedirs(model_dir, exist_ok=True)

    save_path = os.path.join(model_dir, f"lstm_autoencoder_standard.pt")

    torch.save({
        "model_state_dict": model_standard.state_dict(),
        "scaler": scaler,
        "config": {
            "window_size": config.WINDOW_SIZE,
            "hidden_dim": config.HIDDEN_DIM,
            "omega_range": config.OMEGA_RANGE,
            "test_omega": config.TEST_OMEGA,
            "dt": config.DT,
        }
    }, save_path)

    logger.info(f"Saved model checkpoint to {save_path}")

    save_path = os.path.join(model_dir, f"lstm_autoencoder_pinn.pt")

    torch.save({
        "model_state_dict": model_pinn.state_dict(),
        "scaler": scaler,
        "config": {
            "window_size": config.WINDOW_SIZE,
            "hidden_dim": config.HIDDEN_DIM,
            "omega_range": config.OMEGA_RANGE,
            "test_omega": config.TEST_OMEGA,
            "dt": config.DT,
            "physics_weight": config.PHYSICS_LOSS_WEIGHT,
        }
    }, save_path)

    logger.info(f"Saved model checkpoint to {save_path}")

    return model_pinn, model_standard, history_pinn, history_standard


def make_calibration_loader(omega, config, scaler, device):
    """Return a DataLoader of clean windows at a fixed omega for threshold calibration."""
    x_cal = simulate_harmonic_oscillator(
        timesteps=config.N_CAL_TIMESTEPS,
        dt=config.DT,
        omega=omega,
        noise_std=config.TEST_NOISE,
        seed=config.RANDOM_STATE + 200,
        amplitude=3.0,
        phase=0.5,
    )
    X_np, _, _ = prepare_data(x_cal, config.WINDOW_SIZE, scaler=scaler)
    X_tensor = torch.from_numpy(X_np).float()
    omega_tensor = torch.full((len(X_tensor),), omega, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, X_tensor, omega_tensor)
    return DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
    )


def run_multi_eval(config, model_pinn, model_standard, scaler, device):
    """
    Multi-frequency / multi-seed quantitative evaluation.

    For each omega in config.TEST_OMEGAS and each of config.NUM_TEST_SEEDS seeds:
      - Calibrates per-omega thresholds on a held-out clean signal.
      - Injects anomalies at varying positions (controlled by seed_offset).
      - Collects detection AUC, precision, recall, F1 per anomaly type.

    Saves:
      multi_eval_all_seeds.csv    — one row per (model, omega, seed, anomaly_type)
      multi_eval_aggregated.csv   — mean ± std over seeds per (model, omega, anomaly_type)

    Returns: (agg_df, all_results_df)
    """
    records = []

    for omega in config.TEST_OMEGAS:
        logger.info("=== Multi-eval: ω=%.2f ===", omega)

        cal_loader = make_calibration_loader(omega, config, scaler, device)
        tb_pinn = set_threshold(
            model_pinn, cal_loader, omega, config.DT, scaler, device,
            config.WINDOW_SIZE,
            physics_loss_weight=config.PHYSICS_LOSS_WEIGHT,
        )
        tb_std = set_threshold(
            model_standard, cal_loader, omega, config.DT, scaler, device,
            config.WINDOW_SIZE,
            physics_loss_weight=0.0,
        )

        for seed_idx in range(config.NUM_TEST_SEEDS):
            seed_offset = seed_idx * 100
            x_clean = simulate_harmonic_oscillator(
                timesteps=config.TIMESTEPS,
                dt=config.DT,
                omega=omega,
                noise_std=config.TEST_NOISE,
                seed=config.RANDOM_STATE + 100 + seed_idx,
                amplitude=3.0,
                phase=0.5,
            )

            pinn_res = run_anomaly_test_suite(
                config, model_pinn, x_clean, scaler, device,
                model_name="Physics-Informed", omega=omega, dt=config.DT,
                seed_offset=seed_offset, verbose=False,
            )
            std_res = run_anomaly_test_suite(
                config, model_standard, x_clean, scaler, device,
                model_name="Standard", omega=omega, dt=config.DT,
                seed_offset=seed_offset, verbose=False,
            )

            eval_df = run_detection_evaluation(
                pinn_results=pinn_res,
                standard_results=std_res,
                threshold_bundle_pinn=tb_pinn,
                threshold_bundle_std=tb_std,
                config=config,
                save=False,
                verbose=False,
            )
            eval_df['omega'] = omega
            eval_df['seed'] = seed_idx
            records.append(eval_df)

        logger.info("  ω=%.2f complete (%d seeds)", omega, config.NUM_TEST_SEEDS)

    all_results = pd.concat(records, ignore_index=True)

    agg = (
        all_results
        .groupby(['model', 'omega', 'anomaly_type'], sort=False)
        .agg(
            auc_mean=('auc', 'mean'),
            auc_std=('auc', 'std'),
            precision_mean=('precision', 'mean'),
            precision_std=('precision', 'std'),
            recall_mean=('recall', 'mean'),
            f1_mean=('f1', 'mean'),
        )
        .reset_index()
    )

    agg.to_csv(os.path.join(config.RESULTS_DIR, 'multi_eval_aggregated.csv'), index=False)
    all_results.to_csv(os.path.join(config.RESULTS_DIR, 'multi_eval_all_seeds.csv'), index=False)
    logger.info("Saved multi-eval results to %s", config.RESULTS_DIR)

    return agg, all_results


def run_full_test_suite(config: Config, train_again=False):
    """
    Main function to run comprehensive anomaly detection test suite.
    """
    if MLFLOW_AVAILABLE:
        mlflow.set_experiment(config.MLFLOW_EXPERIMENT)
        run_name = (
            f"seg{config.NUM_SEGMENTS}_ts{config.TIMESTEPS_PER_SEGMENT}"
            f"_hd{config.HIDDEN_DIM}_plw{config.PHYSICS_LOSS_WEIGHT}"
        )
        mlflow.start_run(run_name=run_name)
        mlflow.log_params({
            "num_segments":            config.NUM_SEGMENTS,
            "timesteps_per_segment":   config.TIMESTEPS_PER_SEGMENT,
            "window_size":             config.WINDOW_SIZE,
            "batch_size":              config.BATCH_SIZE,
            "hidden_dim":              config.HIDDEN_DIM,
            "num_epochs":              config.NUM_EPOCHS,
            "learning_rate":           config.LEARNING_RATE,
            "physics_loss_weight":     config.PHYSICS_LOSS_WEIGHT,
            "omega_range":             str(config.OMEGA_RANGE),
            "noise_range":             str(config.NOISE_RANGE),
            "test_omega":              config.TEST_OMEGA,
            "dt":                      config.DT,
            "random_state":            config.RANDOM_STATE,
        })
        logger.info("MLflow run started: %s", run_name)

    set_all_seeds(config.RANDOM_STATE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: %s", device)

    g = torch.Generator()
    g.manual_seed(config.RANDOM_STATE)

    n_train_seg = int(config.NUM_SEGMENTS * 0.8)
    n_val_seg = config.NUM_SEGMENTS - n_train_seg

    logger.info("=== 1. Generating Multi-Frequency Training Data ===")
    X_train_np, omega_train_np, scaler = build_multi_segment_dataset(
        num_segments=n_train_seg,
        timesteps_per_segment=config.TIMESTEPS_PER_SEGMENT,
        dt=config.DT,
        omega_range=config.OMEGA_RANGE,
        noise_range=config.NOISE_RANGE,
        window_size=config.WINDOW_SIZE,
        scaler=None,
        seed=config.RANDOM_STATE,
    )
    X_val_np, omega_val_np, _ = build_multi_segment_dataset(
        num_segments=n_val_seg,
        timesteps_per_segment=config.TIMESTEPS_PER_SEGMENT,
        dt=config.DT,
        omega_range=config.OMEGA_RANGE,
        noise_range=config.NOISE_RANGE,
        window_size=config.WINDOW_SIZE,
        scaler=scaler,
        seed=config.RANDOM_STATE + 999,
    )

    logger.info("Train windows: %d  |  Val windows: %d", len(X_train_np), len(X_val_np))

    X_train_tensor = torch.from_numpy(X_train_np).float()
    omega_train_tensor = torch.from_numpy(omega_train_np).float()
    X_val_tensor = torch.from_numpy(X_val_np).float()
    omega_val_tensor = torch.from_numpy(omega_val_np).float()

    _pin = device.type == "cuda"
    _nw  = config.NUM_WORKERS
    _pw  = _nw > 0

    train_dataset = TensorDataset(X_train_tensor, X_train_tensor, omega_train_tensor)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        generator=g,
        worker_init_fn=seed_worker,
        num_workers=_nw,
        pin_memory=_pin,
        persistent_workers=_pw,
    )

    val_dataset = TensorDataset(X_val_tensor, X_val_tensor, omega_val_tensor)
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        generator=g,
        worker_init_fn=seed_worker,
        num_workers=_nw,
        pin_memory=_pin,
        persistent_workers=_pw,
    )
    if train_again:
        logger.info("=== 2. Training Both Models ===")
        model_pinn, model_standard, history_pinn, history_standard = train_both_models(
            config, train_loader, val_loader, device, scaler,
            mlflow_logging=MLFLOW_AVAILABLE,
        )
        logger.info("=== 2b. Plotting Training Diagnostics ===")
        plot_training_diagnostics(
            history_pinn=history_pinn,
            history_standard=history_standard,
            X_train_np=X_train_np,
            model_pinn=model_pinn,
            model_standard=model_standard,
            device=device,
            scaler=scaler,
            save_dir=config.RESULTS_DIR,
            window_size=config.WINDOW_SIZE,
            timesteps_per_segment=config.TIMESTEPS_PER_SEGMENT,
        )
    else:
        model_dir = os.path.join(config.RESULTS_DIR, "saved_models")
        model_standard, scaler, cfg = load_model(os.path.join(model_dir, "lstm_autoencoder_standard.pt"), device)
        model_pinn, scaler, cfg = load_model(os.path.join(model_dir, "lstm_autoencoder_pinn.pt"), device)

    
    logger.info("=== 3. Fitting Threshold Bundles ===")
    threshold_bundle_pinn = set_threshold(
        model_pinn, val_loader, config.TEST_OMEGA, config.DT, scaler, device,
        config.WINDOW_SIZE,
        physics_loss_weight=config.PHYSICS_LOSS_WEIGHT,
        save_path=os.path.join(config.RESULTS_DIR, "threshold_bundle_pinn.npy"),
    )
    threshold_bundle_std = set_threshold(
        model_standard, val_loader, config.TEST_OMEGA, config.DT, scaler, device,
        config.WINDOW_SIZE,
        physics_loss_weight=0.0,
        save_path=os.path.join(config.RESULTS_DIR, "threshold_bundle_standard.npy"),
    )

    logger.info("=== 4. Generating Test Baseline Signal ===")
    x_test_baseline = simulate_harmonic_oscillator(
        timesteps=config.TIMESTEPS,
        dt=config.DT,
        omega=config.TEST_OMEGA,
        noise_std=config.TEST_NOISE,
        seed=config.RANDOM_STATE + 100,
        amplitude=3.0,
        phase=0.5,
    )

    logger.info("=== 5. Running Comprehensive Anomaly Test Suite ===")
    pinn_results, standard_results = compare_models_on_anomalies(
        config=config,
        model_pinn=model_pinn,
        model_standard=model_standard,
        x_clean=x_test_baseline,
        scaler=scaler,
        device=device,
        omega=config.TEST_OMEGA,
        dt=config.DT,
    )

    logger.info("=== 6. Running Detection Evaluation ===")
    eval_df = run_detection_evaluation(
        pinn_results=pinn_results,
        standard_results=standard_results,
        threshold_bundle_pinn=threshold_bundle_pinn,
        threshold_bundle_std=threshold_bundle_std,
        config=config,
    )

    logger.info("=== 7. Multi-Frequency / Multi-Seed Evaluation ===")
    multi_eval_agg, _ = run_multi_eval(config, model_pinn, model_standard, scaler, device)
    plot_multi_freq_auc(multi_eval_agg, save_dir=config.RESULTS_DIR)

    logger.info("=== 8. Generating Visualizations ===")
    create_comprehensive_report(
        pinn_results=pinn_results,
        standard_results=standard_results,
        x_clean=x_test_baseline,
        save_dir=config.RESULTS_DIR
    )
    plot_2d_scatter_with_threshold(
        pinn_results=pinn_results,
        std_results=standard_results,
        threshold_bundle_pinn=threshold_bundle_pinn,
        threshold_bundle_std=threshold_bundle_std,
        save_dir=config.RESULTS_DIR,
        window_size=config.WINDOW_SIZE,
    )
    plot_roc_curves(
        eval_df=eval_df,
        save_dir=config.RESULTS_DIR,
        pinn_results=pinn_results,
        std_results=standard_results,
        threshold_bundle_pinn=threshold_bundle_pinn,
        threshold_bundle_std=threshold_bundle_std,
        config=config,
    )
    plot_score_distributions(
        pinn_results=pinn_results,
        std_results=standard_results,
        threshold_bundle_pinn=threshold_bundle_pinn,
        threshold_bundle_std=threshold_bundle_std,
        save_dir=config.RESULTS_DIR,
        window_size=config.WINDOW_SIZE,
    )
    logger.info("=== 8. Quantitative Analysis ===")
    knn_data = run_full_quantitative_analysis(
        pinn_results=pinn_results,
        standard_results=standard_results,
        save_dir=config.RESULTS_DIR,
        window_size=config.WINDOW_SIZE,
    )
    plot_macro_class_confusion(knn_data, save_dir=config.RESULTS_DIR)
    plot_micro_class_confusion(knn_data, save_dir=config.RESULTS_DIR)

    logger.info("=== 9. End-to-end Pipeline Classification Evaluation ===")
    from sklearn.neighbors import KNeighborsClassifier
    from src.quantitative_metrics import GMMClassifierWrapper

    knn_pinn = KNeighborsClassifier(n_neighbors=5).fit(
        knn_data['X_pinn'], knn_data['y_pinn_micro']
    )
    knn_std = KNeighborsClassifier(n_neighbors=5).fit(
        knn_data['X_std'], knn_data['y_std_micro']
    )
    e2e_data = run_e2e_classification(
        pinn_results=pinn_results,
        standard_results=standard_results,
        threshold_bundle_pinn=threshold_bundle_pinn,
        threshold_bundle_std=threshold_bundle_std,
        config=config,
        classifier_pinn=knn_pinn,
        classifier_std=knn_std,
        micro_order=knn_data['micro_order'],
    )
    plot_e2e_confusion(e2e_data, save_dir=config.RESULTS_DIR)

    logger.info("=== 10. End-to-end Pipeline — GMM Unsupervised Classifier ===")
    gmm_res = knn_data['gmm_results']
    gmm_clf_pinn = GMMClassifierWrapper(
        gmm_res['micro']['gmm_pinn'],
        gmm_res['micro']['map_pinn'],
        confidence_threshold=config.GMM_CONFIDENCE_THRESHOLD,
    )
    gmm_clf_std = GMMClassifierWrapper(
        gmm_res['micro']['gmm_std'],
        gmm_res['micro']['map_std'],
        confidence_threshold=config.GMM_CONFIDENCE_THRESHOLD,
    )
    e2e_data_gmm = run_e2e_classification(
        pinn_results=pinn_results,
        standard_results=standard_results,
        threshold_bundle_pinn=threshold_bundle_pinn,
        threshold_bundle_std=threshold_bundle_std,
        config=config,
        classifier_pinn=gmm_clf_pinn,
        classifier_std=gmm_clf_std,
        micro_order=knn_data['micro_order'],
    )
    plot_e2e_confusion(e2e_data_gmm, save_dir=config.RESULTS_DIR, suffix='gmm')
    plot_gmm_precision_coverage(gmm_res, save_dir=config.RESULTS_DIR)

    logger.info("=== Test Suite Complete ===")
    logger.info(f"All results saved to: {config.RESULTS_DIR}")

    if MLFLOW_AVAILABLE:
        for _, row in eval_df.iterrows():
            prefix = "pinn" if row["model"] == "physics_informed" else "std"
            atype = row["anomaly_type"]
            mlflow.log_metrics({
                f"{prefix}.{atype}.auc":       float(row["auc"]) if row["auc"] is not None else float("nan"),
                f"{prefix}.{atype}.precision": row["precision"],
                f"{prefix}.{atype}.recall":    row["recall"],
                f"{prefix}.{atype}.f1":        row["f1"],
            })
        # Multi-eval: log macro AUC per omega
        for omega in config.TEST_OMEGAS:
            for model_tag, prefix in [("physics_informed", "pinn"), ("standard", "std")]:
                omega_rows = multi_eval_agg[
                    (multi_eval_agg['model'] == model_tag) &
                    (multi_eval_agg['omega'] == omega)
                ]
                if not omega_rows.empty:
                    mlflow.log_metric(
                        f"multi.{prefix}.macro_auc.w{omega:.1f}",
                        float(omega_rows['auc_mean'].mean()),
                    )
        mlflow.log_artifacts(config.RESULTS_DIR)
        mlflow.end_run()
        logger.info("MLflow run complete.")


if __name__ == "__main__":
    setup_logging(level=logging.INFO)
    app_config = Config()
    set_all_seeds(app_config.RANDOM_STATE)
    
    logger.info("="*60)
    logger.info("COMPREHENSIVE ANOMALY DETECTION TEST SUITE")
    logger.info("="*60)
    
    run_full_test_suite(app_config, train_again = False)
