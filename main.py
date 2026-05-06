import sys
import torch
import torch.nn as nn
import numpy as np
import os, sys,logging
from torch.utils.data import DataLoader, TensorDataset

from src.utils import setup_logging
from src.config import Config
from src.model import LSTMAutoencoder
from src.train import train_model_with_validation
from src.dataset import simulate_harmonic_oscillator, prepare_data, build_multi_segment_dataset
from src.utils import set_all_seeds, seed_worker
from src.test_suite_runner import compare_models_on_anomalies
from src.visualise import (create_comprehensive_report, plot_2d_scatter_with_threshold,
                           plot_roc_curves, plot_macro_class_confusion,
                           plot_micro_class_confusion, plot_e2e_confusion,
                           plot_score_distributions, plot_training_diagnostics)
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

def train_both_models(config, train_loader, val_loader, device, scaler):
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
        stop_early=True
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
        stop_early=True
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


def run_full_test_suite(config: Config, train_again = False):
    """
    Main function to run comprehensive anomaly detection test suite.
    """
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

    X_train_tensor = torch.from_numpy(X_train_np).float().to(device)
    omega_train_tensor = torch.from_numpy(omega_train_np).float().to(device)
    X_val_tensor = torch.from_numpy(X_val_np).float().to(device)
    omega_val_tensor = torch.from_numpy(omega_val_np).float().to(device)

    train_dataset = TensorDataset(X_train_tensor, X_train_tensor, omega_train_tensor)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        generator=g,
        worker_init_fn=seed_worker,
    )

    val_dataset = TensorDataset(X_val_tensor, X_val_tensor, omega_val_tensor)
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        generator=g,
        worker_init_fn=seed_worker,
    )
    if train_again:
        logger.info("=== 2. Training Both Models ===")
        model_pinn, model_standard, history_pinn, history_standard = train_both_models(
            config, train_loader, val_loader, device, scaler
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

    logger.info("=== Test Suite Complete ===")
    logger.info(f"All results saved to: {config.RESULTS_DIR}")


if __name__ == "__main__":
    setup_logging(level=logging.INFO)
    app_config = Config()
    set_all_seeds(app_config.RANDOM_STATE)
    
    logger.info("="*60)
    logger.info("COMPREHENSIVE ANOMALY DETECTION TEST SUITE")
    logger.info("="*60)
    
    run_full_test_suite(app_config, train_again = True)
