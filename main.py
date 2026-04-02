import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset

from src.config_large_window import Config
from src.model import LSTMAutoencoder
from src.train import train_model_with_validation
from src.dataset import simulate_harmonic_oscillator, prepare_data
from src.utils import set_all_seeds, seed_worker
from src.test_suite_runner import compare_models_on_anomalies
from src.visualise import create_comprehensive_report
from src.quantitative_metrics import run_full_quantitative_analysis
from torch.serialization import add_safe_globals
from sklearn.preprocessing import StandardScaler

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
    print("\n=== Training Physics-Informed Model ===")
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
    
    print(f"Physics-Informed trained. Best epoch: {best_epoch_pinn}")
    
    print("\n=== Training Standard Model ===")
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
    
    print(f"Standard trained. Best epoch: {best_epoch_standard}")

    model_dir = os.path.join(config.RESULTS_DIR, "saved_models")
    os.makedirs(model_dir, exist_ok=True)

    save_path = os.path.join(model_dir, f"lstm_autoencoder_standard.pt")

    torch.save({
        "model_state_dict": model_standard.state_dict(),
        "scaler": scaler,
        "config": {
            "window_size": config.WINDOW_SIZE,
            "hidden_dim": config.HIDDEN_DIM,
            "omega": config.OMEGA,
            "dt": config.DT
        }
    }, save_path)

    print(f"Saved model checkpoint to {save_path}")

    save_path = os.path.join(model_dir, f"lstm_autoencoder_pinn.pt")

    torch.save({
        "model_state_dict": model_pinn.state_dict(),
        "scaler": scaler,
        "config": {
            "window_size": config.WINDOW_SIZE,
            "hidden_dim": config.HIDDEN_DIM,
            "omega": config.OMEGA,
            "dt": config.DT,
            "physics_weight": config.PHYSICS_LOSS_WEIGHT
        }
    }, save_path)

    print(f"Saved model checkpoint to {save_path}")
    
    return model_pinn, model_standard


def run_full_test_suite(config: Config, train_again = False):
    """
    Main function to run comprehensive anomaly detection test suite.
    """
    set_all_seeds(config.RANDOM_STATE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    g = torch.Generator()
    g.manual_seed(config.RANDOM_STATE)
    if config.SINGLE_FREQUENCY:
        print("\n=== 1. Generating Clean Training Data ===")
        x_clean_full = simulate_harmonic_oscillator(
            timesteps=config.TIMESTEPS,
            dt=config.DT,
            omega=config.OMEGA,
            noise_std=config.NOISE,
            seed=config.RANDOM_STATE,
            amplitude=3.0,
            phase=0.5
        )
    else:
        segments = []
        for i in range(10):
            x_segment = simulate_harmonic_oscillator(
                timesteps=config.TIMESTEPS // 10,
                dt=config.DT,
                omega=config.OMEGA,
                noise_std=config.NOISE,
                seed=config.RANDOM_STATE + i,
                amplitude=np.random.uniform(2.0, 4.0),  # Vary amplitude
            phase=np.random.uniform(0, 2*np.pi)     # Vary phase
            )
            segments.append(x_segment)

        x_clean_full = np.concatenate(segments)
    
    train_size = int(0.8 * len(x_clean_full))
    x_train_raw = x_clean_full[:train_size]
    x_val_raw = x_clean_full[train_size:]
    
    print(f"Train size: {len(x_train_raw)}, Val size: {len(x_val_raw)}")
    
    X_train_np, scaler, _ = prepare_data(x_train_raw, config.WINDOW_SIZE)
    X_train_tensor = torch.from_numpy(X_train_np).float().to(device)
    
    X_val_np, _, _ = prepare_data(x_val_raw, config.WINDOW_SIZE, scaler=scaler)
    X_val_tensor = torch.from_numpy(X_val_np).float().to(device)
    
    train_dataset = TensorDataset(X_train_tensor, X_train_tensor)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        generator=g,
        worker_init_fn=seed_worker
    )
    
    val_dataset = TensorDataset(X_val_tensor, X_val_tensor)
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        generator=g,
        worker_init_fn=seed_worker
    )
    if train_again:
        print("\n=== 2. Training Both Models ===")
        model_pinn, model_standard = train_both_models(
            config, train_loader, val_loader, device, scaler
        )
    else:
        model_dir = os.path.join(config.RESULTS_DIR, "saved_models")
        model_standard, scaler, cfg = load_model(os.path.join(model_dir, "lstm_autoencoder_standard.pt"), device)
        model_pinn, scaler, cfg = load_model(os.path.join(model_dir, "lstm_autoencoder_pinn.pt"), device)
    
    print("\n=== 3. Generating Test Baseline Signal ===")
    x_test_baseline = simulate_harmonic_oscillator(
        timesteps=config.TIMESTEPS,
        dt=config.DT,
        omega=config.OMEGA,
        noise_std=config.NOISE,
        seed=config.RANDOM_STATE + 100,
        amplitude = 3.0,
        phase=0.5
    )
    
    print("\n=== 4. Running Comprehensive Anomaly Test Suite ===")
    pinn_results, standard_results = compare_models_on_anomalies(
        config=config,
        model_pinn=model_pinn,
        model_standard=model_standard,
        x_clean=x_test_baseline,
        scaler=scaler,
        device=device,
        omega=config.OMEGA,
        dt=config.DT
    )
    
    print("\n=== 5. Generating Visualizations ===")
    create_comprehensive_report(
        pinn_results=pinn_results,
        standard_results=standard_results,
        x_clean=x_test_baseline,
        save_dir=config.RESULTS_DIR
    )
    
    print("\n=== 6. Quantitative Analysis ===")
    
    run_full_quantitative_analysis(
        pinn_results=pinn_results,
        standard_results=standard_results,
        save_dir=config.RESULTS_DIR
    )
    
    print("\n=== Test Suite Complete ===")
    print(f"All results saved to: {config.RESULTS_DIR}")


if __name__ == "__main__":
    app_config = Config()
    set_all_seeds(app_config.RANDOM_STATE)
    
    print("\n" + "="*60)
    print("COMPREHENSIVE ANOMALY DETECTION TEST SUITE")
    print("="*60)
    
    run_full_test_suite(app_config, train_again = False)
