# src/utils.py
import torch
import numpy as np
import random
import os, sys, logging
from src.config import Config

logger = logging.getLogger(__name__)

def setup_logging(level=logging.INFO):
    """
    Call once from main.py. All modules use logging.getLogger(__name__)
    and inherit this configuration automatically.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(Config.RESULTS_DIR, "run.log"), mode="w")
        ]
    )

def stitch_windows(windows, window_size):
    """
    windows: (num_windows, window_size, 1)
    returns: (signal_length,)
    """
    num_windows = windows.shape[0]
    signal_len = num_windows + window_size - 1

    recon = torch.zeros(signal_len)
    counts = torch.zeros(signal_len)

    for i in range(num_windows):
        recon[i:i+window_size] += windows[i, :, 0]
        counts[i:i+window_size] += 1

    return recon / counts


def set_all_seeds(seed=42):
    """
    Set all random seeds for reproducibility.
    Call this at the very beginning of your script.
    
    Args:
        seed: Random seed value
    """
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    
    # PyTorch backends
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Python hash seed (for reproducibility across runs)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    logger.info("All random seeds set to: %s", seed)
    logger.info("CUDA deterministic mode: %s", torch.backends.cudnn.deterministic)
    logger.info("CUDA benchmark mode: %s", torch.backends.cudnn.benchmark)


def seed_worker(worker_id):
    """
    Seed function for DataLoader workers.
    Use with DataLoader's worker_init_fn parameter.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def unit_leakage_test_suite(dt, omega_true, amplitudes=[0.5, 1.0, 2.0], scalers=["StandardScaler", "MinMaxScaler"], batch_size=5):
    """
    Comprehensive check for unit leakage in physics loss.
    - dt: time step
    - omega_true: physical frequency
    - amplitudes: list of trajectory amplitudes to test
    - scalers: list of scalers to test
    - batch_size: number of trajectories in a batch
    """
    T = 5.0
    t = np.arange(0, T, dt)

    for scaler_type in scalers:
        # Initialize scaler
        if scaler_type == "StandardScaler":
            from sklearn.preprocessing import StandardScaler
            scaler_class = StandardScaler
        elif scaler_type == "MinMaxScaler":
            from sklearn.preprocessing import MinMaxScaler
            scaler_class = MinMaxScaler
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")

        for A in amplitudes:
            # Create batched trajectories with different phases
            batch = np.zeros((batch_size, len(t)))
            for i in range(batch_size):
                phase = i * np.pi / 4  # different phase for each trajectory
                batch[i] = A * np.cos(omega_true * t + phase)

            # Scale each trajectory
            scaled_batch = []
            scalers_used = []
            for traj in batch:
                scaler = scaler_class()
                scaled_traj = scaler.fit_transform(traj.reshape(-1, 1)).flatten()
                scaled_batch.append(scaled_traj)
                scalers_used.append(scaler)
            scaled_batch = np.stack(scaled_batch, axis=0)

            # Convert to torch tensor
            x_tensor = torch.tensor(scaled_batch, dtype=torch.float32).unsqueeze(-1)  # [batch, seq_len, 1]
            dt_torch = torch.tensor(dt)

            # Compute physics loss for correct omega
            phy_loss_correct = torch.mean(
                torch.stack([
                    calculate_physics_loss(
                        x_tensor[i:i+1], 
                        omega=omega_true, 
                        dt=dt_torch, 
                        scaler=scalers_used[i]
                    )
                    for i in range(batch_size)
                ])
            ).item()

            # Compute physics loss for wrong omega
            omega_wrong = omega_true * 2.0
            phy_loss_wrong = torch.mean(
                torch.stack([
                    calculate_physics_loss(
                        x_tensor[i:i+1], 
                        omega=omega_wrong, 
                        dt=dt_torch, 
                        scaler=scalers_used[i]
                    )
                    for i in range(batch_size)
                ])
            ).item()

            logger.info(f"[Scaler: %s, Amp: %s] Correct omega loss: %0.8f, Wrong omega loss: %.8f", scaler_type, A, phy_loss_correct, phy_loss_wrong)

            # Visual check: plot residuals for wrong omega for first trajectory
            with torch.no_grad():
                x_scaled_s = x_tensor[0].squeeze(-1)
                d2x_scaled_dt2 = (x_scaled_s[2:] - 2*x_scaled_s[1:-1] + x_scaled_s[:-2]) / dt**2
                alpha = torch.tensor(scalers_used[0].scale_[0] if scaler_type=="StandardScaler" else scalers_used[0].data_max_[0]-scalers_used[0].data_min_[0])
                beta = torch.tensor(scalers_used[0].mean_[0] if scaler_type=="StandardScaler" else scalers_used[0].data_min_[0])
                x_phys_torch = alpha * x_scaled_s[1:-1] + beta
                d2x_phys_dt2 = alpha * d2x_scaled_dt2
                residual = d2x_phys_dt2 + (omega_wrong**2) * x_phys_torch

            plt.figure(figsize=(6,3))
            plt.plot(residual.numpy(), label=f"Residual (wrong ω)")
            plt.title(f"Residuals Check | Scaler: {scaler_type} | Amp: {A}")
            plt.xlabel("Time Step")
            plt.ylabel("Residual")
            plt.grid(True)
            plt.show()

            # Basic automatic check
            if phy_loss_wrong <= phy_loss_correct*10:
                logger.info("WARNING: Physics loss not sensitive to omega. Possible unit leakage!")
            else:
                logger.info("Physics loss responds correctly. No unit leakage detected.\n")


def test_physics_loss_units(dt, omega_true, scaler_type="StandardScaler"):
    """
    Quick test to detect unit leakage.
    Runs calculate_physics_loss on a perfect harmonic oscillator.
    """
    import torch
    from src.train import calculate_physics_loss
    
    # Simulate perfect oscillator
    T = 5.0
    t = np.arange(0, T, dt)
    x_phys = np.cos(omega_true * t)  # amplitude=1, simple harmonic
    
    # Scale the data
    if scaler_type == "StandardScaler":
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
    elif scaler_type == "MinMaxScaler":
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
    else:
        raise ValueError("Unknown scaler type")
    
    x_scaled = scaler.fit_transform(x_phys.reshape(-1,1)).flatten()
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    dt_torch = torch.tensor(dt)
    
    # Physics loss with correct omega
    phy_loss_correct = calculate_physics_loss(
        reconstruction=x_tensor,
        omega=omega_true,
        dt=dt_torch,
        scaler=scaler
    )
    
    # Physics loss with wrong omega
    phy_loss_wrong = calculate_physics_loss(
        reconstruction=x_tensor,
        omega=omega_true*2.0,
        dt=dt_torch,
        scaler=scaler
    )
    
    logger.info(f"Physics loss with correct omega (%s): %.8f", scaler_type, phy_loss_correct.item())
    logger.info(f"Physics loss with wrong omega (%s):   %.8f", scaler_type, phy_loss_wrong.item())
    
    if phy_loss_wrong <= phy_loss_correct*10:
        logger.info("WARNING: Physics loss is not sensitive to omega! Potential unit leakage.")
    else:
        logger.info("Physics loss responds correctly to omega. No unit leakage detected.")


def physics_residual_physical(x_phys, omega, dt):
    d2x = (x_phys[2:] - 2*x_phys[1:-1] + x_phys[:-2]) / dt**2
    return d2x + omega**2 * x_phys[1:-1]
