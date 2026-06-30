# config.py
import os
class Config:
    """
    Configuration settings for the Harmonic Oscillator Anomaly Detection Model.
    """
    # --- Reproducibility Settings ---
    RANDOM_STATE = 42

    # --- Training Data Settings ---
    # Training signal is composed of NUM_SEGMENTS independent segments,
    # each with omega and noise drawn uniformly from the ranges below.
    TIMESTEPS_PER_SEGMENT = 500
    NUM_SEGMENTS = 20              # 80% train / 20% val split by segment count
    DT = 0.01
    OMEGA_RANGE = (1.0, 4.0)
    NOISE_RANGE = (0.05, 0.2)

    # --- Test Signal Settings ---
    # The test suite and threshold calibration use a fixed representative frequency.
    TEST_OMEGA = 2.86
    TEST_NOISE = 0.1
    TIMESTEPS = 1000               # length of the test baseline signal

    # --- Multi-frequency / multi-seed evaluation ---
    TEST_OMEGAS = [1.2, 2.0, 2.86, 3.5]   # frequencies spanning OMEGA_RANGE
    NUM_TEST_SEEDS = 10                     # independent test seeds per omega
    N_CAL_TIMESTEPS = 2000                  # clean calibration signal length per omega

    # --- DataLoader ---
    NUM_WORKERS = 4

    # --- Anomaly Injection Settings ---
    NUM_ANOMALIES = 1
    SEVERITY = 2

    # --- Data Prep ---
    WINDOW_SIZE = 30
    BATCH_SIZE = 64

    # --- Model Hyperparameters ---
    HIDDEN_DIM = 128
    NUM_EPOCHS = 2000
    LEARNING_RATE = 1e-3

    # --- Loss Function Settings ---
    RECONSTRUCTION_LOSS_WEIGHT = 1.0
    # Physics residuals scale as omega^2, so with OMEGA_RANGE=(1,4) the residual
    # magnitude varies ~16x across segments. Expect this weight to need retuning
    # after switching to multi-frequency training.
    PHYSICS_LOSS_WEIGHT = 1e-4 

    GMM_CONFIDENCE_THRESHOLD = 0.7

    RESULTS_DIR = 'results'
    os.makedirs(RESULTS_DIR, exist_ok=True)

    MLFLOW_EXPERIMENT = "harmonic_oscillator_anomaly_detection"
