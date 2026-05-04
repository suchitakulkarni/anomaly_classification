# config.py
import os
class Config:
    """
    Configuration settings for the Harmonic Oscillator Anomaly Detection Model.
    """
    # --- Reproducibility Settings ---
    RANDOM_STATE = 42         # Master seed for reproducibility
    
    # --- Data Simulation Settings ---
    TIMESTEPS = 1000          
    DT = 0.01                 
    OMEGA = 2.0               # Oscillation frequency
    NUM_ANOMALIES = 1        # Number of anomalies original 20
    SEVERITY = 2              # Perturbation strength original 0.5
    NOISE = 0.1
    SINGLE_FREQUENCY = True

    # --- Data Prep
    WINDOW_SIZE = 30 #20           # Length of the rolling window sequence original is 20
    BATCH_SIZE = 64 #64           
    #TEST_SIZE = 0.3           
    # RANDOM_STATE already defined above

    # --- Model Hyperparameters ---
    #HIDDEN_DIM = 256 #128 #256          # Single frequency use 256
    HIDDEN_DIM = 128 #128 #256          # Multi frequency use 128
    NUM_EPOCHS = 2000         # original was 50
    LEARNING_RATE = 5e-5 #5e-4      # original was 1e-3
    

    # --- Loss Function Settings ---
    RECONSTRUCTION_LOSS_WEIGHT = 1 #1.0 
    # single frequency with or without noise, use physics weights around 1e-3
    #PHYSICS_LOSS_WEIGHT = 0.001 #For single frequency weights around this range work great, all post plots are made with 0.004; increase this to 0.04 to see anomaly detection performance tank
    PHYSICS_LOSS_WEIGHT = 0.004 #this makes anomaly detection for single frequency worse
    #PHYSICS_LOSS_WEIGHT = 0.006 #For mixture of frequencies this is the current best
    
    RESULTS_DIR = 'results'
    os.makedirs(RESULTS_DIR, exist_ok = True)
