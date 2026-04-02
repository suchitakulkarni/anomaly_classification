import numpy as np
import torch
import os
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class AnomalyTestResult:
    """Stores results for a single anomaly type test."""
    anomaly_type: str
    x_anomalous: np.ndarray
    anomaly_indices: np.ndarray
    mse_values: np.ndarray
    physics_values: np.ndarray
    reconstruction: np.ndarray
    model_name: str


def run_anomaly_test_suite(config, model, x_clean, scaler, device, 
                           model_name="Model", omega=2.0, dt=0.01):
    """
    Runs comprehensive anomaly testing on a trained model.
    
    Args:
        config: Configuration object
        model: Trained LSTM autoencoder
        x_clean: Clean baseline signal for generating anomalies
        scaler: Data scaler used during training
        device: torch device
        model_name: Name identifier for the model
        omega: Natural frequency
        dt: Time step
    
    Returns:
        Dictionary mapping anomaly type to AnomalyTestResult
    """
    from src.evaluate import reconstruct_signal
    from src.train import calculate_physics_loss
    from src.dataset import (
        inject_amplitude_spikes,
        inject_frequency_violations,
        inject_phase_discontinuities,
        inject_damping_violations,
        inject_missing_data,
        inject_white_noise_bursts,
        inject_harmonic_contamination,
        inject_dc_offset_shifts,
        inject_amplitude_modulation,
        simulate_harmonic_oscillator
    )
    
    results = {}
    
    anomaly_configs = [
        ("baseline", simulate_harmonic_oscillator, {'dt':0.01, 'omega':2.0}), ###### BE VERY VERY VERY VERY careful with this, these must be same as defaults of the signal generating function

        ("amplitude_spikes", inject_amplitude_spikes, 
         {"num_anomalies": 5, "severity": 5.0, "seed": config.RANDOM_STATE}),
        
        ("frequency_violations", inject_frequency_violations, 
         {"num_anomalies": 5, "duration": 20, "omega_base": omega, 
          "dt": dt, "seed": config.RANDOM_STATE + 1}),
        
        ("phase_discontinuities", inject_phase_discontinuities, 
         {"num_anomalies": 5, "phase_shift_range": (np.pi/4, np.pi), 
          "seed": config.RANDOM_STATE + 2}),
        
        ("damping_violations", inject_damping_violations, 
         {"num_anomalies": 5, "duration": 30, "damping_factor": 0.95, 
          "dt": dt, "seed": config.RANDOM_STATE + 3}),
        
        ("missing_data", inject_missing_data, 
         {"num_anomalies": 5, "gap_duration": 15, "seed": config.RANDOM_STATE + 4}),
        
        ("white_noise_bursts", inject_white_noise_bursts, 
         {"num_anomalies": 5, "duration": 20, "noise_amplitude": 1.0, 
          "seed": config.RANDOM_STATE + 5}),
        
        ("harmonic_contamination", inject_harmonic_contamination, 
         {"num_anomalies": 5, "duration": 30, "omega_base": omega, 
          "harmonic_order": 2, "dt": dt, "seed": config.RANDOM_STATE + 6}),
        
        ("dc_offset_shifts", inject_dc_offset_shifts, 
         {"num_anomalies": 5, "offset_range": (-1.5, 1.5), 
          "seed": config.RANDOM_STATE + 7}),
        
        ("amplitude_modulation", inject_amplitude_modulation, 
         {"num_anomalies": 5, "duration": 40, "mod_freq": 0.5, 
          "dt": dt, "seed": config.RANDOM_STATE + 8}),
    ]
    
    print(f"\n=== Running Anomaly Test Suite for {model_name} ===")
    
    for anomaly_type, inject_fn, kwargs in anomaly_configs:
        print(f"Testing: {anomaly_type}...")
        if anomaly_type == 'baseline':
            x_anom = inject_fn(**kwargs)
            anom_idxs = [0]
        else:
            x_anom, anom_idxs = inject_fn(x_clean, **kwargs)
        
        recon, mse_vals, phy_vals = reconstruct_signal(
            model,
            x_scaled=x_anom,
            window_size=config.WINDOW_SIZE,
            scaler=scaler,
            device=device,
            omega=omega,
            dt=dt,
            physics_loss_fn=calculate_physics_loss
        )
        
        results[anomaly_type] = AnomalyTestResult(
            anomaly_type=anomaly_type,
            x_anomalous=x_anom,
            anomaly_indices=anom_idxs,
            mse_values=mse_vals,
            physics_values=phy_vals,
            reconstruction=recon,
            model_name=model_name
        )
    
    print(f"Completed {len(results)} anomaly tests.\n")
    return results


def compare_models_on_anomalies(config, model_pinn, model_standard, 
                                x_clean, scaler, device, omega=2.0, dt=0.01):
    """
    Runs test suite on both physics-informed and standard models.
    
    Returns:
        Tuple of (pinn_results, standard_results)
    """
    pinn_results = run_anomaly_test_suite(
        config, model_pinn, x_clean, scaler, device,
        model_name="Physics-Informed", omega=omega, dt=dt
    )
    
    standard_results = run_anomaly_test_suite(
        config, model_standard, x_clean, scaler, device,
        model_name="Standard", omega=omega, dt=dt
    )
    
    return pinn_results, standard_results


def aggregate_anomaly_metrics(results: Dict[str, AnomalyTestResult]) -> Dict:
    """
    Aggregates statistics across all anomaly types.
    
    Returns dictionary with mean/median MSE and physics loss per anomaly type.
    """
    aggregated = {}
    
    for anom_type, result in results.items():
        aggregated[anom_type] = {
            "mean_mse": np.mean(result.mse_values),
            "median_mse": np.median(result.mse_values),
            "std_mse": np.std(result.mse_values),
            "mean_physics": np.mean(result.physics_values),
            "median_physics": np.median(result.physics_values),
            "std_physics": np.std(result.physics_values),
            "num_anomalies": len(result.anomaly_indices)
        }
    
    return aggregated
