from src.train import calculate_physics_loss

import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import matplotlib.pyplot as plt 
import os
from src.dataset import (simulate_harmonic_oscillator, prepare_data,
                         inject_frequency_violations)

def evaluate_model(model, data_loader, criterion, device):
    """
    Evaluates the model and computes the reconstruction error and reconstructed windows.
    Returns: 
    - all_errors (1D numpy array of mean-per-window errors)
    - all_reconstructed_windows (3D numpy array, N x seq_len x 1)
    """
    model.eval()
    all_errors = []
    all_reconstructed_windows = []
    with torch.no_grad():
        for data in data_loader:
            data_window = data[0].to(device)
            reconstruction = model(data_window)
            
            loss_elementwise = criterion(reconstruction, data_window) 
            error_per_window = torch.mean(loss_elementwise, dim=[1, 2])
            
            all_errors.extend(error_per_window.cpu().numpy())
            all_reconstructed_windows.append(reconstruction.cpu().numpy())
            
    all_reconstructed_windows = np.concatenate(all_reconstructed_windows, axis=0)
    return np.array(all_errors), all_reconstructed_windows

def evaluate_model_with_physics(model, data_loader, omega, dt, scaler, device, window_size):
    """
    Evaluates model and computes both reconstruction and physics errors.
    Returns:
    - reco_errors: 1D numpy array of MSE per window
    - phys_errors: 1D numpy array of physics loss per window
    - window_centers: 1D numpy array of center indices for each window
    - reconstructed_windows: 3D numpy array
    """
    model.eval()
    reco_errors = []
    phys_errors = []
    window_centers = []
    all_reconstructed_windows = []
    
    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader):
            data_window = data[0].to(device)
            reconstruction = model(data_window)
            
            # Reconstruction error (MSE)
            mse_per_window = torch.mean((reconstruction - data_window) ** 2, dim=[1, 2])
            reco_errors.extend(mse_per_window.cpu().numpy())
            
            # Physics error per window
            for i in range(reconstruction.shape[0]):
                recon_single = reconstruction[i:i+1]  # Keep batch dim
                phys_loss = calculate_physics_loss(recon_single, omega, dt, scaler)
                phys_errors.append(phys_loss.cpu().item())
                
                # Calculate window center index
                global_window_idx = batch_idx * data_loader.batch_size + i
                center_idx = global_window_idx * window_size + window_size // 2
                window_centers.append(center_idx)
            
            all_reconstructed_windows.append(reconstruction.cpu().numpy())
    
    all_reconstructed_windows = np.concatenate(all_reconstructed_windows, axis=0)
    return (
        np.array(reco_errors), 
        np.array(phys_errors), 
        np.array(window_centers),
        all_reconstructed_windows
    )


def reconstruct_full_signal(
    model,
    x_clean,
    window_size,
    scaler,
    device
):
    """
    Reconstruct a full 1D signal using a trained LSTM autoencoder.

    x_clean: 1D numpy array (unscaled)
    returns: 1D numpy array (reconstructed, unscaled)
    """
    model.eval()

    # Prepare sliding windows using the SAME scaler as training
    X_np, _, _ = prepare_data(
        x_clean,
        window_size,
        scaler=scaler
    )

    X = torch.from_numpy(X_np).float().to(device)

    with torch.no_grad():
        X_recon = model(X)  # (num_windows, window_size, 1)

    # Stitch windows
    recon = torch.zeros(len(x_clean), device="cpu")
    counts = torch.zeros(len(x_clean), device="cpu")

    for i in range(X_recon.shape[0]):
        recon[i:i+window_size] += X_recon[i, :, 0].cpu()
        counts[i:i+window_size] += 1

    recon /= counts

    # Unscale to original units
    recon_unscaled = scaler.inverse_transform(recon.numpy().reshape(-1, 1)).flatten()

    return recon_unscaled

'''def reconstruct_signal(
    model,
    x_scaled,
    window_size,
    scaler,
    device,
    omega,
    dt,
    physics_loss_fn,
):
    model.eval()

    # ensure input is CPU numpy
    if torch.is_tensor(x_scaled):
        x_scaled_np = x_scaled.detach().cpu().numpy()
    else:
        x_scaled_np = np.array(x_scaled)

    # prepare sliding windows (scaled)
    X_np, _, _ = prepare_data(x_scaled_np, window_size, scaler=scaler)
    X_tensor = torch.from_numpy(X_np).float().to(device)

    with torch.no_grad():
        X_recon = model(X_tensor)  # [num_windows, window_size, 1]

    # stitch windows back (still scaled)
    recon_scaled = torch.zeros(len(x_scaled), device="cpu")
    counts = torch.zeros(len(x_scaled), device="cpu")

    for i in range(X_recon.shape[0]):
        start = i
        end = i + window_size
        if end > len(x_scaled):
            end = len(x_scaled)

        win_len = end - start
        if win_len <= 0:
            continue

        recon_scaled[start:end] += X_recon[i, :win_len, 0].cpu()
        counts[start:end] += 1

    counts[counts == 0] = 1
    recon_scaled /= counts

    # inverse scale reconstruction and ground truth
    recon_unscaled = scaler.inverse_transform(
        recon_scaled.numpy().reshape(-1, 1)
    ).flatten()

    x_true_unscaled = x_scaled_np.flatten()

    # MSE in physical coordinates
    mse = np.mean((recon_unscaled - x_true_unscaled) ** 2)

    # physics loss expects a tensor with shape [batch, T, 1]
    # here we treat the full sequence as a single batch
    recon_scaled_tensor = recon_scaled.view(1, -1, 1).to(device)

    phy_loss = physics_loss_fn(
        reconstruction=recon_scaled_tensor,
        omega=omega,
        dt=dt,
        scaler=scaler,
    ).item()

    return recon_unscaled, mse, phy_loss'''

def reconstruct_signal(
    model,
    x_scaled,
    window_size,
    scaler,
    device,
    omega,
    dt,
    physics_loss_fn,
):
    """
    Reconstruct signal using sliding windows and compute per-window MSE and physics loss.

    Returns:
        recon_unscaled : np.array
            Full reconstructed signal in physical units.
        mse_array : np.array, shape [num_windows]
            MSE for each window in physical units.
        phy_loss_array : np.array, shape [num_windows]
            Physics loss for each window.
    """
    model.eval()

    # ensure input is CPU numpy
    if torch.is_tensor(x_scaled):
        x_scaled_np = x_scaled.detach().cpu().numpy()
    else:
        x_scaled_np = np.array(x_scaled)

    # prepare sliding windows (scaled)
    X_np, _, _ = prepare_data(x_scaled_np, window_size, scaler=scaler)
    X_tensor = torch.from_numpy(X_np).float().to(device)

    num_windows = X_tensor.shape[0]

    # arrays to store per-window losses
    mse_list = []
    phy_list = []

    # array to stitch reconstruction
    recon_scaled = torch.zeros(len(x_scaled), device="cpu")
    counts = torch.zeros(len(x_scaled), device="cpu")

    with torch.no_grad():
        X_recon = model(X_tensor)  # [num_windows, window_size, 1]

    for i in range(num_windows):
        start = i
        end = i + window_size
        if end > len(x_scaled):
            end = len(x_scaled)
        win_len = end - start
        if win_len <= 0:
            continue

        # stitch window back
        recon_scaled[start:end] += X_recon[i, :win_len, 0].cpu()
        counts[start:end] += 1

        # compute MSE in physical units for this window
        recon_window_unscaled = scaler.inverse_transform(
            X_recon[i, :win_len, 0].cpu().numpy().reshape(-1, 1)
        ).flatten()
        x_true_window = x_scaled_np[start:end].flatten()
        mse_window = np.mean((recon_window_unscaled - x_true_window) ** 2)
        mse_list.append(mse_window)

        # compute physics loss for this window
        recon_window_tensor = X_recon[i, :win_len, :].unsqueeze(0).to(device)  # shape [1, win_len, 1]
        phy_window = physics_loss_fn(
            reconstruction=recon_window_tensor,
            omega=omega,
            dt=dt,
            scaler=scaler
        ).item()
        phy_list.append(phy_window)

    # finalize full reconstruction
    counts[counts == 0] = 1
    recon_scaled /= counts
    recon_unscaled = scaler.inverse_transform(
        recon_scaled.numpy().reshape(-1, 1)
    ).flatten()

    return recon_unscaled, np.array(mse_list), np.array(phy_list)
