# utils.py (Complete, Final Version)

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
import torch.nn as nn
import matplotlib.pyplot as plt 
import os


'''def calculate_physics_loss(reconstruction, omega, dt, scaler):
    """
    Physics loss computed in physical coordinates.
    Works for StandardScaler and MinMaxScaler.
    """
    x_scaled = reconstruction.squeeze(-1)

    d2x_scaled_dt2 = (
        x_scaled[:, 2:]
        - 2.0 * x_scaled[:, 1:-1]
        + x_scaled[:, :-2]
    ) / (dt ** 2)

    device = reconstruction.device
    dtype = reconstruction.dtype

    # Detect scaler type
    if isinstance(scaler, StandardScaler):  # StandardScaler
        alpha = torch.tensor(scaler.scale_[0], device=device, dtype=dtype)
        beta = torch.tensor(scaler.mean_[0], device=device, dtype=dtype)
    else:  # MinMaxScaler
        alpha = torch.tensor(
            scaler.data_max_[0] - scaler.data_min_[0],
            device=device,
            dtype=dtype
        )
        beta = torch.tensor(scaler.data_min_[0], device=device, dtype=dtype)

    x_phys = alpha * x_scaled[:, 1:-1] + beta
    d2x_phys_dt2 = alpha * d2x_scaled_dt2

    residual = d2x_phys_dt2 + (omega ** 2) * x_phys
    return torch.mean(residual ** 2)'''

def inverse_scale(x_scaled, scaler):
    """
    Inverse-scale a tensor using a fitted scaler.
    Supports StandardScaler and MinMaxScaler.
    """
    device = x_scaled.device
    dtype = x_scaled.dtype

    if isinstance(scaler, StandardScaler):
        alpha = torch.tensor(scaler.scale_[0], device=device, dtype=dtype)
        beta = torch.tensor(scaler.mean_[0], device=device, dtype=dtype)
    elif isinstance(scaler, MinMaxScaler):
        alpha = torch.tensor(
            scaler.data_max_[0] - scaler.data_min_[0],
            device=device,
            dtype=dtype
        )
        beta = torch.tensor(scaler.data_min_[0], device=device, dtype=dtype)
    else:
        raise TypeError("Unsupported scaler type")

    return alpha * x_scaled + beta


def calculate_physics_loss(reconstruction, omega, dt, scaler):
    """
    Physics loss computed fully in physical coordinates.
    """
    # remove feature dimension
    x_scaled = reconstruction.squeeze(-1)

    # inverse scale first
    x_phys = inverse_scale(x_scaled, scaler)

    # second time derivative in physical coordinates
    d2x_phys_dt2 = (
        x_phys[:, 2:]
        - 2.0 * x_phys[:, 1:-1]
        + x_phys[:, :-2]
    ) / (dt ** 2)

    # physics residual
    residual = d2x_phys_dt2 + (omega ** 2) * x_phys[:, 1:-1]

    return torch.mean(residual ** 2)





def train_model(model, train_loader, criterion, optimizer, num_epochs, 
                physics_loss_weight, dt, device, scaler):

    model.train()
    history = {'mse_loss': [], 'physics_loss': [], 'total_loss': []}

    for epoch in range(1, num_epochs + 1):
        epoch_total = 0.0
        epoch_mse = 0.0
        epoch_phy = 0.0

        for data in train_loader:
            data_window = data[0].to(device)
            optimizer.zero_grad()

            reconstruction = model(data_window)

            # Reconstruction loss (MSE)
            mse_loss = torch.mean(criterion(reconstruction, data_window))
            if isinstance(scaler, StandardScaler):
                omega_wrong = 2.0 * scaler.scale_[0]    # wrong by amplitude scale
            elif isinstance(scaler, MinMaxScaler):
                omega_wrong = 2.0 / (scaler.data_max_[0] - scaler.data_min_[0])
            else:
                raise TypeError("Unknown scaler")            
            # Physics loss
            if physics_loss_weight > 0:
                phy_loss = calculate_physics_loss(
                    reconstruction=reconstruction,
                    omega=2.0,
                    dt=dt, 
                    scaler=scaler
                )
            else:
                phy_loss = 0.0

            # Combined loss
            total_loss = mse_loss + physics_loss_weight * phy_loss

            # Backprop
            total_loss.backward()
            optimizer.step()

            # Accumulate epoch totals (as scalars)
            epoch_total += total_loss.item()
            epoch_mse += mse_loss.item()
            #epoch_phy += (phy_loss.item()  * physics_loss_weight) if not isinstance(phy_loss, float) else phy_loss
            epoch_phy += phy_loss.item() if not isinstance(phy_loss, float) else phy_loss

        # Average per epoch
        n = len(train_loader)
        history['mse_loss'].append(epoch_mse / n)
        history['physics_loss'].append(epoch_phy / n)
        history['total_loss'].append(epoch_total / n)

        if epoch == num_epochs:
            print(f"Epoch {epoch:03d}/{num_epochs} | Loss: {epoch_total/n:.6f}")

    return model, history

def compute_residual(reconstruction, omega, dt, scaler):
    """
    Compute physics residual: d**2x/dt**2 + omega**2 x
    """
    x_scaled = reconstruction.squeeze(-1)
    d2x_scaled_dt2 = (x_scaled[:,2:] - 2*x_scaled[:,1:-1] + x_scaled[:,:-2]) / dt**2

    # Convert to physical units
    if isinstance(scaler, StandardScaler):
        alpha = torch.tensor(scaler.scale_[0], device=reconstruction.device, dtype=reconstruction.dtype)
        beta = torch.tensor(scaler.mean_[0], device=reconstruction.device, dtype=reconstruction.dtype)
    else:
        alpha = torch.tensor(scaler.data_max_[0] - scaler.data_min_[0], device=reconstruction.device, dtype=reconstruction.dtype)
        beta = torch.tensor(scaler.data_min_[0], device=reconstruction.device, dtype=reconstruction.dtype)

    x_phys = alpha * x_scaled[:,1:-1] + beta
    d2x_phys_dt2 = alpha * d2x_scaled_dt2

    residual = d2x_phys_dt2 + (omega**2) * x_phys
    return residual

def train_model_with_validation(model, train_loader, val_loader, criterion, optimizer, 
                                num_epochs, physics_loss_weight, dt, device, scaler,
                                patience=100, min_delta=1e-5, stop_early = False):
    """
    Training with validation monitoring and early stopping.
    
    Args:
        patience: Number of epochs to wait for improvement before stopping
        min_delta: Minimum change to qualify as improvement
    
    Returns:
        model: Trained model
        history: Dict with train/val losses
        best_epoch: Epoch with best validation loss
    """
    model.train()
    history = {
        'train_mse': [], 'train_physics': [], 'train_total': [],
        'val_mse': [], 'val_physics': [], 'val_total': [],
        'overfitting_gap': []  # Difference between train and val loss
    }
    
    best_val_loss = float('inf')
    best_model_state = None
    best_epoch = 0
    epochs_without_improvement = 0
    
    for epoch in range(1, num_epochs + 1):
        # ========== TRAINING PHASE ==========
        model.train()
        epoch_train_total = 0.0
        epoch_train_mse = 0.0
        epoch_train_phy = 0.0
        
        for data in train_loader:
            data_window = data[0].to(device)
            optimizer.zero_grad()
            
            reconstruction = model(data_window)
            mse_loss = torch.mean(criterion(reconstruction, data_window))
            
            if physics_loss_weight > 0:
                phy_loss = calculate_physics_loss(
                    reconstruction=reconstruction,
                    omega=2.0,
                    dt=dt,
                    scaler=scaler
                )
            else:
                phy_loss = 0.0
            
            total_loss = mse_loss + physics_loss_weight * phy_loss
            total_loss.backward()
            optimizer.step()
            
            epoch_train_total += total_loss.item()
            epoch_train_mse += mse_loss.item()
            epoch_train_phy += phy_loss.item() if not isinstance(phy_loss, float) else phy_loss
        
        # Average training losses
        n_train = len(train_loader)
        avg_train_mse = epoch_train_mse / n_train
        avg_train_phy = epoch_train_phy / n_train
        avg_train_total = epoch_train_total / n_train
        
        # ========== VALIDATION PHASE ==========
        model.eval()
        epoch_val_total = 0.0
        epoch_val_mse = 0.0
        epoch_val_phy = 0.0
        
        with torch.no_grad():
            for data in val_loader:
                data_window = data[0].to(device)
                reconstruction = model(data_window)
                
                mse_loss = torch.mean(criterion(reconstruction, data_window))
                
                if physics_loss_weight > 0:
                    phy_loss = calculate_physics_loss(
                        reconstruction=reconstruction,
                        omega=2.0,
                        dt=dt,
                        scaler=scaler
                    )
                else:
                    phy_loss = 0.0
                
                total_loss = mse_loss + physics_loss_weight * phy_loss
                
                epoch_val_total += total_loss.item()
                epoch_val_mse += mse_loss.item()
                epoch_val_phy += phy_loss.item() if not isinstance(phy_loss, float) else phy_loss
        
        # Average validation losses
        n_val = len(val_loader)
        avg_val_mse = epoch_val_mse / n_val
        avg_val_phy = epoch_val_phy / n_val
        avg_val_total = epoch_val_total / n_val
        
        # Calculate overfitting gap
        overfitting_gap = avg_val_total - avg_train_total
        
        # Store history
        history['train_mse'].append(avg_train_mse)
        history['train_physics'].append(avg_train_phy)
        history['train_total'].append(avg_train_total)
        history['val_mse'].append(avg_val_mse)
        history['val_physics'].append(avg_val_phy)
        history['val_total'].append(avg_val_total)
        history['overfitting_gap'].append(overfitting_gap)
        # Print progress every 50 epochs
        if epoch % 10 == 0 or epoch == num_epochs:
            print(f"Epoch {epoch:03d}/{num_epochs} | "
                  f"Train Loss: {avg_train_total:.6f} | "
                  f"Val Loss: {avg_val_total:.6f} | "
                  f"Val phy Loss: {avg_val_phy:.6f} | "
                  f"Gap: {overfitting_gap:.6f} | "
                  f"Best: {best_epoch}")
        
        # Early stopping check
        if stop_early:
            if avg_val_total < best_val_loss - min_delta:
            #if avg_val_mse < avg_val_mse - min_delta:
                best_val_loss = avg_val_total
                best_model_state = model.state_dict().copy()
                best_epoch = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"\nEarly stopping at epoch {epoch}. Best epoch was {best_epoch}.")
                break
        
        
        # Early stopping
        #if epochs_without_improvement >= patience:
        #    print(f"\nEarly stopping at epoch {epoch}. Best epoch was {best_epoch}.")
        #    break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model from epoch {best_epoch}")
    
    return model, history, best_epoch
