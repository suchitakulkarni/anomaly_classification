import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def simulate_harmonic_oscillator(timesteps=1000, dt=0.01, omega=2.0, noise_std=0.0, seed=None, amplitude=1.0, phase=0.0):
    """
    Simulates a simple harmonic oscillator with optional measurement noise.

    amplitude: initial amplitude of oscillation
    phase: initial phase in radians
    """
    if seed is not None:
        np.random.seed(seed)

    x = np.zeros(timesteps)
    v = np.zeros(timesteps)

    x[0] = amplitude * np.cos(phase)
    v[0] = -amplitude * omega * np.sin(phase)

    for t in range(1, timesteps):
        a = -omega**2 * x[t-1]
        v[t] = v[t-1] + a * dt
        x[t] = x[t-1] + v[t] * dt

    if noise_std > 0:
        x += np.random.normal(0, noise_std, timesteps)

    return x


def inject_amplitude_spikes(x, num_anomalies=10, severity=2.0, seed=None):
    """Injects amplitude spikes into the signal."""
    if seed is not None:
        np.random.seed(seed)
    
    x_anomalous = x.copy()
    anomaly_indices = np.random.choice(len(x), num_anomalies, replace=False)
    for idx in anomaly_indices:
        if idx < len(x) - 1:
            x_anomalous[idx] += severity * np.random.randn()
    return x_anomalous, anomaly_indices


def inject_frequency_violations(x, num_anomalies=10, duration=20, omega_base=2.0, dt=0.01, seed=None):
    """
    Injects frequency-violating anomalies by locally changing oscillation frequency.
    """
    if seed is not None:
        np.random.seed(seed)
    
    x_anomalous = x.copy()
    
    max_start = len(x) - duration - 1
    if max_start <= 0:
        return x_anomalous, np.array([])
    
    anomaly_indices = np.random.choice(max_start, num_anomalies, replace=False)
    
    for start_idx in anomaly_indices:
        omega_anomalous = omega_base * np.random.uniform(0.5, 2.0)
        A = x_anomalous[start_idx]
        t_segment = np.arange(duration) * dt
        phase_offset = np.arccos(A / (np.abs(A) + 1e-8))
        x_anomalous[start_idx:start_idx + duration] = (
            A * np.cos(omega_anomalous * t_segment + phase_offset)
        )
    
    return x_anomalous, anomaly_indices


def inject_phase_discontinuities(x, num_anomalies=10, phase_shift_range=(np.pi/4, np.pi), seed=None):
    """
    Injects sudden phase jumps that violate smooth oscillation.
    """
    if seed is not None:
        np.random.seed(seed)
    
    x_anomalous = x.copy()
    anomaly_indices = np.random.choice(len(x) - 1, num_anomalies, replace=False)
    anomaly_indices = np.sort(anomaly_indices)
    
    for idx in anomaly_indices:
        phase_shift = np.random.uniform(*phase_shift_range)
        current_val = x_anomalous[idx]
        next_val = x_anomalous[idx + 1]
        x_anomalous[idx + 1] = current_val * np.cos(phase_shift) - next_val * np.sin(phase_shift)
    
    return x_anomalous, anomaly_indices


def inject_damping_violations(x, num_anomalies=10, duration=30, damping_factor=0.95, dt=0.01, seed=None):
    """
    Injects artificial damping or amplification that violates energy conservation.
    """
    if seed is not None:
        np.random.seed(seed)
    
    x_anomalous = x.copy()
    
    max_start = len(x) - duration - 1
    if max_start <= 0:
        return x_anomalous, np.array([])
    
    anomaly_indices = np.random.choice(max_start, num_anomalies, replace=False)
    
    for start_idx in anomaly_indices:
        factor = np.random.choice([
            np.random.uniform(0.85, 0.98),
            np.random.uniform(1.02, 1.15)
        ])
        envelope = factor ** np.arange(duration)
        x_anomalous[start_idx:start_idx + duration] *= envelope
    
    return x_anomalous, anomaly_indices


def inject_missing_data(x, num_anomalies=10, gap_duration=15, seed=None):
    """
    Injects missing data gaps (sets values to 0).
    Low physics loss, high MSE expected.
    """
    if seed is not None:
        np.random.seed(seed)
    
    x_anomalous = x.copy()
    max_start = len(x) - gap_duration - 1
    if max_start <= 0:
        return x_anomalous, np.array([])
    
    anomaly_indices = np.random.choice(max_start, num_anomalies, replace=False)
    
    for start_idx in anomaly_indices:
        x_anomalous[start_idx:start_idx + gap_duration] = 0.0
    
    return x_anomalous, anomaly_indices


def inject_white_noise_bursts(x, num_anomalies=10, duration=20, noise_amplitude=1.0, seed=None):
    """
    Injects bursts of white noise.
    High MSE, variable physics loss depending on intensity.
    """
    if seed is not None:
        np.random.seed(seed)
    
    x_anomalous = x.copy()
    max_start = len(x) - duration - 1
    if max_start <= 0:
        return x_anomalous, np.array([])
    
    anomaly_indices = np.random.choice(max_start, num_anomalies, replace=False)
    
    for start_idx in anomaly_indices:
        noise = np.random.normal(0, noise_amplitude, duration)
        x_anomalous[start_idx:start_idx + duration] += noise
    
    return x_anomalous, anomaly_indices


def inject_harmonic_contamination(x, num_anomalies=10, duration=30, omega_base=2.0, 
                                  harmonic_order=2, dt=0.01, seed=None):
    """
    Adds higher harmonic components (2*omega, 3*omega, etc.).
    Medium MSE, high physics loss.
    """
    if seed is not None:
        np.random.seed(seed)
    
    x_anomalous = x.copy()
    max_start = len(x) - duration - 1
    if max_start <= 0:
        return x_anomalous, np.array([])
    
    anomaly_indices = np.random.choice(max_start, num_anomalies, replace=False)
    
    for start_idx in anomaly_indices:
        t = np.arange(duration) * dt
        amplitude = np.random.uniform(0.3, 0.8)
        phase = np.random.uniform(0, 2*np.pi)
        
        contamination = amplitude * np.sin(harmonic_order * omega_base * t + phase)
        x_anomalous[start_idx:start_idx + duration] += contamination
    
    return x_anomalous, anomaly_indices


def inject_dc_offset_shifts(x, num_anomalies=10, offset_range=(-1.5, 1.5), seed=None):
    """
    Sudden baseline shifts (DC offset changes).
    Medium MSE, high physics loss (violates zero-mean oscillation).
    """
    if seed is not None:
        np.random.seed(seed)
    
    x_anomalous = x.copy()
    anomaly_indices = np.random.choice(len(x) - 50, num_anomalies, replace=False)
    anomaly_indices = np.sort(anomaly_indices)
    
    for start_idx in anomaly_indices:
        offset = np.random.uniform(*offset_range)
        duration = np.random.randint(30, 100)
        end_idx = min(start_idx + duration, len(x))
        x_anomalous[start_idx:end_idx] += offset
    
    return x_anomalous, anomaly_indices


def inject_amplitude_modulation(x, num_anomalies=10, duration=40, 
                                mod_freq=0.5, dt=0.01, seed=None):
    """
    Applies slow amplitude modulation (creates beating pattern).
    Low MSE, medium physics loss.
    """
    if seed is not None:
        np.random.seed(seed)
    
    x_anomalous = x.copy()
    max_start = len(x) - duration - 1
    if max_start <= 0:
        return x_anomalous, np.array([])
    
    anomaly_indices = np.random.choice(max_start, num_anomalies, replace=False)
    
    for start_idx in anomaly_indices:
        t = np.arange(duration) * dt
        modulation = 0.5 + 0.5 * np.cos(mod_freq * t)
        x_anomalous[start_idx:start_idx + duration] *= modulation
    
    return x_anomalous, anomaly_indices


def inject_combined_physics_violations(x, num_anomalies=10, omega_base=2.0, dt=0.01, seed=None):
    """
    Combines multiple types of physics violations for comprehensive testing.
    """
    if seed is not None:
        np.random.seed(seed)
    
    x_anomalous = x.copy()
    
    n_freq = num_anomalies // 3
    n_phase = num_anomalies // 3
    n_damp = num_anomalies - n_freq - n_phase
    
    x_anomalous, freq_idxs = inject_frequency_violations(
        x_anomalous, n_freq, duration=20, omega_base=omega_base, dt=dt, seed=seed
    )
    
    x_anomalous, phase_idxs = inject_phase_discontinuities(
        x_anomalous, n_phase, phase_shift_range=(np.pi/4, np.pi), seed=seed+1 if seed else None
    )
    
    x_anomalous, damp_idxs = inject_damping_violations(
        x_anomalous, n_damp, duration=30, damping_factor=0.95, dt=dt, seed=seed+2 if seed else None
    )
    
    anomaly_dict = {
        'frequency': freq_idxs,
        'phase': phase_idxs,
        'damping': damp_idxs,
        'all': np.concatenate([freq_idxs, phase_idxs, damp_idxs])
    }
    
    return x_anomalous, anomaly_dict


def create_rolling_windows(data, window_size):
    """Converts a 1D time series into a 2D array of overlapping sequences."""
    X = []
    for i in range(len(data) - window_size + 1):
        X.append(data[i:i+window_size])
    return np.array(X)


def prepare_data(data, window_size, scaler=None):
    """Scales the data and creates rolling windows."""
    data_reshaped = data.reshape(-1, 1) 
    
    if scaler is None:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data_reshaped).flatten()
    else:
        scaled_data = scaler.transform(data_reshaped).flatten()
        
    windows = create_rolling_windows(scaled_data, window_size)
    windows_tensor = windows[:, :, np.newaxis] 
    
    return windows_tensor, scaler, scaled_data
