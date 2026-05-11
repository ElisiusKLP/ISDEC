import numpy as np
import mne


def _validate_band_range(freq_range: tuple[float, float], sfreq: float) -> None:
    """Validate that a frequency range is usable for band-pass filtering."""
    fmin, fmax = freq_range
    if fmin <= 0:
        raise ValueError(f"freq_range lower bound must be > 0, got {fmin}")
    if fmax <= fmin:
        raise ValueError(f"freq_range upper bound must be greater than lower bound, got {freq_range}")
    nyquist = 0.5 * sfreq
    if fmax >= nyquist:
        raise ValueError(f"freq_range upper bound must be < Nyquist ({nyquist}), got {fmax}")


def transform_to_band_power(
    x: np.ndarray,
    sfreq: float,
    bands: dict[str, tuple[float, float]] | None = None,
) -> np.ndarray:
    """
    Transform EEG data into band power features using MNE PSD.

    Parameters
    ----------
    x : np.ndarray
        Shape (epochs, channels, timepoints)
    sfreq : float
        Sampling frequency in Hz
    bands : dict
        Frequency bands, e.g. {"delta": (1,4), "theta": (4,8), ...}

    Returns
    -------
    X_bp : np.ndarray
        Shape (epochs, channels * n_bands)
    """

    if bands is None:
        bands = {
            "delta": (1, 4),
            "theta": (4, 8),
            "alpha": (8, 13),
            "beta": (13, 30),
            "gamma": (30, 45),
        }

    n_epochs, n_channels, _ = x.shape

    # Compute PSD using MNE (Welch)
    psd, freqs = mne.time_frequency.psd_array_welch(
        x,
        sfreq=sfreq,
        fmin=min(b[0] for b in bands.values()),
        fmax=max(b[1] for b in bands.values()),
        n_fft=256,
        verbose=False,
    )
    # psd shape: (epochs, channels, freqs)

    # Initialize output
    n_bands = len(bands)
    bandpower = np.zeros((n_epochs, n_channels, n_bands))

    # Compute band power
    for i, (band_name, (fmin, fmax)) in enumerate(bands.items()):
        freq_mask = (freqs >= fmin) & (freqs <= fmax)

        # Average power in band
        bandpower[:, :, i] = psd[:, :, freq_mask].mean(axis=-1)

    # Flatten channels × bands into features
    X_bp = bandpower.reshape(n_epochs, n_channels * n_bands)

    return X_bp

def downsample_time(
    x: np.ndarray,
    original_sfreq: float,
    target_sfreq: float,
) -> np.ndarray:
    """
    Downsample EEG data in time dimension.

    Parameters
    ----------
    x : np.ndarray
        Shape (epochs, channels, timepoints)
    original_sfreq : float
        Original sampling frequency in Hz
    target_sfreq : float
        Target sampling frequency in Hz

    Returns
    -------
    x_downsampled : np.ndarray
        Shape (epochs, channels, new_timepoints)
    """
    from scipy.signal import resample

    n_epochs, n_channels, n_timepoints = x.shape
    duration = n_timepoints / original_sfreq  # in seconds
    new_n_timepoints = int(duration * target_sfreq)

    x_downsampled = resample(x, new_n_timepoints, axis=-1)

    return x_downsampled

def transform_to_phase(x: np.ndarray, sfreq: float, freq_range: tuple[float, float]) -> np.ndarray:
    """
    Extract temporal phase features from EEG data using Hilbert transform.

    Parameters
    ----------
    x : np.ndarray
        Shape (epochs, channels, timepoints)
    sfreq : float
        Sampling frequency in Hz
    freq_range : tuple[float, float]
        Frequency range for bandpass filtering before Hilbert transform

    Returns
    -------
    phase_features : np.ndarray
        Shape (epochs, channels * 2, timepoints) - concatenated sin and cos of phase
    """
    from scipy.signal import butter, filtfilt, hilbert

    if x.ndim != 3:
        raise ValueError(f"Expected x with shape (epochs, channels, timepoints), got {x.shape}")

    _validate_band_range(freq_range, sfreq)

    b, a = butter(
        4,
        [freq_range[0] / (0.5 * sfreq), freq_range[1] / (0.5 * sfreq)],
        btype="band",
    )
    x_filtered = filtfilt(b, a, x, axis=-1)

    # Compute analytic signal and extract phase.
    analytic_signal = hilbert(x_filtered, axis=-1)
    phase = np.angle(analytic_signal)

    # Encode phase as sin and cos
    sin_phase = np.sin(phase)
    cos_phase = np.cos(phase)

    # Concatenate sin and cos features
    phase_features = np.concatenate([sin_phase, cos_phase], axis=1)  # shape: (epochs, channels*2, timepoints)

    return phase_features

def transform_to_band_phase(
    x: np.ndarray,
    sfreq: float,
    bands: dict[str, tuple[float, float]] | None = None
) -> np.ndarray:
    """
    Create combined features of band power and temporal phase components.

    The phase component is computed per channel and per band, then averaged over time to create fixed-size features.
    
    """
    if x.ndim != 3:
        raise ValueError(f"Expected x with shape (epochs, channels, timepoints), got {x.shape}")

    if bands is None:
        bands = {
            "delta": (1, 4),
            "theta": (4, 8),
            "alpha": (8, 13),
            "beta": (13, 30),
            "gamma": (30, 45),
        }

    # Per-band phase features.
    phase_feature_blocks = []
    for _, band_range in bands.items():
        phase_features = transform_to_phase(x, sfreq=sfreq, freq_range=band_range)
        # Collapse the temporal axis so each band contributes a fixed number of features.
        phase_features_flat = phase_features.mean(axis=-1)
        phase_feature_blocks.append(phase_features_flat)

    band_phase_features = np.concatenate(phase_feature_blocks, axis=1)
    # flatten channels × band_phase into features

    band_phase_features = band_phase_features.reshape(band_phase_features.shape[0], -1)

    return band_phase_features

def transform_to_band_power_with_phase(
    x: np.ndarray,
    sfreq: float,
    bands: dict[str, tuple[float, float]] | None = None
) -> np.ndarray:
    """
    Create combined features of band power and band phase components.
    """
    if x.ndim != 3:
        raise ValueError(f"Expected x with shape (epochs, channels, timepoints), got {x.shape}")

    if bands is None:
        bands = {
            "delta": (1, 4),
            "theta": (4, 8),
            "alpha": (8, 13),
            "beta": (13, 30),
            "gamma": (30, 45),
        }

    # Existing power features.
    bandpower_features = transform_to_band_power(x, sfreq=sfreq, bands=bands)

    # Per-band phase features.
    phase_feature_blocks = []
    for _, band_range in bands.items():
        phase_features = transform_to_phase(x, sfreq=sfreq, freq_range=band_range)
        # Collapse the temporal axis so each band contributes a fixed number of features.
        phase_features_flat = phase_features.mean(axis=-1)
        phase_feature_blocks.append(phase_features_flat)

    return np.concatenate([bandpower_features, *phase_feature_blocks], axis=1)