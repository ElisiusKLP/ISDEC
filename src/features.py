import numpy as np
import mne

def transform_to_band_power(
    x: np.ndarray,
    sfreq: float,
    bands: dict = None,
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