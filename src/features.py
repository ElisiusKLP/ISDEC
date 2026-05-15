import numpy as np
import mne
from typing import Literal
import pywt

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
    bands: dict[str, tuple[float, float]],
    mean: bool = True,
    stack_channels: bool = True
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
        If mean=True: shape (epochs, channels * n_bands)
        If mean=False: shape (epochs, channels * total_band_freq_bins)
    """

    if bands is None:
        raise ValueError("bands dictionary must be provided")

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

    if mean:
        n_bands = len(bands)
        bandpower = np.zeros((n_epochs, n_channels, n_bands))
        for i, (_, (fmin, fmax)) in enumerate(bands.items()):
            freq_mask = (freqs >= fmin) & (freqs <= fmax)
            bandpower[:, :, i] = psd[:, :, freq_mask].mean(axis=-1)
        if stack_channels:
            return bandpower.reshape(n_epochs, n_channels * n_bands)
        else:
            return bandpower  # shape (epochs, channels, bands)

    # Keep the full band-limited PSD bins instead of averaging within each band.
    spectrum_blocks = []
    for _, (fmin, fmax) in bands.items():
        freq_mask = (freqs >= fmin) & (freqs <= fmax)
        spectrum_blocks.append(psd[:, :, freq_mask])

    full_spectrum = np.concatenate(spectrum_blocks, axis=-1)
    if stack_channels:
        return full_spectrum.reshape(n_epochs, n_channels * full_spectrum.shape[-1])
    else:
        return full_spectrum  # shape (epochs, channels, total_band_freq_bins)

def transform_to_time_frequency(
    x: np.ndarray, 
    sfreq: float,
    algorithm: Literal["morlet", "dwt"] = "morlet",
    in_bands: bool = False,
    downsample_to_freq: float | None = None,
    bands: dict[str, tuple[float, float]] | None = None,
    n_freqs: int = 20,
    ) -> np.ndarray:
    """Transform EEG data into time-frequency features.

    For Morlet transforms:
    - in_bands=False returns flattened power across all frequency bins and timepoints.
    - in_bands=True returns flattened band-aggregated amplitude across timepoints.
    """
    if x.ndim != 3:
        raise ValueError(f"Expected x with shape (epochs, channels, timepoints), got {x.shape}")
    if n_freqs < 2:
        raise ValueError(f"n_freqs must be >= 2, got {n_freqs}")

    n_epochs, n_channels, n_timepoints = x.shape

    if algorithm == "morlet":

        # specifiy the frequency intervals for the Morlet wavelets. Logarithmic spacing is common for EEG.
        freqs = np.logspace(np.log10(1), np.log10(100), num=n_freqs)
        n_cycles = freqs / 2.0  # example: more cycles for higher freqs

        if in_bands:
            if bands is None:
                raise ValueError("bands dictionary must be provided when in_bands=True")

            complex_tfr = mne.time_frequency.tfr_array_morlet(
                x,
                sfreq=sfreq,
                freqs=freqs,
                n_cycles=n_cycles,
                output="complex",
                verbose=False,
            )
            amplitude = np.abs(complex_tfr)

            band_amplitude_blocks = []
            for band_name, (fmin, fmax) in bands.items():
                _validate_band_range((fmin, fmax), sfreq)
                freq_mask = (freqs >= fmin) & (freqs <= fmax)
                if not np.any(freq_mask):
                    raise ValueError(
                        f"Band '{band_name}' ({fmin}, {fmax}) has no sampled Morlet frequencies. "
                        f"Increase n_freqs or adjust bands."
                    )
                band_amplitude_blocks.append(amplitude[:, :, freq_mask, :].mean(axis=2))

            band_amplitude = np.stack(band_amplitude_blocks, axis=2)
            # Optionally downsample the time axis of the band-amplitude representation
            if downsample_to_freq is not None:
                from scipy.signal import resample
                if downsample_to_freq <= 0:
                    raise ValueError(f"downsample_to_freq must be > 0, got {downsample_to_freq}")
                new_n_timepoints = max(1, int(np.round(n_timepoints * (downsample_to_freq / sfreq))))
                band_amplitude = resample(band_amplitude, new_n_timepoints, axis=-1)

            return band_amplitude.reshape(n_epochs, n_channels * band_amplitude.shape[2] * band_amplitude.shape[3])

        power = mne.time_frequency.tfr_array_morlet(
            x,
            sfreq=sfreq,
            freqs=freqs,
            n_cycles=n_cycles,
            output="power",
            verbose=False,
        )
        # power shape: (epochs, channels, freqs, timepoints)
        if downsample_to_freq is not None:
            from scipy.signal import resample
            if downsample_to_freq <= 0:
                raise ValueError(f"downsample_to_freq must be > 0, got {downsample_to_freq}")
            new_n_timepoints = max(1, int(np.round(n_timepoints * (downsample_to_freq / sfreq))))
            power = resample(power, new_n_timepoints, axis=-1)

        return power.reshape(n_epochs, n_channels * power.shape[2] * power.shape[3])

    elif algorithm == "dwt":
        wavelet_name = "db4"
        max_level = pywt.dwt_max_level(n_timepoints, 8)
        if max_level < 1:
            raise ValueError(
                "Time series is too short for a discrete wavelet transform with the chosen wavelet"
            )

        epoch_feature_blocks = []
        for epoch in range(n_epochs):
            channel_blocks = []
            for channel in range(n_channels):
                coeffs = pywt.wavedec(
                    x[epoch, channel],
                    wavelet=wavelet_name,
                    level=max_level,
                    mode="symmetric",
                )
                channel_blocks.append(np.concatenate([np.asarray(coeff) for coeff in coeffs], axis=-1))
            epoch_feature_blocks.append(np.concatenate(channel_blocks, axis=-1))

        return np.stack(epoch_feature_blocks, axis=0)
    else:
        raise ValueError(f"Unsupported time-frequency algorithm: {algorithm}")

def mutual_info_feature_selection(X: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    """Select top k features based on mutual information with the target."""
    from sklearn.feature_selection import mutual_info_classif

    mi = mutual_info_classif(X, y)
    top_k_indices = np.argsort(mi)[-k:]
    return X[:, top_k_indices]

def pca_feature_selection(X: np.ndarray, n_components: int) -> np.ndarray:
    """Reduce dimensionality to n_components using PCA."""
    from sklearn.decomposition import PCA

    pca = PCA(n_components=n_components)
    return pca.fit_transform(X)

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