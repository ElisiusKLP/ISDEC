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

    n_epochs, n_channels, n_timepoints = x.shape

    # choose n_fft safely so it does not exceed the signal length (Welch requirement)
    n_fft_param = min(256, n_timepoints)

    # Compute PSD using MNE (Welch)
    psd, freqs = mne.time_frequency.psd_array_welch(
        x,
        sfreq=sfreq,
        fmin=min(b[0] for b in bands.values()),
        fmax=max(b[1] for b in bands.values()),
        n_fft=n_fft_param,
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


def transform_to_band_power_mean_sd(
    x: np.ndarray,
    sfreq: float,
    bands: dict[str, tuple[float, float]],
    stack_channels: bool = True,
) -> np.ndarray:
    """
    Compute per-band mean and standard deviation of the PSD for each channel.

    Returns concatenated mean and std per band (mean then std) so the final
    feature order per channel is [band1_mean, band2_mean, ..., band1_std, band2_std, ...].
    """
    if x.ndim != 3:
        raise ValueError(f"Expected x with shape (epochs, channels, timepoints), got {x.shape}")

    if bands is None:
        raise ValueError("bands dictionary must be provided")

    n_epochs, n_channels, n_timepoints = x.shape

    # choose n_fft safely so it does not exceed the signal length
    n_fft_param = min(256, n_timepoints)

    # compute PSD across the full epoch
    psd, freqs = mne.time_frequency.psd_array_welch(
        x,
        sfreq=sfreq,
        fmin=min(b[0] for b in bands.values()),
        fmax=max(b[1] for b in bands.values()),
        n_fft=n_fft_param,
        verbose=False,
    )

    n_bands = len(bands)
    mean_block = np.zeros((n_epochs, n_channels, n_bands))
    std_block = np.zeros((n_epochs, n_channels, n_bands))

    for i, (_, (fmin, fmax)) in enumerate(bands.items()):
        freq_mask = (freqs >= fmin) & (freqs <= fmax)
        vals = psd[:, :, freq_mask]
        mean_block[:, :, i] = vals.mean(axis=-1)
        std_block[:, :, i] = vals.std(axis=-1)

    combined = np.concatenate([mean_block, std_block], axis=2)  # (epochs, channels, n_bands*2)

    if stack_channels:
        return combined.reshape(n_epochs, n_channels * combined.shape[2])
    else:
        return combined


def transform_to_band_power_stats_window(
    x: np.ndarray,
    sfreq: float,
    bands: dict[str, tuple[float, float]],
    stats: list[str] = ["mean", "std"],
    n_windows: int = 6,
    stack_channels: bool = True,
) -> np.ndarray:
    """
    Split each epoch into `n_windows` temporal windows and compute per-window
    band mean and std of the PSD. Concatenates windowed stats to produce
    temporal-localised bandpower features while keeping dimensionality manageable.
    """
    if x.ndim != 3:
        raise ValueError(f"Expected x with shape (epochs, channels, timepoints), got {x.shape}")

    if bands is None:
        raise ValueError("bands dictionary must be provided")

    n_epochs, n_channels, n_timepoints = x.shape

    # create windows as index ranges
    indices = np.array_split(np.arange(n_timepoints), n_windows)

    window_blocks = []
    for idx in indices:
        if len(idx) == 0:
            # in case n_windows > n_timepoints
            continue
        start, end = idx[0], idx[-1] + 1
        x_win = x[:, :, start:end]

        # compute PSD for window; ensure n_fft does not exceed window length
        win_len = x_win.shape[-1]
        n_fft_param = min(256, win_len)
        psd, freqs = mne.time_frequency.psd_array_welch(
            x_win,
            sfreq=sfreq,
            fmin=min(b[0] for b in bands.values()),
            fmax=max(b[1] for b in bands.values()),
            n_fft=n_fft_param,
            verbose=False,
        )

        n_bands = len(bands)
        mean_block = np.zeros((n_epochs, n_channels, n_bands))
        std_block = np.zeros((n_epochs, n_channels, n_bands))
        for i, (_, (fmin, fmax)) in enumerate(bands.items()):
            freq_mask = (freqs >= fmin) & (freqs <= fmax)
            vals = psd[:, :, freq_mask]
            if "mean" in stats:
                mean_block[:, :, i] = vals.mean(axis=-1)
            if "std" in stats:
                std_block[:, :, i] = vals.std(axis=-1)

        combined = np.concatenate([mean_block, std_block], axis=2)  # (epochs, channels, n_bands*2)
        window_blocks.append(combined)

    # concatenate windows along the band/stat axis
    all_windows = np.concatenate(window_blocks, axis=2)  # (epochs, channels, n_windows*n_bands*2)

    if stack_channels:
        return all_windows.reshape(n_epochs, n_channels * all_windows.shape[2])
    else:
        return all_windows

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

        # Specify Morlet frequencies with a Nyquist-safe upper bound.
        max_freq = min(100.0, (0.5 * sfreq) - 1e-6)
        if max_freq <= 1.0:
            raise ValueError(
                f"Sampling frequency too low for Morlet range: sfreq={sfreq}, max usable freq={max_freq}"
            )
        freqs = np.logspace(np.log10(1.0), np.log10(max_freq), num=n_freqs)
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


def tfr_mortlet_to_cnn(
    x: np.ndarray,
    sfreq: float,
    target_size: int = 256,
    n_freqs: int = 256,
    collapse_channels: bool = True,
    pre_downsample_to_sfreq: float | None = 128.0,
) -> np.ndarray:
    """Compute Morlet time-frequency power and return images for CNN.

    Returns array shaped (epochs, H, W, C) where H=n_freqs, W=target_size.
    If `collapse_channels` is True the channel axis is averaged to produce C=1.
    This builds on `transform_to_time_frequency` and performs minimal reshaping.
    """
    if x.ndim != 3:
        raise ValueError(f"Expected x with shape (epochs, channels, timepoints), got {x.shape}")

    # Optional pre-downsampling reduces Morlet compute significantly while
    # preserving the final CNN image size via the later time-axis resample.
    effective_sfreq = float(sfreq)
    if pre_downsample_to_sfreq is not None:
        if pre_downsample_to_sfreq <= 0:
            raise ValueError(
                f"pre_downsample_to_sfreq must be > 0, got {pre_downsample_to_sfreq}"
            )
        if pre_downsample_to_sfreq < sfreq:
            x = downsample_time(
                x,
                original_sfreq=float(sfreq),
                target_sfreq=float(pre_downsample_to_sfreq),
            )
            effective_sfreq = float(pre_downsample_to_sfreq)

    n_epochs, n_channels, n_timepoints = x.shape

    # compute downsample_to_freq so the function will produce ~target_size timepoints
    downsample_to_freq = effective_sfreq * float(target_size) / float(n_timepoints)

    flat = transform_to_time_frequency(
        x,
        sfreq=effective_sfreq,
        algorithm="morlet",
        in_bands=False,
        downsample_to_freq=downsample_to_freq,
        n_freqs=n_freqs,
    )

    # flat has shape (epochs, channels * freqs * timepoints)
    per_epoch = flat.shape[1]
    if per_epoch % n_channels != 0:
        raise ValueError("Unexpected output size from transform_to_time_frequency")
    per_channel = per_epoch // n_channels
    if per_channel % n_freqs != 0:
        raise ValueError("Cannot infer timepoints from transform output; adjust n_freqs or target_size")
    new_timepoints = per_channel // n_freqs

    power = flat.reshape(n_epochs, n_channels, n_freqs, new_timepoints)

    # convert to images with channel last: (epochs, freqs, time, channels)
    images = np.transpose(power, (0, 2, 3, 1))

    if collapse_channels:
        images = images.mean(axis=-1, keepdims=True)

    return images

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

def transform_to_dwt_hierarchical(
    x: np.ndarray,
    sfreq: float,
    stats: list[str] = ["mean"],
) -> np.ndarray:
    """
    Transform EEG data into DWT hierarchical features.
    
    Instead of concatenating all raw DWT coefficients, this aggregates coefficients
    by decomposition level, computing statistics (mean, std, max, 75th percentile)
    for each level. This preserves multi-scale time-frequency structure while
    reducing dimensionality.

    Parameters
    ----------
    x : np.ndarray
        Shape (epochs, channels, timepoints)
    sfreq : float
        Sampling frequency in Hz (not used but kept for API consistency)

    Returns
    -------
    features : np.ndarray
        Shape (epochs, channels * n_levels * 4) where 4 statistics per level
    """
    if x.ndim != 3:
        raise ValueError(f"Expected x with shape (epochs, channels, timepoints), got {x.shape}")

    wavelet_name = "db4"
    n_epochs, n_channels, n_timepoints = x.shape
    
    epoch_features = []
    for epoch in range(n_epochs):
        channel_features = []
        for channel in range(n_channels):
            max_level = pywt.dwt_max_level(n_timepoints, 8)
            if max_level < 1:
                raise ValueError(
                    "Time series is too short for a discrete wavelet transform with the chosen wavelet"
                )
            
            coeffs = pywt.wavedec(
                x[epoch, channel],
                wavelet=wavelet_name,
                level=max_level,
                mode="symmetric",
            )
            
            # Compute statistics per decomposition level
            level_stats = []
            for level_coeff in coeffs:
                level_coeff = np.asarray(level_coeff)
                if "mean" in stats:
                    level_stats.append(level_coeff.mean())
                if "std" in stats:
                    level_stats.append(level_coeff.std())
                if "max" in stats:
                    level_stats.append(np.max(level_coeff))
                if "75th_percentile" in stats:
                    level_stats.append(np.percentile(level_coeff, 75))
            
            channel_features.append(level_stats)
        
        epoch_features.append(np.concatenate(channel_features))
    
    return np.array(epoch_features)

def wavelet_channels_by_mutual_info(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    k_channels: int = 4,
    algorithm: str = "dwt",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Select top k channels based on mutual information with target, then extract DWT features.
    
    This method identifies which channels are most discriminative for the classification task,
    reducing noise from uninformative channels before feature extraction.

    Parameters
    ----------
    x_train : np.ndarray
        Training data, shape (epochs, channels, timepoints)
    y_train : np.ndarray
        Training labels, shape (epochs,)
    x_test : np.ndarray
        Test data, shape (epochs, channels, timepoints)
    k_channels : int
        Number of channels to select (default: 4)
    algorithm : str
        Feature extraction algorithm ("dwt" or "dwt_hierarchical", default: "dwt")

    Returns
    -------
    X_train_features : np.ndarray
        Training features from selected channels
    X_test_features : np.ndarray
        Test features from selected channels
    """
    from sklearn.feature_selection import mutual_info_classif

    if x_train.ndim != 3 or x_test.ndim != 3:
        raise ValueError("Expected x with shape (epochs, channels, timepoints)")
    
    if x_train.shape[1] != x_test.shape[1]:
        raise ValueError("Train and test must have the same number of channels")
    
    n_epochs_train, n_channels, n_timepoints = x_train.shape
    n_epochs_test = x_test.shape[0]
    
    # Extract initial DWT features for all channels to compute MI
    wavelet_name = "db4"
    max_level = pywt.dwt_max_level(n_timepoints, 8)
    
    if max_level < 1:
        raise ValueError(
            "Time series is too short for a discrete wavelet transform with the chosen wavelet"
        )
    
    train_features_all = []
    for epoch in range(n_epochs_train):
        channel_coeffs = []
        for channel in range(n_channels):
            coeffs = pywt.wavedec(
                x_train[epoch, channel],
                wavelet=wavelet_name,
                level=max_level,
                mode="symmetric",
            )
            channel_coeffs.append(np.concatenate([np.asarray(c) for c in coeffs]))
        train_features_all.append(np.concatenate(channel_coeffs))
    
    train_features_all = np.array(train_features_all)
    
    # Compute mutual information for each channel
    mi_scores = mutual_info_classif(train_features_all, y_train, random_state=42)
    coeffs_per_channel = train_features_all.shape[1] // n_channels
    channel_mi = np.zeros(n_channels)
    
    for ch in range(n_channels):
        channel_mi[ch] = mi_scores[ch * coeffs_per_channel:(ch + 1) * coeffs_per_channel].mean()
    
    # Select top k channels
    selected_channels = np.argsort(channel_mi)[-k_channels:]
    selected_channels = np.sort(selected_channels)  # Sort for consistent ordering
    
    # Extract features using only selected channels
    def extract_selected_channels(X):
        n_epochs = X.shape[0]
        features = []
        for epoch in range(n_epochs):
            channel_coeffs = []
            for channel in selected_channels:
                coeffs = pywt.wavedec(
                    X[epoch, channel],
                    wavelet=wavelet_name,
                    level=max_level,
                    mode="symmetric",
                )
                channel_coeffs.append(np.concatenate([np.asarray(c) for c in coeffs]))
            features.append(np.concatenate(channel_coeffs))
        return np.array(features)
    
    x_train_features = extract_selected_channels(x_train)
    x_test_features = extract_selected_channels(x_test)
    
    return x_train_features, x_test_features

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

def smooth_pseudo_wigner_ville_distribution(x: np.ndarray, sfreq: float) -> np.ndarray:
    n_epochs, n_channels, n_timepoints = x.shape
    
    # Patch for scipy compatibility
    from scipy.integrate import trapezoid
    import scipy.integrate
    scipy.integrate.trapz = trapezoid  # type: ignore
    
    from scipy.signal.windows import hamming
    scipy.signal.hamming = hamming  # type: ignore
    
    from tftb.processing import WignerVilleDistribution
    from scipy.ndimage import gaussian_filter
    
    tfr_list = []
    for epoch in range(n_epochs):
        epoch_tfr_list = []
        for channel in range(n_channels):
            # Process single channel at a time
            signal = x[epoch, channel, :]
            # downsample long signals to cap memory/time
            MAX_SIGNAL_LEN = 2048
            if signal.shape[-1] > MAX_SIGNAL_LEN:
                from scipy.signal import resample
                signal_proc = resample(signal, MAX_SIGNAL_LEN)
            else:
                signal_proc = signal

            wvd = WignerVilleDistribution(signal_proc, fs=sfreq)
            result = wvd.run()
            # normalize result into a 2D ndarray (freq, time)
            if isinstance(result, np.ndarray):
                channel_tfr = result
            else:
                # result may be a list/sequence of arrays (possibly variable lengths)
                rows = [np.asarray(r) for r in result]
                # if rows already 2D-compatible, try vstack
                try:
                    channel_tfr = np.vstack(rows)
                except Exception:
                    # pad 1D rows to same length
                    rows1 = [r.ravel() for r in rows]
                    maxlen = max(r.shape[0] for r in rows1)
                    padded_rows = [np.pad(r, (0, maxlen - r.shape[0]), mode="constant") for r in rows1]
                    channel_tfr = np.vstack(padded_rows)
            epoch_tfr_list.append(channel_tfr)
        
        # Pad channel tfrs to common shape then stack
        max_freq = max(arr.shape[0] for arr in epoch_tfr_list)
        max_time = max(arr.shape[1] for arr in epoch_tfr_list)
        padded = []
        for arr in epoch_tfr_list:
            pad_freq = max_freq - arr.shape[0]
            pad_time = max_time - arr.shape[1]
            pad_width = ((0, pad_freq), (0, pad_time))
            padded.append(np.pad(arr, pad_width, mode="constant", constant_values=0))

        epoch_tfr = np.stack(padded, axis=0)
        tfr_list.append(epoch_tfr)
    
    tfr = np.stack(tfr_list, axis=0)
    
    # Apply gaussian filter
    smoothed_tfr = gaussian_filter(tfr, sigma=(0, 0, 1, 0))
    
    return smoothed_tfr

def transform_to_wigner_ville_features(
    x: np.ndarray, 
    sfreq: float,
    mode: Literal["stats", "full"],
    stats: list[str] = ["mean"]) -> np.ndarray:

    """
    Extract features from the smoothed pseudo-Wigner-Ville distribution.
    """

    tfr = smooth_pseudo_wigner_ville_distribution(x, sfreq)
    n_epochs, n_channels, n_freqs, n_timepoints = tfr.shape

    if mode == "stats":
        if stats is None:
            raise ValueError("stats parameter must be specified when mode='stats'")
        stat_blocks = []
        for channel in range(n_channels):
            channel_tfr = tfr[:, channel, :, :]
            for stat in stats:
                if stat == "mean":
                    stat_blocks.append(channel_tfr.mean(axis=-1))
                elif stat == "std":
                    stat_blocks.append(channel_tfr.std(axis=-1))
                else:
                    raise ValueError(f"Unsupported stats option: {stat}")
        return np.concatenate(stat_blocks, axis=1)  # shape (epochs, channels*freqs)
    elif mode == "full":
        return tfr.reshape(n_epochs, n_channels * n_freqs * n_timepoints)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    