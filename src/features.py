import numpy as np
import mne
from typing import Literal
import pywt
from scipy.signal import resample_poly
from scipy.stats import entropy

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
    n_overlap: int = True,
    stack_channels: bool = True
) -> np.ndarray:
    """Compute band-power features from EEG using Welch PSD.

    This computes the power spectral density (PSD) with MNE's
    Welch implementation and either averages power within each
    supplied frequency band or returns the full band-limited PSD
    bins depending on ``mean``.

    Args:
        x (np.ndarray): EEG array shaped (epochs, channels, timepoints).
        sfreq (float): Sampling frequency in Hz.
        bands (dict[str, tuple[float, float]]): Mapping of band name
            to (fmin, fmax) in Hz, e.g. {"delta": (1, 4)}.
        mean (bool): If True, average PSD within each band and return
            one value per band; if False, return all frequency bins
            inside each band. Defaults to True.
        n_overlap (int | bool): Number of timepoints to overlap between
            Welch segments. If False or 0, no overlap is used. Defaults
            to True (interpreted below when computing n_overlap).
        stack_channels (bool): If True, flatten channels into the
            last feature axis (shape (epochs, channels * features)).

    Returns:
        np.ndarray: Feature array. If ``mean`` is True the shape is
            (epochs, channels * n_bands) when ``stack_channels`` is
            True, or (epochs, channels, n_bands) otherwise. If
            ``mean`` is False the returned shape is
            (epochs, channels * total_band_freq_bins) when
            ``stack_channels`` is True, or (epochs, channels,
            total_band_freq_bins) otherwise.
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
        n_overlap=n_fft_param // 5 if n_overlap else 0,
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
    max_signal_len: int = 512,
    ) -> np.ndarray:
    """Compute time-frequency representations for EEG signals.

    Supports Morlet continuous wavelet transform (CWT) and a
    discrete wavelet transform (DWT) path. For Morlet, the function
    can either return the full power tensor or aggregate power within
    supplied bands.

    Args:
        x (np.ndarray): EEG array shaped (epochs, channels, timepoints).
        sfreq (float): Sampling frequency in Hz.
        algorithm (str): Either "morlet" (default) or "dwt".
        in_bands (bool): When using the Morlet algorithm, if True
            aggregate Morlet amplitudes into the provided ``bands``
            and return band-wise amplitudes; if False return power
            across all sampled frequencies and timepoints.
        downsample_to_freq (float | None): Optional temporal
            downsampling frequency (Hz) applied to the time axis of
            the TFR output. If None no temporal resampling is done.
        bands (dict | None): Mapping of band name to (fmin, fmax).
            Required when ``in_bands`` is True.
        n_freqs (int): Number of Morlet center frequencies to sample
            between 1 Hz and a Nyquist-safe upper bound. Must be >=2.
        max_signal_len (int): If >0 and the input time series is
            longer than this value the signal will be resampled to
            ``max_signal_len`` before computing Morlet transforms to
            limit memory usage.

    Returns:
        np.ndarray: Flattened feature array. For Morlet with
            ``in_bands=False`` returns an array of shape
            (epochs, channels * freqs * timepoints) after optional
            downsampling. For Morlet with ``in_bands=True`` returns
            (epochs, channels * n_bands * new_timepoints). For DWT
            returns stacked coefficients per epoch and channel.
    """
    if x.ndim != 3:
        raise ValueError(f"Expected x with shape (epochs, channels, timepoints), got {x.shape}")
    if n_freqs < 2:
        raise ValueError(f"n_freqs must be >= 2, got {n_freqs}")

    n_epochs, n_channels, n_timepoints = x.shape

    if algorithm == "morlet":

        # Cap signal length before expensive Morlet computation to prevent memory overflow
        if max_signal_len > 0 and n_timepoints > max_signal_len:
            from scipy.signal import resample
            x = resample(x, max_signal_len, axis=-1)
            n_timepoints = max_signal_len

        # Specify Morlet frequencies with a Nyquist-safe upper bound.
        max_freq = min(100.0, (0.5 * sfreq) - 1e-6)
        if max_freq <= 1.0:
            raise ValueError(
                f"Sampling frequency too low for Morlet range: sfreq={sfreq}, max usable freq={max_freq}"
            )
        # log-spaced center frequencies between 1 Hz and the safe max
        freqs = np.logspace(np.log10(1.0), np.log10(max_freq), num=n_freqs)
        n_cycles = freqs / 2.0  # example: more cycles for higher freqs

        if in_bands:
            if bands is None:
                raise ValueError("bands dictionary must be provided when in_bands=True")

            # compute complex-valued TFR to extract amplitude per freq/time
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

        # compute power TFR (used when not aggregating into bands)
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

        # collect per-epoch concatenated DWT coefficients
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

def transform_to_morlet_stats(
    x: np.ndarray,
    sfreq: float,
    bands: dict[str, tuple[float, float]] | None = None,
    freqs_per_band: int = 5,
    decim: int = 1,
    n_cycles_scale: float = 2.0,
    entropy_bins: int = 64,
    stack_channels: bool = True,
) -> np.ndarray:
    """Extract compact band-power statistics using Morlet CWT.

    The function computes Morlet time-frequency power, averages
    frequencies within canonical EEG bands to produce band-power time
    series, and then summarizes each band-time series by its mean,
    standard deviation and Shannon entropy. The resulting features
    are concatenated per channel and band.

    Args:
        x (np.ndarray): EEG array shaped (epochs, channels, timepoints).
        sfreq (float): Sampling frequency in Hz.
        bands (dict | None): Mapping of band name to (fmin, fmax).
            Defaults to canonical EEG bands if None.
        freqs_per_band (int): Number of Morlet center frequencies to
            sample inside each band. Must be >=1.
        decim (int): Temporal decimation factor passed to MNE TFR
            computation to reduce memory usage. Must be >=1.
        n_cycles_scale (float): Controls wavelet duration. Per-
            frequency cycles are computed as ``freq / n_cycles_scale``
            then clipped to a reasonable range.
        entropy_bins (int): Number of histogram bins for Shannon
            entropy computation across time.
        stack_channels (bool): If True flatten channels into the
            feature dimension producing shape (epochs, channels *
            bands * 3). If False returns (epochs, channels, bands, 3).

    Returns:
        np.ndarray: Feature array of shape (epochs, channels * n_bands * 3)
            when ``stack_channels`` is True, otherwise
            (epochs, channels, n_bands, 3). The last dimension for
            each band contains [mean, std, entropy].
    """

    if x.ndim != 3:
        raise ValueError(
            f"Expected x with shape "
            f"(epochs, channels, timepoints), got {x.shape}"
        )

    if sfreq <= 0:
        raise ValueError(f"sfreq must be > 0, got {sfreq}")

    if freqs_per_band < 1:
        raise ValueError(
            f"freqs_per_band must be >= 1, got {freqs_per_band}"
        )

    if decim < 1:
        raise ValueError(f"decim must be >= 1, got {decim}")

    if bands is None:
        bands = {
            "delta": (1.0, 4.0),
            "theta": (4.0, 8.0),
            "alpha": (8.0, 13.0),
            "beta": (13.0, 30.0),
            "gamma": (30.0, 100.0),
        }

    # Build frequency grid

    freq_blocks = []
    band_slices = {}

    start_idx = 0

    nyquist = sfreq / 2.0

    for band_name, (fmin, fmax) in bands.items():

        if fmin >= fmax:
            raise ValueError(
                f"Invalid band '{band_name}': ({fmin}, {fmax})"
            )

        if fmax >= nyquist:
            raise ValueError(
                f"Band '{band_name}' exceeds Nyquist frequency "
                f"({nyquist:.2f} Hz)"
            )

        band_freqs = np.logspace(
            np.log10(fmin),
            np.log10(fmax),
            freqs_per_band,
        )

        end_idx = start_idx + len(band_freqs)

        band_slices[band_name] = slice(start_idx, end_idx)

        freq_blocks.append(band_freqs)

        start_idx = end_idx

    freqs = np.concatenate(freq_blocks)

    # Morlet wavelet settings
    min_cycles = 3
    max_cycles = 12

    n_cycles = freqs / n_cycles_scale
    n_cycles = np.clip(n_cycles, min_cycles, max_cycles)

    # Hard safety cap based on signal length
    max_wavelet_length = x.shape[-1] / 3  # conservative

    # wavelet length ≈ n_cycles * sfreq / freq
    wavelet_lengths = n_cycles * sfreq / freqs

    too_long = wavelet_lengths > max_wavelet_length

    if np.any(too_long):
        scale = max_wavelet_length / wavelet_lengths[too_long]
        n_cycles[too_long] *= scale

    # Compute TFR power (epochs, channels, freqs, time)

    power = mne.time_frequency.tfr_array_morlet(
        x,
        sfreq=sfreq,
        freqs=freqs,
        n_cycles=n_cycles,
        output="power",
        decim=decim,
        n_jobs=1,
        verbose=False,
    )

    # Aggregate frequencies into band-power time series

    band_power_series = []

    for band_name in bands:

        freq_slice = band_slices[band_name]

        # Mean power within band (shape: epochs, channels, time)
        band_power = power[:, :, freq_slice, :].mean(axis=2)

        band_power_series.append(band_power)

    # Shape: (epochs, channels, bands, time)
    band_power_series = np.stack(band_power_series, axis=2)

    # Feature extraction

    feature_blocks = []

    n_epochs, n_channels, n_bands, _ = band_power_series.shape

    for band_idx in range(n_bands):

        band_ts = band_power_series[:, :, band_idx, :]

        # Mean across time

        mean_feat = band_ts.mean(axis=-1)

        # Standard deviation across time

        std_feat = band_ts.std(axis=-1)

        # Shannon entropy across time

        entropy_feat = np.zeros(
            (n_epochs, n_channels),
            dtype=np.float32,
        )

        for epoch_idx in range(n_epochs):
            for channel_idx in range(n_channels):

                ts = band_ts[epoch_idx, channel_idx]

                hist, _ = np.histogram(
                    ts,
                    bins=entropy_bins,
                    density=True,
                )

                hist = hist + 1e-12
                hist = hist / hist.sum()

                entropy_feat[epoch_idx, channel_idx] = entropy(
                    hist,
                    base=2,
                )

        # Stack features (shape: epochs, channels, 3)
        band_features = np.stack(
            [
                mean_feat,
                std_feat,
                entropy_feat,
            ],
            axis=-1,
        )

        feature_blocks.append(band_features)

    # Final feature tensor (shape: epochs, channels, bands, features)
    features = np.stack(feature_blocks, axis=2)

    # Flatten (epochs, channels * bands * features)
    if stack_channels:
        return features.reshape(n_epochs, -1)
    else:
        # shape (epochs, channels, bands, features)
        return features

def tfr_mortlet_to_cnn(
    x: np.ndarray,
    sfreq: float,
    target_size: int = 256,
    n_freqs: int = 256,
    collapse_channels: bool = True,
    pre_downsample_to_sfreq: float | None = 128.0,
    max_signal_len: int = 512,
) -> np.ndarray:
    """Produce CNN-ready TFR images using Morlet power.

    This function computes a Morlet time-frequency representation and
    reshapes it into image tiles of shape (epochs, freqs, time, channels)
    suitable for CNN input. Optionally channels are averaged into a
    single image channel.

    Args:
        x (np.ndarray): EEG array shaped (epochs, channels, timepoints).
        sfreq (float): Sampling frequency in Hz.
        target_size (int): Desired temporal width (pixels) of output
            images. The function computes a downsample frequency so
            that the resulting time axis approximates ``target_size``.
        n_freqs (int): Number of frequency bins (image height).
        collapse_channels (bool): If True average channels producing
            images with a single channel (C=1).
        pre_downsample_to_sfreq (float | None): If provided and less
            than ``sfreq``, the input will be downsampled before the
            Morlet computation to reduce compute/memory.
        max_signal_len (int): If >0, signals longer than this value
            will be resampled to ``max_signal_len`` before Morlet
            computation.

    Returns:
        np.ndarray: Images shaped (epochs, n_freqs, target_time, C)
            where C is 1 if channels were collapsed, otherwise the
            original channel count.
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
        max_signal_len=max_signal_len,
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

def transform_to_dwt_level_stats(
    x: np.ndarray,
    sfreq: float,
    downsample_to_freq: float | None = None,
):
    """Extract DWT-based features and summary statistics per level.

    Performs a multi-level discrete wavelet decomposition per epoch
    and channel, maps DWT levels to approximate frequency bands and
    optionally downsamples the time axis. Summary statistics (mean,
    std and entropy) are produced per level.

    Args:
        x (np.ndarray): EEG array shaped (epochs, channels, timepoints).
        sfreq (float): Sampling frequency in Hz.
        downsample_to_freq (float | None): Optional target temporal
            sampling frequency (Hz) to resample the DWT-derived time
            series to. If None no downsampling is applied.

    Returns:
        tuple: The implementation computes several derived tensors:
            - features_4d: (epochs, channels, timepoints_downsampled, levels)
            - features_3d: (epochs, channels, levels * stats_per_level)
            - features_2d: (epochs, n_features) flattened for ML models
            The function currently returns the computed summaries.
    """

    wavelet_name = "db4"
    n_epochs, n_channels, n_timepoints = x.shape

    # DWT setup
    max_level = pywt.dwt_max_level(n_timepoints, filter_len=8)

    if max_level < 1:
        raise ValueError("Signal too short for DWT")

    # Frequency mapping for each DWT level (each level j ≈ [fs/2^(j+1), fs/2^j])
    level_freqs = {}
    for j in range(1, max_level + 1):
        f_low = sfreq / (2 ** (j + 1))
        f_high = sfreq / (2 ** j)
        level_freqs[j] = (f_low, f_high)

    # Keep only levels that intersect 0–100 Hz
    valid_levels = [
        j for j, (f_low, f_high) in level_freqs.items()
        if f_high <= 100 and f_high > 0
    ]

    if len(valid_levels) == 0:
        raise ValueError("No DWT levels fall in 0–100 Hz range")

    n_levels = len(valid_levels)
    print(f"Using DWT levels: {valid_levels} with frequency ranges {[level_freqs[j] for j in valid_levels]} Hz", flush=True)

    # Output tensor: (epochs, channels, time, levels)
    features = np.zeros(
        (n_epochs, n_channels, n_timepoints, n_levels),
        dtype=np.float32
    )

    # DWT decomposition
    for ep in range(n_epochs):
        for ch in range(n_channels):

            signal = x[ep, ch]

            coeffs = pywt.wavedec(
                signal,
                wavelet=wavelet_name,
                level=max_level
            )

            # coeffs = [cA_n, cD_n, ..., cD_1]

            for i, level in enumerate(valid_levels):

                coeff_idx = max_level - level + 1

                coeffs_filtered = [np.zeros_like(c) for c in coeffs]
                coeffs_filtered[coeff_idx] = coeffs[coeff_idx]

                reconstructed = pywt.waverec(
                    coeffs_filtered,
                    wavelet_name
                )[:n_timepoints]

                # band power (simple, robust)
                # store reconstructed time-series for this DWT level
                features[ep, ch, :reconstructed.shape[0], i] = reconstructed

    # Optional downsample reconstructed level time-series in time
    if downsample_to_freq is not None:
        ratio = downsample_to_freq / sfreq
        new_t = int(n_timepoints * ratio)

        downsampled = np.zeros((n_epochs, n_channels, new_t, n_levels), dtype=np.float32)

        for i in range(n_levels):
            for ch in range(n_channels):
                # resample each channel/level time-series to the target rate
                downsampled[:, ch, :, i] = resample_poly(
                    features[:, ch, :, i],
                    up=int(downsample_to_freq),
                    down=int(sfreq),
                    axis=-1,
                )[:, :new_t]

        features = downsampled

    # Summarise each level across time: mean, std, entropy
    # features shape: (epochs, channels, time, levels)
    # compute mean and std across time axis
    means = features.mean(axis=2)
    stds = features.std(axis=2)

    # Shannon entropy computed on power distribution across time
    eps = 1e-12
    power = (features.astype(np.float64) ** 2)
    power_sum = power.sum(axis=2, keepdims=True)
    probs = power / (power_sum + eps)
    probs = np.clip(probs, eps, None)
    entropy = -np.sum(probs * np.log(probs), axis=2)

    # combine stats in order [mean, std, entropy] per level
    stats_array = [means, stds]
    stats_array = [means, stds]
    stats_per_level = np.stack(
        stats_array, axis=-1
    )  # (epochs, channels, levels, n_stats)

    # Flatten for ML: (epochs, n_features)
    features_2d = stats_per_level.reshape(n_epochs, -1)

    # flatten to 3d for CNNs: (epochs, channels, levels*stats)
    features3d = stats_per_level.reshape(n_epochs, n_channels, n_levels * len(stats_array))
    features3d = stats_per_level.reshape(n_epochs, n_channels, n_levels * len(stats_array))

    return stats_per_level, features3d, features_2d

def select_channels_by_mutual_info(
    x_train: np.ndarray,
    y_train: np.ndarray,
    k_channels: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Select top-k channels by mutual information with the target.

    Args:
        x_train (np.ndarray): Training data shaped (epochs, channels, timepoints).
        y_train (np.ndarray): Labels shaped (epochs,) or compatible.
        k_channels (int): Number of channels to select.

    Returns:
        tuple: (new_x_train, selected_indices)
            - new_x_train (np.ndarray): Data sliced to selected channels
              shaped (epochs, k_channels, timepoints).
            - selected_indices (np.ndarray): Indices of the selected
              channels in the original channel axis.
    """
    from sklearn.feature_selection import mutual_info_classif

    if x_train.ndim != 3:
        raise ValueError(f"Expected x_train with shape (epochs, channels, timepoints), got {x_train.shape}")
    
    n_epochs, n_channels, n_timepoints = x_train.shape
    
    # Flatten time dimension for MI computation
    x_flat = x_train.reshape(n_epochs, n_channels * n_timepoints)
    
    mi_scores = mutual_info_classif(x_flat, y_train, random_state=42)
    
    # Average MI scores across timepoints for each channel
    mi_per_channel = np.zeros(n_channels)
    for ch in range(n_channels):
        mi_per_channel[ch] = mi_scores[ch * n_timepoints:(ch + 1) * n_timepoints].mean()
    
    # Select top k channels
    selected_indices = np.argsort(mi_per_channel)[-k_channels:]
    
    # Extract selected channels
    new_x_train = x_train[:, selected_indices, :]

    return new_x_train, selected_indices

def downsample_time(
    x: np.ndarray,
    original_sfreq: float,
    target_sfreq: float,
) -> np.ndarray:
    """Resample the time axis of EEG data to a new sampling rate.

    Args:
        x (np.ndarray): EEG array shaped (epochs, channels, timepoints).
        original_sfreq (float): Original sampling frequency (Hz).
        target_sfreq (float): Desired sampling frequency (Hz).

    Returns:
        np.ndarray: Resampled data shaped (epochs, channels, new_timepoints).
    """
    from scipy.signal import resample

    n_epochs, n_channels, n_timepoints = x.shape
    duration = n_timepoints / original_sfreq  # in seconds
    new_n_timepoints = int(duration * target_sfreq)

    x_downsampled = resample(x, new_n_timepoints, axis=-1)

    return x_downsampled

def transform_to_wigner_ville_features(
    x: np.ndarray,
    sfreq: float,
    mode: Literal["stats", "full"],
    stats: list[str] = ["mean"],
    average_over: Literal["time", "frequency"] = "time",
    downsample_to_freq: float | None = None,
    freq_aggregation: Literal["none", "log_bins", "bands", "cnn"] = "none",
    cnn_target_size: int = 256,
    n_freq_bins: int = 20,
    bands: dict[str, tuple[float, float]] | None = None,
    max_signal_len: int = 512,
) -> np.ndarray:

    """
    NOT CURRENTLY USED - this was experimental. Kept for future reference.
    Extract features from the smoothed pseudo-Wigner-Ville distribution.

    This routine computes either a full time-frequency representation
    (mode="full") or summary statistics across time or frequency
    (mode="stats"). Frequency aggregation options include logarithmic
    binning or aggregation into supplied bands.

    Args:
        x (np.ndarray): EEG array shaped (epochs, channels, timepoints).
        sfreq (float): Sampling frequency in Hz.
        mode (str): "stats" to return aggregated statistics, or "full"
            to return a flattened TFR representation.
        stats (list[str]): List of statistics to compute when ``mode``
            is "stats", e.g. ["mean", "std"].
        average_over (str): Whether to average statistics over the
            "time" axis (default) producing per-frequency features,
            or over "frequency" producing per-timepoint features.
        downsample_to_freq (float | None): Optional temporal downsampling
            frequency (Hz) applied after TFR computation.
        freq_aggregation (str): How to aggregate frequency axis: "none",
            "log_bins", "bands" or "cnn".
        cnn_target_size (int): When producing CNN-style outputs,
            target temporal size used for downsampling.
        n_freq_bins (int): Number of logarithmic frequency bins when
            using "log_bins" aggregation.
        bands (dict | None): Frequency bands mapping when using
            "bands" aggregation.
        max_signal_len (int): Limit input signal length to prevent
            excessive memory use.

    Returns:
        np.ndarray: Depending on ``mode`` and aggregation options either
            aggregated statistics or a flattened full TFR representation.
    """

    if mode == "stats":
        if stats is None:
            raise ValueError("stats parameter must be specified when mode='stats'")
        if average_over not in ("time", "frequency"):
            raise ValueError("average_over must be 'time' or 'frequency'")

        # Patch for scipy compatibility, matching the full-mode path.
        from scipy.integrate import trapezoid
        import scipy.integrate
        scipy.integrate.trapz = trapezoid  # type: ignore

        from scipy.signal.windows import hamming
        scipy.signal.hamming = hamming

        from scipy.signal import resample
        from tftb.processing import WignerVilleDistribution

        def _signal_to_tfr(signal: np.ndarray) -> np.ndarray:
            if max_signal_len > 0 and signal.shape[-1] > max_signal_len:
                signal = resample(signal, max_signal_len)
            wvd = WignerVilleDistribution(signal, fs=sfreq)
            tfr, _, _ = wvd.run()
            return np.asarray(tfr)

        epoch_features: list[np.ndarray] = []
        for epoch in range(x.shape[0]):
            channel_feature_blocks: list[np.ndarray] = []
            for channel in range(x.shape[1]):
                channel_tfr = _signal_to_tfr(x[epoch, channel, :])
                channel_freqs = channel_tfr.shape[0]
                nyquist = 0.5 * sfreq
                freq_vector = np.linspace(0, nyquist, channel_freqs)

                if freq_aggregation == "log_bins":
                    if n_freq_bins < 2:
                        raise ValueError(f"n_freq_bins must be >= 2, got {n_freq_bins}")
                    min_freq = 1.0
                    log_edges = np.logspace(np.log10(min_freq), np.log10(nyquist), n_freq_bins + 1)
                    aggregated_blocks: list[np.ndarray] = []
                    for bin_idx in range(n_freq_bins):
                        fmin, fmax = log_edges[bin_idx], log_edges[bin_idx + 1]
                        freq_mask = (freq_vector >= fmin) & (freq_vector < fmax)
                        if not np.any(freq_mask):
                            continue
                        band_tfr = channel_tfr[freq_mask, :].mean(axis=0)
                        if average_over == "time":
                            aggregated_blocks.append(np.array([band_tfr.mean() if "mean" in stats else 0.0]))
                            if "std" in stats:
                                aggregated_blocks.append(np.array([band_tfr.std()]))
                        else:
                            if "mean" in stats:
                                aggregated_blocks.append(band_tfr.mean(axis=0))
                            if "std" in stats:
                                aggregated_blocks.append(band_tfr.std(axis=0))
                    channel_feature_blocks.append(np.concatenate(aggregated_blocks, axis=0))

                elif freq_aggregation == "bands":
                    if bands is None:
                        raise ValueError("bands dict must be provided when freq_aggregation='bands'")
                    band_feature_blocks: list[np.ndarray] = []
                    for _, (fmin, fmax) in bands.items():
                        freq_mask = (freq_vector >= fmin) & (freq_vector < fmax)
                        if not np.any(freq_mask):
                            continue
                        band_tfr = channel_tfr[freq_mask, :].mean(axis=0)
                        if average_over == "time":
                            if "mean" in stats:
                                band_feature_blocks.append(np.array([band_tfr.mean()]))
                            if "std" in stats:
                                band_feature_blocks.append(np.array([band_tfr.std()]))
                        else:
                            if "mean" in stats:
                                band_feature_blocks.append(band_tfr.mean(axis=0))
                            if "std" in stats:
                                band_feature_blocks.append(band_tfr.std(axis=0))
                    channel_feature_blocks.append(np.concatenate(band_feature_blocks, axis=0))

                else:
                    channel_blocks: list[np.ndarray] = []
                    for stat in stats:
                        if stat == "mean":
                            channel_blocks.append(channel_tfr.mean(axis=-1 if average_over == "time" else -2))
                        elif stat == "std":
                            channel_blocks.append(channel_tfr.std(axis=-1 if average_over == "time" else -2))
                        else:
                            raise ValueError(f"Unsupported stats option: {stat}")
                    channel_feature_blocks.append(np.concatenate(channel_blocks, axis=0))

            epoch_features.append(np.concatenate(channel_feature_blocks, axis=0))

        return np.stack(epoch_features, axis=0)

    elif mode == "full":
        tfr = smooth_pseudo_wigner_ville_distribution(x, sfreq, max_signal_len=max_signal_len)
        n_epochs, n_channels, n_freqs, n_timepoints = tfr.shape
        nyquist = 0.5 * sfreq
        freq_vector = np.linspace(0, nyquist, n_freqs)

        if freq_aggregation == "log_bins":
            if n_freq_bins < 2:
                raise ValueError(f"n_freq_bins must be >= 2, got {n_freq_bins}")
            min_freq = 1.0
            log_edges = np.logspace(np.log10(min_freq), np.log10(nyquist), n_freq_bins + 1)
            aggregated_tfr = np.zeros((n_epochs, n_channels, n_freq_bins, n_timepoints))
            
            for bin_idx in range(n_freq_bins):
                fmin, fmax = log_edges[bin_idx], log_edges[bin_idx + 1]
                freq_mask = (freq_vector >= fmin) & (freq_vector < fmax)
                if freq_mask.sum() > 0:
                    aggregated_tfr[:, :, bin_idx, :] = tfr[:, :, freq_mask, :].mean(axis=2)
            
            tfr = aggregated_tfr
            
        elif freq_aggregation == "bands":
            if bands is None:
                raise ValueError("bands dict must be provided when freq_aggregation='bands'")
            n_bands = len(bands)
            aggregated_tfr = np.zeros((n_epochs, n_channels, n_bands, n_timepoints))
            
            for band_idx, (band_name, (fmin, fmax)) in enumerate(bands.items()):
                freq_mask = (freq_vector >= fmin) & (freq_vector < fmax)
                if freq_mask.sum() > 0:
                    aggregated_tfr[:, :, band_idx, :] = tfr[:, :, freq_mask, :].mean(axis=2)
            
            tfr = aggregated_tfr

        elif freq_aggregation == "cnn":
            # downsample both frequency and time dimensions to produce a fixed-size image per channel
            from scipy.signal import resample
            tfr = resample(tfr, cnn_target_size, axis=2)
            tfr = resample(tfr, cnn_target_size, axis=3)
        
        return tfr.reshape(n_epochs, n_channels * tfr.shape[2] * tfr.shape[3])
    
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    