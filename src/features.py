from etils.etree import stack
import numpy as np
import mne
from typing import Literal
import pywt
import numpy as np
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
    n_overlap : int
        Number of timepoints to overlap between segments

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
    max_signal_len: int = 512,
    ) -> np.ndarray:
    """Transform EEG data into time-frequency features.

    For Morlet transforms:
    - in_bands=False returns flattened power across all frequency bins and timepoints.
    - in_bands=True returns flattened band-aggregated amplitude across timepoints.
    - max_signal_len: cap signal length before expensive TFR computation to prevent memory overflow
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
    """
    Transform EEG signals into compact bandpower statistics using
    Morlet continuous wavelet transforms (CWT).

    Pipeline
    --------
    1. Compute Morlet TFR power
    2. Average frequencies within each canonical EEG band
    3. Produce a band-power time series
    4. Compute summary statistics across time:
        - mean
        - standard deviation
        - Shannon entropy

    Parameters
    ----------
    x : np.ndarray
        EEG array with shape:
            (epochs, channels, timepoints)

    sfreq : float
        Sampling frequency in Hz.

    bands : dict[str, tuple[float, float]]
        EEG frequency bands.
        Defaults to canonical EEG bands.

    freqs_per_band : int
        Number of Morlet center frequencies sampled
        within each band.

    decim : int
        Temporal decimation factor during TFR computation.
        Larger values reduce memory usage.

    n_cycles_scale : float
        Controls wavelet duration:
            n_cycles = clip(freq / n_cycles_scale, 3, 12)

    entropy_bins : int
        Number of histogram bins used for Shannon entropy.

    Returns
    -------
    np.ndarray
        Shape:
            (epochs, channels * n_bands * 3)

        Features are ordered as:
            [mean, std, entropy]
        for each band and channel.
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

    # ---------------------------------------------------------
    # Build frequency grid
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # Morlet wavelet settings
    # ---------------------------------------------------------


    min_cycles = 3
    max_cycles = 12

    n_cycles = freqs / n_cycles_scale
    n_cycles = np.clip(n_cycles, min_cycles, max_cycles)

    # --- HARD SAFETY CAP based on signal length ---
    max_wavelet_length = x.shape[-1] / 3  # conservative

    # wavelet length ≈ n_cycles * sfreq / freq
    wavelet_lengths = n_cycles * sfreq / freqs

    too_long = wavelet_lengths > max_wavelet_length

    if np.any(too_long):
        scale = max_wavelet_length / wavelet_lengths[too_long]
        n_cycles[too_long] *= scale

    # ---------------------------------------------------------
    # Compute TFR power
    # Shape:
    # (epochs, channels, freqs, time)
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # Aggregate frequencies into band-power time series
    # ---------------------------------------------------------

    band_power_series = []

    for band_name in bands:

        freq_slice = band_slices[band_name]

        # Mean power within band
        #
        # Shape:
        # (epochs, channels, time)
        #
        band_power = power[:, :, freq_slice, :].mean(axis=2)

        band_power_series.append(band_power)

    # Shape:
    # (epochs, channels, bands, time)
    #
    band_power_series = np.stack(band_power_series, axis=2)

    # ---------------------------------------------------------
    # Feature extraction
    # ---------------------------------------------------------

    feature_blocks = []

    n_epochs, n_channels, n_bands, _ = band_power_series.shape

    for band_idx in range(n_bands):

        band_ts = band_power_series[:, :, band_idx, :]

        # ---------------------------------------------
        # Mean across time
        # ---------------------------------------------

        mean_feat = band_ts.mean(axis=-1)

        # ---------------------------------------------
        # Standard deviation across time
        # ---------------------------------------------

        std_feat = band_ts.std(axis=-1)

        # ---------------------------------------------
        # Shannon entropy across time
        # ---------------------------------------------

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

        # ---------------------------------------------
        # Stack features
        # Shape:
        # (epochs, channels, 3)
        # ---------------------------------------------

        band_features = np.stack(
            [
                mean_feat,
                std_feat,
                entropy_feat,
            ],
            axis=-1,
        )

        feature_blocks.append(band_features)

    # ---------------------------------------------------------
    # Final feature tensor
    #
    # Shape:
    # (epochs, channels, bands, features)
    # ---------------------------------------------------------

    features = np.stack(feature_blocks, axis=2)

    # Flatten:
    #
    # (epochs, channels * bands * features)
    #
    if stack_channels:
        return features.reshape(n_epochs, -1)
    else:
        # shape (epochs, channels, bands, features)
        return features

def transform_to_stft_bands_stats(
    x: np.ndarray,
    sfreq: float,
    bands: dict[str, tuple[float, float]] | None = None,
    n_fft: int = 256,
    n_overlap: int | None = None,
    window: str = "hann",
    downsample_to_freq: float | None = None,
    entropy_bins: int = 64,
    stack_channels: bool = True,
):
    """
    Compute STFT-based band-power statistics using MNE's STFT (falls back to SciPy).

    For each epoch and channel the short-time Fourier transform is computed and the
    power spectral density over time is aggregated into the requested frequency
    bands. For each band we compute three summary statistics across time:
    - mean
    - standard deviation
    - Shannon entropy (computed from the time-power histogram)

    Returns a triple similar to other feature helpers:
      - stats_per_band: (epochs, channels, bands, 3)
      - features3d: (epochs, channels, bands*3)
      - features_2d: (epochs, channels * bands * 3)
    """
    if x.ndim != 3:
        raise ValueError(f"Expected x with shape (epochs, channels, timepoints), got {x.shape}")

    if bands is None:
        bands = {
            "delta": (1.0, 4.0),
            "theta": (4.0, 8.0),
            "alpha": (8.0, 13.0),
            "beta": (13.0, 30.0),
            "gamma": (30.0, 100.0),
        }

    n_epochs, n_channels, n_timepoints = x.shape

    nperseg = min(n_fft, n_timepoints)
    noverlap = n_overlap if n_overlap is not None else max(0, nperseg // 2)

    # Compute STFT using SciPy (SciPy's `stft` is robust and available).
    from scipy.signal import stft as _scipy_stft

    # Temporary containers
    freqs_cache = None
    times_cache = None
    epoch_channel_power = []  # collect per-epoch/channel power arrays to determine time axis length

    for ep in range(n_epochs):
        channel_power_list = []
        for ch in range(n_channels):
            sig = x[ep, ch, :]

            # Use SciPy STFT
            f, t, Zxx = _scipy_stft(sig, fs=sfreq, nperseg=nperseg, noverlap=noverlap, window=window, boundary=None)

            power = (np.abs(Zxx) ** 2).astype(np.float32)  # shape (freqs, times)

            if freqs_cache is None:
                freqs_cache = f
                times_cache = t

            channel_power_list.append(power)

        epoch_channel_power.append(channel_power_list)

    # Determine STFT time axis length
    if times_cache is None or freqs_cache is None:
        raise RuntimeError("STFT computation failed; no frequency/time grid was produced")

    n_times_stft = times_cache.shape[0]
    n_freqs = freqs_cache.shape[0]

    # Build band-time series by averaging power within each band
    band_power_series = np.zeros((n_epochs, n_channels, len(bands), n_times_stft), dtype=np.float32)

    band_names = list(bands.keys())
    for b_idx, (band_name, (fmin, fmax)) in enumerate(bands.items()):
        freq_mask = (freqs_cache >= fmin) & (freqs_cache <= fmax)
        if not np.any(freq_mask):
            raise ValueError(f"Band '{band_name}' ({fmin}, {fmax}) has no STFT frequency bins. Consider changing n_fft or bands.")

        for ep in range(n_epochs):
            for ch in range(n_channels):
                power = epoch_channel_power[ep][ch]
                # power shape: (freqs, times)
                band_power_series[ep, ch, b_idx, :] = power[freq_mask, :].mean(axis=0)

    # Optional temporal downsample of the band_power_series
    if downsample_to_freq is not None:
        if downsample_to_freq <= 0:
            raise ValueError(f"downsample_to_freq must be > 0, got {downsample_to_freq}")
        from scipy.signal import resample
        new_n_timepoints = max(1, int(np.round(n_timepoints * (downsample_to_freq / sfreq))))
        band_power_series = resample(band_power_series, new_n_timepoints, axis=-1)

    # Compute summary statistics across time (mean, std, entropy)
    means = band_power_series.mean(axis=-1)
    stds = band_power_series.std(axis=-1)

    # Shannon entropy computed from histogram of power over time
    eps = 1e-12
    entropy_block = np.zeros((n_epochs, n_channels, len(bands)), dtype=np.float32)

    for ep in range(n_epochs):
        for ch in range(n_channels):
            for b_idx in range(len(band_names)):
                ts = band_power_series[ep, ch, b_idx, :]
                hist, _ = np.histogram(ts, bins=entropy_bins, density=True)
                hist = hist + eps
                hist = hist / hist.sum()
                entropy_block[ep, ch, b_idx] = entropy(hist, base=2)

    stats_per_band = np.stack([means, stds, entropy_block], axis=-1)  # (epochs, channels, bands, 3)

    features3d = stats_per_band.reshape(n_epochs, n_channels, len(band_names) * 3)
    features_2d = stats_per_band.reshape(n_epochs, -1)

    return stats_per_band, features3d, features_2d

def tfr_mortlet_to_cnn(
    x: np.ndarray,
    sfreq: float,
    target_size: int = 256,
    n_freqs: int = 256,
    collapse_channels: bool = True,
    pre_downsample_to_sfreq: float | None = 128.0,
    max_signal_len: int = 512,
) -> np.ndarray:
    """Compute Morlet time-frequency power and return images for CNN.

    Returns array shaped (epochs, H, W, C) where H=n_freqs, W=target_size.
    If `collapse_channels` is True the channel axis is averaged to produce C=1.
    This builds on `transform_to_time_frequency` and performs minimal reshaping.
    max_signal_len: cap signal length before Morlet computation to prevent memory overflow.
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

def transform_to_dwt_level_stats(
    x: np.ndarray,
    sfreq: float,
    downsample_to_freq: float | None = None,
):
    """
    DWT-based EEG feature extraction.

    Parameters
    ----------
    x : np.ndarray
        Shape (epochs, channels, timepoints)

    sfreq : float
        Sampling frequency (Hz)

    downsample_to_freq : float | None
        Optional temporal downsampling after feature extraction

    Returns
    -------
    features_4d : np.ndarray
        (epochs, channels, timepoints_downsampled, dwt_levels_used)

    features_3d : np.ndarray
        (epochs, channels, dwt_levels_used * stats_per_level)

    features_2d : np.ndarray
        Flattened version for ML models
        (epochs, n_features)
    """

    wavelet_name = "db4"
    n_epochs, n_channels, n_timepoints = x.shape

    # ------------------------------------------------------------
    # DWT setup
    # ------------------------------------------------------------
    max_level = pywt.dwt_max_level(n_timepoints, filter_len=8)

    if max_level < 1:
        raise ValueError("Signal too short for DWT")

    # ------------------------------------------------------------
    # Frequency mapping for each DWT level
    # ------------------------------------------------------------
    # Each level j corresponds approximately to:
    # [fs/2^(j+1), fs/2^j]

    level_freqs = {}
    for j in range(1, max_level + 1):
        f_low = sfreq / (2 ** (j + 1))
        f_high = sfreq / (2 ** j)
        level_freqs[j] = (f_low, f_high)

    # ------------------------------------------------------------
    # Keep only levels that intersect 0–100 Hz
    # ------------------------------------------------------------
    valid_levels = [
        j for j, (f_low, f_high) in level_freqs.items()
        if f_high <= 100 and f_high > 0
    ]

    if len(valid_levels) == 0:
        raise ValueError("No DWT levels fall in 0–100 Hz range")

    n_levels = len(valid_levels)

    # ------------------------------------------------------------
    # Output tensor: (epochs, channels, time, levels)
    # ------------------------------------------------------------
    features = np.zeros(
        (n_epochs, n_channels, n_timepoints, n_levels),
        dtype=np.float32
    )

    # ------------------------------------------------------------
    # DWT decomposition
    # ------------------------------------------------------------
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

    # ------------------------------------------------------------
    # Optional downsampling in time domain
    # ------------------------------------------------------------
    if downsample_to_freq is not None:

        ratio = downsample_to_freq / sfreq
        new_t = int(n_timepoints * ratio)

        downsampled = np.zeros(
            (n_epochs, n_channels, new_t, n_levels),
            dtype=np.float32
        )

        for i in range(n_levels):
            for ch in range(n_channels):
                downsampled[:, ch, :, i] = resample_poly(
                    features[:, ch, :, i],
                    up=int(downsample_to_freq),
                    down=int(sfreq),
                    axis=-1
                )[:, :new_t]

        features = downsampled

    # ------------------------------------------------------------
    # Summarise each level across time: mean, std, entropy
    # ------------------------------------------------------------
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

def select_channels_by_mutual_info(
    x_train: np.ndarray,
    y_train: np.ndarray,
    k_channels: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Select top k channels based on mutual information with target.
    Returns:
    new_x_train : np.ndarray
        Shape (epochs, k_channels, timepoints)
    channel_indices : np.ndarray
        Indices of the selected channels in the original data
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

def smooth_pseudo_wigner_ville_distribution(
    x: np.ndarray,
    sfreq: float,
    max_signal_len: int = 512,
) -> np.ndarray:
    n_epochs, n_channels, n_timepoints = x.shape
    
    # Patch for scipy compatibility
    from scipy.integrate import trapezoid
    import scipy.integrate
    scipy.integrate.trapz = trapezoid  # type: ignore
    
    from scipy.signal.windows import hamming
    scipy.signal.hamming = hamming
    
    from tftb.processing import WignerVilleDistribution
    from scipy.ndimage import gaussian_filter
    
    tfr_list = []
    for epoch in range(n_epochs):
        epoch_tfr_list = []
        for channel in range(n_channels):
            # Process single channel at a time
            signal = x[epoch, channel, :]
            # downsample long signals to cap memory/time
            # Downsample long signals to cap memory/time before the WVD step.
            if max_signal_len > 0 and signal.shape[-1] > max_signal_len:
                from scipy.signal import resample
                signal_proc = resample(signal, max_signal_len)
            else:
                signal_proc = signal

            wvd = WignerVilleDistribution(signal_proc, fs=sfreq)
            tfr, ts, freqs = wvd.run()
            
            channel_tfr = np.asarray(tfr)
            
            # normalize result into a 2D ndarray (freq, time)
            if isinstance(channel_tfr, np.ndarray):
                channel_tfr = channel_tfr
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
    smoothed_tfr = gaussian_filter(tfr, sigma=(0, 0, 1, 1))  # smooth only in freq and time dimensions
    
    return smoothed_tfr

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
    Extract features from the smoothed pseudo-Wigner-Ville distribution.

    Parameters
    ----------
    x, sfreq: as usual
    mode: "stats" returns aggregated statistics, "full" returns flattened TFR
    stats: list of statistics to compute when mode=="stats" (e.g. ["mean", "std"])
    average_over: whether to average statistics over the temporal axis ("time", default)
        which yields per-frequency features, or over the frequency axis ("frequency")
        which yields per-timepoint features.
    downsample_to_freq: optional temporal downsampling frequency (Hz)
    freq_aggregation: "none" (no freq aggregation), "log_bins" (aggregate into n_freq_bins 
        logarithmic intervals), or "bands" (aggregate into predefined frequency bands)
    n_freq_bins: number of logarithmic frequency bins (used if freq_aggregation="log_bins")
    bands: dict of frequency bands (used if freq_aggregation="bands"), 
        e.g. {"delta": (1, 4), "theta": (4, 8), ...}
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

    