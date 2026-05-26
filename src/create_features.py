import numpy as np
from features import (
    transform_to_band_power,
    downsample_time,
    transform_to_time_frequency,
    tfr_mortlet_to_cnn,
    transform_to_wigner_ville_features,
    transform_to_dwt_level_stats,
    transform_to_morlet_stats,
    select_channels_by_mutual_info,
)

# ===============================
# == Feature Creation Function ==
# ===============================

def create_features(
    x: np.ndarray,
    feature_type: str,
    y_train: np.ndarray | None = None,
    x_reference: np.ndarray | None = None,
):
    """Create features from raw EEG data based on the specified feature type."""
    bands = {
            "delta": (1.0, 4.0),
            "theta": (4.0, 8.0),
            "alpha": (8.0, 13.0),
            "beta": (13.0, 30.0),
            "gamma": (30.0, 100.0),
        }

    mi_k_channels = 16

    if feature_type == "tfr_morlet":
        return transform_to_time_frequency(x, sfreq=256, algorithm="morlet")
    elif feature_type == "tfr_morlet_cnn":
        # Return images shaped (epochs, H, W, C) suitable for CNN input
        return tfr_mortlet_to_cnn(x, sfreq=256, target_size=256, n_freqs=256, collapse_channels=True, max_signal_len=512)
    elif feature_type == "tfr_morlet_bands":
        tfr = transform_to_time_frequency(
            x, sfreq=256, algorithm="morlet",
            in_bands=True, bands=bands,
            downsample_to_freq=1,  # example downsampling to reduce dimensionality
            n_freqs=20
        )
        return tfr
    elif feature_type == "tfr_morlet_bands_stats":
        return transform_to_morlet_stats(
            x, sfreq=256, bands=bands,
            stack_channels=True
        )
    elif feature_type == "tfr_morlet_bands_stats_mi":
        if y_train is None:
            raise ValueError("y_train must be provided for tfr_morlet_bands_stats_mi")
        y_train_arr = y_train
        tfr = transform_to_morlet_stats(x, sfreq=256, bands=bands, stack_channels=False)
        tfr = tfr.reshape(tfr.shape[0], tfr.shape[1], -1)
        new_x_train, selected_indices = select_channels_by_mutual_info(
            tfr,
            y_train=y_train_arr,
            k_channels=mi_k_channels,
        )
        if x_reference is not None:
            reference_tfr = transform_to_morlet_stats(x_reference, sfreq=256, bands=bands, stack_channels=False)
            reference_tfr = reference_tfr.reshape(reference_tfr.shape[0], reference_tfr.shape[1], -1)  # (epochs, channels, features)   
            new_x_reference = reference_tfr[:, selected_indices, :]
            return new_x_train.reshape(new_x_train.shape[0], -1), new_x_reference.reshape(new_x_reference.shape[0], -1)

        return new_x_train.reshape(new_x_train.shape[0], -1)
    elif feature_type == "tfr_dwt_cmor":
        return transform_to_time_frequency(x, sfreq=256, algorithm="dwt")
    elif feature_type == "swvd_band_mean":
        return transform_to_wigner_ville_features(
            x, sfreq=256, mode="stats", stats=["mean"],
            freq_aggregation="bands", bands=bands
            )
    elif feature_type == "swvd_logbins_mean":
        return transform_to_wigner_ville_features(
            x, sfreq=256, mode="stats", stats=["mean"],
            freq_aggregation="log_bins", n_freq_bins=10
            )
    elif feature_type == "swvd_full":
        return transform_to_wigner_ville_features(x, sfreq=256, mode="full")
    elif feature_type == "swvd_cnn":
        return transform_to_wigner_ville_features(
            x, sfreq=256, mode="full",
            freq_aggregation="cnn", cnn_target_size=128, max_signal_len=512
        )
    elif feature_type == "dwt_stats":
        _, _, tfr = transform_to_dwt_level_stats(x, sfreq=256)
        print(f"DWT stats shape: {tfr.shape}", flush=True)
        return tfr
    elif feature_type == "dwt_stats_mi":
        if y_train is None:
            raise ValueError("y_train must be provided for dwt_stats_mi")
        y_train_arr = y_train
        _, train_features, _ = transform_to_dwt_level_stats(x, sfreq=256)
        new_x_train, selected_indices = select_channels_by_mutual_info(
            train_features,
            y_train=y_train_arr,
            k_channels=mi_k_channels,
        )
        if x_reference is not None:
            _, reference_features, _ = transform_to_dwt_level_stats(x_reference, sfreq=256)
            new_x_reference = reference_features[:, selected_indices, :]
            return new_x_train.reshape(new_x_train.shape[0], -1), new_x_reference.reshape(new_x_reference.shape[0], -1)

        return new_x_train.reshape(new_x_train.shape[0], -1)
    elif feature_type == "bandpower_nostack":
        return transform_to_band_power(x, sfreq=256, bands=bands,mean=False, stack_channels=False)
    elif feature_type == "bandpower_mean":
        return transform_to_band_power(
            x, sfreq=256, bands=bands,
            mean=True, stack_channels=True
        )
    elif feature_type == "bandpower_mean_mi":
        if y_train is None:
            raise ValueError("y_train must be provided for bandpower_mean_mi")
        y_train_arr = y_train
        bandpower = transform_to_band_power(x, sfreq=256, bands=bands, mean=True, stack_channels=False)
        new_x_train, selected_indices = select_channels_by_mutual_info(
            bandpower,
            y_train=y_train_arr,
            k_channels=mi_k_channels,
        )
        if x_reference is not None:
            reference_bandpower = transform_to_band_power(
                x_reference,
                sfreq=256,
                bands=bands,
                mean=True,
                stack_channels=False,
            )
            new_x_reference = reference_bandpower[:, selected_indices, :]
            return new_x_train.reshape(new_x_train.shape[0], -1), new_x_reference.reshape(new_x_reference.shape[0], -1)

        return new_x_train.reshape(new_x_train.shape[0], -1)
    elif feature_type == "downsample_32hz":
        downsample = downsample_time(x, original_sfreq=256, target_sfreq=32)  # example sfreq, adjust as needed
        return downsample.reshape(downsample.shape[0], -1)
    elif feature_type == "downsample_4hz":
        downsample = downsample_time(x, original_sfreq=256, target_sfreq=4)  # example sfreq, adjust as needed
        return downsample.reshape(downsample.shape[0], -1)
    elif feature_type == "stack":
        return x.reshape(x.shape[0], -1)
    elif feature_type == "mean":
        mean = x.mean(axis=-1)
        return mean.reshape(mean.shape[0], -1)
    elif feature_type == "mean_mi":
        if y_train is None:
            raise ValueError("y_train must be provided for mean_mi")
        y_train_arr = y_train
        mean = x.mean(axis=-1)[:, :, None]
        new_x_train, selected_indices = select_channels_by_mutual_info(
            mean,
            y_train=y_train_arr,
            k_channels=mi_k_channels,
        )
        if x_reference is not None:
            reference_mean = x_reference.mean(axis=-1)[:, :, None]
            new_x_reference = reference_mean[:, selected_indices, :]
            return new_x_train.reshape(new_x_train.shape[0], -1), new_x_reference.reshape(new_x_reference.shape[0], -1)

        return new_x_train.reshape(new_x_train.shape[0], -1)
    else:
        raise ValueError(f"Unknown feature type: {feature_type}")