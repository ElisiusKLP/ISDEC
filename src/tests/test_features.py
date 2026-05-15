"""
Test and visualize feature extraction methods from features.py

This script:
1. Loads raw EEG data from joblib files
2. Applies different feature extraction methods
3. Creates visualizations of the extracted features
4. Saves plots to results/test/features/
"""

import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from tqdm import tqdm
from rich import print

# Import feature extraction functions
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from models import create_features as create_model_features
from features import (
    transform_to_band_power,
    downsample_time,
    transform_to_phase,
    transform_to_band_power_with_phase,
    transform_to_time_frequency,
)

# ============================================================================
# Configuration
# ============================================================================

joblib_dir = Path(__file__).parent.parent.parent / "data" / "derivatives" / "preprocessed" / "joblib"
train_dir = joblib_dir / "training_set"
test_output_dir = Path(__file__).parent.parent.parent / "results" / "test" / "features"

# Create output directory
test_output_dir.mkdir(parents=True, exist_ok=True)

# Default band definitions
DEFAULT_BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 100.0),
}

# ============================================================================
# Visualization Functions
# ============================================================================

def plot_bandpower_features(X_raw: np.ndarray, X_bp: np.ndarray, sfreq: float, subject_name: str):
    """
    Visualize band power features: compare raw signal statistics with extracted features.
    
    Parameters
    ----------
    X_raw : np.ndarray
        Shape (epochs, channels, timepoints) - raw EEG data
    X_bp : np.ndarray
        Shape (epochs, channels * n_bands) - band power features
    sfreq : float
        Sampling frequency
    subject_name : str
        Name of the subject for the title
    """
    n_epochs, n_channels, n_timepoints = X_raw.shape
    n_bands = X_bp.shape[1] // n_channels
    
    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(2, 2, figure=fig)
    
    # Plot 1: Mean raw signal per channel
    ax1 = fig.add_subplot(gs[0, 0])
    mean_raw = X_raw.mean(axis=(0, 2))  # Average across epochs and time
    ax1.bar(range(n_channels), mean_raw)
    ax1.set_xlabel("Channel")
    ax1.set_ylabel("Mean Amplitude")
    ax1.set_title("Mean Raw Signal Amplitude per Channel")
    ax1.grid(alpha=0.3)
    
    # Plot 2: Band power heatmap (mean across epochs)
    ax2 = fig.add_subplot(gs[0, 1])
    bp_reshaped = X_bp.reshape(n_epochs, n_channels, n_bands)
    bp_mean = bp_reshaped.mean(axis=0)  # Average across epochs
    im = ax2.imshow(bp_mean, aspect="auto", cmap="viridis")
    ax2.set_xlabel("Frequency Band")
    ax2.set_ylabel("Channel")
    ax2.set_xticks(range(n_bands))
    ax2.set_xticklabels(DEFAULT_BANDS.keys())
    ax2.set_title("Mean Band Power per Channel and Band")
    plt.colorbar(im, ax=ax2, label="Power (μV²/Hz)")
    
    # Plot 3: Distribution of band power values
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.hist(X_bp.flatten(), bins=50, alpha=0.7, edgecolor='black')
    ax3.set_xlabel("Band Power Value")
    ax3.set_ylabel("Frequency")
    ax3.set_title("Distribution of Band Power Features")
    ax3.grid(alpha=0.3)
    
    # Plot 4: Band power per band (mean across channels and epochs)
    ax4 = fig.add_subplot(gs[1, 1])
    bp_per_band = bp_mean.mean(axis=0)
    ax4.bar(range(n_bands), bp_per_band, color='steelblue')
    ax4.set_xlabel("Frequency Band")
    ax4.set_ylabel("Mean Power")
    ax4.set_xticks(range(n_bands))
    ax4.set_xticklabels(DEFAULT_BANDS.keys())
    ax4.set_title("Mean Band Power across All Channels")
    ax4.grid(alpha=0.3, axis='y')
    
    fig.suptitle(f"Band Power Feature Extraction — {subject_name}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_phase_features(X_raw: np.ndarray, phase_features: np.ndarray, sfreq: float, subject_name: str, freq_range: tuple = (1, 40)):
    """
    Visualize phase features extracted from the signal.
    
    Parameters
    ----------
    X_raw : np.ndarray
        Shape (epochs, channels, timepoints)
    phase_features : np.ndarray
        Shape (epochs, channels * 2, timepoints) - sin and cos phase
    sfreq : float
        Sampling frequency
    subject_name : str
        Name of the subject
    freq_range : tuple
        Frequency range used for phase extraction
    """
    n_epochs, n_channels, n_timepoints = X_raw.shape
    
    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(2, 2, figure=fig)
    
    # Plot 1: Sample raw signal and phase for first channel, first epoch
    ax1 = fig.add_subplot(gs[0, 0])
    time = np.arange(n_timepoints) / sfreq
    ax1_twin = ax1.twinx()
    
    ax1.plot(time, X_raw[0, 0, :], 'b-', alpha=0.6, label='Raw signal')
    sin_phase = phase_features[0, 0, :]  # sin component of first channel
    ax1_twin.plot(time, sin_phase, 'r-', alpha=0.6, label='Phase (sin)')
    
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Raw Signal Amplitude (μV)", color='b')
    ax1_twin.set_ylabel("Phase (sin)", color='r')
    ax1.set_title(f"Raw Signal vs Phase (Ch 0, Epoch 0, {freq_range[0]}-{freq_range[1]} Hz)")
    ax1.grid(alpha=0.3)
    ax1.tick_params(axis='y', labelcolor='b')
    ax1_twin.tick_params(axis='y', labelcolor='r')
    
    # Plot 2: Phase angle distribution
    ax2 = fig.add_subplot(gs[0, 1])
    all_sin = phase_features[:, ::2, :].flatten()  # All sin components
    all_cos = phase_features[:, 1::2, :].flatten()  # All cos components
    
    ax2.scatter(all_cos, all_sin, alpha=0.1, s=1, label='Phase points')
    circle = plt.Circle((0, 0), 1, fill=False, color='red', linestyle='--', label='Unit circle')
    ax2.add_patch(circle)
    ax2.set_xlim(-1.5, 1.5)
    ax2.set_ylim(-1.5, 1.5)
    ax2.set_xlabel("cos(phase)")
    ax2.set_ylabel("sin(phase)")
    ax2.set_title("Phase Distribution (Unit Circle)")
    ax2.set_aspect('equal')
    ax2.grid(alpha=0.3)
    ax2.legend()
    
    # Plot 3: Phase variance per channel
    ax3 = fig.add_subplot(gs[1, 0])
    sin_phases = phase_features[:, ::2, :]  # Shape: (epochs, channels, timepoints)
    cos_phases = phase_features[:, 1::2, :]
    phase_variance = np.var(sin_phases, axis=(0, 2)) + np.var(cos_phases, axis=(0, 2))
    
    ax3.bar(range(n_channels), phase_variance)
    ax3.set_xlabel("Channel")
    ax3.set_ylabel("Variance")
    ax3.set_title("Phase Variance per Channel")
    ax3.grid(alpha=0.3, axis='y')
    
    # Plot 4: Mean absolute phase per channel
    ax4 = fig.add_subplot(gs[1, 1])
    sin_mean_abs = np.abs(sin_phases).mean(axis=(0, 2))
    cos_mean_abs = np.abs(cos_phases).mean(axis=(0, 2))
    
    x = np.arange(n_channels)
    width = 0.35
    ax4.bar(x - width/2, sin_mean_abs, width, label='|sin(phase)|', alpha=0.8)
    ax4.bar(x + width/2, cos_mean_abs, width, label='|cos(phase)|', alpha=0.8)
    ax4.set_xlabel("Channel")
    ax4.set_ylabel("Mean Absolute Value")
    ax4.set_xticks(x)
    ax4.set_title("Mean Absolute Phase Components per Channel")
    ax4.legend()
    ax4.grid(alpha=0.3, axis='y')
    
    fig.suptitle(f"Phase Feature Extraction — {subject_name} ({freq_range[0]}-{freq_range[1]} Hz)", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_combined_features(X_raw: np.ndarray, X_combined: np.ndarray, sfreq: float, subject_name: str):
    """
    Visualize combined band power and phase features.
    
    Parameters
    ----------
    X_raw : np.ndarray
        Shape (epochs, channels, timepoints)
    X_combined : np.ndarray
        Shape (epochs, bandpower_features + phase_features)
    sfreq : float
        Sampling frequency
    subject_name : str
        Name of the subject
    """
    n_epochs, n_channels, _ = X_raw.shape
    n_bands = len(DEFAULT_BANDS)
    n_bandpower_features = n_channels * n_bands
    n_phase_features = X_combined.shape[1] - n_bandpower_features
    phase_features_per_band = n_channels * 2

    if n_phase_features != n_bands * phase_features_per_band:
        raise ValueError(
            "Unexpected phase feature size: "
            f"got {n_phase_features}, expected {n_bands * phase_features_per_band}"
        )
    
    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(2, 2, figure=fig)
    
    # Plot 1: Feature dimension breakdown
    ax1 = fig.add_subplot(gs[0, 0])
    feature_types = ['Band Power', 'Phase']
    feature_counts = [n_bandpower_features, n_phase_features]
    colors = ['steelblue', 'coral']
    ax1.bar(feature_types, feature_counts, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel("Number of Features")
    ax1.set_title("Feature Composition")
    ax1.grid(alpha=0.3, axis='y')
    for i, v in enumerate(feature_counts):
        ax1.text(i, v + 5, str(v), ha='center', fontweight='bold')
    
    # Plot 2: Feature value distributions
    ax2 = fig.add_subplot(gs[0, 1])
    bp_features = X_combined[:, :n_bandpower_features]
    phase_features = X_combined[:, n_bandpower_features:]
    
    ax2.hist(bp_features.flatten(), bins=50, alpha=0.6, label='Band Power', edgecolor='black')
    ax2.hist(phase_features.flatten(), bins=50, alpha=0.6, label='Phase', edgecolor='black')
    ax2.set_xlabel("Feature Value")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Distribution of Combined Features")
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # Plot 3: Mean band power heatmap
    ax3 = fig.add_subplot(gs[1, 0])
    bp_reshaped = bp_features.reshape(n_epochs, n_channels, n_bands)
    bp_mean = bp_reshaped.mean(axis=0)
    im3 = ax3.imshow(bp_mean, aspect="auto", cmap="viridis")
    ax3.set_xlabel("Band")
    ax3.set_ylabel("Channel")
    ax3.set_xticks(range(n_bands))
    ax3.set_xticklabels(list(DEFAULT_BANDS.keys()))
    ax3.set_title("Mean Band Power by Channel")
    plt.colorbar(im3, ax=ax3, label="Power")

    # Plot 4: Mean circular phase structure per band and channel
    ax4 = fig.add_subplot(gs[1, 1])
    phase_blocks = phase_features.reshape(n_epochs, n_bands, n_channels, 2)
    mean_sin = phase_blocks[..., 0].mean(axis=0)
    mean_cos = phase_blocks[..., 1].mean(axis=0)
    mean_phase = np.arctan2(mean_sin, mean_cos)

    im4 = ax4.imshow(mean_phase, aspect="auto", cmap="twilight", vmin=-np.pi, vmax=np.pi)
    ax4.set_xlabel("Channel")
    ax4.set_ylabel("Band")
    ax4.set_xticks(np.arange(0, n_channels, max(1, n_channels // 8)))
    ax4.set_yticks(range(n_bands))
    ax4.set_yticklabels(list(DEFAULT_BANDS.keys()))
    ax4.set_title("Mean Circular Phase by Band and Channel")
    plt.colorbar(im4, ax=ax4, label="Phase (rad)")
    
    fig.suptitle(f"Combined Band Power + Phase Features — {subject_name}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_downsampling_effect(X_raw: np.ndarray, X_downsampled: np.ndarray, original_sfreq: float, target_sfreq: float, subject_name: str):
    """
    Visualize the effect of temporal downsampling.
    
    Parameters
    ----------
    X_raw : np.ndarray
        Shape (epochs, channels, timepoints) - original data
    X_downsampled : np.ndarray
        Shape (epochs, channels, new_timepoints) - downsampled data
    original_sfreq : float
        Original sampling frequency
    target_sfreq : float
        Target sampling frequency
    subject_name : str
        Name of the subject
    """
    n_epochs, n_channels, _ = X_raw.shape
    
    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(2, 2, figure=fig)
    
    # Plot 1: Sample signal - original vs downsampled
    ax1 = fig.add_subplot(gs[0, :])
    epoch_idx, ch_idx = 0, 0
    
    time_orig = np.arange(X_raw.shape[2]) / original_sfreq
    time_down = np.arange(X_downsampled.shape[2]) / target_sfreq
    
    ax1.plot(time_orig, X_raw[epoch_idx, ch_idx, :], 'b-', linewidth=1, label=f'Original ({original_sfreq} Hz)', alpha=0.7)
    ax1.plot(time_down, X_downsampled[epoch_idx, ch_idx, :], 'r-', linewidth=2, label=f'Downsampled ({target_sfreq} Hz)', alpha=0.7)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude (μV)")
    ax1.set_title(f"Original vs Downsampled Signal (Epoch {epoch_idx}, Channel {ch_idx})")
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Plot 2: Sampling rate comparison
    ax2 = fig.add_subplot(gs[1, 0])
    info = [
        f"Original sampling freq: {original_sfreq} Hz",
        f"Target sampling freq: {target_sfreq} Hz",
        f"Original timepoints: {X_raw.shape[2]}",
        f"Downsampled timepoints: {X_downsampled.shape[2]}",
        f"Compression ratio: {X_raw.shape[2] / X_downsampled.shape[2]:.2f}x",
    ]
    ax2.text(0.1, 0.5, '\n'.join(info), fontsize=11, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), family='monospace')
    ax2.axis('off')
    ax2.set_title("Downsampling Parameters")
    
    # Plot 3: Mean signal energy before and after
    ax3 = fig.add_subplot(gs[1, 1])
    energy_orig = np.sqrt((X_raw ** 2).mean(axis=2))  # Shape: (epochs, channels)
    energy_down = np.sqrt((X_downsampled ** 2).mean(axis=2))
    
    ch_range = min(n_channels, 10)
    x = np.arange(ch_range)
    width = 0.35
    ax3.bar(x - width/2, energy_orig[:, :ch_range].mean(axis=0), width, label='Original', alpha=0.8)
    ax3.bar(x + width/2, energy_down[:, :ch_range].mean(axis=0), width, label='Downsampled', alpha=0.8)
    ax3.set_xlabel("Channel")
    ax3.set_ylabel("Mean Signal Energy")
    ax3.set_xticks(x)
    ax3.set_title("Signal Energy Preservation")
    ax3.legend()
    ax3.grid(alpha=0.3, axis='y')
    
    fig.suptitle(f"Temporal Downsampling Effect — {subject_name}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_time_frequency_features(
    X_raw: np.ndarray,
    X_tf: np.ndarray,
    sfreq: float,
    subject_name: str,
    n_freqs: int = 20,
):
    """
    Visualize Morlet time-frequency features.

    Parameters
    ----------
    X_raw : np.ndarray
        Shape (epochs, channels, timepoints) - original data
    X_tf : np.ndarray
        Shape (epochs, channels * n_freqs * timepoints) - flattened time-frequency features
    sfreq : float
        Sampling frequency
    subject_name : str
        Name of the subject
    n_freqs : int
        Number of frequency bins used by the transform
    """
    n_epochs, n_channels, n_timepoints = X_raw.shape
    tf_reshaped = X_tf.reshape(n_epochs, n_channels, n_freqs, n_timepoints)

    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(2, 2, figure=fig)

    # Plot 1: Raw signal and time-frequency map for the first epoch/channel
    ax1 = fig.add_subplot(gs[0, 0])
    time = np.arange(n_timepoints) / sfreq
    ax1.plot(time, X_raw[0, 0, :], color='steelblue', alpha=0.7)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.set_title("Raw Signal (Epoch 0, Channel 0)")
    ax1.grid(alpha=0.3)

    # Plot 2: Time-frequency power for first epoch/channel
    ax2 = fig.add_subplot(gs[0, 1])
    im = ax2.imshow(tf_reshaped[0, 0], aspect='auto', origin='lower', cmap='viridis')
    ax2.set_xlabel("Time Index")
    ax2.set_ylabel("Frequency Bin")
    ax2.set_title("Morlet Time-Frequency Power (Epoch 0, Channel 0)")
    plt.colorbar(im, ax=ax2, label='Power')

    # Plot 3: Mean power per frequency bin across all epochs/channels
    ax3 = fig.add_subplot(gs[1, 0])
    mean_power_per_freq = tf_reshaped.mean(axis=(0, 1, 3))
    ax3.plot(range(n_freqs), mean_power_per_freq, marker='o', color='coral')
    ax3.set_xlabel("Frequency Bin")
    ax3.set_ylabel("Mean Power")
    ax3.set_title("Mean Power per Frequency Bin")
    ax3.grid(alpha=0.3)

    # Plot 4: Distribution of all time-frequency values
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.hist(X_tf.flatten(), bins=50, alpha=0.75, edgecolor='black')
    ax4.set_xlabel("Feature Value")
    ax4.set_ylabel("Frequency")
    ax4.set_title("Distribution of Time-Frequency Features")
    ax4.grid(alpha=0.3)

    fig.suptitle(f"Morlet Time-Frequency Extraction — {subject_name}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_band_aggregated_time_frequency_features(
    X_raw: np.ndarray,
    X_tf: np.ndarray,
    sfreq: float,
    subject_name: str,
    bands: dict[str, tuple[float, float]],
    downsample_to_freq: float,
):
    """
    Visualize band-aggregated Morlet time-frequency features.
    """
    n_epochs, n_channels, n_timepoints = X_raw.shape
    n_bands = len(bands)
    downsampled_n_timepoints = max(1, int(np.round(n_timepoints * (downsample_to_freq / sfreq))))
    tf_reshaped = X_tf.reshape(n_epochs, n_channels, n_bands, downsampled_n_timepoints)

    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(2, 2, figure=fig)

    ax1 = fig.add_subplot(gs[0, 0])
    time = np.arange(n_timepoints) / sfreq
    ax1.plot(time, X_raw[0, 0, :], color='steelblue', alpha=0.7)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.set_title("Raw Signal (Epoch 0, Channel 0)")
    ax1.grid(alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    im = ax2.imshow(tf_reshaped[0, 0], aspect='auto', origin='lower', cmap='viridis')
    ax2.set_xlabel("Downsampled Time Index")
    ax2.set_ylabel("Band")
    ax2.set_yticks(range(n_bands))
    ax2.set_yticklabels(list(bands.keys()))
    ax2.set_title("Band-Aggregated Morlet Amplitude (Epoch 0, Channel 0)")
    plt.colorbar(im, ax=ax2, label='Amplitude')

    ax3 = fig.add_subplot(gs[1, 0])
    mean_per_band = tf_reshaped.mean(axis=(0, 1, 3))
    ax3.bar(range(n_bands), mean_per_band, color='coral')
    ax3.set_xlabel("Band")
    ax3.set_ylabel("Mean Amplitude")
    ax3.set_xticks(range(n_bands))
    ax3.set_xticklabels(list(bands.keys()))
    ax3.set_title("Mean Band-Aggregated Amplitude")
    ax3.grid(alpha=0.3, axis='y')

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.hist(X_tf.flatten(), bins=50, alpha=0.75, edgecolor='black')
    ax4.set_xlabel("Feature Value")
    ax4.set_ylabel("Frequency")
    ax4.set_title("Distribution of Band-Aggregated Time-Frequency Features")
    ax4.grid(alpha=0.3)

    fig.suptitle(f"Morlet Band-Aggregated Time-Frequency Extraction — {subject_name}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_dwt_time_frequency_features(
    X_raw: np.ndarray,
    X_dwt: np.ndarray,
    subject_name: str,
):
    """
    Visualize DWT-based features with a compact summary plot.
    """
    n_epochs, n_channels, n_timepoints = X_raw.shape
    feature_lengths = np.array([X_dwt.shape[1]] * n_epochs)

    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(2, 2, figure=fig)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(X_raw[0, 0, :], color='steelblue', alpha=0.7)
    ax1.set_xlabel("Time Index")
    ax1.set_ylabel("Amplitude")
    ax1.set_title("Raw Signal (Epoch 0, Channel 0)")
    ax1.grid(alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(X_dwt.flatten(), bins=50, alpha=0.75, edgecolor='black', color='darkseagreen')
    ax2.set_xlabel("Feature Value")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Distribution of DWT Features")
    ax2.grid(alpha=0.3)

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.bar([0], [X_dwt.shape[1]], color='slateblue')
    ax3.set_xticks([0])
    ax3.set_xticklabels(["DWT"])
    ax3.set_ylabel("Feature Length")
    ax3.set_title("DWT Feature Vector Length")
    ax3.grid(alpha=0.3, axis='y')

    ax4 = fig.add_subplot(gs[1, 1])
    info = [
        f"Epochs: {n_epochs}",
        f"Channels: {n_channels}",
        f"Original timepoints: {n_timepoints}",
        f"DWT feature length: {X_dwt.shape[1]}",
        f"Mean feature length per epoch: {feature_lengths.mean():.0f}",
    ]
    ax4.text(0.05, 0.5, '\n'.join(info), fontsize=11, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), family='monospace')
    ax4.axis('off')
    ax4.set_title("DWT Summary")

    fig.suptitle(f"DWT Time-Frequency Extraction — {subject_name}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_dwt_hierarchical_features(
    X_raw: np.ndarray,
    X_dwt_hier: np.ndarray,
    subject_name: str,
):
    """
    Visualize DWT hierarchical features.
    
    Parameters
    ----------
    X_raw : np.ndarray
        Shape (epochs, channels, timepoints) - original data
    X_dwt_hier : np.ndarray
        Shape (epochs, channels * (max_level+1) * 4) - hierarchical DWT features
    subject_name : str
        Name of the subject
    """
    n_epochs, n_channels, n_timepoints = X_raw.shape
    
    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(2, 2, figure=fig)
    
    # Plot 1: Raw signal
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(X_raw[0, 0, :], color='steelblue', alpha=0.7)
    ax1.set_xlabel("Time Index")
    ax1.set_ylabel("Amplitude (μV)")
    ax1.set_title("Raw Signal (Epoch 0, Channel 0)")
    ax1.grid(alpha=0.3)
    
    # Plot 2: Distribution of hierarchical features
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(X_dwt_hier.flatten(), bins=50, alpha=0.75, edgecolor='black', color='seagreen')
    ax2.set_xlabel("Feature Value")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Distribution of DWT Hierarchical Features")
    ax2.grid(alpha=0.3)
    
    # Plot 3: Feature statistics
    ax3 = fig.add_subplot(gs[1, 0])
    import pywt
    max_level = pywt.dwt_max_level(n_timepoints, 8)
    features_per_channel = (max_level + 1) * 4
    
    stats = [
        ('Mean', X_dwt_hier.mean()),
        ('Std', X_dwt_hier.std()),
        ('Min', X_dwt_hier.min()),
        ('Max', X_dwt_hier.max()),
    ]
    labels, values = zip(*stats)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    ax3.bar(labels, values, color=colors, alpha=0.7, edgecolor='black')
    ax3.set_ylabel("Value")
    ax3.set_title("Feature Statistics")
    ax3.grid(alpha=0.3, axis='y')
    
    # Plot 4: Summary info
    ax4 = fig.add_subplot(gs[1, 1])
    info = [
        f"Epochs: {n_epochs}",
        f"Channels: {n_channels}",
        f"Original timepoints: {n_timepoints}",
        f"DWT max level: {max_level}",
        f"Stats per level: 4 (mean, std, max(abs), p75(abs))",
        f"Features per channel: {features_per_channel}",
        f"Total features: {X_dwt_hier.shape[1]}",
    ]
    ax4.text(0.05, 0.5, '\n'.join(info), fontsize=10, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), family='monospace')
    ax4.axis('off')
    ax4.set_title("DWT Hierarchical Summary")
    
    fig.suptitle(f"DWT Hierarchical Features — {subject_name}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_channel_selection_by_mutual_info_features(
    X_train: np.ndarray,
    X_train_selected: np.ndarray,
    X_test: np.ndarray,
    X_test_selected: np.ndarray,
    selected_channels: np.ndarray,
    subject_name: str,
):
    """
    Visualize channel selection by mutual information.
    
    Parameters
    ----------
    X_train : np.ndarray
        Shape (epochs, channels, timepoints) - original training data
    X_train_selected : np.ndarray
        Shape (epochs, k_channels * n_levels) - selected channel features from training
    X_test : np.ndarray
        Shape (epochs, channels, timepoints) - original test data
    X_test_selected : np.ndarray
        Shape (epochs, k_channels * n_levels) - selected channel features from test
    selected_channels : np.ndarray
        Indices of selected channels
    subject_name : str
        Name of the subject
    """
    n_channels = X_train.shape[1]
    k_channels = len(selected_channels)
    
    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(2, 2, figure=fig)
    
    # Plot 1: Channel selection visualization
    ax1 = fig.add_subplot(gs[0, 0])
    channel_labels = [f"Ch {i}" for i in range(n_channels)]
    colors = ['green' if i in selected_channels else 'lightgray' for i in range(n_channels)]
    ax1.bar(range(n_channels), [1]*n_channels, color=colors, edgecolor='black')
    ax1.set_xlabel("Channel")
    ax1.set_ylabel("Selected")
    ax1.set_xticks(range(n_channels))
    ax1.set_xticklabels(channel_labels, rotation=45)
    ax1.set_title("Channel Selection by Mutual Information")
    ax1.set_ylim([0, 1.2])
    ax1.grid(alpha=0.3, axis='y')
    
    # Plot 2: Feature distributions
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(X_train_selected.flatten(), bins=50, alpha=0.6, label='Training', edgecolor='black', color='steelblue')
    ax2.hist(X_test_selected.flatten(), bins=50, alpha=0.6, label='Test', edgecolor='black', color='coral')
    ax2.set_xlabel("Feature Value")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Distribution of Selected Channel Features")
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # Plot 3: Mean signal per selected channel
    ax3 = fig.add_subplot(gs[1, 0])
    train_means = X_train[:, selected_channels, :].mean(axis=2)  # (epochs, k_channels)
    mean_by_channel = train_means.mean(axis=0)  # (k_channels,)
    ax3.bar(range(k_channels), mean_by_channel, color='steelblue', alpha=0.7, edgecolor='black')
    ax3.set_xlabel("Selected Channel Index")
    ax3.set_ylabel("Mean Amplitude (μV)")
    ax3.set_title("Mean Signal per Selected Channel")
    ax3.grid(alpha=0.3, axis='y')
    
    # Plot 4: Summary info
    ax4 = fig.add_subplot(gs[1, 1])
    info = [
        f"Total channels: {n_channels}",
        f"Selected channels: {k_channels}",
        f"Selected channel indices: {selected_channels.tolist()}",
        f"Train features shape: {X_train_selected.shape}",
        f"Test features shape: {X_test_selected.shape}",
        f"Train feature mean: {X_train_selected.mean():.4f}",
        f"Test feature mean: {X_test_selected.mean():.4f}",
    ]
    ax4.text(0.05, 0.5, '\n'.join(info), fontsize=10, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), family='monospace')
    ax4.axis('off')
    ax4.set_title("Channel Selection Summary")
    
    fig.suptitle(f"Channel Selection by Mutual Information — {subject_name}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


# ============================================================================
# Main Testing Function
# ============================================================================

def test_feature_extraction(n_subjects: int = 5, sfreq: float = 200.0):
    """
    Test all feature extraction methods on real data.
    
    Parameters
    ----------
    n_subjects : int
        Number of subjects to test on
    sfreq : float
        Sampling frequency in Hz
    """
    print(f"[bold]Testing Feature Extraction Methods[/bold]")
    print(f"Output directory: {test_output_dir}")
    print(f"Number of subjects: {n_subjects}")
    print(f"Sampling frequency: {sfreq} Hz\n")
    
    # Get training files
    train_files = sorted(list(train_dir.glob("*.joblib")))[:n_subjects]
    
    if len(train_files) == 0:
        print("[red]ERROR: No joblib files found in {train_dir}[/red]")
        return
    
    print(f"Found {len(train_files)} training files\n")
    
    for file_idx, file in enumerate(tqdm(train_files, desc="Processing subjects")):
        subject_name = file.stem
        print(f"\n[bold green]Subject {file_idx + 1}/{len(train_files)}: {subject_name}[/bold green]")
        
        # Load data
        data = joblib.load(file)
        X = data["x"]  # Shape: (epochs, channels, timepoints)
        
        print(f"  Data shape: {X.shape}")
        n_epochs, n_channels, n_timepoints = X.shape
        
        # Create subject output directory
        subject_dir = test_output_dir / subject_name
        subject_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # ================================================================
            # Test 1: Band Power Extraction
            # ================================================================
            print(f"  Testing band power extraction...", end=" ")
            X_bp = transform_to_band_power(X, sfreq=sfreq, bands=DEFAULT_BANDS)
            print(f"✓ shape: {X_bp.shape}")
            
            fig = plot_bandpower_features(X, X_bp, sfreq, subject_name)
            fig_path = subject_dir / f"{subject_name}_bandpower.png"
            fig.savefig(fig_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved: {fig_path.name}")
            
        except Exception as e:
            print(f"✗ Error: {e}")
        
        try:
            # ================================================================
            # Test 2: Phase Extraction
            # ================================================================
            print(f"  Testing phase extraction...", end=" ")
            phase_features = transform_to_phase(X, sfreq=sfreq, freq_range=(1.0, 40.0))
            print(f"✓ shape: {phase_features.shape}")
            
            fig = plot_phase_features(X, phase_features, sfreq, subject_name, freq_range=(1.0, 40.0))
            fig_path = subject_dir / f"{subject_name}_phase.png"
            fig.savefig(fig_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved: {fig_path.name}")
            
        except Exception as e:
            print(f"✗ Error: {e}")
        
        try:
            # ================================================================
            # Test 3: Downsampling
            # ================================================================
            print(f"  Testing temporal downsampling...", end=" ")
            X_downsampled = downsample_time(X, original_sfreq=sfreq, target_sfreq=10.0)
            print(f"✓ shape: {X_downsampled.shape}")
            
            fig = plot_downsampling_effect(X, X_downsampled, original_sfreq=sfreq, target_sfreq=10.0, subject_name=subject_name)
            fig_path = subject_dir / f"{subject_name}_downsampling.png"
            fig.savefig(fig_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            print(f"    Saved: {fig_path.name}")
            
        except Exception as e:
            print(f"✗ Error: {e}")
        
        try:
            # ================================================================
            # Test 4: Combined Features
            # ================================================================
            print(f"  Testing combined features...", end=" ")
            X_combined = transform_to_band_power_with_phase(
                X,
                sfreq=sfreq,
                bands=DEFAULT_BANDS,
            )
            print(f"✓ shape: {X_combined.shape}")
            
            fig = plot_combined_features(X, X_combined, sfreq, subject_name)
            fig_path = subject_dir / f"{subject_name}_combined.png"
            fig.savefig(fig_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            print(f"    Saved: {fig_path.name}")
            
        except Exception as e:
            print(f"✗ Error: {e}")

        try:
            # ================================================================
            # Test 5: Morlet Time-Frequency Extraction
            # ================================================================
            print(f"  Testing Morlet time-frequency extraction...", end=" ")
            X_tf = create_model_features(X, feature_type="tfr_morlet")
            if X_tf is None:
                raise ValueError("Time-frequency transform returned None")
            print(f"✓ shape: {X_tf.shape}")

            fig = plot_time_frequency_features(X, X_tf, sfreq, subject_name, n_freqs=20)
            fig_path = subject_dir / f"{subject_name}_time_frequency_morlet.png"
            fig.savefig(fig_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            print(f"    Saved: {fig_path.name}")

        except Exception as e:
            print(f"✗ Error: {e}")

        try:
            # ================================================================
            # Test 6: Band-Aggregated Morlet Time-Frequency Extraction
            # ================================================================
            print(f"  Testing Morlet band-aggregated time-frequency extraction...", end=" ")
            X_tf_bands = create_model_features(X, feature_type="tfr_morlet_bands")
            if X_tf_bands is None:
                raise ValueError("Band-aggregated time-frequency transform returned None")
            print(f"✓ shape: {X_tf_bands.shape}")

            fig = plot_band_aggregated_time_frequency_features(
                X,
                X_tf_bands,
                sfreq,
                subject_name,
                bands=DEFAULT_BANDS,
                downsample_to_freq=4,
            )
            fig_path = subject_dir / f"{subject_name}_time_frequency_morlet_bands.png"
            fig.savefig(fig_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            print(f"    Saved: {fig_path.name}")

        except Exception as e:
            print(f"✗ Error: {e}")

        try:
            # ================================================================
            # Test 7: DWT Time-Frequency Extraction
            # ================================================================
            print(f"  Testing DWT time-frequency extraction...", end=" ")
            X_tf_dwt = create_model_features(X, feature_type="tfr_dwt_cmor")
            if X_tf_dwt is None:
                raise ValueError("DWT time-frequency transform returned None")
            print(f"✓ shape: {X_tf_dwt.shape}")

            fig = plot_dwt_time_frequency_features(X, X_tf_dwt, subject_name)
            fig_path = subject_dir / f"{subject_name}_time_frequency_dwt.png"
            fig.savefig(fig_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            print(f"    Saved: {fig_path.name}")

        except Exception as e:
            print(f"✗ Error: {e}")

        try:
            # ================================================================
            # Test 8: DWT Hierarchical Extraction
            # ================================================================
            print(f"  Testing DWT hierarchical extraction...", end=" ")
            X_dwt_hier = create_model_features(X, feature_type="dwt_hierarchical")
            if X_dwt_hier is None:
                raise ValueError("DWT hierarchical transform returned None")
            print(f"✓ shape: {X_dwt_hier.shape}")

            fig = plot_dwt_hierarchical_features(X, X_dwt_hier, subject_name)
            fig_path = subject_dir / f"{subject_name}_dwt_hierarchical.png"
            fig.savefig(fig_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            print(f"    Saved: {fig_path.name}")

        except Exception as e:
            print(f"✗ Error: {e}")

        try:
            # ================================================================
            # Test 9: Channel Selection by Mutual Information
            # ================================================================
            print(f"  Testing channel selection by mutual information...", end=" ")
            from features import select_channels_by_mutual_info
            
            # For testing purposes, split data into train/test and create synthetic labels
            n_split = max(1, n_epochs // 2)
            X_train_split = X[:n_split]
            X_test_split = X[n_split:]
            
            # Create synthetic labels (simple binary classification)
            y_train_split = np.random.randint(0, 2, n_split)
            
            # Perform channel selection
            X_train_selected, X_test_selected = select_channels_by_mutual_info(
                X_train_split, y_train_split, X_test_split, k_channels=4
            )
            
            if X_train_selected is None or X_test_selected is None:
                raise ValueError("Channel selection returned None")
            print(f"✓ train shape: {X_train_selected.shape}, test shape: {X_test_selected.shape}")
            
            # Infer selected channels (those with highest MI)
            from sklearn.feature_selection import mutual_info_classif
            import pywt
            
            # Flatten training data for MI calculation
            X_train_flat = X_train_split.reshape(n_split, -1)
            mi_scores = mutual_info_classif(X_train_flat, y_train_split, random_state=42)
            
            # Group MI scores by channel (average across timepoints)
            n_timepoints_split = X_train_split.shape[2]
            mi_per_channel = []
            for ch in range(n_channels):
                ch_start = ch * n_timepoints_split
                ch_end = (ch + 1) * n_timepoints_split
                mi_per_channel.append(mi_scores[ch_start:ch_end].mean())
            
            selected_channels = np.argsort(mi_per_channel)[-4:]  # Top 4 channels
            
            fig = plot_channel_selection_by_mutual_info_features(
                X_train_split, X_train_selected, X_test_split, X_test_selected,
                selected_channels, subject_name
            )
            fig_path = subject_dir / f"{subject_name}_channel_selection_mi.png"
            fig.savefig(fig_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            print(f"    Saved: {fig_path.name}")

        except Exception as e:
            print(f"✗ Error: {e}")
    
    print(f"\n[bold green]✓ Testing complete! Plots saved to:{test_output_dir}[/bold green]")


def test_dwt_hierarchical():
    """
    Test transform_to_dwt_hierarchical feature extraction.
    
    This test verifies:
    - Correct output shape (epochs, channels * n_levels * 4)
    - No NaN or inf values in output
    - Proper handling of different array shapes
    """
    from features import transform_to_dwt_hierarchical
    
    print("\n" + "="*80)
    print("Testing: transform_to_dwt_hierarchical")
    print("="*80)
    
    # Test 1: Basic shape validation
    print("\n[Test 1] Shape validation")
    n_epochs, n_channels, n_timepoints = 10, 8, 512
    x = np.random.randn(n_epochs, n_channels, n_timepoints)
    
    features = transform_to_dwt_hierarchical(x, sfreq=256)
    
    assert features.ndim == 2, f"Expected 2D output, got {features.ndim}D"
    assert features.shape[0] == n_epochs, f"Expected {n_epochs} epochs, got {features.shape[0]}"
    
    # Calculate expected number of features per channel
    import pywt
    max_level = pywt.dwt_max_level(n_timepoints, 8)
    expected_features_per_channel = (max_level + 1) * 4

    expected_total_features = n_channels * expected_features_per_channel

    assert features.shape[1] == expected_total_features, \
        f"Expected {expected_total_features} features for {n_channels} channels and {n_timepoints} timepoints, got {features.shape[1]}"
    
    print(f"✓ Input shape: {x.shape}")
    print(f"✓ Output shape: {features.shape}")
    print(f"✓ Features per channel: {expected_features_per_channel} ({max_level + 1} levels × 4 stats)")
    print(f"✓ Total expected features: {expected_total_features} for {n_channels} channels")
    
    # Test 2: No NaN or inf values
    print("\n[Test 2] Data integrity")
    assert not np.any(np.isnan(features)), "Output contains NaN values"
    assert not np.any(np.isinf(features)), "Output contains inf values"
    print(f"✓ No NaN or inf values")
    print(f"✓ Min value: {features.min():.6f}, Max value: {features.max():.6f}")
    
    # Test 3: Different input shapes
    print("\n[Test 3] Different input shapes")
    test_shapes = [(5, 4, 256), (1, 16, 1024), (20, 2, 128)]
    
    for shape in test_shapes:
        x_test = np.random.randn(*shape)
        features_test = transform_to_dwt_hierarchical(x_test, sfreq=256)
        assert features_test.shape[0] == shape[0], f"Epochs mismatch for shape {shape}"
        assert not np.any(np.isnan(features_test)), f"NaN found for shape {shape}"
        print(f"  ✓ Shape {shape} → features shape {features_test.shape}")
    
    # Test 4: Consistency
    print("\n[Test 4] Consistency")
    x_const = np.random.randn(5, 4, 256)
    features1 = transform_to_dwt_hierarchical(x_const, sfreq=256)
    features2 = transform_to_dwt_hierarchical(x_const, sfreq=256)
    assert np.allclose(features1, features2), "Same input produced different outputs"
    print(f"✓ Deterministic: same input produces identical output")
    
    print("\n✓ All tests passed for transform_to_dwt_hierarchical\n")


def test_channel_selection_by_mutual_info():
    """
    Test select_channels_by_mutual_info feature extraction.
    
    This test verifies:
    - Correct number of channels selected
    - Output shape consistency (train/test match)
    - No NaN or inf values
    """
    from features import select_channels_by_mutual_info
    
    print("\n" + "="*80)
    print("Testing: select_channels_by_mutual_info")
    print("="*80)
    
    # Test 1: Basic shape and channel selection
    print("\n[Test 1] Channel selection and shape validation")
    n_channels = 8
    k_channels = 4
    n_epochs_train, n_epochs_test = 30, 10
    n_timepoints = 512
    
    # Create synthetic data
    x_train = np.random.randn(n_epochs_train, n_channels, n_timepoints)
    x_test = np.random.randn(n_epochs_test, n_channels, n_timepoints)
    y_train = np.random.randint(0, 2, n_epochs_train)
    
    # Make first few channels more informative
    for i in range(2):
        x_train[:, i, :] += y_train[:, np.newaxis] * 0.5
        x_test[:, i, :] += np.random.randint(0, 2, n_epochs_test)[:, np.newaxis] * 0.5
    
    X_train_feat, X_test_feat = select_channels_by_mutual_info(
        x_train, y_train, x_test, k_channels=k_channels
    )
    
    print(f"✓ Train input shape: {x_train.shape}")
    print(f"✓ Test input shape: {x_test.shape}")
    print(f"✓ Selected {k_channels} out of {n_channels} channels")
    
    # Check output shapes
    assert X_train_feat.shape[0] == n_epochs_train, "Train epochs mismatch"
    assert X_test_feat.shape[0] == n_epochs_test, "Test epochs mismatch"
    assert X_train_feat.shape[1] == X_test_feat.shape[1], "Train/test feature dimension mismatch"
    
    print(f"✓ Train features shape: {X_train_feat.shape}")
    print(f"✓ Test features shape: {X_test_feat.shape}")
    
    # Test 2: Data integrity
    print("\n[Test 2] Data integrity")
    assert not np.any(np.isnan(X_train_feat)), "Train features contain NaN"
    assert not np.any(np.isinf(X_train_feat)), "Train features contain inf"
    assert not np.any(np.isnan(X_test_feat)), "Test features contain NaN"
    assert not np.any(np.isinf(X_test_feat)), "Test features contain inf"
    print(f"✓ No NaN or inf in train features")
    print(f"✓ No NaN or inf in test features")
    
    # Test 3: Feature count validation
    print("\n[Test 3] Feature count validation")
    import pywt
    max_level = pywt.dwt_max_level(n_timepoints, 8)
    coeffs = pywt.wavedec(
        np.random.randn(n_timepoints),
        wavelet="db4",
        level=max_level,
        mode="symmetric",
    )
    coeffs_per_channel = sum(c.size for c in coeffs)
    expected_features = k_channels * coeffs_per_channel
    
    assert X_train_feat.shape[1] == expected_features, \
        f"Expected {expected_features} features for {k_channels} selected channels and {n_timepoints} timepoints, got {X_train_feat.shape[1]}"
    print(f"✓ Feature count correct: {k_channels} channels × {coeffs_per_channel} coefficients per channel")
    
    # Test 4: Different k_channels values
    print("\n[Test 4] Different k_channels values")
    for k in [2, 4, 6]:
        X_tr, X_te = select_channels_by_mutual_info(
            x_train, y_train, x_test, k_channels=k
        )
        assert X_tr.shape[0] == n_epochs_train
        assert X_te.shape[0] == n_epochs_test
        assert X_tr.shape[1] == X_te.shape[1]
        print(f"  ✓ k_channels={k}: train {X_tr.shape}, test {X_te.shape}")
    
    # Test 5: Input validation
    print("\n[Test 5] Input validation")
    
    # Mismatched channels
    x_train_bad = np.random.randn(20, 8, 256)
    x_test_bad = np.random.randn(10, 7, 256)
    y_train_bad = np.random.randint(0, 2, 20)
    
    try:
        select_channels_by_mutual_info(x_train_bad, y_train_bad, x_test_bad, k_channels=4)
        assert False, "Should have raised ValueError for mismatched channels"
    except ValueError as e:
        print(f"✓ Correctly caught channel mismatch")
    
    # Wrong input dimensions
    x_train_2d = np.random.randn(20, 256)
    try:
        select_channels_by_mutual_info(x_train_2d, y_train_bad, x_test, k_channels=4)
        assert False, "Should have raised ValueError for 2D input"
    except ValueError as e:
        print(f"✓ Correctly caught wrong dimensionality")
    
    print("\n✓ All tests passed for select_channels_by_mutual_info\n")


if __name__ == "__main__":
    # Run new feature tests
    #test_dwt_hierarchical()
    #test_channel_selection_by_mutual_info()
    
    # Test on first 5 subjects by default
    # Adjust n_subjects parameter to test on more subjects
    test_feature_extraction(n_subjects=5, sfreq=256.0)
