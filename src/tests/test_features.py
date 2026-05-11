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
from features import (
    transform_to_band_power,
    downsample_time,
    transform_to_phase,
    transform_to_band_power_with_phase,
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
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45),
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
                phase_freq_range=(1.0, 40.0),
                phase_target_sfreq=10.0,
            )
            print(f"✓ shape: {X_combined.shape}")
            
            fig = plot_combined_features(X, X_combined, sfreq, subject_name)
            fig_path = subject_dir / f"{subject_name}_combined.png"
            fig.savefig(fig_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            print(f"    Saved: {fig_path.name}")
            
        except Exception as e:
            print(f"✗ Error: {e}")
    
    print(f"\n[bold green]✓ Testing complete! Plots saved to:{test_output_dir}[/bold green]")


if __name__ == "__main__":
    # Test on first 5 subjects by default
    # Adjust n_subjects parameter to test on more subjects
    test_feature_extraction(n_subjects=5, sfreq=256.0)
