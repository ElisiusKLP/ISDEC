from pathlib import Path
import re
from typing import Optional, Dict, List

import joblib
import mne
import tqdm
import typer
from mne.preprocessing import ICA
from mne_icalabel import label_components
from rich import print

from dataset import mne_to_dict


mne_dir = Path("data/derivatives/raw/mne")
output_dir = Path("data/derivatives/preprocessed").absolute()
output_dir.mkdir(parents=True, exist_ok=True)


def describe_ica(ica_obj: ICA) -> None:
    """Print key numeric attributes and shapes of an MNE ICA object."""
    print("Inspecting ICA object:")
    attrs = [
        "n_components_",
        "n_pca_components_",
        "n_channels_",
        "mixing_matrix_",
        "unmixing_matrix_",
        "pca_components_",
        "pca_explained_variance_",
        "exclude",
    ]
    for attr in attrs:
        if hasattr(ica_obj, attr):
            val = getattr(ica_obj, attr)
            try:
                if hasattr(val, "shape"):
                    print(f" - {attr}: shape={val.shape}, dtype={getattr(val,'dtype', type(val))}")
                else:
                    print(f" - {attr}: {val}")
            except Exception as e:
                print(f" - {attr}: <error reading: {e}>")
        else:
            print(f" - {attr}: <missing>")

def label_ica_exclusion(ica: ICA, epochs: mne.Epochs, threshold: float = 0.8) -> List[int]:
    """Return list of ICA component indices to exclude based on ICLabel."""
    ic_labels = label_components(epochs, ica, "iclabel")
    labels = ic_labels["labels"]
    labels_probs = ic_labels["y_pred_proba"]
    for label, prob in zip(labels, labels_probs):
        print(f"Component labeled as {label} with probability {prob:.2f}")
    exclude_idx = [
        idx
        for idx, label in enumerate(labels)
        if label not in ["brain", "other"] and labels_probs[idx] >= threshold
    ]
    print(f"Excluding components {exclude_idx}")
    return exclude_idx


def QC(
    clean: mne.Epochs, 
    raw: mne.BaseEpochs, 
    ica: Optional[ICA], 
    subject_id: str, 
    ICLabel_exclusions: Dict[int, Dict],
    set_type: str
) -> None:
    """Save QC plots and ICLabel exclusions. If `ica` is None, skip ICA plots."""
    qc_dir = Path("results/preprocessing/") / set_type

    # RAW PSD
    raw_plt = raw.compute_psd(fmax=100).plot(show=False)
    raw_path = qc_dir / "psd_raw" / f"psd-raw_sub-{subject_id}.png"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_plt.savefig(raw_path.resolve(), bbox_inches="tight")

    # RAW EVOKED
    # create the evokeds for each condition and plot them
    try:
        evokeds_dict = {cond: raw[cond].average() for cond in raw.event_id.keys()}
        evoked_plt = mne.viz.plot_compare_evokeds(evokeds_dict, show=False, combine="mean")
        evoked_path = qc_dir / "evoked_raw" / f"evoked-raw_sub-{subject_id}.png"
        evoked_path.parent.mkdir(parents=True, exist_ok=True)
        # save the figure and not list
        print(f"evoked_plt type: {type(evoked_plt)}")
        print(f"evoked obj {evoked_plt}")
        evoked_plt[0].savefig(evoked_path.resolve(), bbox_inches="tight")
    except Exception as e:
        raise ValueError(f"Failed to plot evoked potentials: {e}")

    # CLEAN EVOKED
    try:
        clean_evokeds_dict = {cond: clean[cond].average() for cond in clean.event_id.keys()}
        clean_evoked_plt = mne.viz.plot_compare_evokeds(clean_evokeds_dict, show=False, combine="mean")
        clean_evoked_path = qc_dir / "evoked_clean" / f"evoked-clean_sub-{subject_id}.png"
        clean_evoked_path.parent.mkdir(parents=True, exist_ok=True)
        clean_evoked_plt[0].savefig(clean_evoked_path.resolve(), bbox_inches="tight")
    except Exception as e:
        raise ValueError(f"Failed to plot clean evoked potentials: {e}")

    # CLEAN PSD
    psd_plt = clean.compute_psd(fmax=100).plot(show=False)
    psd_path = qc_dir / "psd_clean" / f"psd-clean_sub-{subject_id}.png"
    psd_path.parent.mkdir(parents=True, exist_ok=True)
    psd_plt.savefig(psd_path.resolve(), bbox_inches="tight")

    if ica is not None:
        try:
            ica_plt = ica.plot_components(show=False)
            ica_path = qc_dir / "ica" / f"ica_sub-{subject_id}.png"
            ica_path.parent.mkdir(parents=True, exist_ok=True)
            ica_plt.savefig(ica_path.resolve(), bbox_inches="tight")
        except Exception as e:
            print(f"Failed to plot ICA components: {e}")
    else:
        print("Skipping ICA component plot (ICA not used)")

    iclabel_path = qc_dir / "iclabel" / f"iclabel_sub-{subject_id}.txt"
    iclabel_path.parent.mkdir(parents=True, exist_ok=True)
    with open(iclabel_path, "w") as f:
        f.write(f"ICA Exclusions for subject {subject_id}:\n")
        if ICLabel_exclusions:
            for idx, info in ICLabel_exclusions.items():
                f.write(f"Component {idx}: Label={info['label']}, Probability={info['prob']:.2f}\n")
        else:
            f.write("ICA not applied for this subject.\n")

    print(f"Saved QC plots and ICLabel exclusions for subject {subject_id}")


def preprocess(input_dir: Path, set_type: str, use_ica: bool = True) -> None:
    """Preprocess all .fif epoch files in `input_dir`.

    If `use_ica` is True, fit ICA on the training set and save it; for non-training
    partitions the ICA from the training set will be loaded and applied. If False,
    ICA and ICLabel steps are skipped entirely.
    """
    all_files = list(input_dir.glob("*.fif"))

    for file in tqdm.tqdm(all_files, desc="Preprocessing files"):
        match = re.search(r"epochs_sub-(\d{2})-epo\.fif", file.name)
        if match:
            sub_id = match.group(1)
            print(f"Found sub_id: {sub_id} in file: {file.name}")
        else:
            print(f"No match for file: {file.name}")

        subject_file = mne.read_epochs(file)

        epochs = subject_file.copy().filter(l_freq=1.0, h_freq=100.0, method="iir")
        epochs.set_eeg_reference(ref_channels="average")

        ica: Optional[ICA] = None

        if use_ica and set_type == "training_set":
            ica = ICA(
                n_components=20,
                max_iter="auto",
                random_state=2001,
                method="infomax",
                fit_params=dict(extended=True),
            )

            ica_filename = output_dir / "ica" / set_type / f"ica_sub-{sub_id}-ica.fif"
            ica_filename.parent.mkdir(parents=True, exist_ok=True)

            ica.fit(epochs.copy())
            describe_ica(ica)
            print(f"ICA n_components_={getattr(ica, 'n_components_', None)} for subject {sub_id}")
            ica.save(ica_filename, overwrite=True)
            print(f"Saved ICA solution to {ica_filename}")
        elif use_ica and set_type != "training_set":
            # if validation_set or test_set, load ICA solution from training set
            ica_filename = output_dir / "ica" / "training_set" / f"ica_sub-{sub_id}-ica.fif"
            if not ica_filename.exists():
                raise ValueError(f"ICA file not found: {ica_filename}")
            ica = mne.preprocessing.read_ica(ica_filename)
            print(f"Loaded ICA solution from {ica_filename}")
        else:
            print("ICA disabled for this run; skipping ICA and ICLabel steps.")

        ICLabel_exclusions: Dict[int, Dict] = {}

        if ica is not None:
            # ICLabeling
            ic_labels = label_components(epochs, ica, "iclabel")
            labels = ic_labels["labels"]
            labels_probs = ic_labels["y_pred_proba"]
            for label, prob in zip(labels, labels_probs):
                print(f"Component labeled as {label} with probability {prob:.2f}")
            threshold = 0.8
            keep_labels = ["brain", "other", "muscle"]
            exclude_idx = [
                idx for idx, label in enumerate(labels) if label not in keep_labels and labels_probs[idx] >= threshold
            ]
            print(f"Excluding components {exclude_idx}")

            ica.exclude = exclude_idx
            ica.apply(epochs)

            ICLabel_exclusions = {
                idx: {"label": labels[idx], "prob": labels_probs[idx]} for idx in exclude_idx
            }
        else:
            ICLabel_exclusions = {}

        # Baseline correction
        epochs.apply_baseline((-0.5, 0))

        # Filter
        epochs.filter(l_freq=1.0, h_freq=40.0, method="iir")

        print(f"ICLabel_exclusions: {ICLabel_exclusions}")
        QC(
            clean=epochs, 
            raw=subject_file, 
            ica=ica,
            subject_id=sub_id, 
            ICLabel_exclusions=ICLabel_exclusions,
            set_type=set_type
        )

        # Save preprocessed epochs
        preprocessed_dir = output_dir / "mne" / set_type / file.name
        preprocessed_dir.parent.mkdir(parents=True, exist_ok=True)
        epochs.save(preprocessed_dir, overwrite=True)
        print(f"Saved preprocessed epochs to {preprocessed_dir}")

        # Save preprocessed data to joblib
        joblib_filename = output_dir / "joblib" / set_type / f"preprocessed_sub-{sub_id}.joblib"
        joblib_filename.parent.mkdir(parents=True, exist_ok=True)
        preprocessed_data = mne_to_dict(epochs, sub_id)
        joblib.dump(preprocessed_data, joblib_filename)


def main(ica: bool = typer.Option(True, "--ica/--no-ica", help="Include ICA in preprocessing.")) -> None:
    """Run preprocessing across data partitions. Use `--no-ica` to skip ICA steps."""
    data_partitions = ["training_set", "validation_set", "test_set"]
    for partition in data_partitions:
        print(f"Preprocessing {partition} (ICA={'enabled' if ica else 'disabled'})...")
        input_dir = mne_dir / partition
        preprocess(input_dir, partition, use_ica=ica)


if __name__ == "__main__":
    typer.run(main)
