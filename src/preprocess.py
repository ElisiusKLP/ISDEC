from docutils.utils.math.latex2mathml import over
import mne
from pathlib import Path
import re
import tqdm
import joblib
from mne.preprocessing import ICA
from mne_icalabel import label_components
from dataset import mne_to_dict

#Load mne objects

mne_dir = Path("data/derivatives/raw/mne")
output_dir = Path("data/derivatives/preprocessed").absolute()
output_dir.mkdir(parents=True, exist_ok=True)

def describe_ica(ica_obj):
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

def preprocess(input_dir, set_type: str):
    all_files = list(input_dir.glob('*.fif'))

    for file in tqdm.tqdm(all_files, desc="Preprocessing files"):
        match = re.search(r"epochs_sub-(\d{2})-epo\.fif", file.name)
        if match:
            sub_id = match.group(1)
            print(f"Found sub_id: {sub_id} in file: {file.name}")
        else:
            print(f"No match for file: {file.name}")

        subject_file = mne.read_epochs(file)

        epochs = subject_file.copy().filter(
            l_freq=1.0, 
            h_freq=100.0,
            method="iir",
        ) # ICLabel is trained on this this bandpass filter

        epochs.set_eeg_reference(ref_channels="average")

        #ICA should be done before baseline correcting
        if set_type == "training_set": # Only do ICA on training set to avoid data leakage
            ica = ICA(
                n_components=20, 
                max_iter="auto",
                random_state=2001,
                method="infomax", # ICLabel method
                fit_params=dict(extended=True) #ICLabel method
            )
            
            ica_filename = output_dir / "ica" / set_type / f"ica_sub-{sub_id}-ica.fif"
            ica_filename.parent.mkdir(parents=True, exist_ok=True) # Create the directory if it doesn't exist

            ica.fit(epochs.copy())
            describe_ica(ica)
            print(f"ICA n_components_={getattr(ica, 'n_components_', None)} for subject {sub_id}")
            ica.save(ica_filename, overwrite=True)
            print(f"Saved ICA solution to {ica_filename}")
        else:
            # load ICA solution from training set
            ica_filename = output_dir / "ica" / "training_set" / f"ica_sub-{sub_id}-ica.fif"
            if not ica_filename.exists():
                raise ValueError(f"ICA file not found: {ica_filename}")
            ica = mne.preprocessing.read_ica(ica_filename)
            print(f"Loaded ICA solution from {ica_filename}")

        # ICLabeling
        ic_labels = label_components(epochs, ica, 'iclabel')
        labels = ic_labels['labels']
        labels_probs = ic_labels['y_pred_proba']
        # print each labal and its probability
        for label, prob in zip(labels, labels_probs):
            print(f"Component labeled as {label} with probability {prob:.2f}")
        threshold = 0.8
        exclude_idx = [
            idx for idx, label in enumerate(labels) if label not in ["brain", "other"]
            and labels_probs[idx] >= threshold
        ]
        print(f"Excluding components {exclude_idx}")

        # Apply ICA
        ica.exclude = exclude_idx
        ica.apply(epochs)

        # Baseline correction
        epochs.apply_baseline((-0.5, 0))

        # Filter
        epochs.filter(
            l_freq=1.0, 
            h_freq=40.0,
            method="iir",
        )

        # Plot the epochs for QC
        ICLabel_exclusions = {
            idx: {
                "label": labels[idx], 
                "prob": labels_probs[idx]
                } for idx in exclude_idx
            }
        print(f"ICLabel_exclusions: {ICLabel_exclusions}")
        QC(
            clean=epochs, 
            raw=subject_file,
            ica=ica,
            subject_id=sub_id, 
            ICLabel_exclusions=ICLabel_exclusions
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

# https://mne.tools/mne-icalabel/stable/generated/examples/00_iclabel.html#sphx-glr-generated-examples-00-iclabel-py 

def label_ica_exclusion(ica, epochs, threshold=0.8):
    ic_labels = label_components(epochs, ica, 'iclabel')
    labels = ic_labels['labels']
    labels_probs = ic_labels['y_pred_proba']
    for label, prob in zip(labels, labels_probs):
        print(f"Component labeled as {label} with probability {prob:.2f}")
    exclude_idx = [
        idx for idx, label in enumerate(labels) if label not in ["brain", "other"]
        and labels_probs[idx] >= threshold
    ]
    print(f"Excluding components {exclude_idx}")
    return exclude_idx

def QC(clean, raw, ica, subject_id, ICLabel_exclusions: dict):
    """Save plots to investigate the prepreocessing quality"""
    qc_dir = Path("results/preprocessing")

    # Plot the raw epochs for QC
    raw_plt = raw.compute_psd(fmax=100).plot(show=False)
    raw_path = qc_dir / "psd_raw" / f"psd-raw_sub-{subject_id}.png"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_plt.savefig(raw_path.resolve(), bbox_inches="tight")

    # Plot the epochs for QC
    psd_plt = clean.compute_psd(fmax=100).plot(show=False)
    psd_path = qc_dir / "psd_clean" / f"psd-clean_sub-{subject_id}.png"
    psd_path.parent.mkdir(parents=True, exist_ok=True)
    psd_plt.savefig(psd_path.resolve(), bbox_inches="tight")
    
    # Plot ICA components
    ica_plt = ica.plot_components(show=False)
    ica_path = qc_dir / "ica" / f"ica_sub-{subject_id}.png"
    ica_path.parent.mkdir(parents=True, exist_ok=True)
    ica_plt.savefig(ica_path.resolve(), bbox_inches="tight")

    # Save the ICLabel exclusions as txt
    iclabel_path = qc_dir / "iclabel" / f"iclabel_sub-{subject_id}.txt"
    iclabel_path.parent.mkdir(parents=True, exist_ok=True)
    with open(iclabel_path, "w") as f:
        f.write(f"ICA Exclusions for subject {subject_id}:\n")
        for idx, info in ICLabel_exclusions.items():
            f.write(f"Component {idx}: Label={info['label']}, Probability={info['prob']:.2f}\n")

    print(f"Saved QC plots and ICLabel exclusions for subject {subject_id}")


if __name__ == "__main__":
    data_partitions = ["training_set", "validation_set", "test_set"]
    for partition in data_partitions:
        print(f"Preprocessing {partition}...")
        input_dir = mne_dir / partition
        preprocess(input_dir, partition)