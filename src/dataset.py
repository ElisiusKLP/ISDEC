from scipy.constants import h
from pathlib import Path
import os
import mne
import scipy.io as sio
import numpy as np
import h5py
from tqdm import tqdm
import joblib
import re

def create_dataset(data_dir, output_dir):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    split_name = data_dir.name.lower().replace(" ", "_")
    output_dir.mkdir(parents=True, exist_ok=True) # Create the output directory if it doesn't exist
    mne_dir = output_dir / "mne" / split_name
    joblib_dir = output_dir / "joblib" / split_name
    mne_dir.mkdir(parents=True, exist_ok=True)
    joblib_dir.mkdir(parents=True, exist_ok=True)
    
    all_files = list(data_dir.glob('*.mat'))

    for mat_file in tqdm(all_files, desc="Processing files"):
        # Find subject id from filename
        pattern = r"Data_Sample(\d{2})\.mat"
        match = re.search(pattern, mat_file.name)
        if match is None:
            raise ValueError(f"Unexpected filename format: {mat_file.name}")
        subject_id = match.group(1)

        abs_mat_file = mat_file.resolve()
        print(f"\nLoading data from {abs_mat_file}...")

        # Convert .mat to MNE epochs
        epochs = convert_mat_to_mne(abs_mat_file)
        print(f"Created epochs type {type(epochs)} with shape {epochs.get_data().shape} for subject {subject_id}")

        # Save epochs to .fif
        fif_filename = f"epochs_sub-{subject_id}-epo.fif"
        fif_file = mne_dir / fif_filename
        epochs.save(fif_file, overwrite=True)
        print(f"Saved epochs to {fif_file}")

        # Extract raw data arrays
        subject_data = mne_to_dict(epochs, subject_id, class_names=list(epochs.event_id.keys()))

        # Save raw data to joblib
        joblib_filename = f"raw_sub{subject_id}.joblib"

        joblib_file = joblib_dir / joblib_filename

        joblib.dump(subject_data, joblib_file)
        print(f"Saved raw data to {joblib_file}")

def convert_mat_to_mne(mat_file):
    class_names = None

    try:
        # First try to load matlab file with scipy sio.loadmat
        mat_data = sio.loadmat(mat_file)
        print("keys in .mat file:", mat_data.keys())

        # Extracting data and labels
        if 'epo_train' in mat_data:
            raw = mat_data['epo_train']
        elif 'epo_validation' in mat_data:
            raw = mat_data['epo_validation']
        elif 'epo_test' in mat_data:
            raw = mat_data['epo_test']
        else:
            raise ValueError(f"Neither 'epo_train', 'epo_validation', nor 'epo_test' found in {mat_file}")

        epo = raw[0, 0]  # MATLAB struct array cell

        x = epo['x']  # expected shape (n_times, n_channels, n_epochs)
        x_t = np.transpose(x, (2, 1, 0))  # MNE format: (n_epochs, n_channels, n_times)

        sfreq = float(epo["fs"][0][0])  # sampling frequency

        ch_names = [
            str(np.ravel(c)[0])  # unwrap nested array -> string
            for c in np.ravel(epo["clab"])
        ]
        # events
        y = epo["y"]
        labels = np.argmax(y, axis=0).astype(int)

        if "className" in epo.dtype.names:
            class_names = [str(c[0]) for c in epo["className"][0]]

        montage = mat_to_mne_montage(mat_data=mat_data, set_type="train")

    except NotImplementedError:
        print("Detected MATLAB v7.3 file, loading with h5py...")
        with h5py.File(mat_file, "r") as h5f:
            if "epo_train" in h5f:
                set_type = "train"
                epo = h5f["epo_train"]
            elif "epo_validation" in h5f:
                set_type = "validation"
                epo = h5f["epo_validation"]
            elif "epo_test" in h5f:
                set_type = "test"
                print(f"Decoding test set {mat_file} with h5py, which may be missing labels...")
                epo = h5f["epo_test"]
                
            else:
                raise ValueError(f"Neither 'epo_train', 'epo_validation', nor 'epo_test' found in {mat_file}")

            x = np.array(epo["x"])
            # Test-set v7.3 files already store (n_epochs, n_channels, n_times).
            if x.shape[0] < x.shape[2]:
                x_t = x
            else:
                x_t = np.transpose(x, (2, 1, 0))

            sfreq = float(np.array(epo["fs"]).squeeze())

            def _decode_h5_string(ref):
                if isinstance(ref, h5py.Reference):
                    arr = np.array(h5f[ref]).squeeze()
                    if arr.dtype.kind in {"u", "i"}:
                        return "".join(chr(int(v)) for v in np.ravel(arr) if int(v) != 0)
                    return str(arr)

                if isinstance(ref, bytes):
                    return ref.decode("utf-8", errors="ignore").rstrip("\x00")

                if isinstance(ref, np.ndarray):
                    arr = np.asarray(ref).squeeze()
                    if arr.dtype.kind in {"u", "i"}:
                        return "".join(chr(int(v)) for v in np.ravel(arr) if int(v) != 0)
                    return str(arr)

                return str(ref)

            ch_names = [_decode_h5_string(ref) for ref in np.array(epo["clab"]).squeeze()]

            y = np.array(epo["y"]).squeeze() if "y" in epo else np.array([])

            if "className" in epo:
                class_names = [_decode_h5_string(ref) for ref in np.array(epo["className"]).squeeze()]

            if y.ndim == 2 and y.shape[1] == x_t.shape[0]:
                labels = np.argmax(y, axis=0).astype(int)
            elif y.ndim == 1 and y.size == x_t.shape[0]:
                labels = y.astype(int)
            else:
                print("No per-epoch labels found, assigning label 0 to all epochs.")
                labels = np.zeros(x_t.shape[0], dtype=int)

            montage = mat_to_mne_montage(h5f, set_type=set_type)

    info = mne.create_info(
        ch_names=ch_names,
        sfreq=sfreq,
        ch_types="eeg",
    )

    n_epochs = x_t.shape[0]

    events = np.column_stack([
        np.arange(n_epochs),  # sample index (dummy within-epoch indexing)
        np.zeros(n_epochs, int),
        labels
    ])

    unique_classes = np.unique(labels)

    if class_names is not None and len(class_names) > 0:
        event_id = {
            class_names[i] if i < len(class_names) else f"class_{int(c)}": int(c)
            for i, c in enumerate(unique_classes)
        }
    else:
        event_id = {f"class_{int(c)}": int(c) for c in unique_classes}

    epochs = mne.EpochsArray(
        x_t, info,
        events=events, event_id=event_id,
        tmin=-0.5)

    #montage = mne.channels.make_standard_montage("standard_1020")
    epochs.set_montage(montage)

    return epochs

def mne_to_dict(epochs, subject_id, class_names):
    x = epochs.get_data()
    y = epochs.events[:, -1]
    sfreq = epochs.info['sfreq']

    return {
        'x': x,
        'y': y,
        'sfreq': sfreq,
        'class_names': class_names,
        'subject_id': subject_id
    }

def mat_to_mne_montage(mat_data, set_type: str):

    if set_type == "test":
        pos_3d = np.asarray(mat_data['mnt']['pos_3d'])
        channels = np.asarray(mat_data['mnt']['clab']).squeeze()
        pos = pos_3d
        channel_names = []
        for ch in channels:
            if isinstance(ch, h5py.Reference):
                chars = np.asarray(mat_data[ch]).squeeze()
                channel_names.append("".join(chr(int(c)) for c in np.ravel(chars) if int(c) != 0))
            else:
                channel_names.append(str(ch))
        
    else:
        pos_3d = mat_data['mnt'][0,0][2]
        channels = mat_data['mnt'][0,0][3]

        pos = np.stack((pos_3d[0], pos_3d[1], pos_3d[2]), axis=1)
        channel_names = [ch[0] for ch in channels[0]]

    mne_montage = mne.channels.make_dig_montage(ch_pos=dict(zip(channel_names, pos)), coord_frame='head')
    return mne_montage

def decode_matlab_obj(obj, file):
    """Recursively decode MATLAB HDF5 objects into Python types."""

    # Case 1: Dataset
    if isinstance(obj, h5py.Dataset):
        data = obj[()]

        # --- Case 1a: object references (MATLAB cell arrays / strings)
        if obj.dtype == 'object':
            decoded = []
            for ref in data.flatten():
                if isinstance(ref, h5py.Reference):
                    decoded.append(decode_matlab_obj(file[ref], file))
                else:
                    decoded.append(ref)
            return np.array(decoded, dtype=object).reshape(data.shape)

        # --- Case 1b: char arrays (uint16 → string)
        if data.dtype == np.uint16:
            try:
                return "".join(chr(c) for c in data.flatten())
            except:
                return data

        # --- Case 1c: numeric
        return data

    # Case 2: Group (MATLAB struct)
    elif isinstance(obj, h5py.Group):
        return {key: decode_matlab_obj(obj[key], file) for key in obj.keys()}

    else:
        return obj
    

if __name__ == "__main__":
    data_dir = Path("data/is_dataset").absolute()
    # glob subdir
    sets = data_dir.glob("*/")
    output_dir = Path("data/derivatives/raw").absolute()
    
    for set_dir in sets:
        print(f"Processing dataset in {set_dir}...")
    
        create_dataset(set_dir, output_dir)