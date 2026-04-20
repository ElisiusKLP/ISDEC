import marimo

__generated_with = "0.23.1"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Inspecting the Imagined Speech Dataset
    """)
    return


@app.cell
def _():
    from pathlib import Path
    import os
    import mne
    import scipy.io as sio
    import numpy as np
    import h5py

    return Path, h5py, np, os, sio


@app.cell
def _(Path, os):
    # set workdir
    print("Current working directory:", Path.cwd())
    if Path.cwd().name == "notebooks":
        os.chdir(Path.cwd().parent)
        print("New working directory:", Path.cwd())
    return


@app.cell
def _(Path, h5py, sio):
    # load data
    testdata_dirpath = Path("data/is_dataset/Training_set")
    subject_ids = [f"{i:02d}" for i in range(1, 16)]
    pattern = f"Data_Sample{{subject_id}}.mat"

    # pick on subject
    sub = subject_ids[0]
    filename = pattern.format(subject_id=sub)
    filepath = testdata_dirpath / filename
    print(f"Loading data from {filepath.absolute()}...")

    datafile = sio.loadmat(filepath.absolute())
    print("Keys in the loaded .mat file:", datafile.keys())

    with h5py.File(filepath.absolute(), 'r') as f:
        data = f
        print("Keys in the .mat file:", list(data.keys()))

        epo_test_group = f['epo_test']
        print("Keys in epo_test group:", list(epo_test_group.keys()))

        print(f"epo_test['clab']: {epo_test_group['clab']}")
        print(f"epo_test['y']: {epo_test_group['y']}")

        # decode all

    return (filepath,)


@app.cell
def _(filepath, h5py):
    with h5py.File(filepath.absolute(), 'r') as file:
        def print_attrs(name, obj):
            print(name)
            print(f"  Type: {type(obj)}")
            if hasattr(obj, 'shape'):
                print(f"  Shape: {obj.shape}")
            if hasattr(obj, 'dtype'):
                print(f"  Dtype: {obj.dtype}")

        file.visititems(print_attrs)
    return


@app.cell
def _(filepath, h5py, np):
    def hdf5_to_dict(group):
        """Recursively convert HDF5 group/dataset to a dictionary of NumPy arrays."""
        result = {}
        for key, item in group.items():
            if isinstance(item, h5py.Dataset):
                # Convert dataset to NumPy array
                result[key] = np.array(item)
            elif isinstance(item, h5py.Group):
                # Recursively process group
                result[key] = hdf5_to_dict(item)
        return result

    with h5py.File(filepath.absolute(), 'r') as file2:
        # Optionally, convert the entire file
        data_dict = hdf5_to_dict(file2)
        print("Data converted to dictionary with keys:", data_dict.keys())
    return (data_dict,)


@app.cell
def _(filepath, h5py):
    fr = h5py.File(filepath.absolute(), "r")
    clab_refs = fr["epo_test"]["clab"]

    clab = []
    for i in range(clab_refs.shape[0]):
        ref = clab_refs[i, 0]        # this is an HDF5 reference
        obj = fr[ref]                # follow reference into #refs#
    
        # decode MATLAB char array (uint16 → string)
        string = "".join(chr(c) for c in obj[:].flatten())
        clab.append(string)

    print(clab)
    return


@app.cell
def _(data_dict):
    print(f"data_dict['epo_test'].keys(): {data_dict['epo_test'].keys()}")

    for key in data_dict['epo_test'].keys():
        print(f"Key: {key}, Shape: {data_dict['epo_test'][key].shape}, Dtype: {data_dict['epo_test'][key].dtype}")
    return


@app.cell
def _(data_dict):
    print(f"data_dict['epo_test']['clab']: {data_dict['epo_test']['clab']}")
    print(f"data_dict['epo_test']['y']: {data_dict['epo_test']['y']}")
    return


@app.cell
def _(filepath, h5py, np):
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

    with h5py.File(filepath.absolute(), 'r') as file3:
        decoded = decode_matlab_obj(file3, file3)

    # Now inspect
    print(decoded.keys())
    print(decoded["epo_test"].keys())
    # print shape of each key
    for key2 in decoded["epo_test"].keys():
        print(f"Key: {key2}, Shape: {decoded['epo_test'][key2].shape if hasattr(decoded['epo_test'][key2], 'shape') else 'N/A'}, Dtype: {type(decoded['epo_test'][key2])}")
    print("clab:", decoded["epo_test"]["clab"])
    print("className:", decoded["epo_test"]["className"])
    print("y:", decoded["epo_test"]["y"])
    print("x shape:", decoded["epo_test"]["x"].shape)

    print("\n Montage: ")
    print(f"Mnt keys: {decoded['mnt'].keys()}")
    print(f" Mnt clab: {decoded['mnt']['clab']}")
    print(f" Mnt x: {decoded['mnt']['x'].shape}")

    print("\n Refs: ")
    print(f" Refs keys: {decoded['#refs#'].keys()}")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
