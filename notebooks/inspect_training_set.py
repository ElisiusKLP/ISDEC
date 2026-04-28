import marimo

__generated_with = "0.23.3"
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

    return Path, mne, np, os, sio


@app.cell
def _(Path, os):
    # set workdir
    print("Current working directory:", Path.cwd())
    if Path.cwd().name == "notebooks":
        os.chdir(Path.cwd().parent)
        print("New working directory:", Path.cwd())
    rootdir = Path.cwd()
    return (rootdir,)


@app.cell
def _(Path, sio):
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

    # extract data
    print(f"Header: {datafile['__header__']}")
    raw = datafile["epo_train"]  # shape: (n_channels, n_timepoints)
    epo = raw[0,0]
    print(epo.dtype.names)
    print(f"Shape of epo['x']: {epo['x'].shape}")
    print(f"There are {epo['x'].shape[0]} timepoints, {epo['x'].shape[1]} channels, and {epo['x'].shape[2]} training epochs.")
    print(f"Shape of epo['y']: {epo['y'].shape}")

    # find classnames
    print(f"Shape of epo['className']: {epo['className'].shape}")
    epo["className"][0]
    class_names = [str(c[0]) for c in epo["className"][0]]
    print(f"Class names: {class_names}")
    return class_names, epo


@app.cell
def _(epo, mne, np):

    x = epo["x"]  # (795, 64, 300)

    x_t = np.transpose(x, (2, 1, 0))  # → (300, 64, 795)
    x_t.shape

    sfreq = epo["fs"][0][0]
    print(f"Sampling frequency: {sfreq} Hz")

    ch_names = [
        str(np.ravel(c)[0])   # unwrap nested array → string
        for c in np.ravel(epo["clab"])
    ]
    print(f"Channel names: {ch_names}")

    info = mne.create_info(
        ch_names=ch_names,
        sfreq=sfreq,
        ch_types="eeg"
    )

    # events
    y = epo["y"]  # (5, 300)
    labels = np.argmax(y, axis=0)  # (300,)
    n_epochs = x_t.shape[0]

    events = np.column_stack([
        np.arange(n_epochs),   # sample index (dummy within-epoch indexing)
        np.zeros(n_epochs, int),
        labels
    ])
    print(f"Events array shape: {events.shape}")
    unique_classes = np.unique(labels)

    event_id = {f"class_{c}": c for c in unique_classes}
    print(f"Event ID mapping: {event_id}")

    epochs = mne.EpochsArray(
        x_t, info, 
        events=events, event_id=event_id,
        tmin=-0.5,
        baseline=(-0.5, 0))

    montage = mne.channels.make_standard_montage("standard_1020")
    epochs.set_montage(montage)
    return (epochs,)


@app.cell
def _():
    # save raw epochs to disk
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Plotting Raw Epochs
    """)
    return


@app.cell
def _(epochs):
    epochs.plot(scalings="auto", n_epochs=10, n_channels=64)

    # plot psd over epochs
    epochs.compute_psd().plot()

    # plot psd over channels
    epochs.compute_psd(fmax=128).plot(average=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Evokeds?
    """)
    return


@app.cell
def _(class_names, epochs):
    # plots evokeds for all 5 classes
    plots = []
    # create an evokeds dict
    for i in range(5):
        evoked = epochs[f"class_{i}"].average()
        plot = evoked.plot(titles=f"Evoked response for class '{class_names[i]}'")
        plots.append(plot)
    return (plots,)


@app.cell
def _(plots, rootdir):
    # save all plots into a single PDF file
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt

    figdir = rootdir / "figures"
    figdir.mkdir(exist_ok=True)
    pdf_path = figdir / "evoked_all_classes.pdf"

    with PdfPages(pdf_path) as pdf:
        for it in range(5):
            fig = plots[it]

            pdf.savefig(fig)
            plt.close(fig)

    print(f"Saved to {pdf_path}")
    return


@app.cell
def _(class_names, epochs, mne):
    evokeds_list = [
        epochs[f"class_{i}"].average()
        for i in range(5)
    ]

    evokeds = dict(zip(class_names, evokeds_list))

    mne.viz.plot_compare_evokeds(evokeds, title="Evoked responses for all classes", combine="mean")
    return (evokeds,)


@app.cell
def _(evokeds, mne):
    mne.viz.plot_compare_evokeds(evokeds, combine="median", title="Median")
    return


@app.cell
def _(evokeds, mne):
    mne.viz.plot_compare_evokeds(evokeds, combine="gfp", title="gfp")
    return


@app.cell
def _(evokeds, mne):
    def custom_func(x):
        return x.max(axis=1)


    mne.viz.plot_compare_evokeds(evokeds, combine=custom_func)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


app._unparsable_cell(
    r"""
    def
    """,
    name="_"
)


if __name__ == "__main__":
    app.run()
