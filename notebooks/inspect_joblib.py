import marimo

__generated_with = "0.23.3"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # inspect joblib file
    """)
    return


@app.cell
def _():
    import joblib
    import numpy as np
    from pathlib import Path
    import os

    return Path, joblib, np, os


@app.cell
def _(Path, os):
    # set workdir
    print("Current working directory:", Path.cwd())
    if Path.cwd().name == "notebooks":
        os.chdir(Path.cwd().parent)
        print("New working directory:", Path.cwd())
    return


@app.cell
def _(Path, joblib, np):
    joblib_dir = Path("data/derivatives/raw/joblib")
    #set_str = "validation_set"
    set_str = "training_set"
    set_dir = joblib_dir / set_str
    print("Loading files from:", set_dir)
    all_files = list(set_dir.glob("*.joblib"))

    # pick one subject
    subject_file = all_files[0]
    print("Loading subject file:", subject_file.name)
    data = joblib.load(subject_file)
    print("Data keys:", data.keys())

    # check y labels
    y = data["y"]
    print("Unique labels in y:", np.unique(y))
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
