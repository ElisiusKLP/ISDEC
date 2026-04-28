import marimo

__generated_with = "0.23.3"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Inspecting Preprocessed Joblib data

    Loading and exploring the joblib data to build the random forest classifier.
    """)
    return


@app.cell
def _():
    # imports
    import joblib
    import numpy as np
    from pathlib import Path
    import os

    return Path, joblib, os


@app.cell
def _(Path, os):
    # set workdir
    print("Current working directory:", Path.cwd())
    if Path.cwd().name == "notebooks":
        os.chdir(Path.cwd().parent)
        print("New working directory:", Path.cwd())
    return


@app.cell
def _(Path, joblib):
    dir = Path("data/derivatives/raw/joblib/training_set").resolve()
    all_files = list(dir.glob("*.joblib"))
    print(f"Found {len(all_files)} joblib files in {dir}")

    sub = all_files[0]
    print(f"Loading file: {sub}")
    data_dict = joblib.load(sub)
    print(f"Keys in data_dict: {data_dict.keys()}")

    print(f"X shape: {data_dict['x'].shape}")

    data_dict["class_names"]
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
