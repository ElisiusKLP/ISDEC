# Imagined Speech Decoding

**Optimising EEG Imagined Speech Decoding: A Grid-Search Analysis of Models and Feature Representations**

This project explores the performance of **2,258 model-feature-hyperparameter combinations** for decoding imagined speech from EEG data. We systematically evaluate **Logistic Regression, Random Forest, and SVM** classifiers across a range of feature extraction methods—from time-domain statistics to frequency-domain (Bandpower Mean) and time-frequency-domain (Discrete Wavelet Transform) representations. The study identifies **Random Forest with Bandpower Mean features** as the top-performing pipeline, achieving **38.7% decoding accuracy** in a 5-fold cross-validation (5 words/phrases) and highlights the critical role of spectral information in EEG-based imagined speech classification.

Authors
Elisius Krustrup Lyng Pedersen.
Ditlev Kræn Andersen
2026 @ Cognitive Science, Aarhus University

The final paper might be attached to the repository at a later point.

## Setup

Setup is done with uv.
Install uv by astral if not installed yet.
https://docs.astral.sh/uv/getting-started/installation/

Run ``uv sync`` to initiate the virtual envrionment from the *pyproject.toml* file.

Run ``source .venv/bin/activate`` to activate the virtual environment.

Make sure the environment is activated.
Terminal should have (isdec) infront of user.

### Data 

The dataset used in the current study can be found at:

https://osf.io/pq7vb/files/osfstorage [BCI Competition Committee. (2020). 2020 International BCI Competition. https://doi.org/10.17605/OSF.IO/PQ7VB]

In the sidebar, navigate to Files and download the folder "Track#3 Imagines speech classification".

The contents should be copied into a directory names ``data/is_dataset/`` in the root folder.

If you are interested in the **results** from the Grid-Search all the scores are saved as a csv file in ``results/summary/classification_summary.csv``. Use ``uv run src/summarise_results.py`` to create the aggregated summary data and all figures. 

### Run procedure

When the data is in the right folder you can run the full preprocessing + analysis pipeline using the following scripts from the terminal in the designated order:
1. ``uv run src/dataset.py``
2. ``uv run src/preprocess.py``
3. ``uv run src/schedule.py -s 11 --use-config``
4. ``uv run src/kfold_crossval.py``
5. ``uv run src/summarise_results``

## Scripts

### dataset.py
Run ``uv run src/dataset.py`` to convert the competition dataset matlab files into a python-friendly (numpy) format.

### preprocess.py

Run ``uv run src/preprocess.py`` to perform data preprocessing steps required before training and evaluation.

This script reads the raw dataset from ``data/is_dataset/`` and produces preprocessed and derivative outputs (for example under ``data/preprocessed/`` and ``data/derivatives/``) used by the analysis pipeline.

Typical operations include filtering, resampling, epoching, simple artifact rejection and feature extraction
according to the flags and configuration in the script. Use ``uv run src/preprocess.py --help`` to view
available options such as limiting subjects, changing resample frequency, or forcing a re-run of cached
feature extraction.

### classification.py
Run ``uv run src/classification.py`` to train of training set and classify on validation set. 

Example: ``uv run src/classification.py --model=logistic_regression --feature=bandpower --scale``

Run --help for an overview of the classification cli settings. ``uv run src/classification.py --help``

### schedule.py
Run ``uv run src/schedule.py`` to run a model training and validation schedule of different models and feature extraction types. Schedules are defined by integers in ``src/schedules.json``. It can also run a grid search across possible parameter configuration defined in ``src/schedule_config.json``. Additionally, use ``--use-config`` to expand the grid search over hyperparameter configurations.

Example: ``uv run src/schedule.py --use-config -s 6``

### kfold_crossval.py

Run ``uv run src/kfold_crossval.py`` to run stratified k-fold cross-validation experiments on combined training and validation data.

The script reads preprocessed feature caches from ``data/derivatives/preprocessed/joblib/`` (training and validation),
combines them, and runs k-fold splits across varying sample proportions to evaluate model stability and scaling with data amount.

Results for each proportion and fold are saved under ``results/kfold/<model_name>/<feature_type>/prop_<N>/`` as ``.joblib`` files suitable for later aggregation.
Use ``uv run src/kfold_crossval.py --help`` for runtime options and to adjust the winning model configuration or proportions.

### summarise_results.py

Run ``uv run src/summarise_results.py`` to aggregate classification outputs and produce summary tables and plots.
The script collects per-subject results from ``results/classification/`` (and k-fold results from ``results/kfold/``),
creates aggregated CSV summaries and generates visualizations saved to ``results/summary/``. 

Outputs include CSVs of collected data and aggregated summaris and also figures and tables used for data inspection and reporting in the paper.

