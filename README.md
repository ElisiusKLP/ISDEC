# Imagined Speech Decoding

Authors
Elisius Krustrup Lyng Pedersen.
Ditlev Kræn Andersen
2026 @ Cognitive Science, Aarhus University

## Setup

Setup is done with uv.
Install uv by astral if not installed yet.
https://docs.astral.sh/uv/getting-started/installation/

Run ``uv sync`` to initiate the virtual envrionment from the *pyproject.toml* file.

Make sure the environment is activated.
Terminal should have (isdec) infront of user.


## Scripts

### dataset.py
Run ``uv run src/dataset.py`` to convert the competition dataset matlab files into a python-friendly (numpy) format.

## classification.py
Run ``uv run src/classification.py --model={{model_name}} --feature={{feature_type}} --scale/--no-scale``

Model is either: logistic_regression, svm, random_forest

Feature is either: bandpower, downsample, stack

Run --help for an overview of the classification cli settings. ``uv run src/classification.py --help``



