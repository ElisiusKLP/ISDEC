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
Run ``uv run src/classification.py`` to train of training set and classify on validation set. 

Example: ``uv run src/classification.py --model=logistic_regression --feature=bandpower --scale``

Run --help for an overview of the classification cli settings. ``uv run src/classification.py --help``



