from jax.random import f
from numpy.f2py.rules import k
from pyexpat import model

import models
import numpy as np
import sklearn
import joblib
import typer
from rich import print
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

from classification import MODEL_REGISTRY, get_model_strategy

joblib_dir = Path("data/derivatives/preprocessed/joblib")

train_dir = joblib_dir / "training_set"
val_dir = joblib_dir / "validation_set"

kfold_dir = Path("results/kfold")

WINNING_MODEL = {
    "model_name": "random_forest",
    "feature_type": "bandpower_mean",
    "scale": True,
    "config": {
        "n_estimators": 1000,
        "criterion": "gini",
        "max_depth": None,
        "min_samples_split": 5,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
    }
}

def run_kfold_on_subset_data(model, X, y, n_splits=5):
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2001)

    scores = []
    
    for i, (train_index, test_index) in enumerate(kfold.split(X, y)):
        print(f"Fold {i+1}/{n_splits}")
        print(f"   Train indices: {train_index}")
        print(f"   Test indices: {test_index}")

        # we only split the training set
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]

        model.fit(X_train, y_train)

        y_preds = model.predict(X_test)

        score = accuracy_score(y_test, y_preds)
        print(f"   Accuracy: {score:.4f}")
        scores.append(
            {
                "fold": i+1,
                "accuracy": score,
                "precision": precision_score(y_test, y_preds, average="weighted", zero_division=0),
                "recall": recall_score(y_test, y_preds, average="weighted", zero_division=0),
                "f1": f1_score(y_test, y_preds, average="weighted", zero_division=0)
            }
        )

    return scores

def combine_train_val_sets(X_train, y_train, X_val, y_val):
    X = np.vstack([X_train, X_val])
    y = np.hstack([y_train, y_val])
    print(f"Combined training and validation sets: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Class distribution: {np.bincount(y)}")
    return X, y


def fit_models(
    strategy: models.ModelStrategy,
    enable_feature_cache: bool = True
):
    model_name = strategy.get_name()
    strategy_feature_type = str(getattr(strategy, "feature_type", "stack"))
    train_files = sorted(train_dir.glob("*.joblib"))
    if len(train_files) == 0:
        raise ValueError("No training files collected from glob(*.joblib)")
    
    feature_cache_dir = Path("data/derivatives/features_cache") / strategy_feature_type
    train_feature_cache_dir = feature_cache_dir / "training_set"
    val_feature_cache_dir = feature_cache_dir / "validation_set"

    def _process_subject(file_path: Path, model_strategy: models.ModelStrategy):
        # load model
        model = model_strategy.create_model()

        train_data = joblib.load(file_path)
        X_train = train_data["x"]
        y_train = train_data["y"]

        val_candidates = list(val_dir.glob(f"*{file_path.name}"))
        if len(val_candidates) == 0:
            raise ValueError(f"Did not find any validation file for {file_path.name}")
        val_file = val_candidates[0]
        val_data = joblib.load(val_file)
        x_val = val_data["x"]
        y_val = val_data["y"]

        # Transform to features training data
        train_cache_path = train_feature_cache_dir / f"{file_path.stem}.joblib"
        if enable_feature_cache and train_cache_path.exists():
            print(f"    [Feature extraction... cache hit: {train_cache_path.name}]", flush=True)
            X_train_feats = joblib.load(train_cache_path)
        else:
            print(f"    [Feature extraction...]", end="", flush=True)
            X_train_feats = strategy.transform_train(X_train)
            print(f" done. Shape: {X_train_feats.shape}", flush=True)
            if enable_feature_cache:
                joblib.dump(X_train_feats, train_cache_path, compress=3)
                print(f"    [Saved feature cache: {train_cache_path.name}]", flush=True)

        # Transform to features validation data
        val_cache_path = val_feature_cache_dir / f"{val_file.stem}.joblib"
        if enable_feature_cache and val_cache_path.exists():
            print(f"    [Validation feature extraction... cache hit: {val_cache_path.name}]", flush=True)
            x_val_feats = joblib.load(val_cache_path)
        else:
            print(f"    [Validation feature extraction...]", end="", flush=True)
            x_val_feats = strategy.transform_val(x_val)
            print(f" done.", flush=True)
            if enable_feature_cache:
                joblib.dump(x_val_feats, val_cache_path, compress=3)
                print(f"    [Saved validation cache: {val_cache_path.name}]", flush=True)

        # Combine the training and validation sets for k-fold cross-validation
        X, y = combine_train_val_sets(X_train_feats, y_train, x_val_feats, y_val)

        props = list(np.arange(0.1, 1.01, 0.05))
        print(f"proportions: {props}")

        for prop in props:
            # subset the dataset to the given proportion
            n_samples = int(X.shape[0] * prop)
            print(f"Subsetting to {n_samples} samples ({prop*100:.1f}%)")
            indices = np.random.choice(X.shape[0], n_samples, replace=False)
            X_subset = X[indices]
            y_subset = y[indices]
            print(f"Subset shape: {X_subset.shape}, {y_subset.shape}")

            # run k-fold cross-validation on the subsetted data
            kfold_result = run_kfold_on_subset_data(model, X_subset, y_subset, n_splits=5)

            kfold_result = {
                "model_name": model_name,
                "feature_type": strategy_feature_type,
                "proportion": prop,
                "scores": kfold_result
            }

            joblib_dir = kfold_dir / model_name / strategy_feature_type / f"prop_{int(prop*100)}"
            joblib_dir.mkdir(parents=True, exist_ok=True)

            result_path = joblib_dir / f"{file_path.stem}_prop{int(prop*100)}.joblib"

            joblib.dump(kfold_result, result_path, compress=3)
            print(f"Saved k-fold result for proportion {prop} to {result_path}")

    for file_path in tqdm(train_files, desc=f"Processing {model_name} with {strategy_feature_type}"):
        _process_subject(file_path, strategy)

def main():
    # Load the winning model
    model_strategy = get_model_strategy(
        model_name=str(WINNING_MODEL["model_name"]),
        feature_type=str(WINNING_MODEL["feature_type"]),
        scale=bool(WINNING_MODEL["scale"]),
        config=dict(WINNING_MODEL["config"])
    )

    fit_models(model_strategy, enable_feature_cache=True)
    


if __name__ == "__main__":
    # Run main when executed as a script
    main()