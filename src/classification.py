import click
from matplotlib.pylab import plot
import models
import numpy as np
import re
import sklearn
import joblib
import typer
import time
from rich import print
from pathlib import Path
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm
from models import ModelStrategy
from plotting import plot_confusion_matrices_grid

# https://scikit-learn.org/stable/modules/ensemble.html#random-forest-parameters

joblib_dir = Path("data/derivatives/preprocessed/joblib")

train_dir = joblib_dir / "training_set"

val_dir = joblib_dir / "validation_set"


# ============================================================================
# Model Registry - Map model names to strategies
# ============================================================================

MODEL_REGISTRY = {
    "random_forest": models.RandomForestStrategy,
    "logistic_regression": models.LogisticRegressionStrategy,
    "svm": models.SVMStrategy,
    "eegnet": models.EEGNetStrategy,
}

FEATURE_TYPES = [
    "downsample", 
    "tfr_morlet",
    "tfr_morlet_bands",
    "tfr_dwt_cmor",
    "dwt_hierarchical",
    "dwt_channel_select",
    "tfr_pca",
    "bandpower_mean",
    "bandpower_phase",
    "stack",
    "bandphase"
]

MODEL_CHOICES = tuple(MODEL_REGISTRY.keys())
FEATURE_CHOICES = tuple(FEATURE_TYPES)


def get_model_strategy(model_name: str, scale: bool = True, feature_type: str = "stack") -> ModelStrategy:
    """Get strategy for the specified model"""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}"
        )
    if model_name != "eegnet" and feature_type not in FEATURE_TYPES:
        raise ValueError(
            f"Unknown feature type: {feature_type}. Available: {FEATURE_TYPES}"
        )
    if model_name == "eegnet":
        # EEGNet can take on feature extraction
        if feature_type != "raw" and feature_type in FEATURE_TYPES:
            return MODEL_REGISTRY[model_name](scale=False, feature_type=feature_type)
        else:
        # if EEGNet is not given a feature extraction type, it gets "raw"
            return MODEL_REGISTRY[model_name](scale=False, feature_type="raw")
    return MODEL_REGISTRY[model_name](scale=scale, feature_type=feature_type)


# ============================================================================
# Main training function
# ============================================================================

def fit_model(strategy: ModelStrategy, train_dir: Path | None = None, val_dir: Path | None = None):
    """Train and validate model using the given strategy"""
    if train_dir is None or val_dir is None:
        train_dir, val_dir = strategy.get_data_dirs()

    train_files = sorted(train_dir.glob("*.joblib"))
    if len(train_files) == 0:
        raise ValueError("No training files collected from glob(*.joblib)")

    # Infer class labels once to keep confusion matrices consistent across subjects.
    class_labels_set = set()
    for file in train_files:
        data = joblib.load(file)
        class_labels_set.update(np.unique(data["y"]).tolist())
        val_candidates = list(val_dir.glob(f"*{file.name}"))
        if len(val_candidates) == 0:
            raise ValueError(f"Did not find validation file for {file.name}")
        val_data = joblib.load(val_candidates[0])
        class_labels_set.update(np.unique(val_data["y"]).tolist())
    class_labels = sorted(class_labels_set)
    n_classes = len(class_labels)
    chance_baseline = 1.0 / n_classes
    aggregate_confusion = np.zeros((n_classes, n_classes), dtype=int)
    aggregate_samples = 0
    aggregate_correct = 0
    
    # setup a score logger
    score_logger = []
    strategy_scale = bool(getattr(strategy, "scale", True))
    strategy_feature_type = str(getattr(strategy, "feature_type", "stack"))
    scale_tag = 'scale' if strategy_scale else 'no_scale'
    feature_tag = strategy_feature_type
    run_tag = f"{strategy.get_name()}_{feature_tag}_{scale_tag}"

    # Prepare result and log directories so per-subject logs can be appended during the loop
    confusion_dir = Path(f"results/classification/{strategy.get_name()}/{feature_tag}/{scale_tag}/plots")
    result_dir = Path(f"results/classification/{strategy.get_name()}/{feature_tag}/{scale_tag}/joblib")
    log_dir = Path(f"results/classification/{strategy.get_name()}/{feature_tag}/{scale_tag}/logs")
    confusion_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    score_log_path = log_dir / f"{run_tag}_scores.csv"
    timing_log_path = log_dir / f"{run_tag}_timing.log"
    summary_path = log_dir / f"{run_tag}_summary.txt"

    # Create files with headers if they don't exist so we can append per-subject entries safely
    if not score_log_path.exists():
        with open(score_log_path, "w") as f:
            f.write("subject,score,chance_baseline,majority_baseline,delta_over_chance,delta_over_majority\n")
    if not timing_log_path.exists():
        with open(timing_log_path, "w") as f:
            f.write("subject,train_time_sec,predict_time_sec\n")
    if not summary_path.exists():
        with open(summary_path, "w") as f:
            f.write(f"Per-Subject Summary Log for {run_tag}\n\n")

    for file in tqdm(train_files, desc="Training models on subjects"):
        print(f"Classifying on file: {file.name}")
        match = re.search(r"preprocessed_sub-(\d{2}).joblib", file.name)
        if match:
            sub_id = match.group(1)
            print(f"Found sub_id: {sub_id} in file: {file.name}")
        else:
            print(f"No match for file: {file.name}")
        print(f"Subject ID: {sub_id}")  

        # Load training data
        data = joblib.load(file)
        X_train = data["x"]
        y_train = data["y"]

        # Load validation data first (needed for channel selection feature type)
        val_candidates = list(val_dir.glob(f"*{file.name}"))
        if len(val_candidates) == 0:
            raise ValueError("Did not find any validation file")
        val_file = val_candidates[0]
        print(f"Validating on file: {val_file.resolve()}")
        val_data = joblib.load(val_file)
        x_val = val_data["x"]
        y_val = val_data["y"]

        # Set raw data for strategies that need it (e.g., channel selection)
        set_raw_data = getattr(strategy, "set_raw_data", None)
        if callable(set_raw_data):
            set_raw_data(X_train, y_train, x_val)

        # Transform training data
        print(f"Train shape before feature extraction: \n X_train: {X_train.shape} \n y_train: {y_train.shape}")
        X_train = strategy.transform_train(X_train)
        print(f"Train shape after feature extraction: \n X_train: {X_train.shape} \n y_train: {y_train.shape}")

        set_data_info = getattr(strategy, "set_data_info", None)
        if callable(set_data_info) and getattr(strategy, "input_shape", None) is None:
            set_data_info(X_train, class_labels)

        encode_targets = getattr(strategy, "encode_targets", None)
        if callable(encode_targets):
            y_train_fit = encode_targets(y_train)
        else:
            y_train_fit = y_train
        

        # Train model
        model = strategy.create_model()
        print(f"Training model: {strategy.get_name()}")
        train_start = time.perf_counter()
        model.fit(X_train, y_train_fit)
        train_time = time.perf_counter() - train_start

        # Transform validation data using strategy
        x_val = strategy.transform_val(x_val)
        print(f"Validation shape after feature extraction: \n x_val: {x_val.shape} \n y_val: {y_val.shape}")

        # Evaluate
        predict_start = time.perf_counter()
        raw_predictions = model.predict(x_val)
        predict_time = time.perf_counter() - predict_start
        decode_targets = getattr(strategy, "decode_targets", None)
        if callable(decode_targets):
            y_pred = decode_targets(raw_predictions)
        else:
            y_pred = raw_predictions
        score = accuracy_score(y_val, y_pred)
        class_counts = np.bincount(y_val.astype(int))
        majority_baseline = class_counts.max() / len(y_val)
        delta_over_chance = score - chance_baseline
        delta_over_majority = score - majority_baseline

        # log score for subject
        score_logger.append({
            "subject": file.stem,
            "score": score,
            "chance_baseline": chance_baseline,
            "majority_baseline": majority_baseline,
            "delta_over_chance": delta_over_chance,
            "delta_over_majority": delta_over_majority,
            "train_time": train_time,
            "predict_time": predict_time,
        })

        # Confusion matrix
        confusion_matrix_val = confusion_matrix(y_val, y_pred, labels=class_labels)
        aggregate_confusion += confusion_matrix_val
        aggregate_samples += len(y_val)
        aggregate_correct += np.trace(confusion_matrix_val)
        print(f"Trained on {file.name}, validated on {val_file.name}: {score:.3f}")
        print(
            "Baselines "
            f"(chance={chance_baseline:.3f}, majority={majority_baseline:.3f}) "
            f"| deltas (over chance={delta_over_chance:.3f}, over majority={delta_over_majority:.3f})"
        )
        print(f"Confusion Matrix:\n{confusion_matrix_val}")

        # Save plot
        disp = sklearn.metrics.ConfusionMatrixDisplay(
            confusion_matrix=confusion_matrix_val,
            display_labels=class_labels,
        )
        ax = disp.plot()
        disp.ax_.set_title(f"Confusion Matrix — Trained: {file.name}")

        confusion_dir = Path(f"results/classification/{strategy.get_name()}/{feature_tag}/{scale_tag}/plots")
        confusion_dir.mkdir(parents=True, exist_ok=True)
        fig = disp.ax_.figure
        fig.savefig(confusion_dir / f"{run_tag}_{file.stem}_confusion_matrix.png")

        # write result to joblib
        result_path = result_dir / f"{run_tag}_{file.stem}.joblib"
        joblib.dump({
            "y_true": y_val,
            "y_pred": y_pred,
            "score": score,
            "chance_baseline": chance_baseline,
            "majority_baseline": majority_baseline,
            "delta_over_chance": delta_over_chance,
            "delta_over_majority": delta_over_majority,
            "confusion_matrix": confusion_matrix_val,
            "model_info": {
                "model_name": strategy.get_name(),
                "feature_type": strategy_feature_type,
                "scale": strategy_scale,
            },
            "subject_id": sub_id,
            "train_time": float(train_time),
            "predict_time": float(predict_time),
        }, result_path)
        print(f"Saved joblib results to {result_path.resolve()}")

        # Append per-subject entries to rolling logs so partial results are available immediately
        try:
            with open(score_log_path, "a") as f:
                f.write(
                    f"{file.stem},{score:.4f},{chance_baseline:.4f},{majority_baseline:.4f},"
                    f"{delta_over_chance:.4f},{delta_over_majority:.4f}\n"
                )
        except Exception:
            print(f"Warning: failed to append to score log: {score_log_path}")

        try:
            with open(timing_log_path, "a") as f:
                f.write(f"{file.stem},{train_time:.2f},{predict_time:.2f}\n")
        except Exception:
            print(f"Warning: failed to append to timing log: {timing_log_path}")

        try:
            with open(summary_path, "a") as f:
                f.write(
                    f"Subject: {file.stem} | score: {score:.4f} | "
                    f"train_time: {train_time:.2f}s | predict_time: {predict_time:.2f}s\n"
                )
        except Exception:
            print(f"Warning: failed to append to summary log: {summary_path}")
    
    # save all images into a single grid in a pdf
    tags = ["preprocessed", "raw"]
    set_tag = [tag for tag in tags if tag in train_files[0].stem]
    if isinstance(set_tag, list) and len(set_tag) == 1:
        set_tag = set_tag[0]
    elif isinstance(set_tag, str):
        pass
    else:
        raise ValueError(f"Expected exactly one set tag in {tags} to be in filename, but got: {set_tag}")
    
    confusion_tag = f"{run_tag}_{set_tag}"
    plot_confusion_matrices_grid(img_dir=confusion_dir, set_tag=confusion_tag)

    # Final aggregate summary
    overall_accuracy = aggregate_correct / aggregate_samples if aggregate_samples > 0 else 0.0
    mean_subject_score = float(np.mean([entry["score"] for entry in score_logger]))
    mean_majority_baseline = float(np.mean([entry["majority_baseline"] for entry in score_logger]))
    row_sums = aggregate_confusion.sum(axis=1, keepdims=True)
    normalized_confusion = np.divide(
        aggregate_confusion,
        row_sums,
        out=np.zeros_like(aggregate_confusion, dtype=float),
        where=row_sums != 0,
    )
    summary_lines = [
        "",
        "=== Aggregate Validation Summary ===",
        f"Class labels: {class_labels}",
        f"Overall accuracy (micro): {overall_accuracy:.3f}",
        f"Mean subject accuracy (macro): {mean_subject_score:.3f}",
        f"Theoretical chance baseline: {chance_baseline:.3f}",
        f"Mean majority baseline: {mean_majority_baseline:.3f}",
        f"Overall delta over chance: {overall_accuracy - chance_baseline:.3f}",
        f"Overall delta over majority: {overall_accuracy - mean_majority_baseline:.3f}",
        f"Aggregate confusion matrix:\n{aggregate_confusion}",
        "Row-normalized confusion matrix (rows=true class, cols=predicted class):",
        np.array2string(normalized_confusion, formatter={"float_kind": lambda x: f"{x:.3f}"}),
        "Most frequent off-diagonal confusions per class:",
    ]
    for idx, class_id in enumerate(class_labels):
        off_diag = aggregate_confusion[idx].copy()
        off_diag[idx] = 0
        if off_diag.sum() == 0:
            summary_lines.append(f"  class {class_id}: no off-diagonal confusions")
            continue
        confused_with_idx = int(np.argmax(off_diag))
        confused_with = class_labels[confused_with_idx]
        confusion_count = int(off_diag[confused_with_idx])
        confusion_rate = normalized_confusion[idx, confused_with_idx]
        summary_lines.append(
            f"  class {class_id} -> class {confused_with}: "
            f"count={confusion_count}, rate={confusion_rate:.3f}"
        )

    summary_text = "\n".join(summary_lines)
    print(summary_text)

    summary_dir = Path(f"results/classification/{strategy.get_name()}/{feature_tag}/{scale_tag}/logs")
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / f"{run_tag}_summary.txt"
    with open(summary_path, "w") as f:
        f.write(summary_text.lstrip("\n") + "\n")
    print(f"Saved summary to {summary_path.resolve()}")

    # save scores to a log file
    score_log_dir = Path(f"results/classification/{strategy.get_name()}/{feature_tag}/{scale_tag}/logs")
    score_log_dir.mkdir(parents=True, exist_ok=True)
    score_log_path = score_log_dir / f"{run_tag}_scores.csv"
    with open(score_log_path, "w") as f:
        f.write("subject,score,chance_baseline,majority_baseline,delta_over_chance,delta_over_majority\n")
        for entry in score_logger:
            f.write(
                f"{entry['subject']},{entry['score']:.4f},{entry['chance_baseline']:.4f},"
                f"{entry['majority_baseline']:.4f},{entry['delta_over_chance']:.4f},"
                f"{entry['delta_over_majority']:.4f}\n"
            )
    print(f"Saved scores to {score_log_path.resolve()}")

    # save timing log file
    timing_log_path = score_log_dir / f"{run_tag}_timing.log"
    mean_train_time = float(np.mean([entry["train_time"] for entry in score_logger]))
    mean_predict_time = float(np.mean([entry["predict_time"] for entry in score_logger]))
    with open(timing_log_path, "w") as f:
        f.write("Per-Subject Timing Log\n")
        f.write(f"Model: {strategy.get_name()} | Feature Type: {feature_tag} | Scale: {scale_tag}\n")
        f.write("\n")
        f.write("subject,train_time_sec,predict_time_sec\n")
        for entry in score_logger:
            f.write(
                f"{entry['subject']},{entry['train_time']:.2f},{entry['predict_time']:.2f}\n"
            )
        f.write("\n")
        f.write("=== Timing Summary ===\n")
        f.write(f"Mean training time: {mean_train_time:.2f} seconds\n")
        f.write(f"Mean prediction time: {mean_predict_time:.2f} seconds\n")
    print(f"Saved timing log to {timing_log_path.resolve()}")

def main( 
    model: str = typer.Option(
        "random_forest",
        help="Type of model to train",
        click_type=click.Choice(MODEL_CHOICES),
    ),
    feature: str = typer.Option(
        "stack",
        help="Type of features to create.",
        click_type=click.Choice(FEATURE_CHOICES),
    ),
    scale: bool = typer.Option(True, "--scale/--no-scale", help="Enable feature scaling.")
):
    print(f"Training model: {model}")
    print(f"Feature type: {feature}")
    print(f"Scaling enabled: {scale}")
    strategy = get_model_strategy(model, scale=scale, feature_type=feature)
    print(f"Data kind: {getattr(strategy, 'model_type', 'None')}")

    fit_model(strategy, *strategy.get_data_dirs())


if __name__ == "__main__":
    typer.run(main)