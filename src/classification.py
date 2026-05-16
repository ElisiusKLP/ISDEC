import click
import matplotlib
matplotlib.use("Agg")
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


def _format_config_value(value: object) -> str:
    if isinstance(value, dict):
        return _format_config_tag(value)
    if isinstance(value, list):
        return "-".join(_format_config_value(item) for item in value)
    text = str(value)
    return text.replace(" ", "_").replace("/", "-")


def _format_config_tag(config: dict | None) -> str:
    """Build a deterministic filesystem-safe label for a model config."""
    if not config:
        return "default"

    parts = []
    for key in sorted(config.keys()):
        parts.append(f"{key}={_format_config_value(config[key])}")
    return "__".join(parts)


# ============================================================================
# Model Registry - Map model names to strategies
# ============================================================================

MODEL_REGISTRY = {
    "random_forest": models.RandomForestStrategy,
    "logistic_regression": models.LogisticRegressionStrategy,
    "svc": models.SVCStrategy,
    "linear_svc": models.LinearSVCStrategy,
    "eegnet": models.EEGNetStrategy,
}

FEATURE_TYPES = [
    "downsample", 
    "tfr_morlet",
    "tfr_morlet_bands",
    "tfr_dwt_cmor",
    "dwt_hierarchical_allstats",
    "dwt_hierarchical_mean",
    "dwt_channel_select",
    "tfr_pca",
    "bandpower_mean",
    "bandpower_mean_sd",
    "bandpower_mean_window",
    "bandpower_mean_sd_window",
    "bandpower_phase",
    "stack",
    "bandphase"
]

MODEL_CHOICES = tuple(MODEL_REGISTRY.keys())
FEATURE_CHOICES = tuple(FEATURE_TYPES)


def get_model_strategy(model_name: str, scale: bool = True, feature_type: str = "stack", config: dict | None = None) -> ModelStrategy:
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
            return MODEL_REGISTRY[model_name](scale=False, feature_type=feature_type, config=config)
        else:
        # if EEGNet is not given a feature extraction type, it gets "raw"
            return MODEL_REGISTRY[model_name](scale=False, feature_type="raw", config=config)
    return MODEL_REGISTRY[model_name](scale=scale, feature_type=feature_type, config=config)


# ============================================================================
# Main training function
# ============================================================================

def fit_model(strategy: ModelStrategy, train_dir: Path | None = None, val_dir: Path | None = None, n_jobs: int = 1):
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
    strategy_config = getattr(strategy, "config", None)
    scale_tag = 'scale' if strategy_scale else 'no_scale'
    feature_tag = strategy_feature_type
    config_tag = _format_config_tag(strategy_config)
    run_tag = f"{strategy.get_name()}_{feature_tag}_{scale_tag}_{config_tag}"

    # Prepare result and log directories so per-subject logs can be appended during the loop
    # Structure: no config -> .../scale_tag/ | with config -> .../scale_tag/grid/config_tag/
    if strategy_config and len(strategy_config) > 0:
        base_dir = Path(f"results/classification/{strategy.get_name()}/{feature_tag}/{scale_tag}/grid/{config_tag}")
    else:
        base_dir = Path(f"results/classification/{strategy.get_name()}/{feature_tag}/{scale_tag}")
    confusion_dir = base_dir / "plots"
    result_dir = base_dir / "joblib"
    log_dir = base_dir / "logs"
    confusion_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Check if this configuration has already been run
    existing_results = list(result_dir.glob("*.joblib"))
    if existing_results:
        print(f"[yellow]WARNING[/yellow]: Results for this configuration already exist at {result_dir.resolve()}")
        print(f"[yellow]SKIPPING[/yellow]: To rerun, delete: {base_dir.resolve()}")
        return

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

    # define per-subject worker
    def _process_subject(file_path: Path):
        # create fresh strategy instance per worker to avoid shared-state/pickle issues
        model_name = strategy.get_name()
        strategy_worker = get_model_strategy(model_name, scale=strategy_scale, feature_type=strategy_feature_type, config=strategy_config)

        match = re.search(r"preprocessed_sub-(\d{2}).joblib", file_path.name)
        sub_id = match.group(1) if match else file_path.stem

        data = joblib.load(file_path)
        X_train = data["x"]
        y_train = data["y"]

        val_candidates = list(val_dir.glob(f"*{file_path.name}"))
        if len(val_candidates) == 0:
            raise ValueError(f"Did not find any validation file for {file_path.name}")
        val_file = val_candidates[0]
        val_data = joblib.load(val_file)
        x_val = val_data["x"]
        y_val = val_data["y"]

        # set raw data if needed
        set_raw_data = getattr(strategy_worker, "set_raw_data", None)
        if callable(set_raw_data):
            set_raw_data(X_train, y_train, x_val)

        # feature transform
        X_train_feats = strategy_worker.transform_train(X_train)
        set_data_info = getattr(strategy_worker, "set_data_info", None)
        if callable(set_data_info) and getattr(strategy_worker, "input_shape", None) is None:
            set_data_info(X_train_feats, class_labels)

        encode_targets = getattr(strategy_worker, "encode_targets", None)
        if callable(encode_targets):
            y_train_fit = encode_targets(y_train)
        else:
            y_train_fit = y_train

        model = strategy_worker.create_model()
        t0 = time.perf_counter()
        model.fit(X_train_feats, y_train_fit)
        train_time = time.perf_counter() - t0

        x_val_feats = strategy_worker.transform_val(x_val)

        t1 = time.perf_counter()
        raw_pred = model.predict(x_val_feats)
        predict_time = time.perf_counter() - t1
        decode_targets = getattr(strategy_worker, "decode_targets", None)
        if callable(decode_targets):
            y_pred = decode_targets(raw_pred)
        else:
            y_pred = raw_pred

        score = accuracy_score(y_val, y_pred)
        class_counts = np.bincount(y_val.astype(int))
        majority_baseline = class_counts.max() / len(y_val)
        delta_over_chance = score - chance_baseline
        delta_over_majority = score - majority_baseline

        confusion_matrix_val = confusion_matrix(y_val, y_pred, labels=class_labels)

        # save plot for this subject
        disp = sklearn.metrics.ConfusionMatrixDisplay(
            confusion_matrix=confusion_matrix_val,
            display_labels=class_labels,
        )
        ax = disp.plot()
        disp.ax_.set_title(f"Confusion Matrix — Trained: {file_path.name}")
        fig = disp.ax_.figure
        fig_path = confusion_dir / f"{run_tag}_{file_path.stem}_confusion_matrix.png"
        fig.savefig(fig_path)

        # save joblib
        result_path = result_dir / f"{run_tag}_{file_path.stem}.joblib"
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
                "model_name": strategy_worker.get_name(),
                "feature_type": strategy_feature_type,
                "scale": strategy_scale,
                "config": getattr(strategy_worker, "config", None),
            },
            "subject_id": sub_id,
            "train_time": float(train_time),
            "predict_time": float(predict_time),
        }, result_path)

        return {
            "subject": file_path.stem,
            "score": float(score),
            "chance_baseline": float(chance_baseline),
            "majority_baseline": float(majority_baseline),
            "delta_over_chance": float(delta_over_chance),
            "delta_over_majority": float(delta_over_majority),
            "train_time": float(train_time),
            "predict_time": float(predict_time),
            "confusion_matrix": confusion_matrix_val,
            "result_path": str(result_path.resolve()),
        }

    # run per-subject processing in parallel
    from joblib import Parallel, delayed

    if n_jobs == 1:
        results = [_process_subject(f) for f in tqdm(train_files, desc="Training models on subjects")]
    else:
        results = Parallel(n_jobs=n_jobs)(delayed(_process_subject)(f) for f in train_files)

    # collect results and aggregate
    for res in results:
        score_logger.append({
            "subject": res["subject"],
            "score": res["score"],
            "chance_baseline": res["chance_baseline"],
            "majority_baseline": res["majority_baseline"],
            "delta_over_chance": res["delta_over_chance"],
            "delta_over_majority": res["delta_over_majority"],
            "train_time": res["train_time"],
            "predict_time": res["predict_time"],
        })
        aggregate_confusion += res["confusion_matrix"]
        aggregate_samples += int(res["confusion_matrix"].sum())
        aggregate_correct += np.trace(res["confusion_matrix"])
    
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

    summary_dir = base_dir / "logs"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / f"{run_tag}_summary.txt"
    with open(summary_path, "w") as f:
        f.write(summary_text.lstrip("\n") + "\n")
    print(f"Saved summary to {summary_path.resolve()}")

    # save scores to a log file
    score_log_dir = base_dir / "logs"
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