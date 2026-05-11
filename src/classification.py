import click
from matplotlib.pylab import plot
import models
import numpy as np
import sklearn
import joblib
import typer
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
}

FEATURE_TYPES = [
    "downsample", 
    "bandpower", 
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
    if feature_type not in FEATURE_TYPES:
        raise ValueError(
            f"Unknown feature type: {feature_type}. Available: {FEATURE_TYPES}"
        )
    return MODEL_REGISTRY[model_name](scale=scale, feature_type=feature_type)


# ============================================================================
# Main training function
# ============================================================================

def fit_model(strategy: ModelStrategy, train_dir: Path, val_dir: Path):
    """Train and validate model using the given strategy"""
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

    for file in tqdm(train_files, desc="Training models on subjects"):
        print(f"Classifying on file: {file.name}")

        # Load training data
        data = joblib.load(file)
        X_train = data["x"]
        y_train = data["y"]

        # Transform training data
        print(f"Train shape before feature extraction: \n X_train: {X_train.shape} \n y_train: {y_train.shape}")
        X_train = strategy.transform_train(X_train)
        print(f"Train shape after feature extraction: \n X_train: {X_train.shape} \n y_train: {y_train.shape}")
        

        # Train model
        model = strategy.create_model()
        print(f"Training model: {strategy.get_name()}")
        model.fit(X_train, y_train)

        # Load and transform validation data
        val_file = list(val_dir.glob(f"*{file.name}"))[0]
        print(f"Validating on file: {val_file.resolve()}")
        if not val_file:
            raise ValueError("Did not find any validation file")

        val_data = joblib.load(val_file)
        x_val = val_data["x"]
        y_val = val_data["y"]

        # Transform validation data using strategy
        x_val = strategy.transform_val(x_val)
        print(f"Validation shape after feature extraction: \n x_val: {x_val.shape} \n y_val: {y_val.shape}")

        # Evaluate
        y_pred = model.predict(x_val)
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

    fit_model(strategy, train_dir, val_dir)


if __name__ == "__main__":
    typer.run(main)