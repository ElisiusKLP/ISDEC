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
}


def get_model_strategy(model_name: str, scale: bool = True) -> ModelStrategy:
    """Get strategy for the specified model"""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_name](scale=scale)


# ============================================================================
# Main training function
# ============================================================================

def fit_model(strategy: ModelStrategy, train_dir: Path, val_dir: Path):
    """Train and validate model using the given strategy"""
    train_files = list(train_dir.glob("*.joblib"))
    if len(train_files) == 0:
        raise ValueError("No training files collected from glob(*.joblib)")
    
    # setup a score logger
    score_logger = []

    for file in tqdm(train_files, desc="Training models on subjects"):
        print(f"Classifying on file: {file.name}")

        # Load training data
        data = joblib.load(file)
        X_train = data["x"]
        y_train = data["y"]

        # Transform training data
        X_train = strategy.transform_train(X_train)
        print(f"x_train shape: {X_train.shape}")
        print(f"y_train shape: {y_train.shape}")

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

        # Evaluate
        y_pred = model.predict(x_val)
        score = accuracy_score(y_val, y_pred)

        # log score for subject
        score_logger.append({
            "subject": file.stem,
            "score": score,
        })

        # Confusion matrix
        confusion_matrix_val = confusion_matrix(y_val, y_pred)
        print(f"Trained on {file.name}, validated on {val_file.name}: {score:.3f}")
        print(f"Confusion Matrix:\n{confusion_matrix_val}")

        # Save plot
        disp = sklearn.metrics.ConfusionMatrixDisplay(
            confusion_matrix=confusion_matrix_val
        )
        ax = disp.plot()
        disp.ax_.set_title(f"Confusion Matrix — Trained: {file.name}")

        plt_filepath = Path(f"results/plots/accuracy/{strategy.get_name()}")
        plt_filepath.mkdir(parents=True, exist_ok=True)
        fig = disp.ax_.figure
        fig.savefig(plt_filepath / f"{strategy.get_name()}_{file.stem}_confusion_matrix.png")
    
    # save all images into a single grid in a pdf
    tags = ["preprocessed", "raw"]
    set_tag = [tag for tag in tags if tag in train_files[0].stem]
    if isinstance(set_tag, list) and len(set_tag) == 1:
        set_tag = set_tag[0]
    elif isinstance(set_tag, str):
        pass
    else:
        raise ValueError(f"Expected exactly one set tag in {tags} to be in filename, but got: {set_tag}")
    plot_confusion_matrices_grid(img_dir=plt_filepath, set_tag=set_tag)

    # save scores to a log file
    score_log_dir = Path("results/logs/scores")
    score_log_dir.mkdir(parents=True, exist_ok=True)
    score_log_path = score_log_dir / f"{strategy.get_name()}_scores.csv"
    with open(score_log_path, "w") as f:
        f.write("subject,score\n")
        for entry in score_logger:
            f.write(f"{entry['subject']},{entry['score']:.4f}\n")
    print(f"Saved scores to {score_log_path.resolve()}")

def main(
    model: str = typer.Option("random_forest", help="Type of model to train"),
    scale: bool = typer.Option(True, "--scale/--no-scale", help="Enable feature scaling."),
):
    print(f"Training model: {model}")
    print(f"Scaling enabled: {scale}")

    strategy = get_model_strategy(model, scale=scale)

    fit_model(strategy, train_dir, val_dir)


if __name__ == "__main__":
    typer.run(main)