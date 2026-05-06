from jedi.inference.base_value import Value
import numpy as np
import sklearn
import joblib
import typer
from rich import print
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm

#https://scikit-learn.org/stable/modules/ensemble.html#random-forest-parameters

joblib_dir = Path("data/derivatives/preprocessed/joblib")

train_dir = joblib_dir / "training_set"

val_dir = joblib_dir / "validation_set"

def train_model(config, x, y):
    # find model type from config
    if config.get("model") == "random_forest":
        x = x.reshape(x.shape[0], -1) # (epochs X channels*time)

        model = RandomForestClassifier(n_estimators=50, max_features=None, random_state=2001)
    elif config.get("model") == "logistic_regression":
        model = LogisticRegression(solver='lbfgs', max_iter=1000)

    print(f"Training model: {config.get('model')}")
    print(f"with X shape: {x.shape} and y shape: {y.shape}")
    model.fit(x, y)
    return model


def fit_model(
    config, 
    train_dir, 
    val_dir
    ):
    train_files = list(train_dir.glob('*.joblib'))
    if len(train_files)==0:
        raise ValueError("No training files collected from glob(*.joblib)")

    for file in tqdm(train_files, desc="Training models on subjects"):
        print(f"Classifying on file: {file.name}")
        data = joblib.load(file)
        X_train = data['x']
        y_train = data['y']
        print(f"x_train shape: {X_train.shape}")
        print(f"y_train shape: {y_train.shape}")

        model_fit = train_model(config, X_train, y_train)

        val_file = list(val_dir.glob(f"*{file.name}"))[0]
        print(f"Validating on file: {val_file.resolve()}")
        if not val_file:
            raise ValueError("Did not find any validation file")
        
        val_data = joblib.load(val_file)
        x_val = val_data['x'].reshape(val_data['x'].shape[0], -1)
        y_val = val_data['y']

        y_pred = model_fit.predict(x_val)
        score = accuracy_score(y_val, y_pred)

        # Create confusion matrix and save plot
        confusion_matrix_val = confusion_matrix(y_val, y_pred)
        print(f"Trained on {file.name}, validated on {val_file.name}: {score:.3f}")
        print(f"Confusion Matrix:\n{confusion_matrix_val}")
        disp = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_val)
        ax = disp.plot()
        disp.ax_.set_title(f"Confusion Matrix — Trained: {file.name}")

        # save plot
        plt_filepath = Path("results/plots/accuracy")
        plt_filepath.mkdir(parents=True, exist_ok=True)
        fig = disp.ax_.figure
        fig.savefig(plt_filepath / f"{config.get('model')}_{file.stem}_confusion_matrix_.png")

def main(
    model: str = typer.Option("random_forest", help="Type of model to train (e.g., 'random_forest')")
):
    print(f"Training model: {model}")

    config = {
        "model": model
        }

    fit_model(config, train_dir, val_dir)


if __name__ == "__main__":
    typer.run(main)