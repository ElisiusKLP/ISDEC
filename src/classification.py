from jedi.inference.base_value import Value
import numpy as np
import sklearn
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

#https://scikit-learn.org/stable/modules/ensemble.html#random-forest-parameters

joblib_dir = Path("data/derivatives/raw/joblib")

train_dir = joblib_dir / "training_set"

val_dir = joblib_dir / "validation_set"

def train_model (x, y):
    model = RandomForestClassifier(n_estimators=50, max_features=None, random_state=2001)
    model.fit(x, y)
    return model


def fit_model(train_dir, val_dir):
    train_files = list(train_dir.glob('*.joblib'))
    if len(train_files)==0:
        raise ValueError("No training files collected from glob(*.joblib)")

    for file in train_files:
        print(f"Classifying on file: {file.name}")
        data = joblib.load(file)
        x_train = data['x'].reshape(data['x'].shape[0], -1) # (epochs X channels*time)
        print(f"x_train shape: {x_train.shape}")
        y_train = data['y']

        model_fit = train_model(x_train, y_train)

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
        fig.savefig(plt_filepath / f"confusion_matrix_{file.stem}.png")
        

if __name__ == "__main__":
    fit_model(train_dir, val_dir)