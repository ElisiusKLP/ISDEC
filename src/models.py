from abc import ABC, abstractmethod
from typing import Any, Optional
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from features import (
    transform_to_band_power,
    downsample_time,
    transform_to_band_phase,
    transform_to_band_power_with_phase,
    transform_to_time_frequency,
    pca_feature_selection,
    mutual_info_feature_selection,
    transform_to_dwt_hierarchical,
    select_channels_by_mutual_info,
)

# ============================================================================
# Model Strategies - Each model defines its own transformation and fitting
# ============================================================================

class ModelStrategy(ABC):
    """Base class for model-specific behavior"""

    model_type: str = "classic" # "classic" or "deep"
        
    @abstractmethod
    def transform_train(self, x: np.ndarray) -> np.ndarray:
        """Transform training data"""
        pass

    @abstractmethod
    def transform_val(self, x: np.ndarray) -> np.ndarray:
        """Transform validation data"""
        pass

    @abstractmethod
    def create_model(self) -> Any:
        """Create and return the model"""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return model name"""
        pass

    def get_data_dirs(self) -> tuple[Path, Path]:
        if self.model_type == "classic":
            input_data = "preprocessed"
            joblib_dir = Path(f"data/derivatives/{input_data}/joblib")
        elif self.model_type == "deep":
            input_data = "raw"
            joblib_dir = Path(f"data/derivatives/{input_data}/joblib")
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        return joblib_dir / "training_set", joblib_dir / "validation_set"

    def set_raw_data(self, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray) -> None:
        """Store train/val data for feature extractors that require labels."""
        self._x_train_raw = x_train
        self._y_train = y_train
        self._x_val_raw = x_val
        self._cached_train_features = None
        self._cached_val_features = None

    def _transform_with_feature_type(self, x: np.ndarray, *, is_train: bool) -> np.ndarray:
        """Shared transform path for classic models."""
        feature_type = getattr(self, "feature_type", "stack")

        if feature_type != "dwt_channel_select":
            return create_features(x, feature_type=feature_type)

        cached_train = getattr(self, "_cached_train_features", None)
        cached_val = getattr(self, "_cached_val_features", None)

        if is_train and cached_train is not None:
            return cached_train
        if not is_train and cached_val is not None:
            return cached_val

        x_train_raw = getattr(self, "_x_train_raw", None)
        y_train = getattr(self, "_y_train", None)
        x_val_raw = getattr(self, "_x_val_raw", None)

        assert x_train_raw is not None
        assert y_train is not None
        assert x_val_raw is not None

        x_train_features, x_val_features = create_features(
            x_train_raw,
            feature_type=feature_type,
            y_train=y_train,
            x_reference=x_val_raw,
        )
        self._cached_train_features = x_train_features
        self._cached_val_features = x_val_features
        return x_train_features if is_train else x_val_features


class RandomForestStrategy(ModelStrategy):
    def __init__(
        self,
        n_estimators: int = 300,
        max_features: Optional[str] = None,
        random_state: int = 2001,
        scale: bool = True,
        feature_type: str = "stack"
    ):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.random_state = random_state
        self.scale = scale
        self.feature_type = feature_type
        self._x_train_raw = None
        self._y_train = None
        self._x_val_raw = None
        self._cached_train_features = None
        self._cached_val_features = None

    def transform_train(self, x: np.ndarray) -> np.ndarray:
        """Reshape to (n_samples, n_features)"""
        return self._transform_with_feature_type(x, is_train=True)

    def transform_val(self, x: np.ndarray) -> np.ndarray:
        """Reshape to (n_samples, n_features)"""
        return self._transform_with_feature_type(x, is_train=False)

    def create_model(self) -> Pipeline:
        """Create a pipeline with optional scaling and Random Forest classifier"""
        steps = []
        if self.scale:
            steps.append(("scaler", StandardScaler()))
        steps.append(
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=self.n_estimators,
                    max_features=self.max_features,
                    random_state=self.random_state,
                ),
            )
        )
        return Pipeline(steps=steps)

    def get_name(self) -> str:
        return "random_forest"


class LogisticRegressionStrategy(ModelStrategy):
    def __init__(
        self, 
        solver: str = "saga",
        max_iter: int = 5000, 
        scale: bool = True,
        feature_type: str = "stack",
        l1_ratio: Optional[float] = 1
        ):
        self.solver = solver
        self.max_iter = max_iter
        self.scale = scale
        self.feature_type = feature_type
        self.l1_ratio = l1_ratio
        self._x_train_raw = None
        self._y_train = None
        self._x_val_raw = None
        self._cached_train_features = None
        self._cached_val_features = None

    def transform_train(self, x: np.ndarray) -> np.ndarray:
        """Flatten to (n_samples, n_features)"""
        return self._transform_with_feature_type(x, is_train=True)

    def transform_val(self, x: np.ndarray) -> np.ndarray:
        """Same transformation as training"""
        return self._transform_with_feature_type(x, is_train=False)

    def create_model(self) -> Pipeline:
        """Create a pipeline with optional scaling and Logistic Regression classifier"""
        steps = []
        if self.scale:
            steps.append(("scaler", StandardScaler()))
        steps.append(
            (
                "classifier",
                LogisticRegression(
                    solver=self.solver, 
                    max_iter=self.max_iter,
                    l1_ratio=self.l1_ratio
                ),
            )
        )
        return Pipeline(steps=steps)

    def get_name(self) -> str:
        return "logistic_regression"

class SVMStrategy(ModelStrategy):
    def __init__(self, kernel: str = "rbf", C: float = 1.0, scale: bool = True, feature_type: str = "downsample"):
        self.kernel = kernel
        self.C = C
        self.scale = scale
        self.feature_type = feature_type
        self._x_train_raw = None
        self._y_train = None
        self._x_val_raw = None
        self._cached_train_features = None
        self._cached_val_features = None

    def transform_train(self, x: np.ndarray) -> np.ndarray:
        """Flatten to (n_samples, n_features)"""
        return self._transform_with_feature_type(x, is_train=True)

    def transform_val(self, x: np.ndarray) -> np.ndarray:
        """Same transformation as training"""
        return self._transform_with_feature_type(x, is_train=False)

    def create_model(self) -> Pipeline:
        """Create a pipeline with optional scaling and SVM classifier"""
        from sklearn.svm import SVC

        steps = []
        if self.scale:
            steps.append(("scaler", StandardScaler()))
        steps.append(
            (
                "classifier",
                SVC(kernel=self.kernel, C=self.C),
            )
        )
        return Pipeline(steps=steps)

    def get_name(self) -> str:
        return "svm"


class EEGNetStrategy(ModelStrategy):
    model_type: str = "deep"

    def __init__(
        self,
        epochs: int = 5,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        seed: int = 734,
        scale: bool = False,
        feature_type: str = "raw",
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.seed = seed
        self.scale = scale
        self.feature_type = feature_type
        self.input_shape = None
        self.classes_ = None

    def set_data_info(self, x: np.ndarray, classes: list[int]):
        self.input_shape = x.shape[1:]
        self.classes_ = np.asarray(classes)

    def encode_targets(self, y: np.ndarray) -> np.ndarray:
        if self.classes_ is None:
            raise ValueError("Classes have not been set")
        return np.searchsorted(self.classes_, y)

    def decode_targets(self, y: np.ndarray) -> np.ndarray:
        if self.classes_ is None:
            raise ValueError("Classes have not been set")
        return self.classes_[np.asarray(y, dtype=int)]

    def transform_train(self, x: np.ndarray) -> np.ndarray:
        # downsample to 128 hz from 256
        if self.feature_type == "raw":
            x = downsample_time(x, original_sfreq=256, target_sfreq=128)
            return x
        else:
            return create_features(x, feature_type=self.feature_type)

    def transform_val(self, x: np.ndarray) -> np.ndarray:
        # downsample to 128 hz from 256
        if self.feature_type == "raw":
            x = downsample_time(x, original_sfreq=256, target_sfreq=128)
            return x
        else:
            return create_features(x, feature_type=self.feature_type)

    def create_model(self) -> Any:
        from nn_models import EEGNet
        from nn_train import FlaxSKLearnLikeModel

        if self.input_shape is None or self.classes_ is None:
            raise ValueError("Input shape and classes must be set before creating the model")

        channels, samples = self.input_shape[:2]
        model = EEGNet(
            n_classes=len(self.classes_),
            channels=channels,
            samples=samples,
        )
        return FlaxSKLearnLikeModel(
            model,
            input_shape=self.input_shape,
            epochs=self.epochs,
            batch_size=self.batch_size,
            lr=self.learning_rate,
        )

    def get_name(self) -> str:
        return "eegnet"

# =======
# == Create Features function
# =======

def create_features(
    x: np.ndarray,
    feature_type: str,
    y_train: np.ndarray | None = None,
    x_reference: np.ndarray | None = None,
):
    """Create features from raw EEG data based on the specified feature type."""
    bands = {
            "delta": (1.0, 4.0),
            "theta": (4.0, 8.0),
            "alpha": (8.0, 13.0),
            "beta": (13.0, 30.0),
            "gamma": (30.0, 100.0),
        }

    if feature_type == "tfr_morlet":
        return transform_to_time_frequency(x, sfreq=256, algorithm="morlet")
    elif feature_type == "tfr_morlet_bands":
        tfr = transform_to_time_frequency(
            x, sfreq=256, algorithm="morlet",
            in_bands=True, bands=bands,
            downsample_to_freq=1,  # example downsampling to reduce dimensionality
            n_freqs=20
        )
        return tfr
    elif feature_type == "tfr_dwt_cmor":
        return transform_to_time_frequency(x, sfreq=256, algorithm="dwt")
    elif feature_type == "dwt_hierarchical":
        return transform_to_dwt_hierarchical(x, sfreq=256)
    elif feature_type == "dwt_channel_select":
        if y_train is None:
            raise ValueError("y_train must be provided for dwt_channel_select")
        if x_reference is None:
            raise ValueError("x_reference must be provided for dwt_channel_select")
        return select_channels_by_mutual_info(x, y_train, x_reference, k_channels=4)
    elif feature_type == "tfr_pca":
        tfr = transform_to_time_frequency(x, sfreq=256, algorithm="morlet")
        tfr_pca = pca_feature_selection(tfr, n_components=20)
        return tfr_pca
    elif feature_type == "bandpower_nostack":
        return transform_to_band_power(x, sfreq=256, bands=bands,mean=False, stack_channels=False)
    elif feature_type == "bandpower_mean":
        return transform_to_band_power(
            x, sfreq=256, bands=bands,
            mean=True, stack_channels=True
        )
    elif feature_type == "bandphase":
        return transform_to_band_phase(x, sfreq=256)
    elif feature_type == "bandpower_phase":
        return transform_to_band_power_with_phase(
            x,
            sfreq=256,
        )
    elif feature_type == "downsample":
        downsample = downsample_time(x, original_sfreq=256, target_sfreq=50)  # example sfreq, adjust as needed
        return downsample.reshape(downsample.shape[0], -1)
    elif feature_type == "stack":
        return x.reshape(x.shape[0], -1)
    else:
        raise ValueError(f"Unknown feature type: {feature_type}")