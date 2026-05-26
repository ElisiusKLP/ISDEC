from abc import ABC, abstractmethod
from typing import Any, Optional
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from create_features import create_features
from features import downsample_time

# ============================================================================
# = Model Strategies - Each model defines its own transformation and fitting =
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

        shared_mi_feature_types = {
            "dwt_channel_select",
            "dwt_stats_mi",
            "bandpower_mean_mi",
            "mean_mi",
            "stft_bands_stats_mi",
            "tfr_morlet_bands_stats_mi",
        }

        if feature_type not in shared_mi_feature_types:
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
        config: Optional[dict] = None,
        scale: bool = True,
        feature_type: str = "stack",
    ):
        self.config = config or {}
        self.n_estimators = self.config.get("n_estimators", 300)
        self.max_features = self.config.get("max_features", None)
        self.random_state = self.config.get("random_state", 2001)
        self.criterion = self.config.get("criterion", "gini")
        self.max_depth = self.config.get("max_depth", None)
        self.min_samples_split = self.config.get("min_samples_split", 2)
        self.min_samples_leaf = self.config.get("min_samples_leaf", 1)
        self.max_features = self.config.get("max_features", None)
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
                    criterion=self.criterion,
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    min_samples_leaf=self.min_samples_leaf,
                ),
            )
        )
        return Pipeline(steps=steps)

    def get_name(self) -> str:
        return "random_forest"


class LogisticRegressionStrategy(ModelStrategy):
    def __init__(
        self, 
        config: Optional[dict] = None,
        scale: bool = True,
        feature_type: str = "stack",
        ):
        self.config = config or {}
        self.solver = self.config.get("solver", "saga")
        self.max_iter = self.config.get("max_iter", 10000)
        self.l1_ratio = self.config.get("l1_ratio", 1)
        self.tol = self.config.get("tol", 1e-4)
        self.penalty = self.config.get("penalty", "l1")
        self.C = self.config.get("C", 1.0)
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
                    l1_ratio=self.l1_ratio,
                    tol=self.tol,
                    penalty=self.penalty,
                    C=self.C,
                    random_state=2001
                ),
            )
        )
        return Pipeline(steps=steps)

    def get_name(self) -> str:
        return "logistic_regression"

class SVCStrategy(ModelStrategy):
    def __init__(
        self,
        config: Optional[dict] = None,
        scale: bool = True,
        feature_type: str = "downsample",
    ):
        self.config = config or {}
        self.kernel = self.config.get("kernel", "rbf")
        self.C = self.config.get("C", 1.0)
        self.scale = scale
        self.feature_type = feature_type
        self.random_state = self.config.get("random_state", 2072)
        self.degree = self.config.get("degree", 3)
        self.coef0 = self.config.get("coef0", 0.0)
        self.gamma = self.config.get("gamma", "scale")
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
                SVC(
                    kernel=self.kernel,
                    C=self.C, 
                    random_state=self.random_state,
                    degree=self.degree,
                    coef0=self.coef0,
                    gamma=self.gamma
                    ),
            )
        )
        return Pipeline(steps=steps)

    def get_name(self) -> str:
        return "svc"

class LinearSVCStrategy(ModelStrategy):
    def __init__(
        self,
        config: Optional[dict] = None,
        scale: bool = True,
        feature_type: str = "downsample",
    ):
        self.config = config or {}
        self.C = self.config.get("C", 1.0)
        self.penalty = self.config.get("penalty", "l1")
        self.loss = self.config.get("loss", "squared_hinge")
        self.max_iter = self.config.get("max_iter", 5000)
        self.kernel = "linear"
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
        """Create a pipeline with optional scaling and Linear SVM classifier"""
        from sklearn.svm import LinearSVC

        steps = []
        if self.scale:
            steps.append(("scaler", StandardScaler()))
        steps.append(
            (
                "classifier",
                LinearSVC(C=self.C, max_iter=self.max_iter, random_state=2001, penalty=self.penalty, loss=self.loss),
            )
        )
        return Pipeline(steps=steps)

    def get_name(self) -> str:
        return "linear_svc"

class BaggingRFStrategy(ModelStrategy):
    def __init__(
        self,
        config: Optional[dict] = None,
        scale: bool = True,
        feature_type: str = "stack",
    ):
        self.config = config or {}
        self.n_estimators = self.config.get("n_estimators", 100)
        self.max_samples = self.config.get("max_samples", 0.8)
        self.random_state = self.config.get("random_state", 2001)
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
        """Create a pipeline with optional scaling and Bagging Random Forest classifier"""
        from sklearn.ensemble import BaggingClassifier

        steps = []
        if self.scale:
            steps.append(("scaler", StandardScaler()))
        steps.append(
            (
                "classifier",
                BaggingClassifier(
                    estimator=RandomForestClassifier(n_estimators=self.n_estimators, random_state=self.random_state),
                    n_estimators=self.n_estimators,
                    max_samples=self.max_samples,
                    random_state=self.random_state,
                ),
            )
        )
        return Pipeline(steps=steps)

    def get_name(self) -> str:
        return "bagging_rf"

class BaggingSVCStrategy(ModelStrategy):
    def __init__(
        self,
        config: Optional[dict] = None,
        scale: bool = True,
        feature_type: str = "stack",
    ):
        self.config = config or {}
        self.n_estimators = self.config.get("n_estimators", 100)
        self.max_samples = self.config.get("max_samples", 0.8)
        self.random_state = self.config.get("random_state", 2001)
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
        """Create a pipeline with optional scaling and Bagging SVC classifier"""
        from sklearn.ensemble import BaggingClassifier
        from sklearn.svm import SVC

        steps = []
        if self.scale:
            steps.append(("scaler", StandardScaler()))
        steps.append(
            (
                "classifier",
                BaggingClassifier(
                    estimator=SVC(kernel="rbf", C=1.0, random_state=self.random_state),
                    n_estimators=self.n_estimators,
                    max_samples=self.max_samples,
                    random_state=self.random_state,
                ),
            )
        )
        return Pipeline(steps=steps)

    def get_name(self) -> str:
        return "bagging_svc"

class EEGNetStrategy(ModelStrategy):
    model_type: str = "deep"

    def __init__(
        self,
        config: Optional[dict] = None,
        scale: bool = False,
        feature_type: str = "raw",
    ):
        self.config = config or {}
        self.epochs = self.config.get("epochs", 5)
        self.batch_size = self.config.get("batch_size", 32)
        self.learning_rate = self.config.get("learning_rate", 1e-3)
        self.seed = self.config.get("seed", 734)
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


class CNNStrategy(ModelStrategy):
    model_type: str = "deep"

    def __init__(
        self,
        config: Optional[dict] = None,
        scale: bool = False,
        feature_type: str = "tfr_morlet",
        collapse_channels: bool = True,
    ):
        self.config = config or {}
        self.epochs = self.config.get("epochs", 100)
        self.batch_size = self.config.get("batch_size", 16)
        self.learning_rate = self.config.get("learning_rate", 1e-3)
        self.seed = self.config.get("seed", 734)
        self.scale = scale
        self.feature_type = feature_type
        self.collapse_channels = collapse_channels
        self.input_shape = None
        self.classes_ = None

    def set_data_info(self, x: np.ndarray, classes: list[int]):
        # Accept either raw data (epochs, channels, timepoints) or already-extracted
        # CNN images (epochs, H, W, C). If `x` is raw, compute one example image
        # to determine the required `input_shape`. If `x` is already images,
        # infer `input_shape` directly.
        if x.ndim == 4:
            # already-shaped images: (epochs, H, W, C)
            self.input_shape = x.shape[1:]
        elif x.ndim == 3:
            imgs = create_features(x, feature_type=self.feature_type)
            self.input_shape = imgs.shape[1:]
        else:
            raise ValueError(f"Unexpected input shape for set_data_info: {x.shape}")
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
        # produce images for CNN
        imgs = create_features(x, feature_type=self.feature_type)
        return imgs

    def transform_val(self, x: np.ndarray) -> np.ndarray:
        imgs = create_features(x, feature_type=self.feature_type)
        return imgs

    def create_model(self) -> Any:
        # Import here to avoid top-level dependencies
        from nn_models import CNN as FlaxCNN
        from nn_train import FlaxSKLearnLikeModel

        if self.input_shape is None or self.classes_ is None:
            raise ValueError("Input shape and classes must be set before creating the model")

        model = FlaxCNN(n_classes=len(self.classes_), resolution=self.input_shape[0])
        return FlaxSKLearnLikeModel(model, input_shape=self.input_shape, epochs=self.epochs, batch_size=self.batch_size, lr=self.learning_rate)

    def get_name(self) -> str:
        return "cnn"
