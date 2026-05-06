from abc import ABC, abstractmethod
from typing import Any, Optional
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from features import transform_to_band_power, downsample_time

# ============================================================================
# Model Strategies - Each model defines its own transformation and fitting
# ============================================================================

class ModelStrategy(ABC):
    """Base class for model-specific behavior"""

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


class RandomForestStrategy(ModelStrategy):
    def __init__(
        self,
        n_estimators: int = 50,
        max_features: Optional[str] = None,
        random_state: int = 2001,
        scale: bool = True,
        feature_type: str = "downsample"
    ):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.random_state = random_state
        self.scale = scale
        self.feature_type = feature_type

    def transform_train(self, x: np.ndarray) -> np.ndarray:
        """Reshape to (n_samples, n_features)"""
        return create_features(x, feature_type=self.feature_type)

    def transform_val(self, x: np.ndarray) -> np.ndarray:
        """Reshape to (n_samples, n_features)"""
        return create_features(x, feature_type=self.feature_type)

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
        max_iter: int = 1000, 
        scale: bool = True,
        feature_type: str = "downsample",
        l1_ratio: Optional[float] = 1
        ):
        self.solver = solver
        self.max_iter = max_iter
        self.scale = scale
        self.feature_type = feature_type
        self.l1_ratio = l1_ratio
    def transform_train(self, x: np.ndarray) -> np.ndarray:
        """Flatten to (n_samples, n_features)"""
        return create_features(x, feature_type=self.feature_type)

    def transform_val(self, x: np.ndarray) -> np.ndarray:
        """Same transformation as training"""
        return create_features(x, feature_type=self.feature_type)

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

    def transform_train(self, x: np.ndarray) -> np.ndarray:
        """Flatten to (n_samples, n_features)"""
        return create_features(x, feature_type=self.feature_type)

    def transform_val(self, x: np.ndarray) -> np.ndarray:
        """Same transformation as training"""
        return create_features(x, feature_type=self.feature_type)

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

# =======
# Create Features

def create_features(x: np.ndarray, feature_type: str):
    """Create features from raw EEG data based on the specified feature type."""
    if feature_type == "band_power":
        return transform_to_band_power(x, sfreq=256)  # example sfreq, adjust as needed
    elif feature_type == "downsample":
        downsample = downsample_time(x, original_sfreq=256, target_sfreq=50)  # example sfreq, adjust as needed
        return downsample.reshape(downsample.shape[0], -1)
    elif feature_type == "stack":
        return x.reshape(x.shape[0], -1)