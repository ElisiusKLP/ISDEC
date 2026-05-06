from abc import ABC, abstractmethod
from typing import Any, Optional
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
    ):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.random_state = random_state
        self.scale = scale

    def transform_train(self, x: np.ndarray) -> np.ndarray:
        """Reshape to (n_samples, n_features)"""
        return x.reshape(x.shape[0], -1)

    def transform_val(self, x: np.ndarray) -> np.ndarray:
        """Reshape to (n_samples, n_features)"""
        return x.reshape(x.shape[0], -1)

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
    def __init__(self, solver: str = "lbfgs", max_iter: int = 1000, scale: bool = True):
        self.solver = solver
        self.max_iter = max_iter
        self.scale = scale

    def transform_train(self, x: np.ndarray) -> np.ndarray:
        """Flatten to (n_samples, n_features)"""
        return x.reshape(x.shape[0], -1)

    def transform_val(self, x: np.ndarray) -> np.ndarray:
        """Same transformation as training"""
        return x.reshape(x.shape[0], -1)

    def create_model(self) -> Pipeline:
        """Create a pipeline with optional scaling and Logistic Regression classifier"""
        steps = []
        if self.scale:
            steps.append(("scaler", StandardScaler()))
        steps.append(
            (
                "classifier",
                LogisticRegression(solver=self.solver, max_iter=self.max_iter),
            )
        )
        return Pipeline(steps=steps)

    def get_name(self) -> str:
        return "logistic_regression"