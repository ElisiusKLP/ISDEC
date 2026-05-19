config = {
    "random_forest": {
        "n_estimators": [50, 100, 200, 300, 500, 1000],
        "criterion": ["gini", "entropy"],
        "max_depth": [None, 5, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", 0.5, None],
    },
    "svc": {
        "C": [0.5, 1, 5, 10, 30],
        "kernel": ["linear", "rbf", "poly"],
        "degree": [2, 3, 4],
        "coef0": [0.0, 0.1, 0.5],
        "gamma": ["scale", "auto"]
    },
    "linear_svc": {
        "C": [0.5, 1, 10, 50, 100],
        "penalty": ["l1", "l2"],
        "loss": ["hinge", "squared_hinge"],
        "max_iter": [10000]
    },
    "logistic_regression": {
        "C": [0.5, 1, 10, 50, 100],
        "penalty": ["l1", "l2"],
        "solver": ["saga"],
        "max_iter": [5000],
        "tol": [1e-4, 1e-3, 1e-2]
    },
    "bagging_rf": {
        "n_estimators": [175, 200, 225, 250, 300, 500],
        "max_samples": [0.1, 0.25, 0.5, 0.75, 1.0],
        "max_features": [0.5,1.0]
    },
    "bagging_svc": {
        "n_estimators": [10, 20, 50, 100],
        "max_samples": [0.5, 0.75, 1.0],
        "max_features": [0.5, 0.75, 1.0]
    }
}