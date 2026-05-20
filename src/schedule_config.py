config = {
    "random_forest": {
        "n_estimators": [50, 300, 500, 1000],
        "criterion": ["gini"],
        "max_depth": [None, 5, 25],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "max_features": ["sqrt", None],
    },
    "svc": {
        "C": [0.5, 1, 3, 20],
        "kernel": ["linear", "rbf", "poly"],
        "degree": [2, 3, 4],
        "coef0": [0.0, 0.1, 0.5],
        "gamma": ["scale", "auto"]
    },
    "logistic_regression": {
        "C": [0.5, 1, 10, 50],
        "penalty": ["l1", "l2"],
        "solver": ["saga"],
        "max_iter": [5000],
        "tol": [1e-4, 1e-2]
    }
}