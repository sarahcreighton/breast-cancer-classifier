"""
Modelling functions
- model constructors
- hyperparameter grids (for tuning)
- helper utilities related to modelling logic (not training itself)

Author: SC 2026-03-03

TO DO: documentation (comments, docstrings, README)
"""

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


#--- MODEL CONSTRUCTORS ---#
def logistic_pipeline(penalty="l2", C=1.0, random_state=42):
    """
    Logistic regression pipeline using scikit-learn >=1.3 "future" syntax

    Parameters:
    -----------
    penalty : str, default="l2"
        Supported: "l1", "l2", "elasticnet", None
        - "l1"          => elasticnet => l1_ratio=1.0
        - "l2"          => elasticnet => l1_ratio=0.0
        - "elasticnet"  => elasticnet => l1_ratio=0.5 (default)
        - None          => no regularization
    C : float, default=1.0
        Inverse of regularization strength
    random_state : int, default=42, random seed for reproducibility

    Returns:
    --------
    sklearn pipeline with StandardScaler + LogisticRegression ready for modelling
    """
    # Map "classic" penalties to future-proof elasticnet/none parameters
    if penalty == "l1":
        solver = "saga"
        penalty_final = "elasticnet"
        l1_ratio_final = 1.0
    elif penalty == "l2":
        solver = "saga"
        penalty_final = "elasticnet"
        l1_ratio_final = 0.0
    elif penalty == "elasticnet":
        solver = "saga"
        penalty_final = "elasticnet"
        l1_ratio_final = 0.5
    elif penalty == None:
        solver = "lbfgs"
        penalty_final = None
        l1_ratio_final = None
    else:
        raise ValueError(f"Unsupported penalty: {penalty}")

    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            penalty = penalty,
            C = C,
            solver = solver,
            l1_ratio = l1_ratio_final,
            max_iter = 5000,
            random_state = random_state
        ))
    ])


def rf_pipeline(n_estimators=200, max_depth=None, random_state=42):
    """
    Pipeline for a Random Forest classifier
    """
    # Note: RF does NOT need scaling, but keeping structure consistent
    return Pipeline([
        ("model", RandomForestClassifier(
            n_estimators = n_estimators,
            max_depth = max_depth,
            random_state = random_state
        ))
    ])

def svm_pipeline(C=1.0, kernel="rbf", random_state=42):
    """
    Pipeline for a Support Vector Machine (SVM) classifier
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(
            C = C,
            kernel = kernel,
            probability = True,
            random_state = random_state
        ))
    ])


#--- HYPERPARAMETER GRIDS ---#

#--- HELPER UTILITIES ---#
