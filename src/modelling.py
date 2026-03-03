"""
Modelling functions
- model constructors
- hyperparameter grids (for tuning)
- helper utilities related to modelling logic (not training itself)

Author: SC 2026-03-03

TO DO: documentation (comments, docstrings, README)
"""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def logistic_pipeline(penalty="l2", C=1.0, random_state=42):
    """
    Train logistic regression with optional L1/L2 regularization
    """
    # solver = "liblinear" if penalty in ["l1", "l2"] else "lbfgs"
    if penalty == "l1":
        solver = "liblinear"
    elif penalty == "l2":
        solver = "lbfgs"
    elif penalty == "none":
        solver = "lbfgs"

    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            penalty = penalty,
            C = C,
            solver = solver,
            max_iter = 1000,
            random_state = random_state
        ))
    ])


def rf_pipeline(n_estimators=200, max_depth=None, random_state=42):
    """
    Train a Random Forest classifier
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
    Train a Support Vector Machine (SVM) classifier
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