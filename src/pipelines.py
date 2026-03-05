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
    Pipeline for Logistic Regression classifer
    """
    solver = "liblinear" if penalty == "l1" else "lbfgs"
    
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
