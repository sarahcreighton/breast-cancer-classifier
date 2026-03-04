"""
Modelling functions
- model constructors
- hyperparameter grids (for tuning)
- helper utilities related to modelling logic (not training itself)

Author: SC 2026-03-03

TO DO: documentation (comments, docstrings, README)
Consider replacing StandardScaler() with more generic scaler_cls()
"""

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


#--- MODEL CONSTRUCTORS ---#
def logistic_pipeline(penalty="l2", C=1.0, random_state=42):
    """
    Pipeline for logistic regression with optional L1/L2 regularization
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
# def logistic_param_grid():
#     return {
#         "model__C": [0.01, 0.1, 1, 10],
#         "model__penalty": ["l1", "l2"]
#     }

# def rf_param_grid():
#     return {
#         "model__n_estimators": [100, 200, 500],
#         "model__max_depth": [None, 5, 10]
#     }

# def svm_param_grid():
#     return {
#         "model__C": [0.1, 1, 10],
#         "model__kernel": ["linear", "rbf"]
#     }

def get_param_grid(model_name: str) -> dict:
    model_name = model_name.lower()

    if model_name == "logistic":
        return {
            "model__C": [0.01, 0.1, 1, 10],
            "model__penalty": ["l1", "l2"],
        }
    elif model_name == "rf":
        return {
            "model__n_estimators": [100, 200],
            "model__max_depth": [None, 5, 10],
        }
    elif model_name == "svm":
        return {
            "model__C": [0.1, 1, 10],
            "model__kernel": ["linear", "rbf"],
        }
    else:
        raise ValueError("Unknown model name.")


#--- HELPER UTILITIES ---#

# generic tuning
def tune_model(pipeline, param_grid, X_train, y_train, cv=5, scoring="roc_auc"):
    """
    Perform grid search and return best estimator.
    """
    grid = GridSearchCV(
        estimator = pipeline,
        param_grid = param_grid,
        cv = cv, 
        scoring = scoring
        # look into n_jobs=-1
    )
    
    grid.fit(X_train, y_train)

    return grid.best_estimator_, grid.best_params_, grid.best_score_


# # coefficient extraction
# def extract_logistic_coefs(pipeline, feature_names):
#     model = pipeline.named_steps["model"]
#     return pd.Series(model.coef_[0], index=feature_names)

def extract_feature_importance(pipeline, feature_names):
    model = pipeline.named_steps["model"]

    if hasattr(model, "coef_"):
        return dict(zip(feature_names, model.coef_.flatten()))
    
    elif hasattr(model, "feature_importances_"):
        return dict(zip(feature_names, model.feature_importances_))
    
    else:
        return None 