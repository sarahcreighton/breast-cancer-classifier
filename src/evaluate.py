"""
Evaluation utilties for trained model

Author: SC 2026-03-03
"""

import pandas as pd
import numpy as np

from sklearn.metrics import(
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from sklearn.model_selection import cross_val_score
from tune import get_param_grid, tune_model


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluate fitted model on hold-out test data, returning metrics.
    Confusion matrix is returned as a labeled DataFrame for easy plotting.

    Returns: 
        metrics_df: pd.DataFrame of metrics
        cm: np.array of confusion matrices
    """
    y_pred = model.predict(X_test)
    
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1] 
        roc = roc_auc_score(y_test, y_proba)
    else:
        y_proba = None
        roc = None
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sensitivity = recall_score(y_test, y_pred)  # tp / (tp + fn)
    specificity = tn / (tn + fp)

    metrics = {
        "Model": model_name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "Sensitivity": round(sensitivity, 4),
        "Specificity": round(specificity, 4),
        "ROC-AUC": round(roc, 4) if roc else None,
        "Precision": round(precision_score(y_test, y_pred), 4),
        "F1-Score": round(f1_score(y_test, y_pred), 4)
    }

    metrics_df = pd.DataFrame([metrics]).set_index("Model")
    cm = np.array([[tn, fp], [fn, tp]])

    return metrics_df, cm


def cross_validate_model(model, X, y, cv=5, scoring="roc_auc", n_jobs=-1):
    """
    Perform cross-validation.

    Parameters:
    -----------
        model:
        X:
        y:
        cv:
        scoring:
        n_jobs:

    Returns:
    --------
        dict with cv_mean, cv_std, cv_scores
    """
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs) 

    return {
        "cv_mean": np.mean(scores),
        "cv_std": np.std(scores),
        "cv_scores": scores
    }


def tune_and_evaluate(model_name, pipe_fn, X_train, y_train, X_test, y_test):
    """
    Tune a model using grid search and evaluate on test data

    Parameters:
    -----------
        model_name: str, name of the model (used for labeling metrics)
        pipe_fn: function returning an unfitted pipeline
        X_train, y_train, X_test, y_test: pd.DataFrame / pd.Series
            train-test splits
    
    Returns:
    --------
    metrics: pd.DataFrame 
        evaluation metrics (single-row)
    cm: np.ndarray
        confusion matrix
    best_model: fitted pipeline
        trained pipeline with best hyperparameters
    best_params: dict
        best hyperparameters from tuning
    best_score: float
        best cross-validation score from tuning
    """
    # initialize pipeline
    pipe = pipe_fn()

    # get hyperparameter grid
    grid = get_param_grid(model_name.lower())

    # tune the model
    model, param, score = tune_model(pipe, grid, X_train, y_train)

    # evaluate on test data
    metrics, cm = evaluate_model(model, X_test, y_test, model_name=f"{model_name} (Tuned)")

    return metrics, cm, model, param, score 