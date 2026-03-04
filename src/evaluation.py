"""
Evaluation utilties for model comparison

*** SUPER IN PROGRESS AND NOT TESTED ***
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


#--- EVALUATE HOLD-OUT TEST DATA ---#
def evaluate_model(model, X_test, y_test):
    """
    Evaluate fitted model on hold-out test data
    Returns dictionary of metrics
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    return {
        "roc_auc": roc_auc_score(y_test, y_proba) if y_proba is not None else None,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred), 
        "f1": f1_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred)
    }


#--- COMPARE MODELS ---#
def compare_models(models_dict, X_train, y_train, X_test, y_test):
    """
    models_dict: {"Logistic": model_object, ...}
    Returns DataFrame of test metrics
    """
    results = []

    for name, model in models_dict.items():
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        metrics["model"] = name
        results.append(metrics)
    
    return pd.DataFrame(results)



#--- CROSS-VALIDATION EVALUATION ---#
def cross_validate_model(model, X, y, cv=5, scoring="roc_auc"):
    """
    Perform cross-validation
    Returns mean and std of selected metric
    """
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring) #look into n_jobs

    return {
        "cv_mean": np.mean(scores),
        "cv_std": np.std(scores),
        "cv_scores": scores
    }