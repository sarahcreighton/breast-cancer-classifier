""" tune.py

Example workflow in notebook:

    # Step 1: define pipeline
    p = logistic_pipeline()

    # Step 2: get hyperparameter grid
    grid = get_param_grid("logistic")

    # Step 3: run grid search
    best_model, best_params, best_score = tune_model(
        p, grid, X_train, y_train)
    print(best_params)
    print(best_score)

    # Step 4: evaluate tuned model
    metrics, cm = evaluate_model(
        best_model, X_test, y_test, model_name="Logistic (Tuned)"
        )
    
    # Step 5: extract feature importance
    impt = extract_feature_importance(best_model, X_train.columns)
    impt_df = (
        pd.Series(impt)
        .sort_values(key=abs, ascending=False)
        .head(10)
    )
    impt_df = (
        extract_feature_importance(best_model, X_train.columns)
        .sort_values(key=abs, ascending=False)
        .head(10)
    )
    impt_df
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV

def get_param_grid(model_name: str) -> dict:
    model_name = model_name.lower()

    if model_name == "logistic":
        return [
            {
                "model__penalty": [None],
                "model__solver": ["lbfgs"],
                "model__C": [0.01, 0.1, 1, 10],
            },
            {
                "model__penalty": ["l1"],
                "model__solver": ["liblinear"],
                "model__C": [0.01, 0.1, 1, 10],
            },
            {
                "model__penalty": ["l2"],
                "model__solver": ["lbfgs"],
                "model__C": [0.01, 0.1, 1, 10],
            }
        ]
    
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
    

def tune_model(pipeline, param_grid, X_train, y_train, cv=5, scoring="roc_auc", n_jobs=-1):
    """
    Perform grid search and return best estimator.
    """
    grid = GridSearchCV(
        estimator = pipeline,
        param_grid = param_grid,
        cv = cv, 
        scoring = scoring,
        n_jobs = n_jobs
    )
    
    grid.fit(X_train, y_train)

    return grid.best_estimator_, grid.best_params_, grid.best_score_

def extract_feature_importance(pipeline, feature_names):
    estimator = pipeline.named_steps["model"]

    if hasattr(estimator, "coef_"):
        importance = estimator.coef_[0]
        # return dict(zip(feature_names, estimator.coef_.flatten()))
    
    elif hasattr(estimator, "feature_importances_"):
        importance = estimator.feature_importances_
        # return dict(zip(feature_names, estimator.feature_importances_))

    else:
        raise ValueError("Model does not support feature importance.")
    
    return pd.Series(importance, index=feature_names).sort_values(ascending=False) 