# tune.py

from sklearn.model_selection import GridSearchCV

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
    model = pipeline.named_steps["model"]

    if hasattr(model, "coef_"):
        return dict(zip(feature_names, model.coef_.flatten()))
    
    elif hasattr(model, "feature_importances_"):
        return dict(zip(feature_names, model.feature_importances_))
    
    else:
        return None 