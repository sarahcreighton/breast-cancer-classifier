# for comparing multiple models
import pandas as pd
from train import train_model
from evaluate import evaluate_model

def compare_models(models_dict, X_train, y_train, X_test, y_test):
    """
    Train and evaluate multiple models.
    
    Parameters:
    -----------
        models_dict: {model_name: pipeline_object}
        X_train, y_train, X_test, y_test: data splits
        
    Returns:
    --------
        results_df: pd.DataFrame of metrics
        cms: confusion matrices
        trained_models: dict of fitted models
    """
    results = []
    trained_models = {}
    cms = {}

    for name, pipe in models_dict.items():

        # Train the model via train.py
        trained_model = train_model(pipe, X_train, y_train)
        trained_models[name] = trained_model

        # Evaluate on hold-out test data
        metrics_df, cm = evaluate_model(trained_model, X_test, y_test, model_name=name)
        results.append(metrics_df)
        cms[name] = cm
    
    results_df = pd.concat(results)
    
    return results_df, cms, trained_models
