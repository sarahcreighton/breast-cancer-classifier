
def train_model(pipeline, X_train, y_train):
    """
    Fit a pipeline and return trained model.

    Parameters:
    -----------
    pipeline: sklearn Pipeline object
        X_train: pd.DataFrame
        y_train: pd.Series

    Returns:
    --------
    model: trained pipeline
    """
    model = pipeline.fit(X_train, y_train)
    return model