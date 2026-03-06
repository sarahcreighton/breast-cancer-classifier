"""
PRE-PROCESSING FUNCTIONS

Example workflow:
    df = load_raw_data(path)
    X, y = get_Xy(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    pipe = logistic_pipeline()
    trained_model = train_model(pipe, X_train, y_train) 
    results = evaluate_model(trained_model, X_test, y_test)

Notes:
    prepare_model_data() produces ML pipeline ready preprocessed data.
    It abstracts a lot, but useful for quick onboarding of new team 
    members to ensure no data leakage. 

    Scaling happens inside pipelines. Do NOT scale the data before 
    passing it to the pipeline. Scaling should only happen once.

Author: SC 2026-03-03
"""

import pandas as pd 
from dataclasses import dataclass
from typing import Tuple
from sklearn.model_selection import train_test_split

#--- Create ModelData class helper ---#
# only used with prepare_model_data()
@dataclass
class ModelData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series

def load_raw_data(path: str) -> pd.DataFrame:
    """
    Load the Wisconsin Breast Cancer Diagnostic .data,
    set appropriate column names, re-map target to categorical 
    """
    measures = ["mean", "error", "worst"]
    features = [
        "radius", "texture", "perimeter", "area", "smoothness", "compactness", 
        "concavity", "concave points", "symmetry", "fractal dimension"
    ]

    # more convenient naming for plots
    columns = ["id", "diagnosis"] + [
        f"{m} {f}" if m in ["mean", "worst"] else f"{f} {m}"
        for m in measures for f in features
    ]

    # load the data with our column names
    df = pd.read_csv(path, names = columns, header = None)
    
    # get rid of unneeded id column if present
    if "id" in df.columns:
        df = df.drop(columns="id") 

    # re-map target to categorical: 1=Malignant, 0=Benign
    df["diagnosis"] = pd.Categorical(
        df["diagnosis"].map({"M": "malignant", "B": "benign"}),
        categories = ["malignant", "benign"]
    )

    return df

def get_Xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separate data into features (X) and target (y).
    Maps 'malignant' to 1 and 'benign' to 0.

    Parameters:
    ----------- 
    df : pd.DataFrame 
        Input df containing features and target

    Returns: 
    --------
    X : pd.DataFrame, feature columns 
    y : pd.Series, target column ('diagnosis')
    """
    df = df.copy()
    
    if "diagnosis" not in df.columns:
        raise ValueError("Column 'diagnosis' not found in DataFrame.")
    
    df["diagnosis"] = df["diagnosis"].map({"malignant":1, "benign": 0})
    
    drop_cols = [col for col in ["id","diagnosis"] if col in df.columns]
    
    X = df.drop(columns = drop_cols)
    y = df["diagnosis"]
    
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Splits features and target into train and test sets, 
    using stratified sampling (project default policy) to
    preserve class balance.

    Parameters:
    -----------
        X : pd.DataFrame, feature columns
        y : pd.Series, target column
        test_size : float, default=0.2
        random_state : int, default=42 (for reproducibility)

    Returns: 
    --------
        X_train, X_test, y_train, y_test : train/test splits
    """
    if len(X) != len(y):
        raise ValueError("X and y must have the same number of rows.")
    
    return train_test_split(
        X, y,
        stratify = y,
        test_size = test_size,
        random_state = random_state
    )

def prepare_model_data(path, test_size=0.2, random_state=42):
    """
    Prepare data for modelling. Load raw data, separate features (X)
    and target (y), perform stratified train/test split.
    
    This function standardizes the safe team workflow:
        load => X/y => stratified split

    Note that scaling happens inside pipelines, not here.

    Parameters:
    -----------
        path : str, path to the raw data (.data)
        test_size : float, default=0.2, fraction of data allocated to test set
        random_state : int, default=42, for reproducibility
    
    Returns: 
    --------
        dataclass containing .X_train, .X_test, .y_train, .y_test
     """
    df = load_raw_data(path)    # load and clean raw data
    X, y = get_Xy(df)           # separate features and target

    # split train/test (stratified)
    X_train, X_test, y_train, y_test = split_data(
        X, y, 
        test_size = test_size, 
        random_state = random_state
    )
  
    return ModelData(
        X_train = X_train,
        X_test = X_test,
        y_train = y_train,
        y_test = y_test
    )