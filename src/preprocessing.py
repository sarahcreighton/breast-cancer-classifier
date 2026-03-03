"""
PRE-PROCESSING FUNCTIONS

Dependences: pandas, sklearn, wdbc-env should take care of this though

Notes:
    Functions can be standalone, but not generally advised, to avoid
    potential data leakage. One exception: load_raw_data for EDA/figures

    prepare_model_data() produces ML pipeline ready preprocessed data 

    TO DO: Have someone else verify the functions work on their machine

Author: SC 2026-03-03
"""

import pandas as pd 
from dataclasses import dataclass
from typing import Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


#--- Create ModelData class helper ---#
@dataclass
class ModelData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    scaler: Optional[object] = None



#--- LOAD RAW DATA ---#
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

    # re-map target to categorical: 0=Malignant, 1=Benign
    df["diagnosis"] = pd.Categorical(
        df["diagnosis"].map({"M": "malignant", "B": "benign"}),
        categories = ["malignant", "benign"]
    )

    return df


#--- GET MODEL DATA ---#
def get_model_data(df: pd.DataFrame, encode_target: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separate data into features (X) and target (y).

    Parameters:
    ----------- 
    df : pd.DataFrame 
        - input df containing features and target
    encode_target : bool, default=True
        - if True maps 'malignant' to 0 and 'benign' to 1
        - if False, returns original target labels

    Returns: 
    --------
    X : pd.DataFrame, feature columns 
    y : pd.Series, target column ('diagnosis')
    """
    df = df.copy()
    
    if "diagnosis" not in df.columns:
        raise ValueError("Column 'diagnosis' not found in DataFrame.")
    
    if encode_target and df["diagnosis"].dtype != "int":
        df["diagnosis"] = df["diagnosis"].map({"malignant":0, "benign": 1})
    
    drop_cols = [col for col in ["id","diagnosis"] if col in df.columns]
    
    X = df.drop(columns = drop_cols)
    y = df["diagnosis"]
    
    return X, y


#--- SPLIT DATA ---#
def split_data(X, y, test_size=0.2, random_state=42):
    """
    Splits features and target into train and test sets, stratified by target.

    Parameters:
    -----------
    X : pd.DataFrame, feature columns
    y : pd.Series, target column
    test_size : float, default=0.2
    random_state : int, default=42 (for reproducibility)

    Returns: 
    --------
    X_train, X_test, y_train, y_test (or X, y) : train/test splits
    """
    if len(X) != len(y):
        raise ValueError("X and y must have the same number of rows.")
    
    return train_test_split(
        X, y,
        stratify = y,
        test_size = test_size,
        random_state = random_state
    )

#--- FEATURE SCALING ---#
def scale_features(X_train, X_test, scaler_cls=StandardScaler):
    """
    Scale numeric columns in X_train and X_test using scaler_cls.
    Scaler is fitted on X_train only; X_test and future data should 
    be transformed using the returned scaler.
    
    Parameters:
    -----------
    X_train, X_test : pd.DataFrame, feature dataframes to scale
    scaler_cls : class, default=StandardScaler
        - Scikit-learn scaler class

    Returns: 
    --------
    X_train_scaled, X_test_scaled : pd.DataFrame
        - scaled train and test features
    scaler : fitted scaler (optional)
        - the fitted scaler on training data
        - needed to correctly transform unseen test or production data
        and to prevent data leakage
    """
    numeric_cols = X_train.select_dtypes(include="number").columns

    scaler = scaler_cls()
    
    # fit scaler on train only, transform both
    X_train_scaled = X_train.copy()
    X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
   
    # transform X_test
    X_test_scaled = X_test.copy()
    X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

    return X_train_scaled, X_test_scaled, scaler


#--- PREPARE FOR PROCESSING/MODELLING ---#
def prepare_model_data(path, test_size=0.2, random_state=42, use_pipeline=True):
    """
    Prepare data for modelling:
        - load raw data
        - separate features (X) and target (y)
        - Train/test split (statified)
        - Scale numeric features if not using a pipeline
    
    Parameters:
    -----------
    path : str, path to the raw data (.data)
    test_size : float, default=0.2
        - fraction of data to allocate to the test set
    random_state : int, default=42
        - random seed for reproducibility
    use_pipeline : bool, default=True
        - if True, scaling is skipped (assumed to happen in the pipeline)
        - if False, numeric features are scaled using StandardScaler.

    Returns: 
    --------
    dict containing X_train, X_test, y_train, y_test, scaler
        - scaler is the fitted scaler if use_pipeline=False, else None
    """
    # loads raw data
    df = load_raw_data(path)

    # separate features and target
    X, y = get_model_data(df)

    # split train/test (stratified)
    X_train, X_test, y_train, y_test = split_data(
        X, y, 
        test_size=test_size, 
        random_state=random_state
    )

    # Scale numeric features only if not using a pipeline
    if use_pipeline:
        scaler = None
    else: 
        X_train, X_test, scaler = scale_features(X_train, X_test)
  
    return ModelData(
        X_train = X_train,
        X_test = X_test,
        y_train = y_train,
        y_test = y_test,
        scaler = scaler
    )

    # {
    #     "X_train": X_train, 
    #     "X_test": X_test, 
    #     "y_train": y_train, 
    #     "y_test": y_test,
    #     "scaler": scaler 
    # }
