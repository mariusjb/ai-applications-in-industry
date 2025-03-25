from typing import List, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def load_and_split_data(
    path: str,
    datetime_col: str = "Datetime", 
    train_frac: float = 0.7, 
    val_frac: float = 0.15, 
    parse_dates: bool = True
):
    """
    Loads a CSV file, sorts by datetime, and splits it into train, validation, and test sets.

    Parameters:
    - path (str): Path to the CSV file.
    - datetime_col (str): Name of the datetime column to parse and sort by.
    - train_frac (float): Fraction of data to allocate to the training set.
    - val_frac (float): Fraction of data to allocate to the validation set.
    - parse_dates (bool): Whether to parse the datetime column as datetime.

    Returns:
    - Tuple of DataFrames: (train_df, val_df, test_df)
    """
    df = pd.read_csv(path, parse_dates=[datetime_col] if parse_dates else None)
    df = df.sort_values(datetime_col).reset_index(drop=True)

    n = len(df)
    train_end = int(train_frac * n)
    val_end = int((train_frac + val_frac) * n)

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    return train_df, val_df, test_df

def add_time_features(df, datetime_col: str = "Datetime") -> pd.DataFrame:
    """
    Extracts (hourly) time-based features from a datetime column.

    Parameters:
    - df (pd.DataFrame): Input DataFrame containing a datetime column.
    - datetime_col (str): Name of the datetime column to extract features from.

    Returns:
    - pd.DataFrame: Copy of the original DataFrame with new time features added.
    """
    assert datetime_col in df.columns, f"Column '{datetime_col}' not found in DataFrame."

    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df[datetime_col]):
        df[datetime_col] = pd.to_datetime(df[datetime_col])

    df["hour"] = df[datetime_col].dt.hour            # Hour of the day (0–23)
    df["dayofweek"] = df[datetime_col].dt.dayofweek  # Day of week (0=Monday, 6=Sunday)
    df["month"] = df[datetime_col].dt.month          # Month (1–12)
    df["day"] = df[datetime_col].dt.day              # Day of the month (1–31)
    df["year"] = df[datetime_col].dt.year            # Year (e.g., 2023)
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)  # 1 if Saturday or Sunday, else 0

    return df


def add_lag_features(df, target_col: str = "PJME_MW", lags: List[int] = [1, 24, 168]) -> pd.DataFrame:
    """
    Adds lagged versions of the target column to the DataFrame as new features.

    Parameters:
    - df (pd.DataFrame): Input DataFrame with a datetime index or time-sorted rows.
    - target_col (str): Name of the column to generate lag features from.
    - lags (list of int): List of lag values to apply (e.g., 1 for 1 step back).

    Returns:
    - pd.DataFrame: A new DataFrame with lag features added.
    """
    assert target_col in df.columns, f"Column '{target_col}' not found in DataFrame."
    assert df.index.is_monotonic_increasing, "DataFrame index must be sorted by time."
    assert all(lag >= 0 for lag in lags), "Lag must be non-negative. Otherwise, lags will result in future leakage."
    assert all(lag != 0 for lag in lags), "Lag must be non-zero. Otherwise, it will duplicate the target column."

    df = df.copy()
    for lag in lags:
        df[f"lag_{lag}"] = df[target_col].shift(lag)
    return df


def add_rolling_features(df, target_col: str = "PJME_MW", windows: List[int] = [24]) -> pd.DataFrame:
    """
    Adds rolling mean and standard deviation features for the given target column.
    Uses shifted values to avoid data leakage from the future.

    Parameters:
    - df (pd.DataFrame): Input time series DataFrame.
    - target_col (str): Name of the column to compute rolling stats on.
    - windows (list of int): Rolling window sizes (in time steps).

    Returns:
    - pd.DataFrame: Copy of the DataFrame with new rolling features added.
    """
    assert target_col in df.columns, f"Column '{target_col}' not found in DataFrame."
    assert df.index.is_monotonic_increasing, "DataFrame index must be sorted by time."
    assert all(window > 0 for window in windows), "Window size must be positive."
        
    df = df.copy()
    for window in windows:
        shifted = df[target_col].shift(1)
        df[f"rolling_mean_{window}"] = shifted.rolling(window=window).mean()
        df[f"rolling_std_{window}"] = shifted.rolling(window=window).std()
    return df

def scale(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    target_col: str = "PJME_MW",
    datetime_col: str = "Datetime", 
    ignore_cols: List[str] = []
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler, StandardScaler]:
    """
    Scales feature columns and the target column using StandardScaler.

    Parameters:
    - train, val, test: DataFrames to scale (already split chronologically).
    - target_col: The name of the target column to scale separately.
    - ignore_cols: Columns to exclude from feature scaling (e.g., identifiers, metadata).

    Returns:
    - train_scaled, val_scaled, test_scaled: Scaled versions of the input DataFrames.
    - feature_scaler: Fitted scaler used on features.
    - target_scaler: Fitted scaler used on the target column.
    """
    train_scaled = train.copy()
    val_scaled = val.copy()
    test_scaled = test.copy()

    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    feature_cols = [
        col for col in train.columns
        if col not in [datetime_col, target_col] + ignore_cols
    ]

    # Scale features (fit on train, transform on val/test)
    train_scaled[feature_cols] = feature_scaler.fit_transform(train[feature_cols])
    val_scaled[feature_cols] = feature_scaler.transform(val[feature_cols])
    test_scaled[feature_cols] = feature_scaler.transform(test[feature_cols])

    train_scaled[target_col] = target_scaler.fit_transform(train[[target_col]])
    val_scaled[target_col] = target_scaler.transform(val[[target_col]])
    test_scaled[target_col] = target_scaler.transform(test[[target_col]])

    return train_scaled, val_scaled, test_scaled, feature_scaler, target_scaler


def prepare_datasets(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    target_col: str = "PJME_MW",
    datetime_col: str = "Datetime"
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Splits input DataFrames into features and targets for train, validation, and test sets.

    Parameters:
    - train, val, test: Preprocessed DataFrames containing features and target.
    - target_col: The name of the target column to predict.

    Returns:
    - Tuple of:
        X_train (features), y_train (target),
        X_val (features), y_val (target),
        X_test (features), y_test (target)
    """
    assert all(target_col in df.columns for df in [train, val, test]), "Target column not found in DataFrame."
    assert all(datetime_col in df.columns for df in [train, val, test]), "Datetime column not found in DataFrame."

    feature_cols = [
        col for col in train.columns
        if col not in [datetime_col, target_col]
    ]

    X_train, y_train = train[feature_cols], train[target_col]
    X_val, y_val = val[feature_cols], val[target_col]
    X_test, y_test = test[feature_cols], test[target_col]

    return X_train, y_train, X_val, y_val, X_test, y_test


def create_sequences(
    df: pd.DataFrame,
    target_col: str = "PJME_MW",
    datetime_col: str = "Datetime",
    window_size: int = 24,
    forecast_steps: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates sequences of feature windows and corresponding target values
    for time series forecasting models (e.g., LSTM, Transformer).
    Supports both single-step and multi-step forecasting.

    Parameters:
    - df: DataFrame containing time-ordered features and target column.
    - target_col: Name of the column to predict.
    - datetime_col: Name of the datetime column.
    - window_size: Number of time steps in each input sequence.
    - forecast_steps: Number of future steps to predict (default: 1).

    Returns:
    - X: NumPy array of shape (samples, window_size, features)
    - y: NumPy array of shape (samples, forecast_steps) or (samples,) for forecast_steps=1
    """
    assert target_col in df.columns, f"Column '{target_col}' not found in DataFrame."
    assert datetime_col in df.columns, f"Column '{datetime_col}' not found in DataFrame."
    assert window_size > 0, "Window size must be positive."
    assert forecast_steps > 0, "Forecast steps must be positive."
    assert df.shape[0] > window_size + forecast_steps - 1, "Not enough data to create sequences."

    df = df.copy()
    df = df.dropna().reset_index(drop=True)
    feature_cols = [col for col in df.columns if col not in [datetime_col, target_col]]

    X, y = [], []
    for i in tqdm(range(window_size, df.shape[0] - forecast_steps + 1)):
        # Input sequence of features
        X_seq = df[feature_cols].iloc[i - window_size : i].values
        
        if forecast_steps == 1:
            # Single step forecast (original behavior)
            y_val = df[target_col].iloc[i]
            X.append(X_seq)
            y.append(y_val)
        else:
            # Multi-step forecast
            y_seq = df[target_col].iloc[i : i + forecast_steps].values
            X.append(X_seq)
            y.append(y_seq)

    X = np.array(X)
    if forecast_steps == 1:
        y = np.array(y)
    else:
        y = np.array(y)
    
    return X, y