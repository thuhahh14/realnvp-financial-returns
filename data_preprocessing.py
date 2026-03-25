import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_fred_sp500(csv_path: str) -> pd.DataFrame:
    """
    Load SP500 data downloaded from FRED.
    Expected columns:
    - DATE
    - SP500
    """
    df = pd.read_csv(csv_path)

    if "DATE" not in df.columns or "SP500" not in df.columns:
        raise ValueError("CSV file must contain columns: DATE and SP500")

    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df["SP500"] = pd.to_numeric(df["SP500"], errors="coerce")

    df = df.dropna().sort_values("DATE").reset_index(drop=True)
    return df


def compute_log_return(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute log-return from SP500 price index.
    r_t = ln(P_t / P_{t-1})
    """
    df = df.copy()
    df["log_return"] = np.log(df["SP500"] / df["SP500"].shift(1))
    df = df.dropna().reset_index(drop=True)
    return df


def create_sliding_windows(series: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    Convert a 1D return series into windows of shape (n_samples, window_size).
    """
    X = []
    for i in range(len(series) - window_size + 1):
        X.append(series[i:i + window_size])
    return np.array(X, dtype=np.float32)


def standardize_train_test(X: np.ndarray, test_size: float = 0.2):
    """
    Split train/test without shuffling, then standardize using train mean/std.
    """
    X_train, X_test = train_test_split(X, test_size=test_size, shuffle=False)

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)

    # tránh chia cho 0
    std[std == 0] = 1.0

    X_train_std = (X_train - mean) / std
    X_test_std = (X_test - mean) / std

    return X_train_std, X_test_std, mean, std
