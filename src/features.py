# src/features.py
import pandas as pd
import numpy as np
def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add basic time-based features for Gaussian Process modeling.
    Here: numeric time index, log price, daily returns.
    """
    df = df.copy()
    df["t"] = (df["Date"] - df["Date"].min()).dt.days.astype(float)
    df["LogClose"] = np.log(df["Close"])
    df["Return"] = df["Close"].pct_change()
    df = df.dropna().reset_index(drop=True)
    return df
