# src/model.py
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF, WhiteKernel, ConstantKernel as C, ExpSineSquared
)

def train_gp_model(train_df):
    """
    Train a Gaussian Process model on log-close vs time.
    This function is well-written and requires no changes.
    """
    X_train = train_df[["t"]].values
    y_train = train_df["LogClose"].values

    # Composite kernel: constant * RBF + periodic + noise
    kernel = (
        C(1.0, (1e-3, 1e3))
        * RBF(length_scale=50.0, length_scale_bounds=(1.0, 1e3))
        + ExpSineSquared(length_scale=30.0, periodicity=252.0,
                         periodicity_bounds=(50, 400))
        + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-8, 1e1))
    )

    gp = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=5,
        normalize_y=True
    )
    gp.fit(X_train, y_train)

    return gp, gp.kernel_


def forecast_gp(model, df_feat, forecast_days=5, train_len=None):
    """
    Forecast future prices using the trained GP model.

    Returns:
        forecast_df: DataFrame with full history + future forecast.
        pred_df: DataFrame with predictions for the test period only.
    """
    last_date = df_feat["Date"].max()
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)

    start_date_t = df_feat["Date"].min()
    future_t = (future_dates - start_date_t).days.astype(float).reshape(-1, 1)

    # Combine historical and future time steps for prediction
    X_all = np.vstack([df_feat[["t"]].values, future_t])
    mu, std = model.predict(X_all, return_std=True)

    # Combine historical and future dates for the final DataFrame
    dates_all = df_feat["Date"].append(pd.Series(future_dates))

    forecast_df = pd.DataFrame({
        "Date": dates_all,
        "Predicted": np.exp(mu),
        "Upper95": np.exp(mu + 1.96 * std),
        "Lower95": np.exp(mu - 1.96 * std)
    })

    pred_df = None
    if train_len is not None and train_len < len(df_feat):
   
        pred_df = forecast_df.iloc[train_len : len(df_feat)]
        pred_df = pred_df.set_index("Date") # Set date index for correct alignment

    return forecast_df.set_index("Date"), pred_df