# app.py
import streamlit as st
from datetime import date
import numpy as np
import pandas as pd

# sklearn GPR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF, WhiteKernel, ConstantKernel as C, ExpSineSquared
)
from sklearn.preprocessing import StandardScaler

# plotting
import plotly.graph_objects as go

# local helpers (you already have these)
from src.data_loader import load_data
# from src.features import add_basic_features # optional
from src.news_fetcher import fetch_nifty_news, get_sentiment_score

st.set_page_config(page_title="Nifty 50 — GPR + News Sentiment", layout="wide")
st.title("📈 Nifty 50 — Gaussian Process Regression (with News Sentiment)")
st.markdown("A probabilistic stock price forecast using GPR, trained on log returns and adjusted by recent news sentiment.")

# ---------------------------
# Sidebar controls
# ---------------------------
st.sidebar.header("Configuration")
symbol = st.sidebar.text_input("Ticker (yfinance)", value="^NSEI")
start_date = st.sidebar.date_input("Data start date", value=date(2018, 1, 1))
end_date = st.sidebar.date_input("Data end date", value=date.today())
forecast_days = st.sidebar.number_input("Forecast horizon (business days)", min_value=1, max_value=60, value=7)
train_ratio = st.sidebar.slider("Train ratio", 0.5, 0.95, 0.85, help="Fraction of the historical data to use for training the model.")
sentiment_strength = st.sidebar.slider(
    "Sentiment strength", 0.0, 0.1, 0.02, 
    help="How much the sentiment score adjusts the forecast. 0 = no effect."
)
retrain_button = st.sidebar.button("Train & Forecast", type="primary")

# ---------------------------
# Helpers / caching
# ---------------------------
@st.cache_data(show_spinner="Loading historical price data...")
def cached_load_data(sym, start, end):
    return load_data(sym, start, end)

@st.cache_data(show_spinner="Fetching news headlines...")
def cached_fetch_news(start_str, end_str):
    try:
        return fetch_nifty_news(start_str, end_str)
    except Exception:
        return []

@st.cache_data(show_spinner="Analyzing sentiment...")
def cached_sentiment(news_list):
    try:
        # Pass only titles and links for efficient caching
        return get_sentiment_score([(n["title"], n.get("link", "")) for n in news_list])
    except Exception:
        return 0.0

# ---------------------------
# Main Application
# ---------------------------
if not retrain_button:
    st.info("Configure parameters in the sidebar and click **Train & Forecast** to begin.")
    st.stop()

# 1. Load and Prepare Data
# In app.py, replace the entire data loading section with this:

# 1. Load and Prepare Data
st.subheader("Data Loading and Preparation")
df_raw = cached_load_data(symbol, start_date, end_date)

if df_raw is None or df_raw.empty:
    st.error("No data found — check ticker/dates and internet connection.")
    st.stop()

# --- FINAL ROBUST FIX ---
# Instead of patching the old DataFrame, we build a new, clean one.
# This avoids all issues with corrupted cache data.
try:
    # STEP 1: Reliably get the 'Date' data. It could be a column or the index.
    if 'Date' in df_raw.columns:
        date_data = df_raw['Date']
    elif isinstance(df_raw.index, pd.DatetimeIndex):
        date_data = df_raw.index.to_series()
    else:
        st.error("Fatal Error: Could not find a 'Date' column or DatetimeIndex.")
        st.stop()

    # STEP 2: Reliably get the 'Close' data, handling duplicates.
    close_data = df_raw['Close']
    if isinstance(close_data, pd.DataFrame):
        st.warning("Duplicate 'Close' columns found. Using the first available.")
        close_data = close_data.iloc[:, 0]

    # STEP 3: Build the new, clean DataFrame.
    df = pd.DataFrame({
        "Date": date_data,
        "Close": close_data
    })

    # STEP 4: Perform conversions and cleaning on the new DataFrame.
    df["Date"] = pd.to_datetime(df["Date"])
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

    # This line will now work correctly.
    df = df.dropna(subset=["Close", "Date"]).reset_index(drop=True)

except Exception as e:
    st.error(f"A fatal error occurred during data preparation: {e}")
    st.info("This is often caused by an old, incompatible data cache. Please try clearing the cache.")
    st.dataframe(df_raw.head()) # Display the raw data for debugging
    st.stop()


st.subheader("Historical Close Price")
st.dataframe(df[["Date", "Close"]].tail(200), use_container_width=True)

# (The rest of your code follows...)

# 2. Feature Engineering
# ---
# IMPROVEMENT: We train on log returns for better stationarity.
# This changes the problem from predicting price to predicting the daily % change.
df["t"] = (df["Date"] - df["Date"].min()).dt.days.astype(float)
df["LogClose"] = np.log(df["Close"])
df["LogReturn"] = df["LogClose"].diff()
df = df.dropna(subset=["LogReturn"]).reset_index(drop=True)

# Set Date as index for easy plotting and slicing
df_feat = df[["Date", "Close", "t", "LogClose", "LogReturn"]].copy().set_index("Date")

# 3. Fetch News and Calculate Sentiment
# ---
# We fetch news for the forecast window to adjust the upcoming prediction.
last_hist_date = df_feat.index[-1]
news_end_date = last_hist_date
news_start_date = news_end_date - pd.Timedelta(days=3) # Look back over the last 3 days

news_start_date_str = news_start_date.strftime("%Y-%m-%d")
news_end_date_str = news_end_date.strftime("%Y-%m-%d")

# The rest of the news fetching logic can now proceed as before
news_list = cached_fetch_news(news_start_date_str, news_end_date_str)
sentiment_score = cached_sentiment(tuple(news_list)) if news_list else 0.0

st.subheader(f"📰 News Sentiment Analysis ({news_start_date_str} → {news_end_date_str})")
if news_list:
    for i, item in enumerate(news_list[:10], 1):
        title = item.get("title") or "No Title"
        link = item.get("link") or item.get("url")
        st.markdown(f"{i}. [{title}]({link})" if link else f"{i}. {title}")
else:
    st.write("No recent news found for the specified lookback period.")

st.markdown(f"**Aggregated Sentiment Score:** `{sentiment_score:.3f}` (Range: -1 to +1)")
st.markdown(f"**Sentiment Strength Multiplier:** `{sentiment_strength}`")


# 4. Prepare Data for GPR Model
# ---
# CRITICAL FIX: We do NOT use sentiment as a feature in the training data to avoid look-ahead bias.
# The model is trained only on the time index 't'.
X = df_feat[["t"]].values
y = df_feat["LogReturn"].values

# Train/test split by time
n_total = len(df_feat)
n_train = int(n_total * train_ratio)
if n_train < 20:
    st.warning(f"Very few training points ({n_train}). Consider increasing the data range or train ratio.")
    st.stop()

X_train, y_train = X[:n_train], y[:n_train]
dates_train = df_feat.index[:n_train]
dates_test = df_feat.index[n_train:]

# Scale the feature (time)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 5. Define Kernel and Train GPR Model
# ---
# Kernel defines the assumptions about the function we're trying to model.
long_term_trend = C(1.0) * RBF(length_scale=50.0)  # RBF for smooth long-term changes
seasonality = ExpSineSquared(length_scale=1.0, periodicity=252.0) # Periodicity of ~1 year (trading days)
noise = WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e5))

kernel = long_term_trend + seasonality + noise
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=True, random_state=42)

with st.spinner("Training Gaussian Process regressor... (this can take a moment)"):
    gp.fit(X_train_scaled, y_train)

st.success("GPR model trained successfully.")
st.write(f"**Optimized Kernel:** `{gp.kernel_}`")


# 6. Create Future Timeline and Predict
# ---
# IMPROVEMENT: Generate future business days only, not weekends/holidays.
future_dates = pd.bdate_range(start=last_hist_date + pd.Timedelta(days=1), periods=forecast_days)
all_dates = df_feat.index.union(future_dates)

t_all = (all_dates - df["Date"].min()).days.astype(float).values.reshape(-1, 1)
t_all_scaled = scaler.transform(t_all)

# Predict mean and std deviation for log returns
mu_log_returns, std_log_returns = gp.predict(t_all_scaled, return_std=True)

# 7. Reconstruct Price Forecast from Log Returns
# ---
# We must convert the predicted returns back into price levels.
last_log_price = df_feat["LogClose"].iloc[-1]
# The first predicted return is for the day after the last historical day
hist_and_future_log_returns = np.concatenate([df_feat["LogReturn"].values, mu_log_returns[len(df_feat):]])
# Reconstruct log prices by cumulatively adding returns
log_price_mu_recon = last_log_price + np.cumsum(hist_and_future_log_returns[len(df_feat):])
log_price_mu_recon = np.insert(log_price_mu_recon, 0, last_log_price) # Add last known price

# Create a full timeline of mean log prices (history + future)
mu_log_price_all = np.concatenate([df_feat["LogClose"].values, log_price_mu_recon[1:]])

# Calculate uncertainty bounds on the price scale
# Uncertainty accumulates over time with returns
cumulative_std = np.sqrt(np.cumsum(std_log_returns[len(df_feat):]**2))
cumulative_std = np.insert(cumulative_std, 0, 0) # No uncertainty on the last known day

# Adjust the future forecast with the sentiment score
sentiment_adjustment = sentiment_score * sentiment_strength
price_mu_final = np.exp(mu_log_price_all)
price_mu_final[len(df_feat):] *= (1 + sentiment_adjustment) # Apply adjustment only to future dates

price_upper = np.exp(mu_log_price_all + 1.96 * np.concatenate([std_log_returns[:len(df_feat)], cumulative_std[1:]]))
price_lower = np.exp(mu_log_price_all - 1.96 * np.concatenate([std_log_returns[:len(df_feat)], cumulative_std[1:]]))
# Also adjust the bounds by sentiment
price_upper[len(df_feat):] *= (1 + sentiment_adjustment)
price_lower[len(df_feat):] *= (1 + sentiment_adjustment)


# 8. Plotting and Metrics
# ---
st.subheader("Forecast vs. Actuals")
fig = go.Figure()

# Actual historical data
fig.add_trace(go.Scatter(x=df_feat.index, y=df_feat["Close"], mode="lines", name="Actual Price", line=dict(color="black")))

# GPR Mean Forecast (History + Future)
fig.add_trace(go.Scatter(x=all_dates, y=price_mu_final, mode="lines", name="GPR Mean Forecast", line=dict(color="orange", dash="solid")))

# Confidence Interval (95% CI)
fig.add_trace(go.Scatter(
    x=list(all_dates) + list(all_dates[::-1]),
    y=list(price_upper) + list(price_lower[::-1]),
    fill="toself",
    fillcolor="rgba(255,165,0,0.2)",
    line=dict(color="rgba(255,255,255,0)"),
    hoverinfo="skip",
    showlegend=True,
    name="95% Confidence Interval"
))

# Vertical line for train/test split
# New, corrected code
# Step 1: Draw the vertical line without the annotation text
fig.add_vline(x=dates_train[-1], line_width=1, line_dash="dash", line_color="gray")

# Step 2: Add the annotation separately for full control
fig.add_annotation(
    x=dates_train[-1],
    y=0.98,  # Position near the top of the plot
    yref="paper",  # Use plot's relative coordinates for y-position
    text="Train/Test Split",
    showarrow=False,
    font=dict(color="gray", size=12),
    xanchor="right",
    yanchor="top"
)

fig.update_layout(
    title=f"{symbol} Forecast (Adjusted by Sentiment: {sentiment_adjustment:.2%})",
    xaxis_title="Date",
    yaxis_title="Price",
    template="plotly_white",
    hovermode="x unified",
    legend=dict(x=0.01, y=0.98, bordercolor="Black", borderwidth=1)
)
st.plotly_chart(fig, use_container_width=True)

# Test set metrics
if len(dates_test) > 0:
    pred_test_price = price_mu_final[n_train:n_total]
    actual_test_price = df_feat["Close"].values[n_train:]
    
    mae = np.mean(np.abs(pred_test_price - actual_test_price))
    rmse = np.sqrt(np.mean((pred_test_price - actual_test_price) ** 2))
    
    st.subheader("Model Performance (on Test Set)")
    col1, col2 = st.columns(2)
    col1.metric("Mean Absolute Error (MAE)", f"{mae:.2f}")
    col2.metric("Root Mean Squared Error (RMSE)", f"{rmse:.2f}")

# Forecast table
st.subheader(f"Forecast Table (Next {forecast_days} Business Days)")
forecast_df = pd.DataFrame({
    "Date": future_dates,
    "Predicted": price_mu_final[-forecast_days:],
    "Upper 95%": price_upper[-forecast_days:],
    "Lower 95%": price_lower[-forecast_days:]
}).set_index("Date")
st.dataframe(forecast_df.style.format("{:.2f}"))