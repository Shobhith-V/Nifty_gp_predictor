import plotly.graph_objects as go
import pandas as pd

def plot_forecast(df_feat, forecast_df, forecast_days, train_len):
    fig = go.Figure()

    # Training data
    fig.add_trace(go.Scatter(
        x=df_feat.index[:train_len],
        y=df_feat["Close"][:train_len],
        mode="lines",
        name="Train"
    ))

    # Test data
    fig.add_trace(go.Scatter(
        x=df_feat.index[train_len:],
        y=df_feat["Close"][train_len:],
        mode="lines",
        name="Test"
    ))

    # Forecast line
    fig.add_trace(go.Scatter(
        x=forecast_df.index,
        y=forecast_df["Predicted"],
        mode="lines",
        name="Forecast"
    ))

    # Confidence intervals
    fig.add_trace(go.Scatter(
        x=forecast_df.index.tolist() + forecast_df.index[::-1].tolist(),
        y=forecast_df["Upper95"].tolist() + forecast_df["Lower95"][::-1].tolist(),
        fill="toself",
        fillcolor="rgba(0, 176, 246, 0.2)",
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip",
        showlegend=True,
        name="95% CI"
    ))

    fig.update_layout(
        title="Nifty Forecast",
        yaxis=dict(
            range=[
                min(float(df_feat["Close"].min()), float(forecast_df["Lower95"].min())),
                max(float(df_feat["Close"].max()), float(forecast_df["Upper95"].max()))
            ]
        )
    )

    return fig
