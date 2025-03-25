import pandas as pd
import plotly.graph_objects as go
import numpy as np


def plot_actual_vs_pred(y_true, y_pred, title="Actual vs Predicted", n=500):
    """
    Plot actual vs predicted values.

    Parameters:
    - y_true: Ground truth values (array-like or pd.Series)
    - y_pred: Predicted values (array-like or pd.Series)
    - title: Plot title
    - n: Number of points to show (default: 500)
    """
    y_true = pd.Series(y_true).reset_index(drop=True)[:n]
    y_pred = pd.Series(y_pred).reset_index(drop=True)[:n]
    index = list(range(n))

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=index,
            y=y_true,
            mode="lines",
            name="Actual",
            line=dict(color="deepskyblue", width=2),
            opacity=0.8,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=index,
            y=y_pred,
            mode="lines",
            name="Predicted",
            line=dict(color="magenta", width=2, dash="dot"),
            opacity=0.7,
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Time Index",
        yaxis_title="Energy Consumption (MW)",
        template="plotly_dark",
        plot_bgcolor="black",
        paper_bgcolor="black",
        font=dict(color="white"),
        legend=dict(x=0, y=1),
        height=500,
        width=1000,
    )

    fig.show()


def plot_multi_step_actual_vs_pred(
    y_true, y_pred, title="Multi-Step Forecast", sample_idx=0, dates=None
):
    """
    Plot actual vs predicted values for a multi-step forecast.
    Works with both single-step and multi-step forecasts.

    Parameters:
    - y_true: Ground truth values
    - y_pred: Predicted values
    - title: Plot title
    - sample_idx: Index of the sample to plot (only used for multi-step)
    - dates: Optional list of dates for x-axis (if None, uses indices)
    """
    if isinstance(y_true, (np.float32, np.float64, float)) or isinstance(
        y_pred, (np.float32, np.float64, float)
    ):
        print(
            "Warning: Single values provided instead of arrays. Cannot create a plot."
        )
        return

    is_multi_step = len(np.array(y_true).shape) > 1 and np.array(y_true).shape[1] > 1

    if is_multi_step:
        # Extract single sample from arrays for multi-step case
        true_sample = y_true[sample_idx]
        pred_sample = y_pred[sample_idx]
    else:
        # For single-step case, use the full arrays
        true_sample = np.array(y_true)
        pred_sample = np.array(y_pred)

    true_sample = np.array(true_sample)
    pred_sample = np.array(pred_sample)

    if dates is not None:
        x_values = dates
    else:
        x_values = list(range(len(true_sample)))

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=true_sample,
            mode="lines+markers",
            name="Actual",
            line=dict(color="deepskyblue", width=2),
            marker=dict(size=6),
            opacity=0.8,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=pred_sample,
            mode="lines+markers",
            name="Predicted",
            line=dict(color="magenta", width=2, dash="dot"),
            marker=dict(size=6),
            opacity=0.7,
        )
    )

    fig.update_layout(
        title=f"{title}" + (f" (Sample #{sample_idx})" if is_multi_step else ""),
        xaxis_title="Forecast Step" if dates is None else "Date",
        yaxis_title="Energy Consumption (MW)",
        template="plotly_dark",
        plot_bgcolor="black",
        paper_bgcolor="black",
        font=dict(color="white"),
        legend=dict(x=0, y=1),
        height=500,
        width=1000,
    )

    fig.show()


def plot_multi_sample_forecasts(y_true, y_pred, num_samples=3, title_prefix="Multi-Step Forecast"):
    """
    Plot multiple sample forecasts to compare model performance on different cases.

    Parameters:
    - y_true: Ground truth values of shape [samples, forecast_steps]
    - y_pred: Predicted values of shape [samples, forecast_steps]
    - num_samples: Number of random samples to plot
    - title_prefix: Prefix for plot titles
    """
    sample_indices = np.random.choice(
        len(y_true), size=min(num_samples, len(y_true)), replace=False
    )

    for idx in sample_indices:
        plot_multi_step_actual_vs_pred(
            y_true,
            y_pred,
            title=f"{title_prefix}",
            sample_idx=idx,
        )
