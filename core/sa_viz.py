"""Shared SA visualization helpers.

Provides reusable charts for any QUBO problem.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_sa_detail(
    energy_history: list[float],
    best_history: list[float],
    temp_history: list[float],
) -> go.Figure:
    """Return an energy convergence graph + temperature graph for SA."""
    steps = list(range(len(energy_history)))

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=("Energy History", "Temperature"),
        vertical_spacing=0.12,
        row_heights=[0.65, 0.35],
    )
    fig.add_trace(
        go.Scatter(
            x=steps, y=energy_history, mode="lines",
            name="Current Energy",
            line=dict(color="#636EFA", width=1, dash="dot"),
            opacity=0.6,
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=steps, y=best_history, mode="lines",
            name="Best Energy",
            line=dict(color="#EF553B", width=2),
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=steps, y=temp_history, mode="lines",
            name="Temperature T",
            line=dict(color="#00CC96", width=2),
        ),
        row=2, col=1,
    )
    fig.update_yaxes(title_text="Energy E = x^T Q x", row=1, col=1)
    fig.update_yaxes(title_text="Temperature T", row=2, col=1)
    fig.update_xaxes(title_text="Step", row=2, col=1)
    fig.update_layout(
        height=520,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=60, b=40),
    )
    return fig


def plot_qubo_matrix(Q: np.ndarray, var_labels: Sequence[str] | None = None) -> go.Figure:
    """Display a QUBO matrix as a heatmap."""
    n = Q.shape[0]
    labels = list(var_labels) if var_labels is not None else [f"x_{i}" for i in range(n)]
    fig = go.Figure(
        go.Heatmap(
            z=Q,
            x=labels,
            y=labels,
            colorscale="RdBu",
            zmid=0,
            text=np.round(Q, 1),
            texttemplate="%{text}",
        )
    )
    fig.update_layout(
        title="QUBO Matrix Q",
        height=380,
        margin=dict(t=60, b=40),
        yaxis=dict(autorange="reversed"),
    )
    return fig
