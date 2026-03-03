"""数分割問題の可視化ヘルパー。"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_convergence(
    energy_history: list[float],
    best_history: list[float],
    temp_history: list[float],
) -> go.Figure:
    """エネルギー収束グラフと温度推移グラフを返す。"""
    steps = list(range(len(energy_history)))

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=("エネルギー推移", "温度推移"),
        vertical_spacing=0.12,
        row_heights=[0.65, 0.35],
    )

    fig.add_trace(
        go.Scatter(
            x=steps,
            y=energy_history,
            mode="lines",
            name="現在エネルギー",
            line=dict(color="#636EFA", width=1, dash="dot"),
            opacity=0.6,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=best_history,
            mode="lines",
            name="ベストエネルギー",
            line=dict(color="#EF553B", width=2),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=temp_history,
            mode="lines",
            name="温度 T",
            line=dict(color="#00CC96", width=2),
        ),
        row=2,
        col=1,
    )

    fig.update_yaxes(title_text="エネルギー E = x^T Q x", row=1, col=1)
    fig.update_yaxes(title_text="温度 T", row=2, col=1)
    fig.update_xaxes(title_text="ステップ", row=2, col=1)
    fig.update_layout(
        height=520,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=60, b=40),
    )
    return fig


def plot_partition(numbers: Sequence[float], best_x: np.ndarray) -> go.Figure:
    """分割結果を棒グラフで可視化する。"""
    labels = [f"n_{i}" for i in range(len(numbers))]
    group_labels = ["グループ A (x=1)" if xi == 1 else "グループ B (x=0)" for xi in best_x]
    colors = ["#636EFA" if xi == 1 else "#EF553B" for xi in best_x]

    fig = go.Figure()
    seen: set[str] = set()
    for i, (num, color, grp) in enumerate(zip(numbers, colors, group_labels)):
        fig.add_trace(
            go.Bar(
                x=[labels[i]],
                y=[num],
                name=grp,
                marker_color=color,
                showlegend=grp not in seen,
                legendgroup=grp,
            )
        )
        seen.add(grp)

    sum_A = sum(n for n, xi in zip(numbers, best_x) if xi == 1)
    sum_B = sum(n for n, xi in zip(numbers, best_x) if xi == 0)
    diff = abs(sum_A - sum_B)
    fig.update_layout(
        title=f"分割結果  |  Σ A = {sum_A:.2f}    Σ B = {sum_B:.2f}    差 = {diff:.4f}",
        xaxis_title="変数",
        yaxis_title="値",
        height=360,
        barmode="group",
        margin=dict(t=60, b=40),
    )
    return fig


def plot_qubo_matrix(Q: np.ndarray, numbers: Sequence[float]) -> go.Figure:
    """QUBO 行列をヒートマップで表示する。"""
    labels = [f"n_{i}" for i in range(len(numbers))]
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
        title="QUBO 行列 Q",
        height=380,
        margin=dict(t=60, b=40),
        yaxis=dict(autorange="reversed"),
    )
    return fig
