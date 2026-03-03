"""数分割問題 — QUBO 変数入力 UI。

数列の入力、QUBO パラメータの調整、QUBO 行列の構築・表示を担当する。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from core.sa_viz import plot_qubo_matrix
from pages.number_partitioning.qubo import PARAMS, build_qubo

_PRESETS: dict[str, list[float]] = {
    "小: [3, 1, 4, 1, 5]": [3, 1, 4, 1, 5],
    "中: [10, 7, 3, 6, 9, 2, 8, 4]": [10, 7, 3, 6, 9, 2, 8, 4],
    "大: [17, 5, 22, 11, 8, 14, 3, 19, 6, 13]": [17, 5, 22, 11, 8, 14, 3, 19, 6, 13],
}


def render_input() -> tuple[list[float], np.ndarray] | None:
    """
    数列入力・ QUBO パラメータ入力・ QUBO 行列の構築と表示を描画する。

    Returns
    -------
    (numbers, Q) または None (構築エラー時)
    """
    # ── 数列入力 ────────────────────────────────────────
    st.subheader("📥 入力データ")

    preset_options = ["カスタム入力"] + list(_PRESETS.keys())
    preset = st.selectbox("プリセット数列", options=preset_options)

    numbers: list[float]
    if preset == "カスタム入力":
        raw = st.text_input("数列 (カンマ区切り)", value="3, 8, 5, 2, 7, 4")
        try:
            numbers = [float(v.strip()) for v in raw.split(",") if v.strip()]
            if len(numbers) < 2:
                st.error("2 つ以上の数値を入力してください。")
                numbers = [3.0, 8.0, 5.0, 2.0, 7.0, 4.0]
        except ValueError:
            st.error("数値として解析できません。")
            numbers = [3.0, 8.0, 5.0, 2.0, 7.0, 4.0]
    else:
        numbers = _PRESETS[preset]
        st.info(f"数列: {numbers}")

    # ── QUBO パラメータスライダー ────────────────────────
    st.subheader("🔢 QUBO パラメータ")
    spec = PARAMS["lam"]
    lam: float = st.slider(
        spec["label"],
        min_value=float(spec["min"]),
        max_value=float(spec["max"]),
        value=float(spec["default"]),
        step=float(spec["step"]),
        key="np__lam",
    )

    # ── QUBO 行列の構築 ───────────────────────────
    Q = build_qubo(numbers, lam=lam)

    # ── メトリクス & 行列表示 ────────────────────────
    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric("変数 (ビット) 数", len(numbers))
    c2.metric("数列の合計 Σ", f"{sum(numbers):.2f}")
    c3.metric("理想的なグループ合計", f"{sum(numbers) / 2:.2f}")

    with st.expander("📐 QUBO 行列を確認", expanded=False):
        tab_heat, tab_raw = st.tabs(["ヒートマップ", "生の値"])
        labels = [f"n_{i}" for i in range(len(numbers))]
        with tab_heat:
            st.plotly_chart(plot_qubo_matrix(Q, var_labels=labels), use_container_width=True)
        with tab_raw:
            st.dataframe(pd.DataFrame(Q, index=labels, columns=labels))

    return numbers, Q

    c1.metric("変数 (ビット) 数", len(numbers))
    c2.metric("数列の合計 Σ", f"{sum(numbers):.2f}")
    c3.metric("理想的なグループ合計", f"{sum(numbers) / 2:.2f}")

    with st.expander("📐 QUBO 行列を確認", expanded=False):
        tab_heat, tab_raw = st.tabs(["ヒートマップ", "生の値"])
        labels = [f"n_{i}" for i in range(len(numbers))]
        with tab_heat:
            st.plotly_chart(plot_qubo_matrix(Q, var_labels=labels), use_container_width=True)
        with tab_raw:
            st.dataframe(pd.DataFrame(Q, index=labels, columns=labels))

    return numbers, qubo_param_values, Q
