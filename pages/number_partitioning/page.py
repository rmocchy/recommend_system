"""Streamlit ページ: 数分割問題 — QUBO × シミュレーテッドアニーリング。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st

from core.sa_sidebar import sa_sidebar
from pages.number_partitioning.input_ui import render_input
from pages.number_partitioning.output_ui import render_output

st.title("✂️ 数分割問題 — QUBO × シミュレーテッドアニーリング")
st.markdown(
    """
与えられた数列を 2 つのグループ A / B に分割し、各グループの合計を等しくする組み合わせを探します。  
QUBO にエンコードし、シミュレーテッドアニーリング (SA) をブラウザ上で実行します。
"""
)

# SA パラメータ (サイドバー)
sa_params = sa_sidebar()

st.divider()

# 入力 UI
input_result = render_input()

# 出力 UI
if input_result is not None:
    numbers, Q = input_result
    st.divider()
    render_output(numbers, Q, sa_params)

