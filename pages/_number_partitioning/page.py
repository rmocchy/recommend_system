"""Streamlit ページ: 数分割問題 — QUBO × シミュレーテッドアニーリング。"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import streamlit as st

from core.sa import simulated_annealing
from pages._number_partitioning.default_qubo import DEFAULT_QUBO_CODE
from pages._number_partitioning.viz import (
    plot_convergence,
    plot_partition,
    plot_qubo_matrix,
)

st.title("✂️ 数分割問題 — QUBO × シミュレーテッドアニーリング")
st.markdown(
    """
与えられた数列を 2 つのグループ A / B に分割し、各グループの合計を等しくする組み合わせを探します。  
QUBO にエンコードし、シミュレーテッドアニーリング (SA) をブラウザ上で実行します。
"""
)

# ─────────────────────────────────────────
# QUBO コードエディタ
# ─────────────────────────────────────────
with st.expander("🖊️ QUBO 定式化コードを編集", expanded=False):
    st.markdown(
        """
`PARAMS` 辞書にパラメータを追加すると、サイドバーのコントロールが**自動で更新**されます。  
`build_qubo(numbers, params)` 関数を書き換えて定式化を変更してください。
"""
    )
    qubo_code: str = st.text_area(
        "QUBO コード",
        value=st.session_state.get("qubo_code", DEFAULT_QUBO_CODE),
        height=420,
        key="qubo_code_editor",
        label_visibility="collapsed",
    ) or DEFAULT_QUBO_CODE
    col_apply, col_reset = st.columns([1, 1])
    with col_apply:
        apply_btn = st.button("✅ コードを適用", type="primary", use_container_width=True)
    with col_reset:
        if st.button("🔄 デフォルトに戻す", use_container_width=True):
            st.session_state["qubo_code"] = DEFAULT_QUBO_CODE
            st.session_state.pop("qubo_params_cache", None)
            st.rerun()

if apply_btn:
    st.session_state["qubo_code"] = qubo_code
    st.session_state.pop("qubo_params_cache", None)

current_code: str = st.session_state.get("qubo_code", DEFAULT_QUBO_CODE)

# ─────────────────────────────────────────
# コードを exec して PARAMS と build_qubo を取得
# ─────────────────────────────────────────
exec_ns: dict = {}
code_error: str | None = None
try:
    exec(compile(current_code, "<qubo_editor>", "exec"), exec_ns)  # noqa: S102
    if "PARAMS" not in exec_ns:
        code_error = "PARAMS 辞書が定義されていません。"
    elif "build_qubo" not in exec_ns:
        code_error = "build_qubo 関数が定義されていません。"
except Exception:
    code_error = traceback.format_exc()

if code_error:
    st.error(f"**コードエラー:**\n```\n{code_error}\n```")
    st.stop()

PARAMS: dict = exec_ns["PARAMS"]
build_qubo_fn = exec_ns["build_qubo"]

# ─────────────────────────────────────────
# サイドバー — 問題設定
# ─────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 問題の設定")

    preset = st.selectbox(
        "プリセット数列",
        options=[
            "カスタム入力",
            "小: [3, 1, 4, 1, 5]",
            "中: [10, 7, 3, 6, 9, 2, 8, 4]",
            "大: [17, 5, 22, 11, 8, 14, 3, 19, 6, 13]",
        ],
    )

    preset_map: dict[str, list[float]] = {
        "小: [3, 1, 4, 1, 5]": [3, 1, 4, 1, 5],
        "中: [10, 7, 3, 6, 9, 2, 8, 4]": [10, 7, 3, 6, 9, 2, 8, 4],
        "大: [17, 5, 22, 11, 8, 14, 3, 19, 6, 13]": [17, 5, 22, 11, 8, 14, 3, 19, 6, 13],
    }

    numbers: list[float]
    if preset == "カスタム入力":
        raw = st.text_input(
            "数列 (カンマ区切り)",
            value="3, 8, 5, 2, 7, 4",
        )
        try:
            numbers = [float(v.strip()) for v in raw.split(",") if v.strip()]
            if len(numbers) < 2:
                st.error("2 つ以上の数値を入力してください。")
                numbers = [3.0, 8.0, 5.0, 2.0, 7.0, 4.0]
        except ValueError:
            st.error("数値として解析できません。")
            numbers = [3.0, 8.0, 5.0, 2.0, 7.0, 4.0]
    else:
        numbers = preset_map[preset]
        st.info(f"数列: {numbers}")

    # ── QUBO パラメータ (PARAMS から動的生成) ──
    if PARAMS:
        st.divider()
        st.header("🔢 QUBO パラメータ")

    qubo_param_values: dict = {}
    for key, spec in PARAMS.items():
        label = spec.get("label", key)
        ptype = spec.get("type", "float")
        default = spec.get("default", 1.0)
        pmin = spec.get("min", 0.0)
        pmax = spec.get("max", 10.0)
        step = spec.get("step", 0.1 if ptype == "float" else 1)

        if ptype == "float":
            qubo_param_values[key] = st.slider(
                label,
                min_value=float(pmin),
                max_value=float(pmax),
                value=float(default),
                step=float(step),
                key=f"qparam_{key}",
            )
        elif ptype == "int":
            qubo_param_values[key] = st.slider(
                label,
                min_value=int(pmin),
                max_value=int(pmax),
                value=int(default),
                step=int(step),
                key=f"qparam_{key}",
            )
        else:
            qubo_param_values[key] = st.text_input(label, value=str(default), key=f"qparam_{key}")

    # ── SA パラメータ ──
    st.divider()
    st.header("🔧 SA パラメータ")

    T_init = st.number_input("初期温度 T₀", min_value=1.0, max_value=1e6, value=1000.0, step=100.0)
    T_min = st.number_input("最低温度 T_min", min_value=1e-10, max_value=10.0, value=1e-4, format="%.1e")
    cooling_rate = st.slider(
        "冷却率 α (T ← α·T)",
        min_value=0.900, max_value=0.9999,
        value=0.995, step=0.001, format="%.4f",
    )
    max_iter = st.number_input(
        "最大イテレーション数",
        min_value=100, max_value=500_000, value=10_000, step=1000,
    )
    seed = st.number_input("乱数シード (0=ランダム)", min_value=0, max_value=99999, value=42, step=1)

    st.divider()
    run_btn = st.button("▶ SA を実行", type="primary", use_container_width=True)

# ─────────────────────────────────────────
# QUBO 行列を構築
# ─────────────────────────────────────────
st.divider()
Q_error: str | None = None
Q: np.ndarray | None = None
try:
    Q = build_qubo_fn(numbers, qubo_param_values)
    if not isinstance(Q, np.ndarray) or Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
        Q_error = "build_qubo は正方な np.ndarray を返す必要があります。"
except Exception:
    Q_error = traceback.format_exc()

if Q_error or Q is None:
    st.error(f"**QUBO 構築エラー:**\n```\n{Q_error}\n```")
    st.stop()

col1, col2, col3 = st.columns(3)
col1.metric("変数 (ビット) 数", len(numbers))
col2.metric("数列の合計 Σ", f"{sum(numbers):.2f}")
col3.metric("理想的なグループ合計", f"{sum(numbers) / 2:.2f}")

with st.expander("📐 QUBO 行列を確認", expanded=False):
    tab_heat, tab_raw = st.tabs(["ヒートマップ", "生の値"])
    with tab_heat:
        st.plotly_chart(plot_qubo_matrix(Q, numbers), use_container_width=True)
    with tab_raw:
        import pandas as pd
        labels = [f"n_{i}" for i in range(len(numbers))]
        st.dataframe(pd.DataFrame(Q, index=labels, columns=labels))

# ─────────────────────────────────────────
# SA 実行
# ─────────────────────────────────────────
if run_btn:
    with st.spinner("シミュレーテッドアニーリングを実行中…"):
        t0 = time.perf_counter()
        result = simulated_annealing(
            Q=Q,
            T_init=float(T_init),
            T_min=float(T_min),
            cooling_rate=float(cooling_rate),
            max_iter=int(max_iter),
            seed=int(seed) if seed != 0 else None,
        )
        elapsed = time.perf_counter() - t0

    best_x = result["best_x"].astype(int)
    best_energy: float = result["best_energy"]
    n_iter: int = result["n_iter"]

    sum_A = sum(n for n, xi in zip(numbers, best_x) if xi == 1)
    sum_B = sum(n for n, xi in zip(numbers, best_x) if xi == 0)
    diff = abs(sum_A - sum_B)

    st.subheader("📊 実行結果")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("最良エネルギー E*", f"{best_energy:.4f}")
    m2.metric("グループ間の差 |ΣA − ΣB|", f"{diff:.4f}")
    m3.metric("実行ステップ数", f"{n_iter:,}")
    m4.metric("実行時間", f"{elapsed * 1000:.1f} ms")

    solution_str = "  ".join(
        [f"`n_{i}={numbers[i]:.0f}` → **{'A' if xi == 1 else 'B'}**" for i, xi in enumerate(best_x)]
    )
    st.markdown(f"**解ベクトル x** : {solution_str}")

    if diff < 1e-6:
        st.success("✅ 完全分割に成功しました！ΣA = ΣB")
    elif diff <= sum(numbers) * 0.05:
        st.warning(f"⚠️ 差 {diff:.4f} の近似解が見つかりました。")
    else:
        st.error(f"❌ 最良解でも差が {diff:.4f} あります。パラメータを調整してみてください。")

    col_g1, col_g2 = st.columns([3, 2])
    with col_g1:
        st.plotly_chart(
            plot_convergence(result["energy_history"], result["best_history"], result["temp_history"]),
            use_container_width=True,
        )
    with col_g2:
        st.plotly_chart(plot_partition(numbers, best_x), use_container_width=True)

else:
    st.info("👈 サイドバーでパラメータを設定し、**SA を実行** ボタンを押してください。")
