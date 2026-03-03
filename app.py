"""
QUBO シミュレーテッドアニーリング - Streamlit アプリ
問題: 数分割問題 (Number Partitioning Problem)
"""

import math
import random
import time

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ─────────────────────────────────────────
# ページ設定
# ─────────────────────────────────────────
st.set_page_config(
    page_title="QUBO シミュレーテッドアニーリング",
    page_icon="🌡️",
    layout="wide",
)

# ─────────────────────────────────────────
# QUBO ヘルパー関数
# ─────────────────────────────────────────

def build_number_partitioning_qubo(numbers: list[float]) -> np.ndarray:
    """
    数分割問題を QUBO 行列に変換する。
    目的関数: minimize (Σ_i (2x_i - 1) * n_i)^2 = x^T Q x + const
    
    QUBO 行列の要素:
        Q[i][i] = 4 * n_i * (n_i - S)
        Q[i][j] = 8 * n_i * n_j  (i ≠ j)
    ここで S = Σ n_i
    """
    n = len(numbers)
    nums = np.array(numbers, dtype=float)
    S = nums.sum()
    Q = np.zeros((n, n))
    for i in range(n):
        Q[i, i] = 4 * nums[i] * (nums[i] - S)
        for j in range(i + 1, n):
            Q[i, j] = 8 * nums[i] * nums[j]
            Q[j, i] = Q[i, j]
    return Q


def qubo_energy(x: np.ndarray, Q: np.ndarray) -> float:
    """QUBO エネルギー E = x^T Q x を計算する。"""
    return float(x @ Q @ x)


def simulated_annealing(
    Q: np.ndarray,
    T_init: float,
    T_min: float,
    cooling_rate: float,
    max_iter: int,
    seed: int | None = None,
) -> dict:
    """
    シミュレーテッドアニーリングで QUBO を最小化する。

    Returns
    -------
    dict:
        best_x          : 最良解 (ndarray)
        best_energy     : 最良エネルギー
        energy_history  : 各ステップのエネルギー推移
        best_history    : 各ステップのベストエネルギー推移
        temp_history    : 各ステップの温度推移
        n_iter          : 実際のイテレーション数
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    n = Q.shape[0]
    x = np_rng.integers(0, 2, n).astype(float)
    current_energy = qubo_energy(x, Q)
    best_x = x.copy()
    best_energy = current_energy

    T = T_init
    energy_history = [current_energy]
    best_history = [best_energy]
    temp_history = [T]

    for step in range(max_iter):
        if T <= T_min:
            break

        # ランダムにビットを 1 つフリップ
        flip_idx = rng.randint(0, n - 1)
        x_new = x.copy()
        x_new[flip_idx] = 1.0 - x_new[flip_idx]

        new_energy = qubo_energy(x_new, Q)
        delta_E = new_energy - current_energy

        # メトロポリス基準で受理/棄却
        if delta_E < 0 or rng.random() < math.exp(-delta_E / T):
            x = x_new
            current_energy = new_energy
            if current_energy < best_energy:
                best_x = x.copy()
                best_energy = current_energy

        T *= cooling_rate
        energy_history.append(current_energy)
        best_history.append(best_energy)
        temp_history.append(T)

    return {
        "best_x": best_x,
        "best_energy": best_energy,
        "energy_history": energy_history,
        "best_history": best_history,
        "temp_history": temp_history,
        "n_iter": len(energy_history) - 1,
    }


# ─────────────────────────────────────────
# 可視化ヘルパー
# ─────────────────────────────────────────

def plot_convergence(energy_history: list, best_history: list, temp_history: list):
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

    # エネルギー推移
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

    # 温度推移
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=temp_history,
            mode="lines",
            name="温度 T",
            line=dict(color="#00CC96", width=2),
            showlegend=True,
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


def plot_partition(numbers: list[float], best_x: np.ndarray):
    """分割結果を棒グラフで可視化する。"""
    labels = [f"n_{i}" for i in range(len(numbers))]
    colors = ["#636EFA" if xi == 1 else "#EF553B" for xi in best_x]
    group_labels = ["グループ A (x=1)" if xi == 1 else "グループ B (x=0)" for xi in best_x]

    fig = go.Figure()
    for i, (num, color, grp) in enumerate(zip(numbers, colors, group_labels)):
        fig.add_trace(
            go.Bar(
                x=[labels[i]],
                y=[num],
                name=grp,
                marker_color=color,
                showlegend=(i == 0 or (i > 0 and group_labels[i] != group_labels[i - 1])),
                legendgroup=grp,
            )
        )

    sum_A = sum(n for n, xi in zip(numbers, best_x) if xi == 1)
    sum_B = sum(n for n, xi in zip(numbers, best_x) if xi == 0)
    fig.update_layout(
        title=f"分割結果  |  Σ A = {sum_A:.2f}    Σ B = {sum_B:.2f}    差 = {abs(sum_A - sum_B):.4f}",
        xaxis_title="変数",
        yaxis_title="値",
        height=360,
        barmode="group",
        margin=dict(t=60, b=40),
    )
    return fig


def plot_qubo_matrix(Q: np.ndarray, numbers: list[float]):
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


# ─────────────────────────────────────────
# UI
# ─────────────────────────────────────────

st.title("🌡️ QUBO シミュレーテッドアニーリング")
st.markdown(
    """
**問題**: **数分割問題** (Number Partitioning Problem)  
与えられた数列を 2 つのグループ A / B に分割し、各グループの合計を等しくする組み合わせを探します。  
QUBO にエンコードし、シミュレーテッドアニーリング (SA) でフロントエンド上で解きます。
"""
)
st.divider()

# ── サイドバー ──────────────────────────
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

    if preset == "カスタム入力":
        raw = st.text_input(
            "数列 (カンマ区切り)",
            value="3, 8, 5, 2, 7, 4",
            help="正の整数または小数をカンマ区切りで入力",
        )
        try:
            numbers = [float(v.strip()) for v in raw.split(",") if v.strip()]
            if len(numbers) < 2:
                st.error("2 つ以上の数値を入力してください。")
                numbers = [3, 8, 5, 2, 7, 4]
        except ValueError:
            st.error("数値として解釈できない文字が含まれています。")
            numbers = [3, 8, 5, 2, 7, 4]
    else:
        preset_map = {
            "小: [3, 1, 4, 1, 5]": [3, 1, 4, 1, 5],
            "中: [10, 7, 3, 6, 9, 2, 8, 4]": [10, 7, 3, 6, 9, 2, 8, 4],
            "大: [17, 5, 22, 11, 8, 14, 3, 19, 6, 13]": [17, 5, 22, 11, 8, 14, 3, 19, 6, 13],
        }
        numbers = preset_map[preset]
        st.info(f"数列: {numbers}")

    st.divider()
    st.header("🔧 SA パラメータ")

    T_init = st.number_input("初期温度 T₀", min_value=1.0, max_value=1e6, value=1000.0, step=100.0)
    T_min = st.number_input("最低温度 T_min", min_value=1e-10, max_value=10.0, value=1e-4, format="%.1e")
    cooling_rate = st.slider("冷却率 α (T ← α·T)", min_value=0.900, max_value=0.9999, value=0.995, step=0.001, format="%.4f")
    max_iter = st.number_input("最大イテレーション数", min_value=100, max_value=500_000, value=10_000, step=1000)
    seed = st.number_input("乱数シード (0=ランダム)", min_value=0, max_value=99999, value=42, step=1)
    use_seed = seed != 0

    st.divider()
    run_btn = st.button("▶ SA を実行", type="primary", use_container_width=True)

# ── メインエリア ──────────────────────────

# QUBO 行列を構築して常に表示
Q = build_number_partitioning_qubo(numbers)

col_info1, col_info2, col_info3 = st.columns(3)
with col_info1:
    st.metric("変数 (ビット) 数", len(numbers))
with col_info2:
    st.metric("数列の合計 Σ", f"{sum(numbers):.2f}")
with col_info3:
    st.metric("理想的なグループ合計", f"{sum(numbers)/2:.2f}")

with st.expander("📐 QUBO 行列を確認", expanded=False):
    tab_heat, tab_raw = st.tabs(["ヒートマップ", "生の値"])
    with tab_heat:
        st.plotly_chart(plot_qubo_matrix(Q, numbers), use_container_width=True)
    with tab_raw:
        st.dataframe(
            Q,
            column_config={str(i): st.column_config.NumberColumn(f"n_{i}", format="%.1f") for i in range(len(numbers))},
        )

st.divider()

# ── SA 実行 ──
if run_btn:
    with st.spinner("シミュレーテッドアニーリングを実行中…"):
        t0 = time.perf_counter()
        result = simulated_annealing(
            Q=Q,
            T_init=T_init,
            T_min=T_min,
            cooling_rate=cooling_rate,
            max_iter=int(max_iter),
            seed=int(seed) if use_seed else None,
        )
        elapsed = time.perf_counter() - t0

    best_x = result["best_x"].astype(int)
    best_energy = result["best_energy"]
    n_iter = result["n_iter"]

    sum_A = sum(n for n, xi in zip(numbers, best_x) if xi == 1)
    sum_B = sum(n for n, xi in zip(numbers, best_x) if xi == 0)
    diff = abs(sum_A - sum_B)

    # ── 結果サマリー ──
    st.subheader("📊 実行結果")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("最良エネルギー E*", f"{best_energy:.4f}")
    m2.metric("グループ間の差 |ΣA - ΣB|", f"{diff:.4f}")
    m3.metric("実行ステップ数", f"{n_iter:,}")
    m4.metric("実行時間", f"{elapsed*1000:.1f} ms")

    # 解ベクトルの表示
    solution_str = "  ".join(
        [f"`n_{i}={numbers[i]:.0f}` → **{'A' if xi==1 else 'B'}**" for i, xi in enumerate(best_x)]
    )
    st.markdown(f"**解ベクトル x** : {solution_str}")

    if diff < 1e-6:
        st.success("✅ 完全分割に成功しました！ΣA = ΣB")
    elif diff <= sum(numbers) * 0.05:
        st.warning(f"⚠️ 差 {diff:.4f} の近似解が見つかりました。")
    else:
        st.error(f"❌ 最良解でも差が {diff:.4f} あります。パラメータを調整してみてください。")

    # ── グラフ ──
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
