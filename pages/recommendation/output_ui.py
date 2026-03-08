"""Recommendation System — SA execution & rich output UI."""

from __future__ import annotations

import time

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core.sa import simulated_annealing
from core.sa_sidebar import SAParams
from core.sa_viz import plot_sa_detail
from pages.recommendation.cards import compact_card_html, item_card_html
from pages.recommendation.items_data import Item

# ── Main output UI ────────────────────────────────────
def render_output(
    items: list[Item],
    budget: float,
    Q: np.ndarray,
    sa_params: SAParams,
) -> None:
    """
    Run SA and display results in a rich shopping-site style layout.
    """
    if not sa_params.run:
        st.info("👈 Press the **Run SA** button in the sidebar.")
        return

    # ── Run SA ────────────────────────────────────────
    with st.spinner("🤖 Computing recommendations with SA…"):
        t0 = time.perf_counter()
        result = simulated_annealing(
            Q=Q,
            T_init=sa_params.T_init,
            T_min=sa_params.T_min,
            cooling_rate=sa_params.cooling_rate,
            max_iter=sa_params.max_iter,
            seed=sa_params.seed,
            timeout_sec=sa_params.timeout_sec,
        )
        elapsed = time.perf_counter() - t0

    if result["timed_out"]:
        st.warning(f"⏱ SA stopped due to timeout ({sa_params.timeout_sec:.0f}s). Showing best solution found so far.")

    best_x = result["best_x"].astype(int)
    best_energy: float = result["best_energy"]

    recommended = [it for it, xi in zip(items, best_x) if xi == 1]
    not_recommended = [it for it, xi in zip(items, best_x) if xi == 0]

    total_price = sum(it.price for it in recommended)

    # ── KPI bar ──────────────────────────────────────────
    k1, k2, k3 = st.columns(3)
    k1.metric("🛒 Recommended Items", f"{len(recommended)}")
    k2.metric("💰 Total Price", f"${total_price:,}")
    k3.metric("⚡ SA Runtime", f"{elapsed * 1000:.0f} ms")

    st.divider()

    # ── Recommended product grid ──────────────────────────────
    st.subheader(f"🛒 Recommended Products ({len(recommended)})")

    if not recommended:
        st.warning("No products were recommended. Try adjusting the SA parameters or QUBO settings.")
    else:
        # Sort by rating and display
        rec_ranked = sorted(recommended, key=lambda x: -x.score)
        cols_per_row = min(4, len(rec_ranked))
        for row_start in range(0, len(rec_ranked), cols_per_row):
            row_items = rec_ranked[row_start: row_start + cols_per_row]
            cols = st.columns(cols_per_row)
            for col, it in zip(cols, row_items):
                with col:
                    st.markdown(
                        item_card_html(it),
                        unsafe_allow_html=True,
                    )
            st.write("")  # gap

    st.divider()

    # ── Cart total bar ─────────────────────────────────────────
    budget_pct = min(total_price / budget * 100, 150) if budget > 0 else 0
    bar_color = "#e74c3c" if total_price > budget else "#2ecc71"
    st.markdown(f"""
<div style="background:#f8f9fa;border-radius:12px;padding:16px 20px;margin-bottom:16px;">
  <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
    <span style="font-weight:700;color:#333;">🛒 Cart Total</span>
    <span style="font-size:20px;font-weight:800;color:{bar_color};">${total_price:,}</span>
  </div>
  <div style="background:#e9ecef;border-radius:8px;height:12px;overflow:hidden;">
    <div style="
      background:linear-gradient(90deg,{bar_color},{bar_color}aa);
      width:{min(budget_pct,100):.1f}%;height:12px;border-radius:8px;
      transition:width .4s;
    "></div>
  </div>
  <div style="display:flex;justify-content:space-between;font-size:12px;color:#888;margin-top:4px;">
    <span>$0</span>
    <span>Budget Limit ${budget:,.0f}</span>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── SA convergence graph ──────────────────────────────────
    with st.expander("📈 Show SA Convergence Graph", expanded=False):
        st.plotly_chart(
            plot_sa_detail(result["energy_history"], result["best_history"], result["temp_history"]),
            use_container_width=True,
        )
        c1, c2 = st.columns(2)
        c1.metric("Best Energy E*", f"{best_energy:.4f}")
        c2.metric("Iterations", f"{result['n_iter']:,}")