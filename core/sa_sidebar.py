"""Shared SA parameter sidebar component.

Call this from any page as follows:

    from core.sa_sidebar import sa_sidebar

    sa_params = sa_sidebar()
    if sa_params.run:
        result = simulated_annealing(Q, **sa_params.sa)
"""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st


@dataclass
class SAParams:
    T_init: float
    T_min: float
    cooling_rate: float
    max_iter: int
    seed: int | None
    timeout_sec: float
    run: bool


def sa_sidebar() -> SAParams:
    """
    Render SA parameters in the left sidebar and return a SAParams object.

    Returns
    -------
    SAParams
    """
    with st.sidebar:
        st.header("🔧 SA Parameters")

        T_init = st.number_input(
            "Initial temperature T₀",
            min_value=1.0, max_value=1e6, value=1000.0, step=100.0,
        )
        T_min = st.number_input(
            "Min temperature T_min",
            min_value=1e-10, max_value=10.0, value=1e-4, format="%.1e",
        )
        cooling_rate = st.slider(
            "Cooling rate α (T ← α·T)",
            min_value=0.900, max_value=0.9999,
            value=0.995, step=0.001, format="%.4f",
        )
        max_iter = st.number_input(
            "Max iterations",
            min_value=100, max_value=500_000, value=10_000, step=1000,
        )
        seed_raw = st.number_input(
            "Random seed (0 = random)",
            min_value=0, max_value=99999, value=42, step=1,
        )
        timeout_sec = st.number_input(
            "Timeout (seconds)",
            min_value=1, max_value=60, value=10, step=1,
            help="SA will stop automatically if it runs longer than this.",
        )
        st.divider()
        run = st.button("▶ Run SA", type="primary", use_container_width=True)

    return SAParams(
        T_init=float(T_init),
        T_min=float(T_min),
        cooling_rate=float(cooling_rate),
        max_iter=int(max_iter),
        seed=int(seed_raw) if seed_raw != 0 else None,
        timeout_sec=float(timeout_sec),
        run=run,
    )
