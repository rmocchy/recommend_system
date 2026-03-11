"""OpenJij SA parameter sidebar component.

Call this from any page as follows::

    from core.openjij_sidebar import OpenjijParams, openjij_sidebar

    openjij_params = openjij_sidebar()
    if openjij_params.run:
        from core.openjij_sa import run_openjij
        result = run_openjij(bqm, openjij_params)
"""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st


@dataclass
class OpenjijParams:
    num_reads: int
    num_sweeps: int
    beta_min: float   # 1/T_start  (low beta = high temperature = exploration)
    beta_max: float   # 1/T_end    (high beta = low temperature = exploitation)
    schedule: str     # "geometric" or "linear"
    seed: int | None
    run: bool

    @property
    def sampler_kwargs(self) -> dict:
        """Return kwargs ready to pass to openjij SASampler.sample().

        Note: openjij's ``schedule`` parameter expects a list of (beta, steps)
        tuples, not a string.  We therefore pass ``beta_min``/``beta_max`` and
        build the schedule ourselves based on ``self.schedule``.
        """
        import numpy as np

        if self.schedule == "geometric":
            betas = np.geomspace(self.beta_min, self.beta_max, self.num_sweeps)
        else:  # linear
            betas = np.linspace(self.beta_min, self.beta_max, self.num_sweeps)

        schedule = [[float(b), 1] for b in betas]

        return {
            "num_reads": self.num_reads,
            "schedule": schedule,
            **({"seed": self.seed} if self.seed is not None else {}),
        }


def openjij_sidebar() -> OpenjijParams:
    """
    Render OpenJij SA parameters in the left sidebar and return an OpenjijParams object.

    Returns
    -------
    OpenjijParams
    """
    with st.sidebar:
        st.header("🔧 OpenJij SA Parameters")

        num_reads = st.number_input(
            "Num reads (independent runs)",
            min_value=1, max_value=200, value=20, step=1,
            help="Number of independent SA runs. Best solution is reported.",
        )
        num_sweeps = st.number_input(
            "Num sweeps (MC steps per read)",
            min_value=100, max_value=100_000, value=1_000, step=100,
            help="Number of Monte Carlo steps per read.",
        )
        st.markdown("**Temperature range** (β = 1/T)")
        beta_min = st.number_input(
            "β_min  (1/T_start, high T = exploration)",
            min_value=1e-6, max_value=1.0, value=0.001, format="%.4f",
        )
        beta_max = st.number_input(
            "β_max  (1/T_end, low T = exploitation)",
            min_value=0.1, max_value=1000.0, value=10.0, step=1.0,
        )
        schedule = st.selectbox(
            "β schedule type",
            options=["geometric", "linear"],
            index=0,
            help="geometric: exponential cooling schedule. linear: linear cooling schedule.",
        )
        seed_raw = st.number_input(
            "Random seed (0 = random)",
            min_value=0, max_value=99999, value=42, step=1,
        )
        st.divider()
        run = st.button("▶ Run OpenJij SA", type="primary", use_container_width=True)

    return OpenjijParams(
        num_reads=int(num_reads),
        num_sweeps=int(num_sweeps),
        beta_min=float(beta_min),
        beta_max=float(beta_max),
        schedule=str(schedule),
        seed=int(seed_raw) if seed_raw != 0 else None,
        run=run,
    )
