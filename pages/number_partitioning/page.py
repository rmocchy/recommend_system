"""Streamlit page: Number Partitioning — QUBO × Simulated Annealing."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st

from core.sa_sidebar import sa_sidebar
from pages.number_partitioning.input_ui import render_input
from pages.number_partitioning.output_ui import render_output

st.title("✂️ Number Partitioning — QUBO × Simulated Annealing")
st.markdown(
    """
Given a list of numbers, find a way to split them into two groups (A / B)  
so that the sum of each group is as equal as possible.  
The problem is encoded as QUBO and solved with Simulated Annealing (SA) in the browser.
"""
)

# SA parameters (sidebar)
sa_params = sa_sidebar()

st.divider()

# Input UI
input_result = render_input()

# Output UI
if input_result is not None:
    numbers, Q = input_result
    st.divider()
    render_output(numbers, Q, sa_params)

