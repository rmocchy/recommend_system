"""Number Partitioning — QUBO formulation module."""

from __future__ import annotations

import numpy as np

# QUBO parameter definition
# Each entry holds: type / label / default / min / max / step
PARAMS: dict = {
    "lam": {
        "type": "float",
        "label": "Penalty coefficient λ",
        "default": 1.0,
        "min": 0.1,
        "max": 10.0,
        "step": 0.1,
    },
}


def build_qubo(numbers: list[float], lam: float = 1.0) -> np.ndarray:
    """
    Build the QUBO matrix for the number partitioning problem.

    Objective: λ * (Σ_i (2x_i - 1) * n_i)^2
    Q[i][i] = λ * 4 * n_i * (n_i - S)
    Q[i][j] = λ * 8 * n_i * n_j  (i ≠ j)
    """
    nums = np.array(numbers, dtype=float)
    n = len(nums)
    S = nums.sum()
    Q = np.zeros((n, n))
    for i in range(n):
        Q[i, i] = lam * 4.0 * nums[i] * (nums[i] - S)
        for j in range(i + 1, n):
            Q[i, j] = lam * 8.0 * nums[i] * nums[j]
            Q[j, i] = Q[i, j]
    return Q
