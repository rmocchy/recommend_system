"""Number Partitioning — QUBO formulation module (PyQUBO)."""

from __future__ import annotations

import numpy as np
from pyqubo import Array, Model # type: ignore
import dimod

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


def build_qubo(numbers: list[float], lam: float = 1.0) -> Model:
    """
    Build a BinaryQuadraticModel for the number partitioning problem using PyQUBO.

    Hamiltonian:
        H = λ * (Σ_i (2x_i - 1) * n_i)^2

    PyQUBO compiles this symbolic expression and returns a dimod BQM
    whose coefficients match the analytic expansion:
        linear bias   : λ * 4 * n_i * (n_i - S)
        quadratic bias: λ * 8 * n_i * n_j   (i < j)
        offset        : λ * S^2

    where S = Σ_i n_i.
    """
    nums = np.array(numbers, dtype=float)
    n = len(nums)

    # ── PyQUBO symbolic formulation ──────────────────────────────
    x = Array.create('x', shape=n, vartype='BINARY')

    # H = λ * (Σ_i (2x_i − 1) * n_i)^2  (lam は float — Placeholder 不要)
    delta = sum((2 * x[i] - 1) * float(nums[i]) for i in range(n))
    H = lam * delta * delta

    return H.compile() # type: ignore


def bqm_to_numpy(bqm: dimod.BinaryQuadraticModel) -> np.ndarray:
    """
    Convert a BinaryQuadraticModel to a symmetric QUBO numpy matrix.

    Variables are sorted and mapped to row/column indices.
    The offset (constant term) is not included.
    """
    labels = sorted(bqm.variables, key=lambda v: int(str(v).split('[')[1].rstrip(']')))  # type: ignore[type-var]
    idx = {v: i for i, v in enumerate(labels)}
    n = len(labels)
    Q = np.zeros((n, n))

    for v, bias in bqm.linear.items():
        Q[idx[v], idx[v]] = bias

    for (u, v), bias in bqm.quadratic.items():
        i, j = idx[u], idx[v]
        Q[i, j] = bias
        Q[j, i] = bias

    return Q
