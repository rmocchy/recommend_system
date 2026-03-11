"""Recommendation System — QUBO formulation module (PyQUBO)."""

from __future__ import annotations

import numpy as np
from pyqubo import Array, Model # type: ignore
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pages.recommendation.items_data import Item


# ── QUBO parameter definitions (for slider UI) ──────────────────
PARAMS: dict[str, dict] = {
    "lambda_required": {
        "type": "float",
        "label": "Required Category λ (λ_req)",
        "default": 5.0,
        "min": 0.1,
        "max": 20.0,
        "step": 0.1,
    },
    "lambda_optional": {
        "type": "float",
        "label": "Optional Category λ (λ_opt)",
        "default": 5.0,
        "min": 0.1,
        "max": 20.0,
        "step": 0.1,
    },
    "lambda_budget": {
        "type": "float",
        "label": "Budget Penalty λ (×10⁻⁶)",
        "default": 1.0,
        "min": 0.0,
        "max": 50.0,
        "step": 0.5,
    },
    "lambda_score": {
        "type": "float",
        "label": "Score Weight λ (λ_score)",
        "default": 1.0,
        "min": 0.0,
        "max": 10.0,
        "step": 0.1,
    },
}


def build_bqm(
    items: list["Item"],
    required_categories: list[str],
    optional_categories: list[str],
    budget_target: float,
    params: dict,
) -> Model:
    """
    Build a PyQUBO Model for the recommendation problem.

    Hamiltonian:
        H = E_req + E_opt + E_bud + E_score

        E_req   = λ_req · Σ_{c ∈ C_req} (Σ_{i ∈ c} x_i − 1)²
        E_opt   = λ_opt · Σ_{c ∈ C_opt} (Σ_{i ∈ c} x_i − 1/2)²
        E_bud   = λ_bud · (Σ_i p_i x_i − B)²
        E_score = −λ_s · Σ_i s_i x_i
    """
    lam_req   = float(params.get("lambda_required", 5.0))
    lam_opt   = float(params.get("lambda_optional", 5.0))
    lam_bud   = float(params.get("lambda_budget", 1.0)) * 1e-6  # UI uses ×10⁻⁶ scale
    lam_score = float(params.get("lambda_score", 1.0))

    n = len(items)
    x = Array.create('x', shape=n, vartype='BINARY')

    # ── E_req : exactly-one per required category ─────────────────
    # λ_req · (Σ_{i ∈ c} x_i − 1)²
    E_req = 0
    for cat in required_categories:
        if any(it.category == cat for it in items):
            S_c = sum(x[i] for i, it in enumerate(items) if it.category == cat)
            delta = S_c - 1
            E_req += lam_req * delta * delta

    # ── E_opt : at-most-one per optional category ─────────────────
    # λ_opt · (Σ_{i ∈ c} x_i − 1/2)²  → cost=0 for k=0 or k=1, penalises k≥2
    E_opt = 0
    for cat in optional_categories:
        if any(it.category == cat for it in items):
            S_c = sum(x[i] for i, it in enumerate(items) if it.category == cat)
            delta = S_c - 0.5
            E_opt += lam_opt * delta * delta

    # ── E_bud : budget constraint ────────────────────────────────
    # λ_bud · (Σ_i p_i x_i − B)²
    spend = sum(float(items[i].price) * x[i] for i in range(n))
    diff = spend - budget_target
    E_bud = lam_bud * diff * diff

    # ── E_score : score maximisation ─────────────────────────────
    # −λ_s · Σ_i s_i x_i
    E_score = -lam_score * sum(float(items[i].score) * x[i] for i in range(n))

    H = E_req + E_opt + E_bud + E_score
    return H.compile() # type: ignore


def build_qubo_matrix(
    items: list["Item"],
    required_categories: list[str],
    optional_categories: list[str],
    budget_target: float,
    params: dict,
) -> np.ndarray:
    """Return the QUBO as a symmetric numpy matrix (used for QUBO preview)."""
    model = build_bqm(items, required_categories, optional_categories, budget_target, params)
    bqm = model.to_bqm()

    n = len(items)
    labels = [f'x[{i}]' for i in range(n)]
    idx = {v: i for i, v in enumerate(labels)}
    Q = np.zeros((n, n))

    for v, bias in bqm.linear.items():
        Q[idx[v], idx[v]] = bias
    for (u, v), bias in bqm.quadratic.items():
        i, j = idx[u], idx[v]
        Q[i, j] += bias
        Q[j, i] += bias

    return Q
