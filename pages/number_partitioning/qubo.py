"""数分割問題 — QUBO 定式化モジュール。"""

from __future__ import annotations

import numpy as np

# QUBO パラメータ定義
# type/label/default/min/max/step を持つ辞書リスト
PARAMS: dict = {
    "lam": {
        "type": "float",
        "label": "ペナルティ係数 λ",
        "default": 1.0,
        "min": 0.1,
        "max": 10.0,
        "step": 0.1,
    },
}


def build_qubo(numbers: list[float], lam: float = 1.0) -> np.ndarray:
    """
    数分割問題の QUBO 行列を構築する。

    目的関数: λ * (Σ_i (2x_i - 1) * n_i)^2
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
