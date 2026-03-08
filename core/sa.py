"""General-purpose Simulated Annealing solver."""

from __future__ import annotations

import math
import random
import time

import numpy as np


def qubo_energy(x: np.ndarray, Q: np.ndarray) -> float:
    """Compute QUBO energy E = x^T Q x."""
    return float(x @ Q @ x)


def simulated_annealing(
    Q: np.ndarray,
    T_init: float,
    T_min: float,
    cooling_rate: float,
    max_iter: int,
    seed: int | None = None,
    timeout_sec: float | None = None,
) -> dict:
    """
    Minimize a QUBO problem using Simulated Annealing.

    Parameters
    ----------
    Q            : QUBO matrix (n×n)
    T_init       : Initial temperature
    T_min        : Minimum temperature (stop condition)
    cooling_rate : Cooling rate α (T ← α·T)
    max_iter     : Maximum number of iterations
    seed         : Random seed (None for random)

    Returns
    -------
    dict:
        best_x          : Best solution found (ndarray)
        best_energy     : Energy of the best solution
        energy_history  : Energy at each step
        best_history    : Best energy at each step
        temp_history    : Temperature at each step
        n_iter          : Actual number of iterations run
        timed_out       : Whether the run was stopped by timeout
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    n = Q.shape[0]
    x = np_rng.integers(0, 2, n).astype(float)
    current_energy = qubo_energy(x, Q)
    best_x = x.copy()
    best_energy = current_energy

    T = T_init
    energy_history: list[float] = [current_energy]
    best_history: list[float] = [best_energy]
    temp_history: list[float] = [T]
    timed_out = False
    t0 = time.perf_counter()

    for _ in range(max_iter):
        if T <= T_min:
            break
        if timeout_sec is not None and time.perf_counter() - t0 >= timeout_sec:
            timed_out = True
            break

        flip_idx = rng.randint(0, n - 1)
        x_new = x.copy()
        x_new[flip_idx] = 1.0 - x_new[flip_idx]

        new_energy = qubo_energy(x_new, Q)
        delta_E = new_energy - current_energy

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
        "timed_out": timed_out,
    }
