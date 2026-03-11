"""Shared wrapper around OpenJij SimulatedAnnealingSampler.

Usage from any page::

    from core.openjij_sa import run_openjij
    from core.openjij_sidebar import OpenjijParams

    result = run_openjij(bqm, openjij_params)
    best_x       = result["best_x"]        # np.ndarray of ints
    penalty      = result["penalty"]        # best energy (includes BQM offset)
    qubo_raw     = result["qubo_raw"]       # x^T Q x  (without BQM offset)
    all_energies = result["all_energies"]   # per-read energies (includes offset)
    elapsed_sec  = result["elapsed_sec"]    # wall-clock time in seconds
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import dimod
import numpy as np
import openjij as oj

from core.openjij_sidebar import OpenjijParams


@dataclass
class OpenjijResult:
    """Return type of run_openjij()."""

    best_x: np.ndarray
    """Best binary solution vector (integer ndarray, 0/1)."""

    penalty: float
    """Best energy (includes BQM offset).  0 means perfect solution."""

    qubo_raw: float
    """x^T Q x  (BQM offset excluded)."""

    all_energies: np.ndarray
    """Per-read energies including BQM offset (one scalar per read)."""

    elapsed_sec: float
    """Wall-clock time of sampler.sample_bqm() call."""


def run_openjij(
    bqm: dimod.BinaryQuadraticModel,
    params: OpenjijParams,
) -> OpenjijResult:
    """
    Run OpenJij SA on a dimod BQM and return structured results.

    Parameters
    ----------
    bqm    : dimod.BinaryQuadraticModel — problem to solve.
    params : OpenjijParams from openjij_sidebar().

    Returns
    -------
    OpenjijResult
        best_x       : Best binary solution (ndarray[int], sorted by variable index)
        penalty      : Best energy (includes BQM offset; 0 = perfect)
        qubo_raw     : x^T Q x without constant offset
        all_energies : Per-read energies (includes offset)
        elapsed_sec  : Solver wall-clock time in seconds
    """
    sampler = oj.SASampler()

    t0 = time.perf_counter()
    sample_set = sampler.sample(bqm, **params.sampler_kwargs)
    elapsed = time.perf_counter() - t0

    best_datum = sample_set.first

    # dimod SampleSet energies include the BQM offset
    penalty: float = float(best_datum.energy)  # type: ignore
    qubo_raw: float = float(best_datum.energy) - float(bqm.offset)  # type: ignore

    n_vars = len(bqm.variables)
    best_x = np.array(
        [best_datum.sample[v] for v in range(n_vars)],  # type: ignore
        dtype=int,
    )
    all_energies: np.ndarray = sample_set.record["energy"]

    return OpenjijResult(
        best_x=best_x,
        penalty=penalty,
        qubo_raw=qubo_raw,
        all_energies=all_energies,
        elapsed_sec=elapsed,
    )
