"""General-purpose SA solver and shared UI components."""

from .sa import qubo_energy, simulated_annealing
from .sa_sidebar import SAParams, sa_sidebar

__all__ = ["qubo_energy", "simulated_annealing", "sa_sidebar", "SAParams"]
