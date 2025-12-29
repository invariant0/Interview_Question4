# econ_models/vfi/simulation/simulator.py
"""
Simulation engine for VFI model validation.

This module provides simulation capabilities for testing whether
computed policies respect grid boundaries.
"""

from typing import Dict

import numpy as np
import tensorflow as tf

from econ_models.core.types import TENSORFLOW_DTYPE, Array, Tensor


class Simulator:
    """
    Simulation engine for policy function validation.

    This class provides methods for simulating the economy under
    computed optimal policies to verify grid boundary conditions.
    """

    @staticmethod
    def simulate_basic_model(
        policy_map: np.ndarray,
        transition_matrix: np.ndarray,
        n_capital: int,
        n_productivity: int,
        t_steps: int = 1000
    ) -> Dict[str, float]:
        """
        Simulate basic model to check boundary hits.

        Args:
            policy_map: Policy function mapping (k, z) -> k' index.
            transition_matrix: Markov transition matrix for productivity.
            n_capital: Number of capital grid points.
            n_productivity: Number of productivity grid points.
            t_steps: Number of simulation periods.

        Returns:
            Dictionary with boundary hit percentages.
        """
        k_idx = n_capital // 2
        z_idx = n_productivity // 2

        hits_min = 0
        hits_max = 0

        for _ in range(t_steps):
            k_idx = policy_map[k_idx, z_idx]

            if k_idx == 0:
                hits_min += 1
            elif k_idx == n_capital - 1:
                hits_max += 1

            z_idx = np.random.choice(n_productivity, p=transition_matrix[z_idx])

        return {
            "min_hit_pct": hits_min / t_steps,
            "max_hit_pct": hits_max / t_steps
        }

    @staticmethod
    def simulate_risky_model(
        pol_k: np.ndarray,
        pol_b: np.ndarray,
        transition_matrix: np.ndarray,
        n_capital: int,
        n_debt: int,
        n_productivity: int,
        t_steps: int = 1000
    ) -> Dict[str, float]:
        """
        Simulate risky debt model to check boundary hits.

        Args:
            pol_k: Policy function for capital, shape (n_k, n_b, n_z).
            pol_b: Policy function for debt, shape (n_k, n_b, n_z).
            transition_matrix: Markov transition matrix for productivity.
            n_capital: Number of capital grid points.
            n_debt: Number of debt grid points.
            n_productivity: Number of productivity grid points.
            t_steps: Number of simulation periods.

        Returns:
            Dictionary with boundary hit percentages for each state.
        """
        k_idx = n_capital // 2
        b_idx = 0
        z_idx = n_productivity // 2

        stats = {"k_min": 0, "k_max": 0, "b_min": 0, "b_max": 0}

        for _ in range(t_steps):
            k_next = pol_k[k_idx, b_idx, z_idx]
            b_next = pol_b[k_idx, b_idx, z_idx]

            if k_next == 0:
                stats["k_min"] += 1
            if k_next == n_capital - 1:
                stats["k_max"] += 1
            if b_next == 0:
                stats["b_min"] += 1
            if b_next == n_debt - 1:
                stats["b_max"] += 1

            k_idx = k_next
            b_idx = b_next
            z_idx = np.random.choice(n_productivity, p=transition_matrix[z_idx])

        return {k: v / t_steps for k, v in stats.items()}