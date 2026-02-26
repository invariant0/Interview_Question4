"""Markov-chain simulator for the basic RBC model.

Extracted from ``BoundaryFinder._simulate_basic`` to enable
independent testing with pre-canned VFI solutions.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from econ_models.vfi.grids.grid_utils import interp_2d_batch


class BasicSimulator:
    """Simulate the basic model to detect boundary hits.

    Uses Markov-chain Z transitions and bilinear interpolation of
    the value function and policy to decide adjust-vs-wait.

    Parameters
    ----------
    n_steps : int
        Number of simulation periods per batch.
    n_batches : int
        Number of parallel simulation trajectories.
    seed : int, optional
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_steps: int = 1000,
        n_batches: int = 1000,
        seed: Optional[int] = None,
    ) -> None:
        self.n_steps = n_steps
        self.n_batches = n_batches
        self.seed = seed

    def run(
        self, solution: Dict[str, Any]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
        """Simulate the basic model and return history + boundary stats.

        Parameters
        ----------
        solution : dict
            Output of ``BasicModelVFI.solve()``.

        Returns
        -------
        history : dict
            ``K`` → capital history array ``(n_batches, n_steps)``.
        stats : dict
            ``min_hit_pct``, ``max_hit_pct`` — boundary-hit rates (%).
        """
        np.random.seed(self.seed)

        v_adjust = solution["V_adjust"]
        v_wait = solution["V_wait"]
        policy_k_values = solution["policy_k_values"]
        k_grid = solution["K"]
        z_grid = solution["Z"]
        transition = solution["transition_matrix"]
        k_min = solution["k_min"]
        k_max = solution["k_max"]
        delta = solution["depreciation_rate"]
        k_ss = solution["k_ss"]

        n_k = len(k_grid)
        n_z = len(z_grid)

        k_sim = np.full(self.n_batches, k_ss)
        z_idx = np.full(self.n_batches, n_z // 2, dtype=int)
        k_history = np.zeros((self.n_batches, self.n_steps))

        for t in range(self.n_steps):
            k_history[:, t] = k_sim
            z_vals = z_grid[z_idx]

            v_adj = interp_2d_batch(
                k_grid, z_grid, v_adjust, k_sim, z_vals
            )
            v_wt = interp_2d_batch(
                k_grid, z_grid, v_wait, k_sim, z_vals
            )
            should_adjust = v_adj > v_wt

            k_depreciated = k_sim * (1.0 - delta)
            k_next = np.copy(k_depreciated)

            if np.any(should_adjust):
                k_target = interp_2d_batch(
                    k_grid,
                    z_grid,
                    policy_k_values,
                    k_sim[should_adjust],
                    z_vals[should_adjust],
                )
                k_next[should_adjust] = np.clip(k_target, k_min, k_max)

            k_sim = np.clip(k_next, k_min, k_max)

            # Markov Z transition
            cum_p = np.cumsum(transition[z_idx], axis=1)
            u = np.random.rand(self.n_batches)
            for b in range(self.n_batches):
                z_idx[b] = min(np.searchsorted(cum_p[b], u[b]), n_z - 1)

        stats = {
            "min_hit_pct": (
                np.mean(k_history <= k_grid[0]) * 100
            ),
            "max_hit_pct": (
                np.mean(k_history >= k_grid[-1]) * 100
            ),
        }
        return {"K": k_history}, stats
