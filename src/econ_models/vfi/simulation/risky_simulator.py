"""Grid-index simulator for the risky-debt model.

Extracted from ``BoundaryFinder._simulate_risky`` to enable
independent testing with pre-canned VFI solutions.

Works entirely on grid indices with direct array lookups (no
interpolation).  Nearest-neighbour accuracy is adequate for
detecting boundary pile-ups and is 10–50× faster than trilinear
interpolation.

Post-default cooldown suppresses false-positive boundary hits
caused by the reset target sitting near index 0–1 on wide grids.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from econ_models.econ import SteadyStateCalculator
from econ_models.config.economic_params import EconomicParams


class RiskySimulator:
    """Grid-index simulator for risky-debt boundary detection.

    Parameters
    ----------
    params : EconomicParams
        Structural economic parameters.
    n_steps : int
        Number of simulation periods per batch.
    n_batches : int
        Number of parallel simulation trajectories.
    seed : int, optional
        Random seed for reproducibility.
    default_cooldown_periods : int
        Observations within this many periods after a default-reset
        are excluded from boundary-hit statistics.
    """

    DEFAULT_COOLDOWN_PERIODS: int = 10

    def __init__(
        self,
        params: EconomicParams,
        n_steps: int = 1000,
        n_batches: int = 1000,
        seed: Optional[int] = None,
        default_cooldown_periods: int = 10,
    ) -> None:
        self.params = params
        self.n_steps = n_steps
        self.n_batches = n_batches
        self.seed = seed
        self.DEFAULT_COOLDOWN_PERIODS = default_cooldown_periods

    @staticmethod
    def _build_depreciated_k_index_map(
        k_grid: np.ndarray, delta: float
    ) -> np.ndarray:
        """Pre-compute nearest-grid-index of depreciated capital.

        For each index *i*, maps to the closest grid index of
        ``k_grid[i] * (1 − δ)``.

        Parameters
        ----------
        k_grid : np.ndarray
            Capital grid, shape ``(n_k,)``.
        delta : float
            Depreciation rate.

        Returns
        -------
        np.ndarray
            Integer index array, shape ``(n_k,)``.
        """
        n_k = len(k_grid)
        k_dep_vals = k_grid * (1.0 - delta)
        k_dep_idx = np.searchsorted(k_grid, k_dep_vals, side="left")
        k_dep_idx = np.clip(k_dep_idx, 0, n_k - 1)
        left = np.clip(k_dep_idx - 1, 0, n_k - 1)
        use_left = np.abs(k_grid[left] - k_dep_vals) < np.abs(
            k_grid[k_dep_idx] - k_dep_vals
        )
        return np.where(use_left, left, k_dep_idx)

    def run(
        self, solution: Dict[str, Any]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
        """Run grid-index simulation and return history + boundary stats.

        Parameters
        ----------
        solution : dict
            Output of ``RiskyDebtModelVFI.solve()``.

        Returns
        -------
        history : dict
            ``K``, ``B`` → continuous history arrays.
        stats : dict
            Boundary-hit percentages and default rate.
        """
        np.random.seed(self.seed)

        # Unpack grids and discrete policies
        k_grid = np.asarray(solution["K"])
        b_grid = np.asarray(solution["B"])
        z_grid = np.asarray(solution["Z"])
        transition = np.asarray(solution["transition_matrix"])

        pol_default = np.asarray(solution["policy_default"])
        pol_adjust = np.asarray(solution["policy_adjust"])
        pol_k_idx = np.asarray(solution["policy_k_idx"])
        pol_b_idx = np.asarray(solution["policy_b_idx"])
        pol_bw_idx = np.asarray(solution["policy_b_wait_idx"])

        n_k, n_b, n_z = len(k_grid), len(b_grid), len(z_grid)
        delta = self.params.depreciation_rate
        k_ss = SteadyStateCalculator.calculate_capital(self.params)

        k_dep_idx = self._build_depreciated_k_index_map(k_grid, delta)
        cum_p = np.cumsum(transition, axis=1)

        # Initialise firms at interior grid points
        ki_init = int(np.argmin(np.abs(k_grid - k_ss)))
        ki_init = max(2, min(ki_init, n_k - 3))
        bi_init = int(np.argmin(np.abs(b_grid)))
        bi_init = max(2, min(bi_init, n_b - 3))
        zi_init = n_z // 2

        ki = np.full(self.n_batches, ki_init, dtype=np.int32)
        bi = np.full(self.n_batches, bi_init, dtype=np.int32)
        zi = np.full(self.n_batches, zi_init, dtype=np.int32)
        cooldown = np.zeros(self.n_batches, dtype=np.int32)

        ki_hist = np.empty((self.n_batches, self.n_steps), dtype=np.int32)
        bi_hist = np.empty((self.n_batches, self.n_steps), dtype=np.int32)
        valid_mask = np.ones((self.n_batches, self.n_steps), dtype=bool)
        n_defaults = 0

        for t in range(self.n_steps):
            ki_hist[:, t] = ki
            bi_hist[:, t] = bi
            valid_mask[cooldown > 0, t] = False
            cooldown = np.maximum(cooldown - 1, 0)

            is_def = pol_default[ki, bi, zi]
            is_adj = pol_adjust[ki, bi, zi]

            # DEFAULT → reset
            def_mask = is_def
            if np.any(def_mask):
                n_defaults += int(np.sum(def_mask))
                ki[def_mask] = ki_init
                bi[def_mask] = bi_init
                cooldown[def_mask] = self.DEFAULT_COOLDOWN_PERIODS

            # ADJUST
            adj_mask = (~def_mask) & is_adj
            if np.any(adj_mask):
                ki_old = ki[adj_mask].copy()
                bi_old = bi[adj_mask].copy()
                zi_old = zi[adj_mask]
                ki[adj_mask] = np.clip(
                    pol_k_idx[ki_old, bi_old, zi_old], 0, n_k - 1
                )
                bi[adj_mask] = np.clip(
                    pol_b_idx[ki_old, bi_old, zi_old], 0, n_b - 1
                )

            # WAIT
            wait_mask = (~def_mask) & (~is_adj)
            if np.any(wait_mask):
                bi[wait_mask] = np.clip(
                    pol_bw_idx[ki[wait_mask], bi[wait_mask], zi[wait_mask]],
                    0,
                    n_b - 1,
                )
                ki[wait_mask] = k_dep_idx[ki[wait_mask]]

            # Markov Z transition (vectorised)
            u = np.random.rand(self.n_batches, 1)
            zi = np.minimum(
                np.sum(u > cum_p[zi], axis=1).astype(np.int32), n_z - 1
            )

        # Boundary-hit statistics
        n_valid = max(np.sum(valid_mask), 1)
        total = self.n_batches * self.n_steps

        stats: Dict[str, float] = {
            "k_min_hit_pct": np.sum((ki_hist <= 1) & valid_mask)
            / n_valid
            * 100,
            "k_max_hit_pct": np.sum((ki_hist >= n_k - 2) & valid_mask)
            / n_valid
            * 100,
            "b_min_hit_pct": np.sum((bi_hist <= 1) & valid_mask)
            / n_valid
            * 100,
            "b_max_hit_pct": np.sum((bi_hist >= n_b - 2) & valid_mask)
            / n_valid
            * 100,
            "default_rate": n_defaults / total * 100,
            "valid_pct": n_valid / total * 100,
            "avg_adjust_rate": 0.0,
        }
        return {"K": k_grid[ki_hist], "B": b_grid[bi_hist]}, stats
