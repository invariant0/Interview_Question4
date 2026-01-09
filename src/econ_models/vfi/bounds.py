# econ_models/vfi/bounds.py
"""
Automated boundary discovery for VFI grids.

This module iteratively solves models with expanding grids to ensure
the simulated economy stays within the computational domain.

Example:
    >>> finder = BoundaryFinder(params, config)
    >>> bounds = finder.find_basic_bounds()
"""

from typing import Dict, Any

from econ_models.config.economic_params import EconomicParams
from econ_models.config.vfi_config import GridConfig
from econ_models.vfi.basic import BasicModelVFI
from econ_models.vfi.risky_debt import RiskyDebtModelVFI
from econ_models.econ import SteadyStateCalculator


class BoundaryFinder:
    """
    Iteratively solve and simulate models to find optimal grid boundaries.

    This class performs a search procedure that expands grid boundaries
    when simulated paths hit the edges, ensuring the computational domain
    is large enough to capture equilibrium dynamics.

    Attributes:
        params: Economic parameters.
        config: Grid configuration.
        threshold: Maximum acceptable boundary hit frequency.
        expansion_factor: Multiplier for boundary expansion.
        margin: Safety margin multiplier for final bounds.
        n_steps: Number of simulation steps per batch.
        n_batches: Number of simulation batches.
    """

    def __init__(
        self,
        params: EconomicParams,
        config: GridConfig,
        threshold: float = 0.01,
        margin: float = 1.1,
        n_steps: int = 1000,
        n_batches: int = 1000,
        seed: int | None = None
    ) -> None:
        """
        Initialize the boundary finder.

        Args:
            params: Economic parameters.
            config: Grid configuration.
            threshold: Maximum acceptable boundary hit rate.
            margin: Safety margin multiplier (e.g., 1.1 = 10% padding).
            n_steps: Number of simulation steps per batch.
            n_batches: Number of simulation batches for statistics.
            seed: Random seed for reproducibility.
        """
        self.params = params
        self.config = config
        self.threshold = threshold
        self.expansion_factor = 1.25
        self.margin = margin
        self.n_steps = n_steps
        self.n_batches = n_batches
        self.seed = seed

    def find_basic_bounds(self) -> Dict[str, Any]:
        """
        Find optimal capital bounds for the basic model.

        Returns:
            Dictionary containing original bounds, margin-adjusted bounds,
            and the final VFI solution.
        """
        k_ss = SteadyStateCalculator.calculate_capital(self.params)
        k_min_scale = 0.5
        k_max_scale = 2.0

        print(f"Starting Boundary Search (Basic). K_ss ~ {k_ss:.2f}")

        final_res = {}
        model = None

        for i in range(10):
            k_min = k_min_scale * k_ss
            k_max = k_max_scale * k_ss
            print(f"  Iter {i+1}: Testing K range [{k_min:.2f}, {k_max:.2f}]")

            model = BasicModelVFI(self.params, self.config, k_bounds=(k_min, k_max))
            res = model.solve()
            history, stats = model.simulate(
                res['V'],
                n_steps=self.n_steps,
                n_batches=self.n_batches,
                seed=self.seed
            )

            expand = False
            if stats['min_hit_pct'] > self.threshold:
                print(f"    -> Hit lower bound ({stats['min_hit_pct']:.1%}). Expanding.")
                k_min_scale /= self.expansion_factor
                expand = True

            if stats['max_hit_pct'] > self.threshold:
                print(f"    -> Hit upper bound ({stats['max_hit_pct']:.1%}). Expanding.")
                k_max_scale *= self.expansion_factor
                expand = True

            final_res = res

            if not expand:
                print("    -> Bounds sufficient.")
                break

        return self._format_basic_result(model, k_min, k_max, final_res)

    def _format_basic_result(
        self,
        model: BasicModelVFI,
        k_min: float,
        k_max: float,
        solution: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Format the boundary search result for basic model."""
        z_min = float(model.z_min)
        z_max = float(model.z_max)

        return {
            "k_bounds_original": (k_min, k_max),
            "z_bounds_original": (z_min, z_max),
            "k_bounds_add_margin": (k_min / self.margin, k_max * self.margin),
            "z_bounds_add_margin": (z_min / self.margin, z_max * self.margin),
            "vfi_solution": solution
        }

    def find_risky_bounds(self) -> Dict[str, Any]:
        """
        Find optimal capital and debt bounds for the risky debt model.

        Returns:
            Dictionary containing original bounds, margin-adjusted bounds,
            and the final VFI solution.
        """
        k_ss = SteadyStateCalculator.calculate_capital(self.params)
        k_min_scale = 0.5
        k_max_scale = 2.0
        b_min_scale = 1.2
        b_max_scale = 2.0

        print(f"Starting Boundary Search (Risky). K_ss ~ {k_ss:.2f}")

        final_res = {}
        model = None
        k_min, k_max = 0.0, 0.0
        b_min, b_max = 0.0, 0.0

        for i in range(15):
            k_min = k_min_scale * k_ss
            k_max = k_max_scale * k_ss
            b_min = -b_min_scale * k_ss
            b_max = b_max_scale * k_ss

            print(
                f"  Iter {i+1}: K[{k_min:.2f}, {k_max:.2f}], "
                f"B[{b_min:.2f}, {b_max:.2f}]"
            )

            model = RiskyDebtModelVFI(
                self.params, self.config,
                k_bounds=(k_min, k_max),
                b_bounds=(b_min, b_max)
            )
            res = model.solve()
            history, stats = model.simulate(
                res['V'],
                res['Q'],
                n_steps=self.n_steps,
                n_batches=self.n_batches,
                seed=self.seed
            )

            expand = self._check_and_expand_risky_bounds(
                stats,
                k_min_scale, k_max_scale,
                b_min_scale, b_max_scale
            )

            if expand:
                k_min_scale, k_max_scale, b_min_scale, b_max_scale = expand
            else:
                print("    -> Bounds sufficient.")
                final_res = res
                break

            final_res = res

        return self._format_risky_result(
            model, k_min, k_max, b_min, b_max, final_res
        )

    def _check_and_expand_risky_bounds(
        self,
        stats: Dict[str, float],
        k_min_scale: float,
        k_max_scale: float,
        b_min_scale: float,
        b_max_scale: float
    ):
        """Check boundary hits and return updated scales if expansion needed."""
        expand = False

        if stats['k_min'] > self.threshold:
            print(f"    -> Hit K lower ({stats['k_min']:.1%}).")
            k_min_scale /= self.expansion_factor
            expand = True

        if stats['k_max'] > self.threshold:
            print(f"    -> Hit K upper ({stats['k_max']:.1%}).")
            k_max_scale *= self.expansion_factor
            expand = True

        if stats['b_min'] > self.threshold:
            print(f"    -> Hit B lower ({stats['b_min']:.1%}).")
            b_min_scale *= self.expansion_factor
            expand = True

        if stats['b_max'] > self.threshold:
            print(f"    -> Hit B upper ({stats['b_max']:.1%}).")
            b_max_scale *= self.expansion_factor
            expand = True

        if expand:
            return k_min_scale, k_max_scale, b_min_scale, b_max_scale
        return None

    def _format_risky_result(
        self,
        model: RiskyDebtModelVFI,
        k_min: float,
        k_max: float,
        b_min: float,
        b_max: float,
        solution: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Format the boundary search result for risky model."""
        z_min = float(model.z_min)
        z_max = float(model.z_max)

        return {
            "k_bounds_original": (k_min, k_max),
            "b_bounds_original": (b_min, b_max),
            "z_bounds_original": (z_min, z_max),
            "k_bounds_add_margin": (k_min / self.margin, k_max * self.margin),
            "b_bounds_add_margin": (b_min * self.margin, b_max * self.margin),
            "z_bounds_add_margin": (z_min / self.margin, z_max * self.margin),
            "vfi_solution": solution
        }
