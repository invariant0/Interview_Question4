"""Automated boundary discovery for VFI grids.

Iteratively solves and simulates models to determine capital and debt
grid bounds that are wide enough to avoid boundary-hit artifacts, but
narrow enough to maintain grid resolution.  Separate methods are
provided for the basic RBC model (:meth:`BoundaryFinder.find_basic_bounds`)
and the risky-debt model (:meth:`BoundaryFinder.find_risky_bounds`).

The refactored version delegates simulation to standalone
:class:`BasicSimulator` and :class:`RiskySimulator` instances,
and optionally accepts a solver factory and simulator via constructor
injection for testability.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional, Tuple

from econ_models.config.economic_params import EconomicParams
from econ_models.config.vfi_config import GridConfig
from econ_models.econ import SteadyStateCalculator
from econ_models.vfi.grids.grid_utils import compute_optimal_chunks
from econ_models.vfi.simulation.basic_simulator import BasicSimulator
from econ_models.vfi.simulation.risky_simulator import RiskySimulator

logger = logging.getLogger(__name__)


class BoundaryFinder:
    """Iteratively solve and simulate to discover optimal grid boundaries.

    Parameters
    ----------
    params : EconomicParams
        Structural economic parameters.
    config : GridConfig
        Grid sizes, tolerances, and iteration limits.
    threshold : float
        Maximum acceptable boundary-hit rate (percent).
    margin : float
        Safety-margin multiplier applied to the discovered bounds
        (e.g. 1.2 → 20 % padding).
    n_steps : int
        Number of simulation periods per batch.
    n_batches : int
        Number of parallel simulation trajectories.
    seed : int, optional
        Random seed for reproducibility.
    k_chunk_size, b_chunk_size : int, optional
        VRAM-tile sizes for the *state* dimensions (risky model only).
    kp_chunk_size, bp_chunk_size : int, optional
        VRAM-tile sizes for the *choice* dimensions (risky model only).
    basic_solver_factory : callable, optional
        Factory for creating basic-model solvers.  Signature:
        ``(params, config, k_bounds) -> solver_with_solve_method``.
        Defaults to constructing ``BasicModelVFI`` directly.
    risky_solver_factory : callable, optional
        Factory for creating risky-model solvers.  Signature:
        ``(params, config, k_bounds, b_bounds, k_cs, b_cs, kp_cs, bp_cs) ->
        solver_with_solve_method``.
        Defaults to constructing ``RiskyDebtModelVFI`` directly.
    basic_simulator : BasicSimulator, optional
        Pre-configured basic-model simulator.  When *None*, one is
        created from *n_steps*, *n_batches*, *seed*.
    risky_simulator : RiskySimulator, optional
        Pre-configured risky-model simulator.  When *None*, one is
        created from *params*, *n_steps*, *n_batches*, *seed*.
    """

    # Post-default cooldown: observations within this many periods
    # after a default-reset are excluded from boundary-hit statistics.
    DEFAULT_COOLDOWN_PERIODS: int = 10

    # Physical caps (multiples of k_ss) beyond which expansion is
    # counter-productive—the ergodic distribution cannot reach there.
    K_MIN_PHYSICAL_MULT: float = 0.001
    K_MAX_PHYSICAL_MULT: float = 30.0
    B_MAX_PHYSICAL_MULT: float = 40.0

    _EXPANSION_FACTOR: float = 1.2

    def __init__(
        self,
        params: EconomicParams,
        config: GridConfig,
        threshold: float = 1.0,
        margin: float = 1.2,
        n_steps: int = 1000,
        n_batches: int = 1000,
        seed: Optional[int] = None,
        k_chunk_size: Optional[int] = None,
        b_chunk_size: Optional[int] = None,
        kp_chunk_size: Optional[int] = None,
        bp_chunk_size: Optional[int] = None,
        basic_solver_factory: Optional[Callable[..., Any]] = None,
        risky_solver_factory: Optional[Callable[..., Any]] = None,
        basic_simulator: Optional[BasicSimulator] = None,
        risky_simulator: Optional[RiskySimulator] = None,
    ) -> None:
        self.params: EconomicParams = params
        self.config: GridConfig = config
        self.threshold: float = threshold
        self.expansion_factor: float = self._EXPANSION_FACTOR
        self.margin: float = margin
        self.n_steps: int = n_steps
        self.n_batches: int = n_batches
        self.seed: Optional[int] = seed
        self.k_chunk_size: Optional[int] = k_chunk_size
        self.b_chunk_size: Optional[int] = b_chunk_size
        self.kp_chunk_size: Optional[int] = kp_chunk_size
        self.bp_chunk_size: Optional[int] = bp_chunk_size

        # Solver factories (lazy imports to avoid circular deps)
        self._basic_solver_factory = basic_solver_factory
        self._risky_solver_factory = risky_solver_factory

        # Simulators
        self._basic_simulator = basic_simulator or BasicSimulator(
            n_steps=n_steps, n_batches=n_batches, seed=seed,
        )
        self._risky_simulator = risky_simulator or RiskySimulator(
            params=params, n_steps=n_steps, n_batches=n_batches, seed=seed,
            default_cooldown_periods=self.DEFAULT_COOLDOWN_PERIODS,
        )

    # ==================================================================
    #  Solver construction helpers
    # ==================================================================

    def _create_basic_solver(
        self, k_bounds: Tuple[float, float]
    ) -> Any:
        """Create a basic-model solver (with lazy import)."""
        if self._basic_solver_factory is not None:
            return self._basic_solver_factory(
                self.params, self.config, k_bounds
            )
        from econ_models.vfi.basic import BasicModelVFI
        return BasicModelVFI(
            self.params, self.config, k_bounds=k_bounds
        )

    def _create_risky_solver(
        self,
        k_bounds: Tuple[float, float],
        b_bounds: Tuple[float, float],
        k_cs: int,
        b_cs: int,
        kp_cs: int,
        bp_cs: int,
    ) -> Any:
        """Create a risky-model solver (with lazy import)."""
        if self._risky_solver_factory is not None:
            return self._risky_solver_factory(
                self.params, self.config,
                k_bounds, b_bounds,
                k_cs, b_cs, kp_cs, bp_cs
            )
        from econ_models.vfi.risky import RiskyDebtModelVFI
        return RiskyDebtModelVFI(
            self.params, self.config,
            k_bounds=k_bounds,
            b_bounds=b_bounds,
            k_chunk_size=k_cs,
            b_chunk_size=b_cs,
            kp_chunk_size=kp_cs,
            bp_chunk_size=bp_cs,
        )

    # ==================================================================
    #  Static utility
    # ==================================================================

    @staticmethod
    def compute_optimal_chunks(
        n_k: int,
        n_b: int,
        n_z: int = 12,
        vram_limit_gb: float = 28.0,
    ) -> Tuple[int, int, int, int]:
        """Return VRAM-aware ``(k_chunk, b_chunk, kp_chunk, bp_chunk)``.

        Delegates to
        :func:`econ_models.vfi.grids.grid_utils.compute_optimal_chunks`.
        """
        return compute_optimal_chunks(
            n_k, n_b, n_z=n_z, vram_limit_gb=vram_limit_gb
        )

    # ==================================================================
    #  Public API — basic model
    # ==================================================================

    def find_basic_bounds(self, max_iters: int = 25) -> Dict[str, Any]:
        """Find optimal capital bounds for the basic model.

        Iteratively solves, simulates, and expands the grid until the
        boundary-hit rate falls below :attr:`threshold`.

        Parameters
        ----------
        max_iters : int
            Maximum number of expansion iterations.

        Returns
        -------
        dict
            ``k_bounds_original``, ``k_bounds_add_margin``,
            ``z_bounds_original``, ``z_bounds_add_margin``,
            ``vfi_solution``.
        """
        k_ss = SteadyStateCalculator.calculate_capital(self.params)
        k_min_scale = 0.5
        k_max_scale = 2.0

        logger.info(
            "Starting Boundary Search (Basic). K_ss ~ %.2f", k_ss
        )

        final_res: Dict[str, Any] = {}
        model: Any = None
        k_min, k_max = 0.0, 0.0

        for i in range(max_iters):
            k_min = k_min_scale * k_ss
            k_max = k_max_scale * k_ss
            logger.info(
                "  Iter %d: Testing K range [%.2f, %.2f]",
                i + 1,
                k_min,
                k_max,
            )

            model = self._create_basic_solver(k_bounds=(k_min, k_max))
            res = model.solve()
            _, stats = self._basic_simulator.run(res)

            expand = False
            if stats["min_hit_pct"] > self.threshold:
                logger.info(
                    "    -> Hit lower bound (%.1f%%). Expanding.",
                    stats["min_hit_pct"],
                )
                k_min_scale /= self.expansion_factor
                expand = True

            if stats["max_hit_pct"] > self.threshold:
                logger.info(
                    "    -> Hit upper bound (%.1f%%). Expanding.",
                    stats["max_hit_pct"],
                )
                k_max_scale *= self.expansion_factor
                expand = True

            final_res = res

            if not expand:
                logger.info("    -> Bounds sufficient.")
                break

        return self._format_basic_result(model, k_min, k_max, final_res)

    # ==================================================================
    #  Public API — risky model
    # ==================================================================

    def _resolve_chunk_sizes(self) -> Tuple[int, int, int, int]:
        """Return explicit chunk sizes, auto-computing if not provided.

        When only state-dimension chunks (k, b) are supplied but
        choice-dimension chunks (kp, bp) are not, the choice dimensions
        default to the full grid size.  This matches the old 2-level
        loop behaviour where the kernel always processed the full
        ``k' × b'`` space for each ``(k_chunk, b_chunk)`` tile.
        """
        n_k = self.config.n_capital
        n_b = self.config.n_debt
        n_z = self.config.n_productivity

        if self.k_chunk_size is not None:
            k_cs = self.k_chunk_size
            b_cs = self.b_chunk_size if self.b_chunk_size is not None else n_b
            kp_cs = self.kp_chunk_size if self.kp_chunk_size is not None else n_k
            bp_cs = self.bp_chunk_size if self.bp_chunk_size is not None else n_b
            return k_cs, b_cs, kp_cs, bp_cs
        return self.compute_optimal_chunks(n_k, n_b, n_z)

    def find_risky_bounds(self, max_iters: int = 25) -> Dict[str, Any]:
        """Find optimal capital and debt bounds for the risky-debt model.

        Iteratively solves, simulates, and expands the grid until the
        boundary-hit rate falls below :attr:`threshold`.  Physical caps
        prevent runaway expansion that degrades grid resolution.

        Parameters
        ----------
        max_iters : int
            Maximum number of expansion iterations.

        Returns
        -------
        dict
            ``k_bounds_original``, ``k_bounds_add_margin``,
            ``b_bounds_original``, ``b_bounds_add_margin``,
            ``z_bounds_original``, ``z_bounds_add_margin``,
            ``vfi_solution``.
        """
        k_ss = SteadyStateCalculator.calculate_capital(self.params)

        k_min_scale = 0.5
        k_max_scale = 3.0
        b_min_scale = -0.5
        b_max_scale = 2.0

        logger.info(
            "Starting Boundary Search (Risky Debt). K_ss ~ %.2f", k_ss
        )

        final_res: Dict[str, Any] = {}
        model: Any = None
        k_min, k_max = 0.0, 0.0
        b_min, b_max = 0.0, 0.0

        for i in range(max_iters):
            k_min = max(k_min_scale * k_ss, self.K_MIN_PHYSICAL_MULT * k_ss)
            k_max = min(k_max_scale * k_ss, self.K_MAX_PHYSICAL_MULT * k_ss)
            b_min = b_min_scale * k_ss
            b_max = min(b_max_scale * k_ss, self.B_MAX_PHYSICAL_MULT * k_ss)

            logger.info(
                "  Iter %d: K [%.4f, %.2f], B [%.2f, %.2f]",
                i + 1,
                k_min,
                k_max,
                b_min,
                b_max,
            )

            try:
                k_cs, b_cs, kp_cs, bp_cs = self._resolve_chunk_sizes()

                model = self._create_risky_solver(
                    k_bounds=(k_min, k_max),
                    b_bounds=(b_min, b_max),
                    k_cs=k_cs,
                    b_cs=b_cs,
                    kp_cs=kp_cs,
                    bp_cs=bp_cs,
                )
                res = model.solve()
                _, stats = self._risky_simulator.run(res)

                logger.info(
                    "    -> K_min=%.2f%%, K_max=%.2f%%, "
                    "B_min=%.2f%%, B_max=%.2f%%, "
                    "Default=%.2f%%, Valid=%.1f%%",
                    stats["k_min_hit_pct"],
                    stats["k_max_hit_pct"],
                    stats["b_min_hit_pct"],
                    stats["b_max_hit_pct"],
                    stats["default_rate"],
                    stats["valid_pct"],
                )

                expand = self._check_and_expand_risky(
                    stats, k_min, k_max, b_max, k_ss,
                    k_min_scale, k_max_scale, b_min_scale, b_max_scale,
                )
                if expand is not None:
                    k_min_scale, k_max_scale, b_min_scale, b_max_scale = (
                        expand
                    )
                    continue

                final_res = res
                logger.info("    -> Bounds sufficient.")
                break

            except Exception:
                logger.exception("    -> Error in iteration %d.", i + 1)
                k_max_scale *= self.expansion_factor
                b_max_scale *= self.expansion_factor
                continue
        else:
            # Loop exhausted without breaking — use last successful result
            pass

        return self._format_risky_result(
            model, k_min, k_max, b_min, b_max, final_res
        )

    def _check_and_expand_risky(
        self,
        stats: Dict[str, float],
        k_min: float,
        k_max: float,
        b_max: float,
        k_ss: float,
        k_min_scale: float,
        k_max_scale: float,
        b_min_scale: float,
        b_max_scale: float,
    ) -> Optional[Tuple[float, float, float, float]]:
        """Check boundary-hit stats and return updated scales, or *None*.

        Returns
        -------
        tuple or None
            Updated ``(k_min_scale, k_max_scale, b_min_scale, b_max_scale)``
            if any expansion is needed; *None* if bounds are sufficient.
        """
        expand = False

        if stats["k_min_hit_pct"] > self.threshold:
            if k_min > self.K_MIN_PHYSICAL_MULT * k_ss + 1e-6:
                logger.info(
                    "    -> Hit K lower (%.2f%%). Expanding.",
                    stats["k_min_hit_pct"],
                )
                k_min_scale /= self.expansion_factor
                expand = True
            else:
                logger.info(
                    "    -> Hit K lower but at physical floor. Accepting."
                )

        if stats["k_max_hit_pct"] > self.threshold:
            if k_max < self.K_MAX_PHYSICAL_MULT * k_ss - 1e-6:
                logger.info("    -> Hit K upper. Expanding.")
                k_max_scale *= self.expansion_factor
                expand = True
            else:
                logger.info(
                    "    -> Hit K upper but at physical cap. Accepting."
                )

        if stats["b_min_hit_pct"] > self.threshold:
            logger.info("    -> Hit B lower. Expanding.")
            b_min_scale -= 0.3
            expand = True

        if stats["b_max_hit_pct"] > self.threshold:
            if b_max < self.B_MAX_PHYSICAL_MULT * k_ss - 1e-6:
                logger.info("    -> Hit B upper. Expanding.")
                b_max_scale *= self.expansion_factor
                expand = True
            else:
                logger.info(
                    "    -> Hit B upper but at physical cap. Accepting."
                )

        if expand:
            return k_min_scale, k_max_scale, b_min_scale, b_max_scale
        return None

    # ==================================================================
    #  Result formatting helpers
    # ==================================================================

    def _format_basic_result(
        self,
        model: Any,
        k_min: float,
        k_max: float,
        solution: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Format boundary-search output for the basic model.

        Parameters
        ----------
        model : BasicModelVFI or None
            The VFI solver from the final iteration.
        k_min, k_max : float
            Discovered capital bounds.
        solution : dict
            VFI solution dictionary.

        Returns
        -------
        dict
            Bounds (original and with margin) plus the VFI solution.
        """
        z_min = float(model.z_min) if model is not None else 0.0
        z_max = float(model.z_max) if model is not None else 0.0

        return {
            "k_bounds_original": (k_min, k_max),
            "z_bounds_original": (z_min, z_max),
            "k_bounds_add_margin": (
                k_min / self.margin,
                k_max * self.margin,
            ),
            "z_bounds_add_margin": (
                z_min / self.margin,
                z_max * self.margin,
            ),
            "vfi_solution": solution,
        }

    def _format_risky_result(
        self,
        model: Any,
        k_min: float,
        k_max: float,
        b_min: float,
        b_max: float,
        solution: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Format boundary-search output for the risky-debt model.

        Negative debt bounds are handled asymmetrically to avoid
        sign-flipping when the margin multiplier is applied.

        Parameters
        ----------
        model : RiskyDebtModelVFI or None
            The VFI solver from the final iteration.
        k_min, k_max, b_min, b_max : float
            Discovered bounds.
        solution : dict
            VFI solution dictionary.

        Returns
        -------
        dict
            Bounds (original and with margin) plus the VFI solution.
        """
        z_min = float(model.z_min) if model is not None else 0.0
        z_max = float(model.z_max) if model is not None else 0.0

        b_min_margin = (
            b_min * self.margin
            if b_min >= 0
            else b_min * (2.0 - 1.0 / self.margin)
        )

        return {
            "k_bounds_original": (k_min, k_max),
            "k_bounds_add_margin": (
                max(0.0, k_min / self.margin),
                k_max * self.margin,
            ),
            "b_bounds_original": (b_min, b_max),
            "b_bounds_add_margin": (b_min_margin, b_max * self.margin),
            "z_bounds_original": (z_min, z_max),
            "z_bounds_add_margin": (
                z_min / self.margin,
                z_max * self.margin,
            ),
            "vfi_solution": solution,
        }
