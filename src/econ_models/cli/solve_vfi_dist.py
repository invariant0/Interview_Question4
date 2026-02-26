# econ_models/cli/solve_vfi_dist.py
"""
Command-line interface for solving economic models using VFI.

This script orchestrates the VFI solving process including
automatic boundary discovery and result persistence.

Example:
    $ python -m econ_models.cli.solve_vfi_dist --model basic
    $ python -m econ_models.cli.solve_vfi_dist --model risky
"""

import argparse
import dataclasses
import logging
import sys
import json
import os
from typing import Optional, Tuple, Dict, Any

from econ_models.config.economic_params import EconomicParams
from econ_models.config.vfi_config import GridConfig, load_grid_config
from econ_models.io.artifacts import save_vfi_results
from econ_models.io.file_utils import load_json_file, save_boundary_to_json

# Path configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname('./')))
CONFIG_PARAMS_FILE = os.path.join(BASE_DIR, "hyperparam_dist/prefixed/vfi_params_dist.json")
ECON_PARAMS_FILE_BASIC = os.path.join(BASE_DIR, "hyperparam_dist/prefixed/econ_params_basic_dist.json")
ECON_PARAMS_FILE_RISKY = os.path.join(BASE_DIR, "hyperparam_dist/prefixed/econ_params_risky_dist.json")

# Saved output paths
SAVE_BOUNDARY_BASIC = os.path.join(BASE_DIR, "hyperparam_dist/autogen/bounds_basic_dist.json")
SAVE_BOUNDARY_RISKY = os.path.join(BASE_DIR, "hyperparam_dist/autogen/bounds_risky_dist.json")


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

def solve_basic_model(args) -> None:
    """Orchestrate solving the Basic RBC Model."""
    from econ_models.vfi.bounds import BoundaryFinder

    logger.info(f"Loading parameters from {ECON_PARAMS_FILE_BASIC}...")
    params = load_json_file(ECON_PARAMS_FILE_BASIC)
    params = EconomicParams(**params)
    config = load_grid_config(CONFIG_PARAMS_FILE, args.model)
    estimate_params = params.estimate_param if params.estimate_param is not None else {}
    productivity_persistence_min = estimate_params["productivity_persistence"]["min"]
    productivity_persistence_max = estimate_params["productivity_persistence"]["max"]
    productivity_std_dev_min = estimate_params["productivity_std_dev"]["min"]
    productivity_std_dev_max = estimate_params["productivity_std_dev"]["max"]
    adjustment_cost_convex_min = estimate_params["adjustment_cost_convex"]["min"]
    adjustment_cost_convex_max = estimate_params["adjustment_cost_convex"]["max"]
    adjustment_cost_fixed_min = estimate_params["adjustment_cost_fixed"]["min"]
    adjustment_cost_fixed_max = estimate_params["adjustment_cost_fixed"]["max"]


    # ── Corner scenarios for k-bound discovery ────────────────────
    #
    # k_max is driven by: high ρ (persistent good shocks) + high σ
    #   (extreme z draws) + low adjustment costs (cheap to invest).
    # k_min is driven by: high ρ (persistent BAD shocks → long
    #   depreciation spells) + high σ + HIGH adjustment costs
    #   (too expensive to invest even as z partially recovers).
    #
    # Both extremes need ρ_max, so the key variation is adjustment
    # cost (low vs high).  We also include ρ_min corners for
    # completeness (they produce the tightest distributions and
    # confirm that they never dominate the envelope).
    #
    # Full 2×2 factorial:
    #   productivity: {ρ_max, σ_max} vs {ρ_min, σ_max}
    #   adj. costs:   {γ_min, F_min} vs {γ_max, F_max}
    # ─────────────────────────────────────────────────────────────
    productivity_levels = [
        ("high_rho", {
            "productivity_persistence": productivity_persistence_max,
            "productivity_std_dev": productivity_std_dev_max,
        }),
        ("low_rho", {
            "productivity_persistence": productivity_persistence_min,
            "productivity_std_dev": productivity_std_dev_max,
        }),
    ]
    adjustment_cost_levels = [
        ("low_adj", {
            "adjustment_cost_convex": adjustment_cost_convex_min,
            "adjustment_cost_fixed": adjustment_cost_fixed_min,
        }),
        ("high_adj", {
            "adjustment_cost_convex": adjustment_cost_convex_max,
            "adjustment_cost_fixed": adjustment_cost_fixed_max,
        }),
    ]
    scenarios = []
    for prod_label, prod_overrides in productivity_levels:
        for adj_label, adj_overrides in adjustment_cost_levels:
            label = f"{prod_label}_{adj_label}"
            overrides = {**prod_overrides, **adj_overrides}
            scenario = dataclasses.replace(params, estimate_param=None, **overrides)
            scenarios.append((label, scenario))

    logger.info(f"Running {len(scenarios)} corner scenarios for basic bound discovery...")
    k_max_list = []
    k_min_list = []
    z_max_list = []
    z_min_list = []
    for label, scenario_params in scenarios:
        logger.info(
            f"  Scenario {label}: rho={scenario_params.productivity_persistence}, "
            f"sigma={scenario_params.productivity_std_dev}, "
            f"gamma={scenario_params.adjustment_cost_convex}, "
            f"F={scenario_params.adjustment_cost_fixed}"
        )
        scenario_config = dataclasses.replace(
            config, 
            n_capital=200,
        )
        logger.info("Starting automatic boundary discovery for Basic Model...")
        finder = BoundaryFinder(scenario_params, scenario_config, n_steps=1000, n_batches=5000, margin= 1.0)
        bounds_result = finder.find_basic_bounds(max_iters=50)

        k_max_list.append(bounds_result['k_bounds_add_margin'][1])
        k_min_list.append(bounds_result['k_bounds_add_margin'][0])
        z_max_list.append(bounds_result['z_bounds_original'][1])
        z_min_list.append(bounds_result['z_bounds_original'][0])
        logger.info(
            f"    → k=[{bounds_result['k_bounds_add_margin'][0]:.2f}, "
            f"{bounds_result['k_bounds_add_margin'][1]:.2f}]"
        )
    k_bounds = (min(k_min_list), max(k_max_list))
    z_bounds = (min(z_min_list), max(z_max_list))
    bounds_data = {
        "k_min": float(k_bounds[0]),
        "k_max": float(k_bounds[1]),
        "z_min": float(z_bounds[0]),
        "z_max": float(z_bounds[1]),
        "rho_min": float(productivity_persistence_min),
        "rho_max": float(productivity_persistence_max),
        "std_min": float(productivity_std_dev_min),
        "std_max": float(productivity_std_dev_max),
        "convex_min": float(adjustment_cost_convex_min),
        "convex_max": float(adjustment_cost_convex_max),
        "fixed_min": float(adjustment_cost_fixed_min),
        "fixed_max": float(adjustment_cost_fixed_max)
    }
    save_boundary_to_json(SAVE_BOUNDARY_BASIC, bounds_data, params)
    logger.info(
        f"\nFinal Boundary Determined: \n"
        f"K={k_bounds}, \n"
        f"Z={z_bounds}\n"
    )


def solve_risky_model(args) -> None:
    """Orchestrate solving the Risky Debt Model.

    Unlike the basic model where parameter→bound mapping is roughly monotonic
    (2 extreme scenarios suffice), the risky model has NON-MONOTONIC interactions
    between equity issuance costs and adjustment costs on the debt dimension:

      - b_max is driven by: large firm (low adj. cost) + expensive equity (high η)
        → firm is forced to finance growth via debt
      - b_min is driven by: small firm (high adj. cost) + cheap equity (low η)
        → firm hoards cash (precautionary savings)
      - k_max is driven by: low adj. cost + low equity cost + high persistence
      - k_min is driven by: high adj. cost + high equity cost + low persistence

    These extremes require OPPOSITE equity cost settings paired with OPPOSITE
    adjustment cost settings, so no single pair of scenarios can capture all
    bound extremes simultaneously.

    Solution: enumerate 2^3 = 8 corner scenarios over 3 parameter groups:
      1. Productivity (ρ, σ):       {high ρ, high σ} vs {low ρ, high σ}
      2. Adjustment costs (ψ₀, ψ₁): {min} vs {max}
      3. Equity costs (η₀, η₁):     {min} vs {max}
    Then take the envelope (min of mins, max of maxes) across all 8 runs.
    """
    from econ_models.vfi.bounds import BoundaryFinder

    logger.info(f"Loading parameters from {ECON_PARAMS_FILE_RISKY}...")
    params = load_json_file(ECON_PARAMS_FILE_RISKY)
    params = EconomicParams(**params)
    config = load_grid_config(CONFIG_PARAMS_FILE, args.model)

    estimate_params = params.estimate_param if params.estimate_param is not None else {}
    productivity_persistence_min = estimate_params["productivity_persistence"]["min"]
    productivity_persistence_max = estimate_params["productivity_persistence"]["max"]
    productivity_std_dev_min = estimate_params["productivity_std_dev"]["min"]
    productivity_std_dev_max = estimate_params["productivity_std_dev"]["max"]
    adjustment_cost_convex_min = estimate_params["adjustment_cost_convex"]["min"]
    adjustment_cost_convex_max = estimate_params["adjustment_cost_convex"]["max"]
    adjustment_cost_fixed_min = estimate_params["adjustment_cost_fixed"]["min"]
    adjustment_cost_fixed_max = estimate_params["adjustment_cost_fixed"]["max"]
    equity_issuance_cost_fixed_min = estimate_params["equity_issuance_cost_fixed"]["min"]
    equity_issuance_cost_fixed_max = estimate_params["equity_issuance_cost_fixed"]["max"]
    equity_issuance_cost_linear_min = estimate_params["equity_issuance_cost_linear"]["min"]
    equity_issuance_cost_linear_max = estimate_params["equity_issuance_cost_linear"]["max"]

    # --- Build 2^3 = 8 corner scenarios ---
    # Dimension levels: (label, param_overrides)
    productivity_levels = [
        ("high_rho", {
            "productivity_persistence": productivity_persistence_max,
            "productivity_std_dev": productivity_std_dev_max,
        }),
        ("low_rho", {
            "productivity_persistence": productivity_persistence_min,
            "productivity_std_dev": productivity_std_dev_max,  # always max σ for widest z
        }),
    ]
    adjustment_cost_levels = [
        ("low_adj", {
            "adjustment_cost_convex": adjustment_cost_convex_min,
            "adjustment_cost_fixed": adjustment_cost_fixed_min,
        }),
        ("high_adj", {
            "adjustment_cost_convex": adjustment_cost_convex_max,
            "adjustment_cost_fixed": adjustment_cost_fixed_max,
        }),
    ]
    equity_cost_levels = [
        ("low_eq", {
            "equity_issuance_cost_fixed": equity_issuance_cost_fixed_min,
            "equity_issuance_cost_linear": equity_issuance_cost_linear_min,
        }),
        ("high_eq", {
            "equity_issuance_cost_fixed": equity_issuance_cost_fixed_max,
            "equity_issuance_cost_linear": equity_issuance_cost_linear_max,
        }),
    ]

    # Full factorial: 2 × 2 × 2 = 8 scenarios
    scenarios = []
    for prod_label, prod_overrides in productivity_levels:
        for adj_label, adj_overrides in adjustment_cost_levels:
            for eq_label, eq_overrides in equity_cost_levels:
                label = f"{prod_label}_{adj_label}_{eq_label}"
                overrides = {**prod_overrides, **adj_overrides, **eq_overrides}
                scenario = dataclasses.replace(params, estimate_param=None, **overrides)
                scenarios.append((label, scenario))

    logger.info(f"Running {len(scenarios)} corner scenarios for risky bound discovery...")

    # Grid resolution for boundary search — large enough that expanding
    # bounds doesn't degrade solution quality.
    BOUND_N_K = 200
    BOUND_N_B = 200
    BOUND_N_Z = config.n_productivity

    # Pre-compute optimal chunk sizes (VRAM-aware, same logic as golden VFI finder)
    k_cs, b_cs, kp_cs, bp_cs = BoundaryFinder.compute_optimal_chunks(
        BOUND_N_K, BOUND_N_B, BOUND_N_Z
    )
    logger.info(
        f"Boundary search grid: ({BOUND_N_K},{BOUND_N_B},{BOUND_N_Z})  "
        f"chunks: k={k_cs}, b={b_cs}, kp={kp_cs}, bp={bp_cs}"
    )

    k_max_list = []
    k_min_list = []
    b_max_list = []
    b_min_list = []
    z_max_list = []
    z_min_list = []
    for label, scenario_params in scenarios:
        logger.info(
            f"\n--- Scenario: {label} ---\n"
            f"  rho={scenario_params.productivity_persistence}, sigma={scenario_params.productivity_std_dev}\n"
            f"  psi0={scenario_params.adjustment_cost_convex}, psi1={scenario_params.adjustment_cost_fixed}\n"
            f"  eta0={scenario_params.equity_issuance_cost_fixed}, eta1={scenario_params.equity_issuance_cost_linear}"
        )
        scenario_config = dataclasses.replace(
            config,
            n_capital=BOUND_N_K,
            n_debt=BOUND_N_B,
        )
        finder = BoundaryFinder(
            scenario_params, scenario_config,
            n_steps=1000, n_batches=5000,
            k_chunk_size=k_cs, b_chunk_size=b_cs,
            kp_chunk_size=kp_cs, bp_chunk_size=bp_cs, margin=1.1
        )
        bounds_result = finder.find_risky_bounds(max_iters=50)

        k_min_list.append(bounds_result['k_bounds_add_margin'][0])
        k_max_list.append(bounds_result['k_bounds_add_margin'][1])
        b_min_list.append(bounds_result['b_bounds_add_margin'][0])
        b_max_list.append(bounds_result['b_bounds_add_margin'][1])
        z_min_list.append(bounds_result['z_bounds_original'][0])
        z_max_list.append(bounds_result['z_bounds_original'][1])

        logger.info(
            f"  → k=[{bounds_result['k_bounds_add_margin'][0]:.4f}, {bounds_result['k_bounds_add_margin'][1]:.4f}], "
            f"b=[{bounds_result['b_bounds_add_margin'][0]:.4f}, {bounds_result['b_bounds_add_margin'][1]:.4f}]"
        )

    # Envelope: take the extremes across all 8 scenarios
    k_bounds = (min(k_min_list), max(k_max_list))
    b_bounds = (min(b_min_list), max(b_max_list))
    z_bounds = (min(z_min_list), max(z_max_list))
   
    logger.info(
        f"\n{'='*60}\n"
        f"Envelope across {len(scenarios)} scenarios:\n"
        f"  K = [{k_bounds[0]:.4f}, {k_bounds[1]:.4f}]\n"
        f"  B = [{b_bounds[0]:.4f}, {b_bounds[1]:.4f}]\n"
        f"  Z = [{z_bounds[0]:.4f}, {z_bounds[1]:.4f}]\n"
        f"{'='*60}"
    )

    bounds_data = {
        "k_min": float(k_bounds[0]),
        "k_max": float(k_bounds[1]),
        "b_min": float(b_bounds[0]),
        "b_max": float(b_bounds[1]),
        "z_min": float(z_bounds[0]),
        "z_max": float(z_bounds[1]),
        "rho_min": float(productivity_persistence_min),
        "rho_max": float(productivity_persistence_max),
        "std_min": float(productivity_std_dev_min),
        "std_max": float(productivity_std_dev_max),
        "convex_min": float(adjustment_cost_convex_min),
        "convex_max": float(adjustment_cost_convex_max),
        "fixed_min": float(adjustment_cost_fixed_min),
        "fixed_max": float(adjustment_cost_fixed_max),
        "eta0_min": float(equity_issuance_cost_fixed_min),
        "eta0_max": float(equity_issuance_cost_fixed_max),
        "eta1_min": float(equity_issuance_cost_linear_min),
        "eta1_max": float(equity_issuance_cost_linear_max),
    }
    save_boundary_to_json(SAVE_BOUNDARY_RISKY, bounds_data, params)
    logger.info(
        f"\nFinal Boundary Determined: \n"
        f"K={k_bounds}, \n"
        f"B={b_bounds}, \n"
        f"Z={z_bounds}\n"
    )


def configure_gpu(gpu_id=None) -> None:
    """Pin this process to a single GPU via CUDA_VISIBLE_DEVICES."""
    if gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        logger.info(f"GPU pinned to device {gpu_id}")


def main():
    """Main entry point for the VFI solver CLI."""
    parser = argparse.ArgumentParser(description="Solve Models via VFI")
    parser.add_argument(
        '--model',
        type=str,
        default='basic',
        choices=['basic', 'risky'],
        help="Model type to solve."
    )
    parser.add_argument(
        '--gpu',
        type=int,
        default=None,
        help="GPU device ID to use (0-3). If not set, uses all GPUs."
    )
    args = parser.parse_args()
    configure_gpu(args.gpu)

    try:
        if args.model == 'basic':
            solve_basic_model(args)
        elif args.model == 'risky':
            solve_risky_model(args)
    except Exception as e:
        logger.error(f"Solver failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
