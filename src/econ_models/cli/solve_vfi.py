# econ_models/cli/solve_vfi.py
"""
Command-line interface for solving economic models using VFI.

This script orchestrates the VFI solving process including
automatic boundary discovery and result persistence.

Example:
    $ python -m econ_models.cli.solve_vfi --model basic
    $ python -m econ_models.cli.solve_vfi --model risky
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

# VFI config
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname('./')))
CONFIG_PARAMS_FILE = os.path.join(BASE_DIR, "hyperparam/prefixed/vfi_params.json")


# BASIC config 
# scinario_basic = [0.6, 0.17, 1.0, 0.02]
scinario_basic = [0.57, 0.2, 0.8, 0.022]
CONFIG_PARAMS_FILE = os.path.join(BASE_DIR, "hyperparam/prefixed/vfi_params.json")
ECON_PARAMS_FILE_BASIC = os.path.join(BASE_DIR, f"hyperparam/prefixed/econ_params_basic_{scinario_basic[0]}_{scinario_basic[1]}_{scinario_basic[2]}_{scinario_basic[3]}.json")
SAVE_BOUNDARY_BASIC = os.path.join(BASE_DIR, f"hyperparam/autogen/bounds_basic_{scinario_basic[0]}_{scinario_basic[1]}_{scinario_basic[2]}_{scinario_basic[3]}.json")
SAVE_GROUND_TRUTH_BASIC = os.path.join(BASE_DIR, f"ground_truth_basic/basic_model_vfi_results_{scinario_basic[0]}_{scinario_basic[1]}_{scinario_basic[2]}_{scinario_basic[3]}.npz")
os.makedirs(os.path.dirname(SAVE_GROUND_TRUTH_BASIC), exist_ok=True)
PARAMS_BASIC = load_json_file(ECON_PARAMS_FILE_BASIC)
PARAMS_BASIC = EconomicParams(**PARAMS_BASIC)

# RISKY config
# scinario_risky = [0.6, 0.17, 1.0, 0.02, 0.1, 0.08]
scinario_risky = [0.5, 0.23, 1.5, 0.01, 0.1, 0.1]

ECON_PARAMS_FILE_RISKY = os.path.join(BASE_DIR, f"hyperparam/prefixed/econ_params_risky_{scinario_risky[0]}_{scinario_risky[1]}_{scinario_risky[2]}_{scinario_risky[3]}_{scinario_risky[4]}_{scinario_risky[5]}.json")
SAVE_BOUNDARY_RISKY = os.path.join(BASE_DIR, f"hyperparam/autogen/bounds_risky_{scinario_risky[0]}_{scinario_risky[1]}_{scinario_risky[2]}_{scinario_risky[3]}_{scinario_risky[4]}_{scinario_risky[5]}.json")
SAVE_GROUND_TRUTH_RISKY = os.path.join(BASE_DIR, f"ground_truth_risky/risky_debt_model_vfi_results_{scinario_risky[0]}_{scinario_risky[1]}_{scinario_risky[2]}_{scinario_risky[3]}_{scinario_risky[4]}_{scinario_risky[5]}.npz")
os.makedirs(os.path.dirname(SAVE_GROUND_TRUTH_RISKY), exist_ok=True)
PARAMS_RISKY = load_json_file(ECON_PARAMS_FILE_RISKY)
PARAMS_RISKY = EconomicParams(**PARAMS_RISKY)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

def solve_basic_model(args) -> None:
    """Orchestrate solving the Basic RBC Model."""
    from econ_models.vfi.basic import BasicModelVFI
    from econ_models.vfi.bounds import BoundaryFinder

    logger.info(f"Loading parameters from {ECON_PARAMS_FILE_BASIC}...")

    config = load_grid_config(CONFIG_PARAMS_FILE, args.model)

    if args.find_bounds:
        config = dataclasses.replace(
            config, 
            n_capital=200,
        )
        logger.info("Starting automatic boundary discovery for Basic Model...")
        finder = BoundaryFinder(PARAMS_BASIC, config, n_steps=1000, n_batches=5000, margin= 1.1)
        bounds_result = finder.find_basic_bounds(max_iters=50)

        k_bounds = bounds_result['k_bounds_add_margin']
        z_bounds = bounds_result['z_bounds_original']
        bounds_data = {
            "k_min": float(k_bounds[0]),
            "k_max": float(k_bounds[1]),
            "z_min": float(z_bounds[0]),
            "z_max": float(z_bounds[1])
        }
        save_boundary_to_json(SAVE_BOUNDARY_BASIC, bounds_data, PARAMS_BASIC)
        logger.info(
            f"\nFinal Boundary Determined: \n"
            f"K={bounds_result['k_bounds_add_margin']}, \n"
            f"Z={bounds_result['z_bounds_original']}\n"
        )
    else:
        logger.info(f"Loading boundaries from {SAVE_BOUNDARY_BASIC}...")
        bounds_json = load_json_file(SAVE_BOUNDARY_BASIC)
        bounds_result = {
            'k_bounds_add_margin': (
                bounds_json['bounds']['k_min'],
                bounds_json['bounds']['k_max']
            )
        }
    if args.solve:
        config = load_grid_config(CONFIG_PARAMS_FILE, args.model)
        logger.info(f"Solving Ground Truth Basic Model with grid size n_capital = {config.n_capital} n_productivity = {config.n_productivity}...")
        solver = BasicModelVFI(
            PARAMS_BASIC, config, k_bounds=bounds_result['k_bounds_add_margin']
        )
        res = solver.solve()

        save_vfi_results(res, SAVE_GROUND_TRUTH_BASIC)
        logger.info(f"VFI Results saved to {SAVE_GROUND_TRUTH_BASIC}")


def solve_risky_model(args) -> None:
    """Orchestrate solving the Risky Debt Model."""
    from econ_models.vfi.risky import RiskyDebtModelVFI
    from econ_models.vfi.bounds import BoundaryFinder

    logger.info(f"Loading parameters from {ECON_PARAMS_FILE_RISKY}...")
    config = load_grid_config(CONFIG_PARAMS_FILE, args.model)
    if args.find_bounds:
        config = dataclasses.replace(
            config, 
            n_capital=100,
            n_debt=100,
        )
        logger.info("Starting automatic boundary discovery for Risky Debt Model with Low Resolution...")
        finder = BoundaryFinder(PARAMS_RISKY, config, n_steps=1000, n_batches=5000, k_chunk_size=80, b_chunk_size=80, margin = 1.1)
        bounds_result = finder.find_risky_bounds(max_iters=50)

        k_bounds = bounds_result['k_bounds_add_margin']
        b_bounds = bounds_result['b_bounds_add_margin']
        z_bounds = bounds_result['z_bounds_original']

        bounds_data = {
            "k_min": float(k_bounds[0]),
            "k_max": float(k_bounds[1]),
            "b_min": float(b_bounds[0]),
            "b_max": float(b_bounds[1]),
            "z_min": float(z_bounds[0]),
            "z_max": float(z_bounds[1])
        }
        save_boundary_to_json(SAVE_BOUNDARY_RISKY, bounds_data, PARAMS_RISKY)
        logger.info(
            f"\nFinal Boundary Determined: \n"
            f"K={bounds_result['k_bounds_add_margin']},\n"
            f"B={bounds_result['b_bounds_add_margin']},\n"
            f"Z={bounds_result['z_bounds_original']}\n"
        )
    else:
        logger.info(f"Loading boundaries from {SAVE_BOUNDARY_RISKY}...")
        bounds_json = load_json_file(SAVE_BOUNDARY_RISKY)
        bounds_result = {
            'k_bounds_add_margin': (
                bounds_json['bounds']['k_min'],
                bounds_json['bounds']['k_max']
            ),
            'b_bounds_add_margin': (
                bounds_json['bounds']['b_min'],
                bounds_json['bounds']['b_max']
            ),
            'z_bounds_original': (
                bounds_json['bounds']['z_min'],
                bounds_json['bounds']['z_max']
            )
        }
    if args.solve:
        config = load_grid_config(CONFIG_PARAMS_FILE, args.model)
        logger.info(f"Solving Ground Truth Risky Debt Model With n_capital = {config.n_capital} and n_debt = {config.n_debt}...")
        # Benchmark-optimised chunk sizes for RTX 5090 (32 GB).
        # Full-grid state dims (k=n, b=n) with small choice tiles (kp=50, bp=50)
        # yielded 2.23× speedup over previous (n/2, n/2, 100, 100) config.
        # Smaller tiles ⇒ smaller reduce_max dimension ⇒ faster XLA kernels,
        # and only ~6 GB VRAM despite the full state-dim coverage.
        _KP_BP_CHUNK = min(config.n_capital // 10, 50)
        solver = RiskyDebtModelVFI(
            PARAMS_RISKY, config,
            k_bounds=bounds_result['k_bounds_add_margin'],
            b_bounds=bounds_result['b_bounds_add_margin'],
            k_chunk_size=config.n_capital,
            b_chunk_size=config.n_debt,
            kp_chunk_size=_KP_BP_CHUNK,
            bp_chunk_size=_KP_BP_CHUNK,
        )
        res = solver.solve()

        save_vfi_results(res, SAVE_GROUND_TRUTH_RISKY)
        logger.info(f"VFI Results saved to {SAVE_GROUND_TRUTH_RISKY}")


def configure_gpu(gpu_id: Optional[int]) -> None:
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
        '--find_bounds',
        action='store_true',
        help="Flag to find boundaries automatically."
    )

    parser.add_argument(
        '--solve',
        action='store_true',
        help="Flag to solve the model."
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
