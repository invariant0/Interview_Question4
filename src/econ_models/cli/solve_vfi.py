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
import logging
import sys
import json
import os
from typing import Optional, Tuple, Dict, Any

from econ_models.config.economic_params import EconomicParams
from econ_models.config.vfi_config import GridConfig, load_grid_config
from econ_models.vfi.basic import BasicModelVFI
from econ_models.vfi.risky_debt import RiskyDebtModelVFI
from econ_models.vfi.bounds import BoundaryFinder
from econ_models.io.artifacts import save_vfi_results
from econ_models.io.file_utils import load_json_file

# Path configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname('./')))
CONFIG_PARAMS_FILE = os.path.join(BASE_DIR, "hyperparam/prefixed/vfi_params.json")
ECON_PARAMS_FILE_BASIC = os.path.join(BASE_DIR, "hyperparam/prefixed/econ_params_basic.json")
ECON_PARAMS_FILE_RISKY = os.path.join(BASE_DIR, "hyperparam/prefixed/econ_params_risky.json")
SAVE_BOUNDARY_BASIC = os.path.join(BASE_DIR, "hyperparam/autogen/bounds_basic.json")
SAVE_BOUNDARY_RISKY = os.path.join(BASE_DIR, "hyperparam/autogen/bounds_risky.json")
SAVE_GROUND_TRUTH_BASIC = os.path.join(BASE_DIR, "ground_truth/basic_model_vfi_results.npz")
SAVE_GROUND_TRUTH_RISKY = os.path.join(BASE_DIR, "ground_truth/risky_debt_model_vfi_results.npz")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_econ_params_from_json(filename: str) -> EconomicParams:
    """
    Load economic parameters from a JSON file.

    Args:
        filename: Path to the JSON file.

    Returns:
        Populated EconomicParams instance.

    Raises:
        SystemExit: If file cannot be read or is invalid.
    """
    if not os.path.exists(filename):
        logger.error(f"Parameter file '{filename}' not found.")
        sys.exit(1)

    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        return EconomicParams(**data)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {filename}: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to load parameters from {filename}: {e}")
        sys.exit(1)


def save_boundary_to_json(
    filename: str,
    bounds_data: Dict[str, float],
    params: EconomicParams
) -> None:
    """
    Save boundary data and source parameters to JSON.

    Args:
        filename: Target file path.
        bounds_data: Dictionary of boundary values.
        params: Economic parameters used to generate bounds.
    """
    export_data = {
        "bounds": bounds_data,
        "source_params": params.__dict__
    }

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    try:
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=4)
        logger.info(f"Boundaries saved to {filename}")
    except IOError as e:
        logger.error(f"Failed to save boundaries to {filename}: {e}")


def solve_basic_model(config_params_file: str, econ_params_file: str) -> None:
    """Orchestrate solving the Basic RBC Model."""
    logger.info(f"Loading parameters from {econ_params_file}...")
    params = load_econ_params_from_json(econ_params_file)
    config = load_grid_config(config_params_file, "basic")

    logger.info("Starting automatic boundary discovery for Basic Model...")
    finder = BoundaryFinder(params, config)
    bounds_result = finder.find_basic_bounds()

    k_bounds = bounds_result['k_bounds_add_margin']
    z_bounds = bounds_result['z_bounds_add_margin']
    bounds_data = {
        "k_min": float(k_bounds[0]),
        "k_max": float(k_bounds[1]),
        "z_min": float(z_bounds[0]),
        "z_max": float(z_bounds[1])
    }
    save_boundary_to_json(SAVE_BOUNDARY_BASIC, bounds_data, params)
    logger.info(
        f"\nFinal Boundary Determined: \n"
        f"K={bounds_result['k_bounds_original']}, \n"
        f"Z={bounds_result['z_bounds_original']}\n"
    )

    logger.info("Solving Ground Truth Basic Model...")
    solver = BasicModelVFI(
        params, config, k_bounds=bounds_result['k_bounds_original']
    )
    res = solver.solve()

    save_vfi_results(res, SAVE_GROUND_TRUTH_BASIC)
    logger.info(f"VFI Results saved to {SAVE_GROUND_TRUTH_BASIC}")


def solve_risky_model(config_params_file: str, econ_params_file: str) -> None:
    """Orchestrate solving the Risky Debt Model."""
    logger.info(f"Loading parameters from {econ_params_file}...")
    params = load_econ_params_from_json(econ_params_file)
    config = load_grid_config(config_params_file, "risky")

    logger.info("Starting automatic boundary discovery for Risky Debt Model...")
    finder = BoundaryFinder(params, config)
    bounds_result = finder.find_risky_bounds()

    k_bounds = bounds_result['k_bounds_add_margin']
    b_bounds = bounds_result['b_bounds_add_margin']
    z_bounds = bounds_result['z_bounds_add_margin']

    bounds_data = {
        "k_min": float(k_bounds[0]),
        "k_max": float(k_bounds[1]),
        "b_min": float(b_bounds[0]),
        "b_max": float(b_bounds[1]),
        "z_min": float(z_bounds[0]),
        "z_max": float(z_bounds[1])
    }
    save_boundary_to_json(SAVE_BOUNDARY_RISKY, bounds_data, params)
    logger.info(
        f"\nFinal Boundary Determined: \n"
        f"K={bounds_result['k_bounds_original']},\n"
        f"B={bounds_result['b_bounds_original']},\n"
        f"Z={bounds_result['z_bounds_original']}\n"
    )

    logger.info("Solving Ground Truth Risky Debt Model...")
    solver = RiskyDebtModelVFI(
        params, config,
        k_bounds=bounds_result['k_bounds_original'],
        b_bounds=bounds_result['b_bounds_original']
    )
    res = solver.solve()

    save_vfi_results(res, SAVE_GROUND_TRUTH_RISKY)
    logger.info(f"VFI Results saved to {SAVE_GROUND_TRUTH_RISKY}")


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
    args = parser.parse_args()

    try:
        if args.model == 'basic':
            solve_basic_model(CONFIG_PARAMS_FILE, ECON_PARAMS_FILE_BASIC)
        else:
            solve_risky_model(CONFIG_PARAMS_FILE, ECON_PARAMS_FILE_RISKY)
    except Exception as e:
        logger.error(f"Solver failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()