# econ_models/cli/train_dl.py
"""
Command-line interface for training Deep Learning economic models.

This script orchestrates the training process including configuration
loading, boundary validation, and model instantiation.

Example:
    $ python -m econ_models.cli.train_dl --model basic
    $ python -m econ_models.cli.train_dl --model risky
"""

import argparse
import sys
import logging
import os
from typing import Any

from econ_models.config.economic_params import EconomicParams, load_economic_params
from econ_models.config.dl_config import DeepLearningConfig, load_dl_config
from econ_models.dl.basic import BasicModelDL
from econ_models.dl.risky import RiskyModelDL
from econ_models.dl.risky_upgrade import RiskyModelDL_UPGRADE
from econ_models.io.file_utils import load_json_file
import numpy as np
import tensorflow as tf
import random

def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # For GPU determinism (optional, slower)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

# Path configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname('./')))
BASIC_ECON_PARAMS_FILE = os.path.join(BASE_DIR, "hyperparam/prefixed/econ_params_basic.json")
RISKY_ECON_PARAMS_FILE = os.path.join(BASE_DIR, "hyperparam/prefixed/econ_params_risky.json")
BASIC_BOUNDS_FILE = os.path.join(BASE_DIR, "hyperparam/autogen/bounds_basic.json")
RISKY_BOUNDS_FILE = os.path.join(BASE_DIR, "hyperparam/autogen/bounds_risky.json")
DL_CONFIG_FILE = os.path.join(BASE_DIR, "hyperparam/prefixed/dl_params.json")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class BoundaryValidator:
    """
    Validates that pre-computed bounds match current parameters.

    This ensures consistency between the VFI-computed boundaries
    and the parameters used for deep learning training.
    """

    @staticmethod
    def validate_and_load(
        bounds_file: str,
        current_params: EconomicParams
    ) -> dict:
        """
        Load bounds and validate parameter consistency.

        Args:
            bounds_file: Path to the bounds JSON file.
            current_params: Current economic parameters to validate against.

        Returns:
            Dictionary of validated boundary values.

        Raises:
            SystemExit: If validation fails or file is missing.
        """
        if not os.path.exists(bounds_file):
            logger.error(
                f"Boundary file '{bounds_file}' missing. "
                "Run 'solve_vfi.py --auto-bounds' first."
            )
            sys.exit(1)

        data = load_json_file(bounds_file)

        if "source_params" not in data or "bounds" not in data:
            logger.error(
                "Invalid boundary file format. "
                "Please re-run 'solve_vfi.py --auto-bounds'."
            )
            sys.exit(1)

        BoundaryValidator._check_parameter_consistency(
            data["source_params"],
            current_params.__dict__
        )

        logger.info("Boundary validation successful.")
        return data["bounds"]

    @staticmethod
    def _check_parameter_consistency(
        stored_params: dict,
        current_params: dict
    ) -> None:
        """Check for mismatches between stored and current parameters."""
        mismatches = []

        for key, val in current_params.items():
            if key in stored_params:
                if stored_params[key] != val:
                    mismatches.append(
                        f"{key}: stored={stored_params[key]}, current={val}"
                    )
            else:
                mismatches.append(f"{key} missing in stored bounds")

        if mismatches:
            logger.critical(
                "Parameter mismatch detected between current config "
                "and pre-computed bounds."
            )
            for m in mismatches:
                logger.error(f"  - {m}")
            logger.error(
                "Action required: Re-run 'solve_vfi.py --auto-bounds' "
                "with the current parameter file."
            )
            sys.exit(1)


def configure_model(args: argparse.Namespace) -> Any:
    """
    Configure and instantiate the appropriate model.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Configured model instance (BasicModelDL or RiskyModelDL).
    """
    logger.info(f"--- Preparing {args.model.capitalize()} Model ---")

    # Load DL configuration
    logger.info(f"Loading DL configuration from {DL_CONFIG_FILE}")
    dl_config = load_dl_config(DL_CONFIG_FILE, args.model)

    # Determine and load economic parameters
    params_file = (
        BASIC_ECON_PARAMS_FILE if args.model == 'basic'
        else RISKY_ECON_PARAMS_FILE
    )
    logger.info(f"Loading economic parameters from {params_file}")
    econ_params = load_economic_params(params_file)

    # Load and validate bounds
    bounds_file = (
        BASIC_BOUNDS_FILE if args.model == 'basic'
        else RISKY_BOUNDS_FILE
    )
    bounds = BoundaryValidator.validate_and_load(bounds_file, econ_params)

    # Apply bounds to DL config
    dl_config.capital_min = bounds['k_min']
    dl_config.capital_max = bounds['k_max']
    dl_config.productivity_min = bounds['z_min']
    dl_config.productivity_max = bounds['z_max']

    if args.model == 'risky' or args.model == 'risky_upgrade':
        dl_config.debt_min = bounds['b_min']
        dl_config.debt_max = bounds['b_max']
        logger.info(
            f"Boundary for state variables: "
            f"K=[{dl_config.capital_min:.2f}, {dl_config.capital_max:.2f}], "
            f"B=[{dl_config.debt_min:.2f}, {dl_config.debt_max:.2f}]"
        )
    else:
        logger.info(
            f"Boundary for state variables: "
            f"K=[{dl_config.capital_min:.2f}, {dl_config.capital_max:.2f}]"
        )

    logger.info(
        f"Config: Batch={dl_config.batch_size}, "
        f"LR={dl_config.learning_rate}, Epochs={dl_config.epochs}"
    )

    # Instantiate model
    if args.model == 'basic':
        return BasicModelDL(econ_params, dl_config)
    elif args.model == 'risky':
        return RiskyModelDL(econ_params, dl_config)
    elif args.model == 'risky_upgrade':
        return RiskyModelDL_UPGRADE(econ_params, dl_config)


def main():
    """Main entry point for the DL training CLI."""
    parser = argparse.ArgumentParser(
        description="Train Economic Deep Learning Models"
    )
    parser.add_argument(
        '--model',
        type=str,
        default='basic',
        choices=['basic', 'risky', 'risky_upgrade'],
        help="Model architecture to train."
    )
    args = parser.parse_args()

    try:
        model = configure_model(args)
        model.train()
    except KeyboardInterrupt:
        logger.warning("\nTraining interrupted by user.")
        sys.exit(0)
    except Exception as e:
        logger.exception(f"Fatal error during execution: {e}")
        sys.exit(1)


if __name__ == "__main__":
    set_seed(42)
    main()