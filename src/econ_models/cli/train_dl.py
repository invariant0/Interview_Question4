# econ_models/cli/train_dl.py
"""
Command-line interface for training Deep Learning economic models.

This script orchestrates the training process including configuration
loading, boundary validation, and model instantiation.

Example:
    $ python -m econ_models.cli.train_dl --model basic
    $ python -m econ_models.cli.train_dl --model risky
    $ python -m econ_models.cli.train_dl --model risky_upgrade
    $ python -m econ_models.cli.train_dl --model risk_free
    $ python -m econ_models.cli.train_dl --model risky_final
"""

import argparse
import sys
import logging
import os
from typing import Any

from econ_models.config.economic_params import EconomicParams
from econ_models.config.dl_config import DeepLearningConfig, load_dl_config
from econ_models.config.bond_config import BondsConfig
from econ_models.io.file_utils import load_json_file
import numpy as np
import random

def configure_gpu(gpu_id=None) -> None:
    """Pin this process to a single GPU via CUDA_VISIBLE_DEVICES.
    
    Must be called before any TF operation that initialises a GPU context.
    """
    if gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        logger.info(f"GPU pinned to device {gpu_id}")


def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    import tensorflow as tf
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # For GPU determinism (optional, slower)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname('./')))
DL_CONFIG_FILE = os.path.join(BASE_DIR, "hyperparam/prefixed/dl_params.json")
econ_list_basic = [[0.6, 0.17, 1.0, 0.02], [0.57, 0.2, 0.8, 0.022]]
econ_list_risky = [[0.6, 0.17, 1.0, 0.02, 0.1, 0.08], [0.5, 0.23, 1.5, 0.01, 0.1, 0.1]]

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

def configure_model(args: argparse.Namespace) -> Any:
    """
    Configure and instantiate the appropriate model.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Configured model instance (BasicModelDL or RiskyModelDL).
    """
    logger.info(f"--- Preparing {args.model.capitalize()} Model ---")

    if args.model in ['basic', 'basic_final']:
        econ_id = args.econ_id
        scinario_basic = econ_list_basic[econ_id]
        BASIC_ECON_PARAMS_FILE = os.path.join(BASE_DIR, f"hyperparam/prefixed/econ_params_basic_{scinario_basic[0]}_{scinario_basic[1]}_{scinario_basic[2]}_{scinario_basic[3]}.json")
        BASIC_BOUNDS_FILE = os.path.join(BASE_DIR, f"hyperparam/autogen/bounds_basic_{scinario_basic[0]}_{scinario_basic[1]}_{scinario_basic[2]}_{scinario_basic[3]}.json")

    elif args.model in ['risky_final', 'risk_free']:
        econ_id = args.econ_id
        scinario_risky = econ_list_risky[econ_id]
        RISKY_ECON_PARAMS_FILE = os.path.join(BASE_DIR, f"hyperparam/prefixed/econ_params_risky_{scinario_risky[0]}_{scinario_risky[1]}_{scinario_risky[2]}_{scinario_risky[3]}_{scinario_risky[4]}_{scinario_risky[5]}.json")
        RISKY_BOUNDS_FILE = os.path.join(BASE_DIR, f"hyperparam/autogen/bounds_risky_{scinario_risky[0]}_{scinario_risky[1]}_{scinario_risky[2]}_{scinario_risky[3]}_{scinario_risky[4]}_{scinario_risky[5]}.json")

    # Load DL configuration
    logger.info(f"Loading DL configuration from {DL_CONFIG_FILE}")
    dl_config = load_dl_config(DL_CONFIG_FILE, args.model)

    # Determine and load economic parameters
    params_file = (
        BASIC_ECON_PARAMS_FILE if 'basic' in args.model
        else RISKY_ECON_PARAMS_FILE
    )
    logger.info(f"Loading economic parameters from {params_file}")
    econ_params = EconomicParams(**load_json_file(params_file))

    # Load and validate bounds
    bounds_file = (
        BASIC_BOUNDS_FILE if 'basic' in args.model 
        else RISKY_BOUNDS_FILE
    )
    bounds = BondsConfig.validate_and_load(bounds_file, econ_params)

    # Apply bounds to DL config
    dl_config.capital_min = bounds['k_min']
    dl_config.capital_max = bounds['k_max']
    dl_config.productivity_min = bounds['z_min']
    dl_config.productivity_max = bounds['z_max']

    if 'risk' in args.model:
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

    # Instantiate model  (lazy imports so TF sees CUDA_VISIBLE_DEVICES)
    from econ_models.dl.basic import BasicModelDL
    from econ_models.dl.basic_final import BasicModelDL_FINAL
    from econ_models.dl.basic_final import OptimizationConfig as BasicFinalOptConfig
    from econ_models.dl.risky_final import RiskyModelDL_FINAL
    from econ_models.dl.risky_final import OptimizationConfig as RiskyOptConfig
    from econ_models.dl.risk_free import RiskFreeModelDL

    # Resolve default pretrained checkpoint dir per model
    pretrained_dir = args.pretrained_checkpoint_dir
    if pretrained_dir is None:
        if args.model == 'basic_final':
            pretrained_dir = 'checkpoints_pretrain/basic'
        else:
            pretrained_dir = 'checkpoints_pretrain/risk_free'

    if args.model == 'basic':
        return BasicModelDL(econ_params, dl_config, bounds)
    elif args.model == 'basic_final':
        opt_config = BasicFinalOptConfig()
        return BasicModelDL_FINAL(
            econ_params, dl_config, bounds,
            pretrained_checkpoint_dir=pretrained_dir,
            pretrained_epoch=args.pretrained_epoch,
            optimization_config=opt_config,
        )
    elif args.model == 'risky_final':
        opt_config = RiskyOptConfig()
        return RiskyModelDL_FINAL(
            econ_params, dl_config, bounds,
            pretrained_checkpoint_dir=pretrained_dir,
            pretrained_epoch=args.pretrained_epoch,
            optimization_config=opt_config,
        )
    elif args.model == 'risk_free':
        return RiskFreeModelDL(econ_params, dl_config, bounds)
    
def main():
    """Main entry point for the DL training CLI."""
    parser = argparse.ArgumentParser(
        description="Train Economic Deep Learning Models"
    )
    parser.add_argument(
        '--model',
        type=str,
        default='basic',
        choices=['basic', 'basic_final', 'risky_final', 'risk_free'],
        help="Model architecture to train. "
             "'basic'/'risk_free': pretrain stage (FOC-based, saves to checkpoints_pretrain/). "
             "'basic_final'/'risky_final': final stage (loads pretrained, saves to checkpoints_final/)."
    )
    parser.add_argument(
        '--pretrained_checkpoint_dir',
        type=str,
        default=None,
        help="Directory containing pretrained checkpoint weights. "
             "Defaults to 'checkpoints_pretrain/risk_free' for risky_final, "
             "'checkpoints_pretrain/basic' for basic_final."
    )
    parser.add_argument(
        '--pretrained_epoch',
        type=int,
        default=480,
        help="Epoch number of the risk-free checkpoint to load (used with --training-mode pretrained)."
    )
    parser.add_argument(
        '--gpu',
        type=int,
        default=None,
        help="GPU device ID to use (0-3). If not set, uses all GPUs."
    )
    parser.add_argument(
        '--econ_id',
        type=int,
        default=0,
        help="Economic scenario ID to use for loading parameters and bounds."
    )
    args = parser.parse_args()
    configure_gpu(args.gpu)
    set_seed(42)
    
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
    main()
