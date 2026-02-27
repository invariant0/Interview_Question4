# econ_models/cli/train_dl.py
"""
Command-line interface for training Deep Learning economic models.

This script orchestrates the training process including configuration
loading, boundary validation, and model instantiation.

Example:
    $ python -m econ_models.cli.train_dl_dist --model basic
    $ python -m econ_models.cli.train_dl_dist --model risky
    $ python -m econ_models.cli.train_dl_dist --model risky_upgrade
    $ python -m econ_models.cli.train_dl_dist --model risk_free
    $ python -m econ_models.cli.train_dl_dist --model risky_final
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

# Path configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname('./')))
BASIC_ECON_PARAMS_FILE = os.path.join(BASE_DIR, "hyperparam_dist/prefixed/econ_params_basic_dist.json")
RISKY_ECON_PARAMS_FILE = os.path.join(BASE_DIR, "hyperparam_dist/prefixed/econ_params_risky_dist.json")
BASIC_BOUNDS_FILE = os.path.join(BASE_DIR, "hyperparam_dist/autogen/bounds_basic_dist.json")
RISKY_BOUNDS_FILE = os.path.join(BASE_DIR, "hyperparam_dist/autogen/bounds_risky_dist.json")
DL_CONFIG_FILE = os.path.join(BASE_DIR, "hyperparam_dist/prefixed/dl_params_dist.json")

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

    # Load DL configuration
    # --config-key overrides the JSON key used for hyperparameters
    config_key = getattr(args, 'config_key', None) or args.model
    logger.info(f"Loading DL configuration from {DL_CONFIG_FILE} (key={config_key})")
    dl_config = load_dl_config(DL_CONFIG_FILE, config_key)

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

    # Resolve checkpoint and log directories (CLI overrides defaults)
    checkpoint_dir = getattr(args, 'checkpoint_dir', None) or None
    log_dir = getattr(args, 'log_dir', None) or None

    # Lazy imports so TF sees CUDA_VISIBLE_DEVICES set by configure_gpu()
    from econ_models.dl_dist.basic import BasicModelDL
    from econ_models.dl_dist.basic_final import BasicModelDL_FINAL, OptimizationConfig
    from econ_models.dl_dist.risky_final import RiskyModelDL_FINAL
    from econ_models.dl_dist.risky_final import OptimizationConfig as RiskyOptimizationConfig
    from econ_models.dl_dist.risk_free import RiskFreeModelDL

    # Instantiate model
    if args.model == 'basic':
        opt_config = OptimizationConfig(
            cross_product_sampling=bool(dl_config.cross_product_sampling),
            batch_size_states=dl_config.batch_size_states,
            batch_size_params=dl_config.batch_size_params,
        )
        extra_kwargs = {}
        if checkpoint_dir:
            extra_kwargs['checkpoint_dir'] = checkpoint_dir
        if log_dir:
            extra_kwargs['log_dir_prefix'] = log_dir
        return BasicModelDL(econ_params, dl_config, bounds, optimization_config=opt_config, **extra_kwargs)
    elif args.model == 'basic_final':
        opt_config = OptimizationConfig(
            cross_product_sampling=bool(dl_config.cross_product_sampling),
            batch_size_states=dl_config.batch_size_states,
            batch_size_params=dl_config.batch_size_params,
        )
        pretrained_dir = getattr(args, 'pretrained_dir', None) or "checkpoints_pretrain_dist/basic"
        pretrained_epoch = args.pretrained_epoch
        extra_kwargs = {}
        if checkpoint_dir:
            extra_kwargs['checkpoint_dir'] = checkpoint_dir
        if log_dir:
            extra_kwargs['log_dir_prefix'] = log_dir
        return BasicModelDL_FINAL(
            econ_params, dl_config, bounds,
            pretrained_checkpoint_dir=pretrained_dir,
            pretrained_epoch=int(pretrained_epoch),
            optimization_config=opt_config,
            **extra_kwargs,
        )
    elif args.model == 'risk_free':
        opt_config = OptimizationConfig(
            cross_product_sampling=bool(dl_config.cross_product_sampling),
            batch_size_states=dl_config.batch_size_states,
            batch_size_params=dl_config.batch_size_params,
        )
        extra_kwargs = {}
        if checkpoint_dir:
            extra_kwargs['checkpoint_dir'] = checkpoint_dir
        if log_dir:
            extra_kwargs['log_dir_prefix'] = log_dir
        return RiskFreeModelDL(econ_params, dl_config, bounds, optimization_config=opt_config, **extra_kwargs)
    elif args.model == 'risky_final':
        opt_config = RiskyOptimizationConfig(
            cross_product_sampling=bool(dl_config.cross_product_sampling),
            batch_size_states=dl_config.batch_size_states,
            batch_size_params=dl_config.batch_size_params,
        )
        pretrained_dir = getattr(args, 'pretrained_dir', None) or "checkpoints_pretrain_dist/risk_free"
        pretrained_epoch = args.pretrained_epoch
        extra_kwargs = {}
        if log_dir:
            extra_kwargs['log_dir_prefix'] = log_dir
        return RiskyModelDL_FINAL(
            econ_params, dl_config, bounds,
            pretrained_checkpoint_dir=pretrained_dir,
            pretrained_epoch=int(pretrained_epoch),
            optimization_config=opt_config,
            **extra_kwargs,
        )
    
def main():
    """Main entry point for the DL training CLI."""
    parser = argparse.ArgumentParser(
        description="Train Economic Deep Learning Models"
    )
    parser.add_argument(
        '--model',
        type=str,
        default='basic',
        choices=['basic', 'basic_final', 'risk_free', 'risky_final'],
        help="Model architecture to train. "
             "'basic': pretrain stage (FOC-based, saves to checkpoints_pretrain_dist/). "
             "'basic_final'/'risky_final': final stage (loads pretrained, saves to checkpoints_final_dist/)."
    )
    parser.add_argument(
        '--config-key',
        type=str,
        default=None,
        help="Override the JSON config key used for hyperparameters "
             "(defaults to --model value). "
             "Useful for running experiments with custom config entries."
    )
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default=None,
        help="Override the checkpoint save directory."
    )
    parser.add_argument(
        '--log-dir',
        type=str,
        default=None,
        help="Override the TensorBoard log directory prefix."
    )
    parser.add_argument(
        '--pretrained-dir',
        type=str,
        default=None,
        help="Override the pretrained checkpoint directory to load from "
             "(basic_final/risky_final). Defaults to "
             "'checkpoints_pretrain_dist/basic' for basic_final, "
             "'checkpoints_pretrain_dist/risk_free' for risky_final."
    )
    parser.add_argument(
        '--pretrained_epoch',
        type=int,
        default=None,
        help="Override the pretrained epoch to load (basic_final/risky_final). "
             "Defaults to 6200 for basic_final, 500 for risky_final."
    )
    parser.add_argument(
        '--gpu',
        type=int,
        default=None,
        help="GPU device ID to use (0-3). If not set, uses all GPUs."
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
