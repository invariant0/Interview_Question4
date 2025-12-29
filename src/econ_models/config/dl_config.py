# econ_models/config/dl_config.py
"""
Configuration for Deep Learning training loops.

This module provides configuration classes for neural network training,
including hyperparameters, domain boundaries, and curriculum learning settings.

Example:
    >>> from econ_models.config.dl_config import load_dl_config
    >>> config = load_dl_config("config/dl.json", "risky")
    >>> print(f"Learning rate: {config.learning_rate}")
"""

from dataclasses import dataclass, fields
from typing import Tuple, Optional
import os
import sys
import logging

from econ_models.config.economic_params import EconomicParams
from econ_models.econ import SteadyStateCalculator

logger = logging.getLogger(__name__)


@dataclass
class DeepLearningConfig:
    """
    Configuration for deep learning training loop and state space boundaries.

    This mutable configuration manages hyperparameters for neural networks,
    training loop settings, and the physical boundaries of the state space.

    Attributes:
        batch_size: Number of samples per training batch.
        learning_rate: Initial learning rate for optimizer.
        verbose: Whether to print progress information.
        epochs: Total number of training epochs.
        steps_per_epoch: Number of gradient steps per epoch.
        capital_min: Minimum capital value for sampling.
        capital_max: Maximum capital value for sampling.
        productivity_min: Minimum productivity value for sampling.
        productivity_max: Maximum productivity value for sampling.
        debt_min: Minimum debt value for sampling (risky model).
        debt_max: Maximum debt value for sampling (risky model).
        tauchen_std_dev_width: Width for productivity discretization.
        capital_min_scale: Multiplier for minimum capital (relative to k_ss).
        capital_max_scale: Multiplier for maximum capital (relative to k_ss).
        debt_max_scale: Multiplier for maximum debt (relative to k_ss).
        debt_min_scale: Multiplier for minimum debt (relative to k_ss).
        boundary_margin: Safety margin for boundaries.
        capital_steady_state: Calculated steady state capital reference.
        value_scale_factor: Scale factor for value function normalization.
        hidden_layers: Tuple of hidden layer sizes.
        activation_function: Activation function name.
        mc_next_candidate_sample: Monte Carlo samples for candidate selection.
        mc_sample_number_bond_priceing: MC samples for bond pricing.
        mc_sample_number_continuation_value: MC samples for continuation value.
        polyak_averaging_decay: Soft update coefficient for target networks.
        gradient_clip_norm: Maximum gradient norm for clipping.
        curriculum_epochs: Epochs for curriculum learning ramp-up.
        curriculum_initial_ratio: Initial sampling ratio for curriculum.
        euler_residual_weight: Weight for Euler equation loss (basic model).
        min_q_price: Minimum bond price for numerical stability.
        epsilon_debt: Small threshold for zero-debt detection.
        epsilon_value_default: Value threshold for default detection.
    """

    # Training hyperparameters
    batch_size: int = 500
    learning_rate: float = 1e-3
    verbose: bool = True
    epochs: int = 10000
    steps_per_epoch: int = 100

    # Domain boundaries
    capital_min: Optional[float] = None
    capital_max: Optional[float] = None
    productivity_min: Optional[float] = None
    productivity_max: Optional[float] = None
    debt_min: Optional[float] = None
    debt_max: Optional[float] = None

    # Boundary heuristics
    tauchen_std_dev_width: float = 3.0
    capital_min_scale: float = 0.1
    capital_max_scale: float = 3.0
    debt_max_scale: float = 15.0
    debt_min_scale: float = 0.0
    boundary_margin: float = 0.05
    capital_steady_state: Optional[float] = None

    # Value function scaling
    value_scale_factor: float = 1.0

    # Neural network architecture
    hidden_layers: Tuple[int, ...] = (64, 64)
    activation_function: str = "swish"

    # Risky model specifics
    mc_next_candidate_sample: int = 400
    mc_sample_number_bond_priceing: int = 30
    mc_sample_number_continuation_value: int = 30
    polyak_averaging_decay: float = 0.005
    gradient_clip_norm: Optional[float] = 1.0

    # Curriculum learning
    curriculum_epochs: int = 400
    curriculum_initial_ratio: float = 0.05

    # Basic model specifics
    euler_residual_weight: float = 10000.0

    # Numerical safety
    min_q_price: float = 1e-6
    epsilon_debt: float = 1e-6
    epsilon_value_default: float = 1e-4

    def update_value_scale(self, params: EconomicParams) -> None:
        """
        Update value scale factor based on steady state dividends.

        This method calculates an appropriate scaling factor for the value
        function based on the present value of steady state dividends,
        which helps neural network training converge more reliably.

        Args:
            params: Economic parameters for steady state calculation.
        """
        self.capital_steady_state = SteadyStateCalculator.calculate_capital(params)
        output_ss = 1.0 * (self.capital_steady_state ** params.capital_share)
        investment_ss = params.depreciation_rate * self.capital_steady_state
        dividends_ss = output_ss - investment_ss

        # Present value of perpetuity of steady state dividends
        self.value_scale_factor = dividends_ss / (1.0 - params.discount_factor)

        if self.verbose:
            self._print_boundary_info()

    def _print_boundary_info(self) -> None:
        """Print boundary configuration information."""
        print("DLConfig Boundaries Updated based on EconomicParams:")
        print(
            f"  K: [{self.capital_min:.4f}, {self.capital_max:.4f}] "
            f"(K_ss={self.capital_steady_state:.4f})"
        )
        if self.debt_max is not None and self.debt_max > 0:
            print(f"  B: [{self.debt_min:.4f}, {self.debt_max:.4f}]")
        print(f"  Z: [{self.productivity_min:.4f}, {self.productivity_max:.4f}]")


def load_dl_config(filename: str, model_type: str) -> DeepLearningConfig:
    """
    Load deep learning configuration from a JSON file.

    Args:
        filename: Path to the JSON configuration file.
        model_type: Key in the JSON file ('basic' or 'risky').

    Returns:
        Populated DeepLearningConfig instance.
    """
    if not os.path.exists(filename):
        logger.warning(
            f"DL config file '{filename}' not found. Using defaults."
        )
        return DeepLearningConfig()

    try:
        with open(filename, 'r') as f:
            import json
            full_data = json.load(f)

        if model_type not in full_data:
            logger.warning(
                f"Key '{model_type}' not in {filename}. Using defaults."
            )
            return DeepLearningConfig()

        model_data = full_data[model_type]
        valid_keys = {f.name for f in fields(DeepLearningConfig)}
        filtered_data = {k: v for k, v in model_data.items() if k in valid_keys}

        # Convert hidden_layers list to tuple if present
        if 'hidden_layers' in filtered_data:
            filtered_data['hidden_layers'] = tuple(filtered_data['hidden_layers'])

        return DeepLearningConfig(**filtered_data)

    except Exception as e:
        logger.error(f"Error reading DL config {filename}: {e}")
        sys.exit(1)