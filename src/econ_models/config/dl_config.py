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
from typing import Tuple, Optional, List, Any
import os
import sys
import json
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
    All parameters are initialized to None and must be loaded from a configuration file.

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
        capital_steady_state: Calculated steady state capital reference.
        value_scale_factor: Scale factor for value function normalization.
        hidden_layers: Tuple of hidden layer sizes.
        activation_function: Activation function name.
        mc_next_candidate_sample: Monte Carlo samples for candidate selection.
        mc_sample_number_bond_priceing: MC samples for bond pricing.
        mc_sample_number_continuation_value: MC samples for continuation value.
        polyak_averaging_decay: Soft update coefficient for target networks.
        gradient_clip_norm: Maximum gradient norm for clipping.
        euler_residual_weight: Weight for Euler equation loss (basic model).
        min_q_price: Minimum bond price for numerical stability.
        epsilon_debt: Small threshold for zero-debt detection.
        epsilon_value_default: Value threshold for default detection.
    """

    # Training hyperparameters
    batch_size: Optional[int] = None
    learning_rate: Optional[float] = None
    verbose: Optional[bool] = None
    epochs: Optional[int] = None
    steps_per_epoch: Optional[int] = None
    use_layer_norm: Optional[bool] = None

    # Domain boundaries
    capital_min: Optional[float] = None
    capital_max: Optional[float] = None
    productivity_min: Optional[float] = None
    productivity_max: Optional[float] = None
    debt_min: Optional[float] = None
    debt_max: Optional[float] = None

    # Boundary heuristics
    capital_steady_state: Optional[float] = None

    # Value function scaling
    value_scale_factor: Optional[float] = None

    # Neural network architecture
    hidden_layers: Optional[Tuple[int, ...]] = None
    activation_function: Optional[str] = None

    # Risky model specifics
    mc_next_candidate_sample: Optional[int] = None
    mc_sample_number_bond_priceing: Optional[int] = None
    mc_sample_number_continuation_value: Optional[int] = None
    polyak_averaging_decay: Optional[float] = None
    gradient_clip_norm: Optional[float] = None

    # Learning-rate and Polyak-averaging decay schedules
    lr_decay_rate: Optional[float] = None
    lr_decay_steps: Optional[int] = None
    lr_policy_scale: Optional[float] = None
    polyak_decay_end: Optional[float] = None
    polyak_decay_epochs: Optional[int] = None
    target_update_freq: Optional[int] = None

    # Basic model specifics
    euler_residual_weight: Optional[float] = None

    # Numerical safety
    min_q_price: Optional[float] = None
    epsilon_debt: Optional[float] = None
    epsilon_value_default: Optional[float] = None

    # Equity FB constraint weight (multiplier on equity FB loss)
    equity_fb_weight: Optional[float] = None

    # Equity FB weight epoch scheduler: linearly anneal from start → end
    equity_fb_weight_start: Optional[float] = None
    equity_fb_weight_end: Optional[float] = None
    equity_fb_start_epoch: Optional[int] = None
    equity_fb_end_epoch: Optional[int] = None

    # Continuous-net negative-value leak: dampens negative V_cont in policy update
    continuous_leak_weight_start: Optional[float] = None
    continuous_leak_weight_end: Optional[float] = None
    continuous_leak_start_epoch: Optional[int] = None
    continuous_leak_end_epoch: Optional[int] = None

    # Bernoulli entropy regularization for capital policy (sigmoid ∈ [0,1])
    entropy_capital_weight_start: Optional[float] = None
    entropy_capital_weight_end: Optional[float] = None
    entropy_capital_start_epoch: Optional[int] = None
    entropy_capital_end_epoch: Optional[int] = None

    # Bernoulli entropy regularization for debt policy (sigmoid ∈ [0,1])
    entropy_debt_weight_start: Optional[float] = None
    entropy_debt_weight_end: Optional[float] = None
    entropy_debt_start_epoch: Optional[int] = None
    entropy_debt_end_epoch: Optional[int] = None

    # Bernoulli entropy regularization for default policy (sigmoid ∈ [0,1])
    entropy_default_weight_start: Optional[float] = None
    entropy_default_weight_end: Optional[float] = None
    entropy_default_start_epoch: Optional[int] = None
    entropy_default_end_epoch: Optional[int] = None

    # Policy warm-up: number of epochs where policy LR linearly ramps from
    # ~0 to full policy LR.  Critics train at full speed during warm-up.
    # Set to 0 to disable warm-up (default behaviour).
    policy_warmup_epochs: Optional[int] = None

    # Policy change-rate scale (legacy, unused with sigmoid+denorm workflow)
    policy_change_scale: Optional[float] = None

    # Cross-product sampling
    cross_product_sampling: Optional[bool] = None
    batch_size_states: Optional[int] = None
    batch_size_params: Optional[int] = None

    def update_value_scale(self, params: EconomicParams) -> None:
        """
        Update value scale factor based on steady state dividends.

        This method calculates an appropriate scaling factor for the value
        function based on the present value of steady state dividends.

        Args:
            params: Economic parameters for steady state calculation.
        """
        self.capital_steady_state = SteadyStateCalculator.calculate_capital(params)
        
        # Ensure steady state calculation succeeded
        if self.capital_steady_state is None:
            logger.error("Failed to calculate capital steady state.")
            return

        output_ss = 1.0 * (self.capital_steady_state ** params.capital_share)
        investment_ss = params.depreciation_rate * self.capital_steady_state
        dividends_ss = output_ss - investment_ss

        # Present value of perpetuity of steady state dividends
        self.value_scale_factor = dividends_ss / (1.0 - params.discount_factor)

        if self.verbose:
            self._print_boundary_info()

    def validate_scheduling_fields(
        self, required_fields: List[str], model_name: str
    ) -> None:
        """Validate that all required scheduling fields are provided in JSON config.

        Args:
            required_fields: List of field names that must not be None.
            model_name: Name of the model (for error messages).

        Raises:
            ValueError: If any required field is None.
        """
        missing = [f for f in required_fields if getattr(self, f) is None]
        if missing:
            raise ValueError(
                f"[{model_name}] Missing required scheduling fields: {missing}. "
                f"All scheduling parameters must be provided in the JSON config file."
            )

    def _print_boundary_info(self) -> None:
        """Print boundary configuration information."""
        # Helper to safely format optional floats
        def fmt(val: Optional[float]) -> str:
            return f"{val:.4f}" if val is not None else "None"

        print("DLConfig Boundaries Updated based on EconomicParams:")
        print(
            f"  K: [{fmt(self.capital_min)}, {fmt(self.capital_max)}] "
            f"(K_ss={fmt(self.capital_steady_state)})"
        )
        if self.debt_max is not None and self.debt_max > 0:
            print(f"  B: [{fmt(self.debt_min)}, {fmt(self.debt_max)}]")
        print(f"  Z: [{fmt(self.productivity_min)}, {fmt(self.productivity_max)}]")


def load_dl_config(filename: str, model_type: str) -> DeepLearningConfig:
    """
    Load deep learning configuration from a JSON file.

    This function strictly requires the file and the specific model key to exist.
    It will not fall back to defaults, as all defaults are None.

    Args:
        filename: Path to the JSON configuration file.
        model_type: Key in the JSON file ('basic' or 'risky').

    Returns:
        Populated DeepLearningConfig instance.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the model_type is not in the file or required fields are missing.
    """
    if not os.path.exists(filename):
        error_msg = f"DL config file '{filename}' not found. Cannot proceed."
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    try:
        with open(filename, 'r') as f:
            full_data = json.load(f)

        if model_type not in full_data:
            error_msg = f"Key '{model_type}' not found in {filename}."
            logger.error(error_msg)
            raise ValueError(error_msg)

        model_data = full_data[model_type]
        valid_keys = {f.name for f in fields(DeepLearningConfig)}
        
        # Filter data to only include valid dataclass fields
        filtered_data = {k: v for k, v in model_data.items() if k in valid_keys}

        # Convert JSON null values to None (they already are, but strip
        # any that slipped through as explicit None entries so the
        # dataclass defaults apply correctly).
        filtered_data = {k: v for k, v in filtered_data.items() if v is not None}

        # Convert hidden_layers list to tuple if present (JSON arrays are lists)
        if 'hidden_layers' in filtered_data and isinstance(filtered_data['hidden_layers'], list):
            filtered_data['hidden_layers'] = tuple(filtered_data['hidden_layers'])

        config = DeepLearningConfig(**filtered_data)
        
        # Validate that essential configuration was actually loaded.
        # Since defaults are None, we must ensure the JSON provided the data.
        # We check for a few critical fields to ensure integrity.
        critical_fields = [
            'batch_size', 'learning_rate', 'epochs', 
            'hidden_layers', 'activation_function'
        ]
        
        missing_fields = [field for field in critical_fields if getattr(config, field) is None]
        
        if missing_fields:
            error_msg = (
                f"Configuration for '{model_type}' in '{filename}' is missing "
                f"required fields: {missing_fields}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        return config

    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {filename}: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading DL config: {e}")
        raise e