"""
Common utilities and data structures for RBC model evaluation.

This module contains shared components used by both value function evaluation
and policy simulation scripts.

Standards:
    - PEP 8 (Style Guide for Python Code)
    - PEP 257 (Docstring Conventions)
    - Clean Code principles (Modularity, Clarity)

Author: AI Assistant
Date: 2025-12-26
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import tensorflow as tf
from numpy.typing import NDArray

# Lazy imports for plotting to speed up initial load
plt = None
cm = None


def _ensure_plotting_imports() -> None:
    """Lazily import matplotlib modules when needed."""
    global plt, cm
    if plt is None:
        import matplotlib.pyplot as _plt
        from matplotlib import cm as _cm
        plt = _plt
        cm = _cm


# Type aliases for clarity
FloatArray = NDArray[np.floating[Any]]
TensorFloat = tf.Tensor


class ImportError(Exception):
    """Raised when required project modules cannot be imported."""


# Attempt to import project modules with informative error handling
try:
    from econ_models.config.dl_config import DeepLearningConfig
    from econ_models.config.economic_params import EconomicParams
    from econ_models.core.math import MathUtils
    from econ_models.core.types import TENSORFLOW_DTYPE
    from econ_models.dl.basic import BasicModelDL
    from econ_models.econ import EconomicLogic
except ModuleNotFoundError as e:
    raise ImportError(
        f"Failed to import required project modules: {e}. "
        "Ensure 'econ_models' package is installed and accessible."
    ) from e


# --- Logging Configuration ---
def configure_logging(level: int = logging.INFO) -> logging.Logger:
    """
    Configure and return the module logger.

    Args:
        level: Logging level (default: INFO).

    Returns:
        Configured logger instance.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


logger = configure_logging()


# --- Configuration Dataclasses ---
@dataclass(frozen=True)
class PathConfig:
    """Immutable configuration for all file paths used in evaluation."""

    econ_params: Path = Path("./hyperparam/prefixed/econ_params_basic.json")
    dl_params: Path = Path("./hyperparam/prefixed/dl_params.json")
    bounds: Path = Path("./hyperparam/autogen/bounds_basic.json")
    vfi_results: Path = Path("./ground_truth/basic_model_vfi_results.npz")
    checkpoints: Path = Path("checkpoints/basic")
    figures: Path = Path("./results/result_effectiveness_dl_basic")

    def validate(self) -> list[str]:
        """
        Validate that required input paths exist.

        Returns:
            List of missing file paths (empty if all exist).
        """
        required = [self.econ_params, self.dl_params, self.bounds]
        return [str(p) for p in required if not p.exists()]

    def ensure_output_dirs(self) -> None:
        """Create output directories if they don't exist."""
        self.figures.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class EvaluationConfig:
    """Configuration parameters for model evaluation."""

    monte_carlo_samples: int = 10_000
    residual_sample_size: int = 500
    simulation_trajectories: int = 10000
    simulation_timesteps: int = 500
    checkpoint_epoch_filter: int = 20
    figure_dpi: int = 300

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.monte_carlo_samples < 100:
            raise ValueError("monte_carlo_samples must be at least 100")
        if self.residual_sample_size < 10:
            raise ValueError("residual_sample_size must be at least 10")


# --- Data Structures ---
class Checkpoint(NamedTuple):
    """Represents a model checkpoint with epoch and weight paths."""

    epoch: int
    value_net_path: Path
    policy_net_path: Path


@dataclass
class VFIData:
    """Container for Value Function Iteration benchmark data."""

    capital_grid: FloatArray
    productivity_grid: FloatArray
    value_function: FloatArray

    @classmethod
    def from_npz(cls, filepath: Path) -> VFIData | None:
        """
        Load VFI data from an NPZ file.

        Args:
            filepath: Path to the NPZ file.

        Returns:
            VFIData instance or None if file doesn't exist.
        """
        if not filepath.exists():
            logger.warning(f"VFI results not found at {filepath}")
            return None

        try:
            data = np.load(filepath)
            return cls(
                capital_grid=data["K"],
                productivity_grid=data["Z"],
                value_function=data["V"],
            )
        except (KeyError, ValueError) as e:
            logger.error(f"Invalid VFI data format: {e}")
            return None

    def create_meshgrid(self) -> tuple[FloatArray, FloatArray]:
        """Create meshgrid from capital and productivity grids."""
        return np.meshgrid(
            self.capital_grid, self.productivity_grid, indexing="ij"
        )

    def get_flattened_tensors(
        self,
    ) -> tuple[TensorFloat, TensorFloat, FloatArray, FloatArray]:
        """
        Create flattened tensors for model evaluation, filtering NaN values.

        Returns:
            Tuple of (k_tensor, z_tensor, v_array, valid_mask).
        """
        k_mesh, z_mesh = self.create_meshgrid()
        v_flat = self.value_function.flatten()

        valid_mask = ~np.isnan(v_flat)

        k_flat = k_mesh.flatten()[valid_mask].reshape(-1, 1)
        z_flat = z_mesh.flatten()[valid_mask].reshape(-1, 1)
        v_valid = v_flat[valid_mask]

        k_tensor = tf.constant(k_flat, dtype=TENSORFLOW_DTYPE)
        z_tensor = tf.constant(z_flat, dtype=TENSORFLOW_DTYPE)

        return k_tensor, z_tensor, v_valid, valid_mask


@dataclass
class TrainingMetrics:
    """Container for tracking training metrics across epochs."""

    epochs: list[int] = field(default_factory=list)
    vfi_mae: list[float] = field(default_factory=list)
    bellman_rmse: list[float] = field(default_factory=list)
    euler_rmse: list[float] = field(default_factory=list)

    def append(
        self,
        epoch: int,
        vfi: float,
        bellman: float,
        euler: float,
    ) -> None:
        """Append metrics for a single epoch."""
        self.epochs.append(epoch)
        self.vfi_mae.append(vfi)
        self.bellman_rmse.append(bellman)
        self.euler_rmse.append(euler)

    def to_dict(self) -> dict[str, list[float]]:
        """Convert metrics to dictionary format."""
        return {
            "vfi_mae": self.vfi_mae,
            "bellman_rmse": self.bellman_rmse,
            "euler_rmse": self.euler_rmse,
        }


@dataclass
class SimulationResult:
    """Container for simulation results."""

    capital_history: FloatArray
    productivity_history: FloatArray
    investment_history: FloatArray
    investment_rate_history: FloatArray
    total_values: FloatArray
    steady_state_capital: float


# --- Utility Functions ---
def load_json_file(filepath: Path) -> dict[str, Any]:
    """
    Load and parse a JSON file with comprehensive error handling.

    Args:
        filepath: Path to the JSON file.

    Returns:
        Dictionary containing the parsed JSON data.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file contains invalid JSON.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Configuration file not found: {filepath}")

    try:
        with filepath.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {filepath}: {e}") from e


def compute_rmse(predictions: FloatArray, targets: FloatArray) -> float:
    """
    Compute Root Mean Square Error between predictions and targets.

    Args:
        predictions: Predicted values.
        targets: Target values.

    Returns:
        RMSE value.
    """
    return float(np.sqrt(np.nanmean((predictions - targets) ** 2)))


def compute_mae(predictions: FloatArray, targets: FloatArray) -> float:
    """Compute Mean Absolute Error."""
    return float(np.nanmean(np.abs(predictions - targets)))


def compute_mape(
    predictions: FloatArray,
    targets: FloatArray,
    epsilon: float = 1e-8,
) -> float:
    """
    Compute Mean Absolute Percentage Error.

    Args:
        predictions: Predicted values.
        targets: Target values.
        epsilon: Small value to avoid division by zero.

    Returns:
        MAPE as a percentage.
    """
    safe_targets = np.where(np.abs(targets) < epsilon, epsilon, targets)
    percentage_errors = np.abs((predictions - targets) / safe_targets) * 100.0
    return float(np.nanmean(percentage_errors))


# --- Model Setup ---
class ModelFactory:
    """Factory class for creating and configuring the DL model."""

    def __init__(self, paths: PathConfig) -> None:
        """
        Initialize the factory with path configuration.

        Args:
            paths: Configuration containing all required file paths.
        """
        self.paths = paths

    def create_model(self) -> BasicModelDL:
        """
        Create and initialize a BasicModelDL instance.

        Returns:
            Configured model with computational graph built.

        Raises:
            FileNotFoundError: If required configuration files are missing.
            ValueError: If configuration data is invalid.
        """
        logger.info("Initializing model configuration...")

        econ_params = self._load_economic_params()
        dl_config = self._load_dl_config()
        self._apply_bounds(dl_config)

        model = BasicModelDL(econ_params, dl_config)
        self._build_graph(model)

        return model

    def _load_economic_params(self) -> EconomicParams:
        """Load and validate economic parameters."""
        data = load_json_file(self.paths.econ_params)
        return EconomicParams(**data)

    def _load_dl_config(self) -> DeepLearningConfig:
        """Load and validate deep learning configuration."""
        raw_data = load_json_file(self.paths.dl_params)

        # Handle nested configuration structure
        config_dict = raw_data.get("basic", raw_data)

        # Convert hidden layers to immutable tuple
        if "hidden_layers" in config_dict:
            config_dict["hidden_layers"] = tuple(config_dict["hidden_layers"])

        return DeepLearningConfig(**config_dict)

    def _apply_bounds(self, config: DeepLearningConfig) -> None:
        """Apply state space bounds to configuration."""
        bounds_data = load_json_file(self.paths.bounds)
        bounds = bounds_data.get("bounds", {})

        config.capital_min = bounds.get("k_min", 0.1)
        config.capital_max = bounds.get("k_max", 10.0)
        config.productivity_min = bounds.get("z_min", 0.5)
        config.productivity_max = bounds.get("z_max", 1.5)

        logger.info(
            f"State bounds configured: "
            f"K=[{config.capital_min:.2f}, {config.capital_max:.2f}], "
            f"Z=[{config.productivity_min:.2f}, {config.productivity_max:.2f}]"
        )

    def _build_graph(self, model: BasicModelDL) -> None:
        """Build computational graph with a dummy forward pass."""
        dummy_k = tf.constant([[1.0]], dtype=TENSORFLOW_DTYPE)
        dummy_z = tf.constant([[1.0]], dtype=TENSORFLOW_DTYPE)
        _ = model.compute_loss(dummy_k, dummy_z)
        logger.debug("Computational graph built successfully")


# --- Checkpoint Management ---
class CheckpointManager:
    """Manages discovery and loading of model checkpoints."""

    CHECKPOINT_PATTERN = re.compile(r"basic_value_net_(\d+)\.weights\.h5")

    def __init__(self, checkpoint_dir: Path, epoch_filter: int = 20) -> None:
        """
        Initialize the checkpoint manager.

        Args:
            checkpoint_dir: Directory containing checkpoint files.
            epoch_filter: Only include checkpoints at epochs divisible by this value.
        """
        self.checkpoint_dir = checkpoint_dir
        self.epoch_filter = epoch_filter

    def discover_checkpoints(self) -> list[Checkpoint]:
        """
        Discover and return sorted list of valid checkpoints.

        Returns:
            List of Checkpoint objects sorted by epoch (ascending).

        Raises:
            FileNotFoundError: If checkpoint directory doesn't exist.
        """
        if not self.checkpoint_dir.exists():
            raise FileNotFoundError(
                f"Checkpoint directory not found: {self.checkpoint_dir}"
            )

        checkpoints = []
        for filepath in self.checkpoint_dir.glob("basic_value_net_*.weights.h5"):
            checkpoint = self._parse_checkpoint(filepath)
            if checkpoint is not None:
                checkpoints.append(checkpoint)

        # Sort by epoch and filter
        checkpoints.sort(key=lambda c: c.epoch)
        filtered = [c for c in checkpoints if c.epoch % self.epoch_filter == 0]

        logger.info(
            f"Discovered {len(filtered)} checkpoints "
            f"(filtered from {len(checkpoints)} total)"
        )

        return filtered

    def _parse_checkpoint(self, value_path: Path) -> Checkpoint | None:
        """Parse a checkpoint from its value network path."""
        match = self.CHECKPOINT_PATTERN.search(value_path.name)
        if not match:
            return None

        epoch = int(match.group(1))
        policy_path = value_path.parent / value_path.name.replace(
            "value_net", "policy_net"
        )

        if not policy_path.exists():
            logger.debug(f"Missing policy network for epoch {epoch}")
            return None

        return Checkpoint(epoch, value_path, policy_path)

    @staticmethod
    def load_weights(
        model: BasicModelDL,
        checkpoint: Checkpoint,
    ) -> bool:
        """
        Load weights from a checkpoint into the model.

        Args:
            model: Model to load weights into.
            checkpoint: Checkpoint containing weight paths.

        Returns:
            True if successful, False otherwise.
        """
        try:
            model.value_net.load_weights(str(checkpoint.value_net_path))
            model.policy_net.load_weights(str(checkpoint.policy_net_path))
            return True
        except Exception as e:
            logger.warning(
                f"Failed to load checkpoint epoch {checkpoint.epoch}: {e}"
            )
            return False


# --- Residual Computation ---
class ResidualCalculator:
    """Computes Bellman and Euler residuals for model evaluation."""

    def __init__(
        self,
        model: BasicModelDL,
        n_samples: int = 10_000,
    ) -> None:
        """
        Initialize the residual calculator.

        Args:
            model: The trained deep learning model.
            n_samples: Number of Monte Carlo samples for expectation estimation.
        """
        self.model = model
        self.n_samples = n_samples

    @tf.function(reduce_retracing=True)
    def compute(
        self,
        k_batch: TensorFloat,
        z_batch: TensorFloat,
    ) -> tuple[TensorFloat, TensorFloat]:
        """
        Compute Bellman and Euler residuals using vectorized Monte Carlo integration.

        This implementation eliminates Python loops by using tensor operations,
        performing a single forward pass for all MC samples.

        Args:
            k_batch: Capital states tensor, shape (batch_size, 1).
            z_batch: Productivity states tensor, shape (batch_size, 1).

        Returns:
            Tuple of (bellman_residuals, euler_residuals), each shape (batch_size, 1).
        """
        # Current period computation
        current_state = self._compute_current_state(k_batch, z_batch)

        # Vectorized Monte Carlo expansion and next period computation
        next_state_expectations = self._compute_next_state_expectations(
            current_state.k_prime, z_batch
        )

        # Calculate residuals
        bellman_residual = self._compute_bellman_residual(
            current_state, next_state_expectations
        )
        euler_residual = self._compute_euler_residual(
            current_state, next_state_expectations
        )

        return bellman_residual, euler_residual

    def _compute_current_state(
        self,
        k_batch: TensorFloat,
        z_batch: TensorFloat,
    ) -> "_CurrentState":
        """Compute all current period quantities."""
        # Normalize inputs
        k_norm = self.model.normalizer.normalize_capital(k_batch)
        z_norm = self.model.normalizer.normalize_productivity(z_batch)
        inputs = tf.concat([k_norm, z_norm], axis=1)

        # Network outputs
        v_current = self.model.value_net(inputs)
        investment_rate = self.model.policy_net(inputs)
        investment = investment_rate * k_batch

        # Economic calculations
        profit = EconomicLogic.production_function(
            k_batch, z_batch, self.model.params
        )
        adj_cost, marginal_adj_cost = EconomicLogic.adjustment_costs(
            investment, k_batch, self.model.params
        )
        cash_flow = EconomicLogic.calculate_cash_flow(profit, investment, adj_cost)

        # Next period capital (deterministic)
        k_prime = investment + (1.0 - self.model.params.depreciation_rate) * k_batch

        return _CurrentState(
            value=v_current,
            cash_flow=cash_flow,
            marginal_adj_cost=marginal_adj_cost,
            k_prime=k_prime,
        )

    def _compute_next_state_expectations(
        self,
        k_prime: TensorFloat,
        z_current: TensorFloat,
    ) -> "_NextStateExpectations":
        """
        Compute expected next period values using vectorized Monte Carlo.

        Expands the batch from (B, 1) to (B * N, 1) to process all samples in parallel.
        """
        batch_size = tf.shape(k_prime)[0]
        total_samples = batch_size * self.n_samples

        # Expand tensors for all MC samples
        k_prime_expanded = tf.repeat(k_prime, repeats=self.n_samples, axis=0)
        z_current_expanded = tf.repeat(z_current, repeats=self.n_samples, axis=0)

        # Sample productivity shocks
        epsilon = self.model.shock_dist.sample(sample_shape=(total_samples, 1))

        # Compute next period productivity
        z_prime_expanded = MathUtils.log_ar1_transition(
            z_current_expanded,
            self.model.params.productivity_persistence,
            epsilon,
        )

        # Compute value and gradients for all samples
        with tf.GradientTape() as tape:
            k_prime_norm = self.model.normalizer.normalize_capital(k_prime_expanded)
            tape.watch(k_prime_norm)

            z_prime_norm = self.model.normalizer.normalize_productivity(
                z_prime_expanded
            )
            inputs_next = tf.concat([k_prime_norm, z_prime_norm], axis=1)

            v_next_expanded = self.model.value_net(inputs_next)

        # Compute dV/dK
        dv_dk_norm = tape.gradient(v_next_expanded, k_prime_norm)
        dv_dk_expanded = dv_dk_norm / self.model.normalizer.k_range

        # Aggregate by averaging across MC samples
        v_next_reshaped = tf.reshape(v_next_expanded, (batch_size, self.n_samples))
        dv_dk_reshaped = tf.reshape(dv_dk_expanded, (batch_size, self.n_samples))

        return _NextStateExpectations(
            expected_value=tf.reduce_mean(v_next_reshaped, axis=1, keepdims=True),
            expected_dv_dk=tf.reduce_mean(dv_dk_reshaped, axis=1, keepdims=True),
        )

    def _compute_bellman_residual(
        self,
        current: "_CurrentState",
        expectations: "_NextStateExpectations",
    ) -> TensorFloat:
        """Compute Bellman equation residual: V - (CF + β * E[V'])."""
        bellman_target = (
            current.cash_flow
            + self.model.params.discount_factor * expectations.expected_value
        )
        return current.value - bellman_target

    def _compute_euler_residual(
        self,
        current: "_CurrentState",
        expectations: "_NextStateExpectations",
    ) -> TensorFloat:
        """
        Compute unit-free Euler equation residual.

        Residual: 1 - (β * E[dV/dK']) / (1 + MC)
        """
        marginal_cost_investment = 1.0 + current.marginal_adj_cost
        euler_fraction = (
            self.model.params.discount_factor * expectations.expected_dv_dk
        ) / (marginal_cost_investment + 1e-8)
        return 1.0 - euler_fraction


@dataclass
class _CurrentState:
    """Internal container for current period state quantities."""

    value: TensorFloat
    cash_flow: TensorFloat
    marginal_adj_cost: TensorFloat
    k_prime: TensorFloat


@dataclass
class _NextStateExpectations:
    """Internal container for next period expectations."""

    expected_value: TensorFloat
    expected_dv_dk: TensorFloat


# --- Visualization Helpers ---
class PlotStyle(Enum):
    """Enumeration of available plot styles."""

    SEABORN = "seaborn-v0_8-whitegrid"
    DEFAULT = "default"


def apply_plot_style(style: PlotStyle) -> None:
    """Apply matplotlib style."""
    _ensure_plotting_imports()
    try:
        plt.style.use(style.value)
    except OSError:
        logger.debug(f"Style {style.value} not available, using default")


def save_figure(fig, output_dir: Path, filename: str, dpi: int = 300) -> Path:
    """Save figure and return path."""
    _ensure_plotting_imports()
    filepath = output_dir / filename
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved figure: {filepath}")
    return filepath
