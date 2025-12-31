"""
Deep Learning RBC Model Evaluation Script.

This module evaluates the performance of a Deep Learning-based Real Business Cycle (RBC)
model against a Value Function Iteration (VFI) benchmark. It generates three figures:

1. Error evolution (MAE, Bellman, Euler) over training epochs.
2. A 2x2 comparative analysis of Value Functions and Error Distributions (MAE/MAPE).
3. Stochastic simulation of trajectories compared to theoretical steady states.

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
import os
import re
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
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
    simulation_trajectories: int = 500
    simulation_timesteps: int = 1000
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


# --- Visualization ---
class PlotStyle(Enum):
    """Enumeration of available plot styles."""

    SEABORN = "seaborn-v0_8-whitegrid"
    DEFAULT = "default"


class FigureGenerator:
    """Generates evaluation figures for the DL model."""

    def __init__(
        self,
        model: BasicModelDL,
        output_dir: Path,
        dpi: int = 300,
    ) -> None:
        """
        Initialize the figure generator.

        Args:
            model: The trained deep learning model.
            output_dir: Directory to save generated figures.
            dpi: Resolution for saved figures.
        """
        _ensure_plotting_imports()
        self.model = model
        self.output_dir = output_dir
        self.dpi = dpi

    def plot_error_evolution(
        self,
        checkpoints: list[Checkpoint],
        vfi_data: VFIData | None,
        eval_config: EvaluationConfig,
    ) -> Path:
        """
        Generate Figure 1: Error evolution over training epochs.

        Args:
            checkpoints: List of model checkpoints to evaluate.
            vfi_data: Optional VFI benchmark data.
            eval_config: Evaluation configuration.

        Returns:
            Path to the saved figure.
        """
        logger.info("Generating Figure 1: Error Evolution...")

        metrics = self._compute_metrics_over_epochs(
            checkpoints, vfi_data, eval_config
        )

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        self._apply_style(PlotStyle.SEABORN)

        plot_configs = [
            ("vfi_mae", "Value MAE vs VFI", "#2E86AB", "o"),
            ("bellman_rmse", "Bellman Residual RMSE", "#A23B72", "s"),
            ("euler_rmse", "Euler Residual RMSE", "#1B998B", "^"),
        ]

        for ax, (key, title, color, marker) in zip(axes, plot_configs):
            data = metrics.to_dict()[key]
            self._plot_metric(ax, metrics.epochs, data, title, color, marker)

        fig.suptitle(
            "Training Convergence and Error Dynamics",
            fontsize=16,
            fontweight="bold",
            y=1.02,
        )
        plt.tight_layout()

        return self._save_figure(fig, "figure1_error_evolution.png")

    def plot_comparative_analysis(
        self,
        checkpoint: Checkpoint,
        vfi_data: VFIData,
    ) -> Path:
        """
        Generate Figure 2: 2x2 comparative analysis of value functions and errors.

        Args:
            checkpoint: Checkpoint to evaluate.
            vfi_data: VFI benchmark data.

        Returns:
            Path to the saved figure.
        """
        logger.info("Generating Figure 2: Comparative Analysis...")

        CheckpointManager.load_weights(self.model, checkpoint)

        # Prepare data
        k_mesh, z_mesh = vfi_data.create_meshgrid()
        v_vfi = vfi_data.value_function

        # Compute DL predictions
        v_dl = self._predict_values_on_grid(k_mesh, z_mesh)

        # Calculate error metrics
        abs_error = np.abs(v_dl - v_vfi)
        pct_error = self._compute_percentage_error(v_dl, v_vfi)
        mae = compute_mae(v_dl, v_vfi)
        mape = compute_mape(v_dl, v_vfi)

        # Create figure
        fig = plt.figure(figsize=(16, 14))
        self._apply_style(PlotStyle.SEABORN)

        self._plot_3d_surface(
            fig, 221, k_mesh, z_mesh, v_dl,
            "Deep Learning Solution", "viridis", "Value V(K,Z)"
        )
        self._plot_3d_surface(
            fig, 222, k_mesh, z_mesh, v_vfi,
            "VFI Benchmark", "viridis", "Value V(K,Z)"
        )
        self._plot_3d_surface(
            fig, 223, k_mesh, z_mesh, abs_error,
            "Absolute Error Distribution", "inferno", "|Error|",
            annotation=self._format_mae_annotation(mae),
        )
        self._plot_3d_surface(
            fig, 224, k_mesh, z_mesh, pct_error,
            "Percentage Error Distribution", "cividis", "Error (%)",
            annotation=self._format_mape_annotation(mape),
        )

        fig.suptitle(
            "Model Accuracy Analysis (DL vs VFI)",
            fontsize=18,
            fontweight="bold",
            y=0.98,
        )
        plt.tight_layout()

        return self._save_figure(fig, "figure2_comparative_analysis.png")

    def plot_simulation(
        self,
        checkpoint: Checkpoint,
        vfi_data: VFIData | None,
        n_trajectories: int = 500,
        timesteps: int = 1000,
    ) -> Path:
        """
        Generate Figure 3: Stochastic simulation analysis.

        Args:
            checkpoint: Checkpoint to evaluate.
            vfi_data: Optional VFI data for comparison.
            n_trajectories: Number of simulation trajectories.
            timesteps: Number of time steps per trajectory.

        Returns:
            Path to the saved figure.
        """
        logger.info("Generating Figure 3: Stochastic Simulation...")

        CheckpointManager.load_weights(self.model, checkpoint)

        # Run simulation
        simulation = self._run_simulation(n_trajectories, timesteps)

        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        self._apply_style(PlotStyle.SEABORN)

        self._plot_trajectories(axes[0], simulation, n_trajectories)
        self._plot_value_distribution(axes[1], simulation, vfi_data)

        fig.suptitle(
            "Dynamic Simulation Analysis",
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )
        plt.tight_layout()

        return self._save_figure(fig, "figure3_simulation.png")

    def _compute_metrics_over_epochs(
        self,
        checkpoints: list[Checkpoint],
        vfi_data: VFIData | None,
        config: EvaluationConfig,
    ) -> TrainingMetrics:
        """Compute evaluation metrics for each checkpoint."""
        metrics = TrainingMetrics()

        # Prepare VFI comparison data
        vfi_tensors = None
        if vfi_data:
            vfi_tensors = vfi_data.get_flattened_tensors()

        # Sample states for residual calculation
        k_sample, z_sample = MathUtils.sample_states(
            config.residual_sample_size,
            self.model.config,
            include_debt=False,
            progress=tf.constant(1.0, TENSORFLOW_DTYPE),
        )

        residual_calc = ResidualCalculator(
            self.model, n_samples=config.monte_carlo_samples
        )

        for checkpoint in checkpoints:
            if not CheckpointManager.load_weights(self.model, checkpoint):
                continue

            # VFI MAE
            vfi_mae = np.nan
            if vfi_tensors:
                k_tf, z_tf, v_vfi, _ = vfi_tensors
                v_pred = self._predict_values(k_tf, z_tf)
                vfi_mae = compute_mae(v_pred, v_vfi)

            # Residual RMSEs
            bellman_res, euler_res = residual_calc.compute(k_sample, z_sample)
            bellman_rmse = compute_rmse(bellman_res.numpy(), np.zeros_like(bellman_res.numpy()))
            euler_rmse = compute_rmse(euler_res.numpy(), np.zeros_like(euler_res.numpy()))

            metrics.append(checkpoint.epoch, vfi_mae, bellman_rmse, euler_rmse)

            if checkpoint.epoch % 100 == 0:
                logger.info(f"Processed epoch {checkpoint.epoch}")

        return metrics

    def _predict_values(
        self,
        k_tensor: TensorFloat,
        z_tensor: TensorFloat,
    ) -> FloatArray:
        """Predict values for given state tensors."""
        k_norm = self.model.normalizer.normalize_capital(k_tensor)
        z_norm = self.model.normalizer.normalize_productivity(z_tensor)
        return self.model.value_net(
            tf.concat([k_norm, z_norm], axis=1)
        ).numpy().flatten()

    def _predict_values_on_grid(
        self,
        k_mesh: FloatArray,
        z_mesh: FloatArray,
    ) -> FloatArray:
        """Predict values on a meshgrid."""
        k_flat = k_mesh.flatten().reshape(-1, 1)
        z_flat = z_mesh.flatten().reshape(-1, 1)

        k_tensor = tf.constant(k_flat, dtype=TENSORFLOW_DTYPE)
        z_tensor = tf.constant(z_flat, dtype=TENSORFLOW_DTYPE)

        v_flat = self._predict_values(k_tensor, z_tensor)
        return v_flat.reshape(k_mesh.shape)

    def _compute_percentage_error(
        self,
        predictions: FloatArray,
        targets: FloatArray,
        epsilon: float = 1e-6,
    ) -> FloatArray:
        """Compute percentage error avoiding division by zero."""
        safe_targets = np.where(np.abs(targets) < epsilon, epsilon, targets)
        return (np.abs(predictions - targets) / np.abs(safe_targets)) * 100.0

    def _run_simulation(
        self,
        n_trajectories: int,
        timesteps: int,
    ) -> "_SimulationResult":
        """Run stochastic simulation from steady state."""
        # Initial conditions
        k_ss = getattr(self.model.config, "capital_steady_state", 1.0)
        z_ss = 1.0

        k_current = tf.fill([n_trajectories, 1], tf.cast(k_ss, TENSORFLOW_DTYPE))
        z_current = tf.fill([n_trajectories, 1], tf.cast(z_ss, TENSORFLOW_DTYPE))

        # Storage
        capital_history = np.zeros((timesteps, n_trajectories))
        reward_history = np.zeros((timesteps, n_trajectories))

        for t in range(timesteps):
            # Get policy
            k_norm = self.model.normalizer.normalize_capital(k_current)
            z_norm = self.model.normalizer.normalize_productivity(z_current)
            inputs = tf.concat([k_norm, z_norm], axis=1)

            investment_rate = self.model.policy_net(inputs)
            investment = investment_rate * k_current

            # Calculate rewards
            profit = EconomicLogic.production_function(
                k_current, z_current, self.model.params
            )
            adj_cost, _ = EconomicLogic.adjustment_costs(
                investment, k_current, self.model.params
            )
            cash_flow = EconomicLogic.calculate_cash_flow(
                profit, investment, adj_cost
            )

            capital_history[t] = k_current.numpy().flatten()
            reward_history[t] = cash_flow.numpy().flatten()

            # Transition dynamics
            k_next = (
                investment + (1.0 - self.model.params.depreciation_rate) * k_current
            )
            epsilon = self.model.shock_dist.sample(sample_shape=(n_trajectories, 1))
            z_next = MathUtils.log_ar1_transition(
                z_current, self.model.params.productivity_persistence, epsilon
            )

            # Apply bounds
            k_current = tf.clip_by_value(
                k_next,
                self.model.config.capital_min,
                self.model.config.capital_max,
            )
            z_current = tf.clip_by_value(
                z_next,
                self.model.config.productivity_min,
                self.model.config.productivity_max,
            )

        # Compute discounted values
        beta = self.model.params.discount_factor
        discounts = np.array([beta**t for t in range(timesteps)])
        cumulative_rewards = np.sum(reward_history * discounts[:, None], axis=0)

        # Terminal value correction for infinite horizon
        terminal_value = (
            reward_history[-1] / (1.0 - beta) * (beta**timesteps)
        )
        total_values = cumulative_rewards + terminal_value

        return _SimulationResult(
            capital_history=capital_history,
            total_values=total_values,
            steady_state_capital=k_ss,
        )

    def _plot_metric(
        self,
        ax: plt.Axes,
        epochs: list[int],
        data: list[float],
        title: str,
        color: str,
        marker: str,
    ) -> None:
        """Plot a single metric on an axis."""
        if np.all(np.isnan(data)):
            ax.text(
                0.5, 0.5, "Data Unavailable",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=12, color="gray",
            )
        else:
            ax.semilogy(
                epochs, data,
                color=color, marker=marker,
                markersize=4, linewidth=1.5, alpha=0.85,
            )

        ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Error (Log Scale)", fontsize=12)
        ax.grid(True, which="both", linestyle="-", alpha=0.3)

    def _plot_3d_surface(
        self,
        fig: plt.Figure,
        position: int,
        x: FloatArray,
        y: FloatArray,
        z: FloatArray,
        title: str,
        colormap: str,
        z_label: str,
        annotation: str | None = None,
    ) -> None:
        """Plot a 3D surface subplot."""
        ax = fig.add_subplot(position, projection="3d")

        surf = ax.plot_surface(
            x, y, z,
            cmap=getattr(cm, colormap),
            edgecolor="none",
            alpha=0.9,
            antialiased=True,
        )

        ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
        ax.set_xlabel("Capital (K)", fontsize=10, labelpad=8)
        ax.set_ylabel("Productivity (Z)", fontsize=10, labelpad=8)
        ax.set_zlabel(z_label, fontsize=10, labelpad=8)
        ax.view_init(elev=25, azim=-55)

        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=12, pad=0.1)

        if annotation:
            ax.text2D(
                0.05, 0.92, annotation,
                transform=ax.transAxes, fontsize=11,
                bbox=dict(
                    boxstyle="round,pad=0.4",
                    facecolor="white",
                    alpha=0.9,
                    edgecolor="gray",
                ),
            )

    def _plot_trajectories(
        self,
        ax: plt.Axes,
        simulation: "_SimulationResult",
        n_trajectories: int,
    ) -> None:
        """Plot capital accumulation trajectories."""
        timesteps = simulation.capital_history.shape[0]
        time = np.arange(timesteps)

        # Individual trajectories (semi-transparent)
        ax.plot(
            time, simulation.capital_history,
            color="steelblue", alpha=0.05, linewidth=0.5,
        )

        # Mean trajectory
        mean_trajectory = np.mean(simulation.capital_history, axis=1)
        ax.plot(
            time, mean_trajectory,
            color="#C41E3A", linewidth=2.5,
            label="Mean Trajectory",
        )

        # Steady state reference
        ax.axhline(
            simulation.steady_state_capital,
            color="black", linestyle="--", linewidth=1.5,
            label=f"Steady State (K={simulation.steady_state_capital:.2f})",
        )

        ax.set_title(
            f"Capital Accumulation ({n_trajectories} Simulations)",
            fontsize=14, fontweight="bold",
        )
        ax.set_xlabel("Time Period", fontsize=12)
        ax.set_ylabel("Capital Stock", fontsize=12)
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(True, alpha=0.3)

    def _plot_value_distribution(
        self,
        ax: plt.Axes,
        simulation: "_SimulationResult",
        vfi_data: VFIData | None,
    ) -> None:
        """Plot distribution of lifetime rewards."""
        ax.hist(
            simulation.total_values,
            bins=35, density=True, alpha=0.75,
            color="forestgreen", edgecolor="darkgreen", linewidth=0.5,
        )

        mean_simulated = np.mean(simulation.total_values)
        ax.axvline(
            mean_simulated,
            color="#C41E3A", linewidth=2.5,
            label=f"Simulated Mean: {mean_simulated:.2f}",
        )

        # Add VFI comparison if available
        if vfi_data is not None:
            vfi_value = self._get_vfi_steady_state_value(
                vfi_data, simulation.steady_state_capital
            )
            if vfi_value is not None:
                ax.axvline(
                    vfi_value,
                    color="royalblue", linestyle="--", linewidth=2,
                    label=f"VFI Theoretical: {vfi_value:.2f}",
                )

        ax.set_title(
            "Distribution of Lifetime Rewards",
            fontsize=14, fontweight="bold",
        )
        ax.set_xlabel("Discounted Value", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(True, alpha=0.3)

    def _get_vfi_steady_state_value(
        self,
        vfi_data: VFIData,
        k_ss: float,
        z_ss: float = 1.0,
    ) -> float | None:
        """Get VFI value at steady state."""
        try:
            k_idx = np.abs(vfi_data.capital_grid - k_ss).argmin()
            z_idx = np.abs(vfi_data.productivity_grid - z_ss).argmin()
            return float(vfi_data.value_function[k_idx, z_idx])
        except (IndexError, ValueError):
            return None

    @staticmethod
    def _format_mae_annotation(mae: float) -> str:
        """Format MAE annotation with formula."""
        return (
            r"$\mathbf{MAE} = \frac{1}{N} \sum |V_{DL} - V_{VFI}|$"
            + f"\nValue: {mae:.6f}"
        )

    @staticmethod
    def _format_mape_annotation(mape: float) -> str:
        """Format MAPE annotation with formula."""
        return (
            r"$\mathbf{MAPE} = \frac{100}{N} \sum \left| \frac{V_{DL} - V_{VFI}}{V_{VFI}} \right|$"
            + f"\nValue: {mape:.4f}%"
        )

    def _apply_style(self, style: PlotStyle) -> None:
        """Apply matplotlib style."""
        try:
            plt.style.use(style.value)
        except OSError:
            logger.debug(f"Style {style.value} not available, using default")

    def _save_figure(self, fig: plt.Figure, filename: str) -> Path:
        """Save figure and return path."""
        filepath = self.output_dir / filename
        fig.savefig(filepath, dpi=self.dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        logger.info(f"Saved figure: {filepath}")
        return filepath


@dataclass
class _SimulationResult:
    """Container for simulation results."""

    capital_history: FloatArray
    total_values: FloatArray
    steady_state_capital: float


# --- Main Execution ---
class EvaluationPipeline:
    """Orchestrates the complete model evaluation pipeline."""

    def __init__(
        self,
        paths: PathConfig | None = None,
        eval_config: EvaluationConfig | None = None,
    ) -> None:
        """
        Initialize the evaluation pipeline.

        Args:
            paths: Path configuration (uses defaults if None).
            eval_config: Evaluation parameters (uses defaults if None).
        """
        self.paths = paths or PathConfig()
        self.eval_config = eval_config or EvaluationConfig()

    def run(self) -> None:
        """Execute the complete evaluation pipeline."""
        logger.info("=" * 60)
        logger.info("Starting RBC Model Evaluation Pipeline")
        logger.info("=" * 60)

        # Validate paths
        self._validate_paths()

        # Setup components
        model = self._setup_model()
        checkpoints = self._load_checkpoints()
        vfi_data = self._load_vfi_data()

        if not checkpoints:
            logger.error("No valid checkpoints found. Please train the model first.")
            return

        # Generate figures
        self._generate_figures(model, checkpoints, vfi_data)

        logger.info("=" * 60)
        logger.info("Evaluation Complete")
        logger.info(f"Figures saved to: {self.paths.figures}")
        logger.info("=" * 60)

    def _validate_paths(self) -> None:
        """Validate required paths exist."""
        missing = self.paths.validate()
        if missing:
            raise FileNotFoundError(
                f"Missing required configuration files: {missing}"
            )
        self.paths.ensure_output_dirs()

    def _setup_model(self) -> BasicModelDL:
        """Initialize and return the model."""
        factory = ModelFactory(self.paths)
        return factory.create_model()

    def _load_checkpoints(self) -> list[Checkpoint]:
        """Load and return model checkpoints."""
        manager = CheckpointManager(
            self.paths.checkpoints,
            epoch_filter=self.eval_config.checkpoint_epoch_filter,
        )
        return manager.discover_checkpoints()

    def _load_vfi_data(self) -> VFIData | None:
        """Load VFI benchmark data if available."""
        return VFIData.from_npz(self.paths.vfi_results)

    def _generate_figures(
        self,
        model: BasicModelDL,
        checkpoints: list[Checkpoint],
        vfi_data: VFIData | None,
    ) -> None:
        """Generate all evaluation figures."""
        generator = FigureGenerator(
            model,
            self.paths.figures,
            dpi=self.eval_config.figure_dpi,
        )

        # Figure 1: Error Evolution
        generator.plot_error_evolution(checkpoints, vfi_data, self.eval_config)

        # Figure 2: Comparative Analysis (requires VFI data)
        if vfi_data:
            generator.plot_comparative_analysis(checkpoints[-1], vfi_data)
        else:
            logger.warning("Skipping Figure 2: VFI data required")

        # Figure 3: Simulation
        generator.plot_simulation(
            checkpoints[-1],
            vfi_data,
            n_trajectories=self.eval_config.simulation_trajectories,
            timesteps=self.eval_config.simulation_timesteps,
        )


def main() -> None:
    """Main entry point for the evaluation script."""
    try:
        pipeline = EvaluationPipeline()
        pipeline.run()
    except FileNotFoundError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Unexpected error during evaluation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()