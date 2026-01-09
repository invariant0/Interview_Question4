"""
Value Function Evaluation Script (Figures 1-2).

This module evaluates the value function learned by the Deep Learning RBC model
against a Value Function Iteration (VFI) benchmark. It generates two figures:

1. Error evolution (MAE, Bellman, Euler) over training epochs.
2. A 2x2 comparative analysis of Value Functions and Error Distributions (MAE/MAPE).

Standards:
    - PEP 8 (Style Guide for Python Code)
    - PEP 257 (Docstring Conventions)
    - Clean Code principles (Modularity, Clarity)

Author: AI Assistant
Date: 2025-12-26
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

from evaluation_common import (
    PathConfig,
    EvaluationConfig,
    VFIData,
    TrainingMetrics,
    Checkpoint,
    ModelFactory,
    CheckpointManager,
    ResidualCalculator,
    PlotStyle,
    FloatArray,
    TensorFloat,
    compute_mae,
    compute_mape,
    compute_rmse,
    apply_plot_style,
    save_figure,
    configure_logging,
    _ensure_plotting_imports,
)

# Import project modules
from econ_models.core.math import MathUtils
from econ_models.core.types import TENSORFLOW_DTYPE
from econ_models.dl.basic import BasicModelDL

logger = configure_logging()

# Lazy imports for plotting
plt = None
cm = None


def _get_plotting_modules():
    """Get matplotlib modules."""
    global plt, cm
    _ensure_plotting_imports()
    import matplotlib.pyplot as _plt
    from matplotlib import cm as _cm
    plt = _plt
    cm = _cm
    return plt, cm


class ValueFunctionEvaluator:
    """Evaluates value function accuracy and generates Figures 1-2."""

    def __init__(
        self,
        model: BasicModelDL,
        output_dir: Path,
        dpi: int = 300,
    ) -> None:
        """
        Initialize the value function evaluator.

        Args:
            model: The trained deep learning model.
            output_dir: Directory to save generated figures.
            dpi: Resolution for saved figures.
        """
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
        plt, _ = _get_plotting_modules()

        metrics = self._compute_metrics_over_epochs(
            checkpoints, vfi_data, eval_config
        )

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        apply_plot_style(PlotStyle.SEABORN)

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

        return save_figure(fig, self.output_dir, "error_evolution.png", self.dpi)

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
        plt, cm = _get_plotting_modules()

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
        apply_plot_style(PlotStyle.SEABORN)

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

        return save_figure(fig, self.output_dir, "comparative_analysis.png", self.dpi)

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

    def _plot_metric(
        self,
        ax,
        epochs: list[int],
        data: list[float],
        title: str,
        color: str,
        marker: str,
    ) -> None:
        """Plot a single metric on an axis."""
        plt, _ = _get_plotting_modules()
        
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
        fig,
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
        plt, cm = _get_plotting_modules()
        
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


class ValueFunctionEvaluationPipeline:
    """Orchestrates the value function evaluation pipeline (Figures 1-2)."""

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
        """Execute the value function evaluation pipeline."""
        logger.info("=" * 60)
        logger.info("Starting Value Function Evaluation Pipeline (Figures 1-2)")
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
        logger.info("Value Function Evaluation Complete")
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
        """Generate value function evaluation figures."""
        evaluator = ValueFunctionEvaluator(
            model,
            self.paths.figures,
            dpi=self.eval_config.figure_dpi,
        )

        # Figure 1: Error Evolution
        evaluator.plot_error_evolution(checkpoints, vfi_data, self.eval_config)

        # Figure 2: Comparative Analysis (requires VFI data)
        if vfi_data:
            evaluator.plot_comparative_analysis(checkpoints[-1], vfi_data)
        else:
            logger.warning("Skipping Figure 2: VFI data required")


def main() -> None:
    """Main entry point for value function evaluation."""
    try:
        pipeline = ValueFunctionEvaluationPipeline()
        pipeline.run()
    except FileNotFoundError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Unexpected error during evaluation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()