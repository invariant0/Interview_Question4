"""
Policy Function Simulation Script (Figures 1-2).

This module simulates the policy function learned by the Deep Learning RBC model
and analyzes its dynamic behavior.

Refactored for compactness and clarity.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple, Dict, List

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from evaluation_common import (
    PathConfig,
    EvaluationConfig,
    Checkpoint,
    ModelFactory,
    CheckpointManager,
    PlotStyle,
    FloatArray,
    apply_plot_style,
    save_figure,
    configure_logging,
)

from econ_models.dl.basic import BasicModelDL

logger = configure_logging()


class PolicyAnalyzer:
    """Analyzes policy function behavior through simulation."""

    def __init__(self, model: BasicModelDL, output_dir: Path, dpi: int = 300):
        self.model = model
        self.output_dir = output_dir
        self.dpi = dpi
        self.cfg = model.config

    def _simulate(self, n_traj: int, steps: int, seed: int | None = None) -> object:
        """Run simulation and return the full history object."""
        history, _ = self.model.simulate(n_steps=steps, n_batches=n_traj, seed=seed)
        return history

    def plot_stationary_distribution(
        self, checkpoint: Checkpoint, n_traj: int = 10000, steps: int = 150
    ) -> Path:
        """Generate Figure 1: Stationary distribution heatmap and snapshots."""
        logger.info("Generating Figure 1: Stationary distribution...")
        CheckpointManager.load_weights(self.model, checkpoint)
        
        # 1. Simulate
        history = self._simulate(n_traj, steps)
        cap_hist = history.capital_history

        # 2. Calculate Stats
        k_min, k_max = self.cfg.capital_min, self.cfg.capital_max
        total = cap_hist.size
        rate_min = (np.sum(cap_hist <= k_min * 1.01) / total) * 100
        rate_max = (np.sum(cap_hist >= k_max * 0.99) / total) * 100

        # 3. Plot
        apply_plot_style(PlotStyle.SEABORN)
        fig, axes = plt.subplots(2, 1, figsize=(10, 12), constrained_layout=True)
        
        self._draw_heatmap(axes[0], cap_hist, steps, k_min, k_max, rate_min, rate_max)
        self._draw_snapshots(axes[1], cap_hist, steps, k_min, k_max)

        fig.suptitle(f"Stationary Distribution (N={n_traj})", fontsize=16, fontweight="bold")
        return save_figure(fig, self.output_dir, "stationary_distribution.png", self.dpi)

    def plot_policy_functions(
        self, checkpoint: Checkpoint, n_traj: int = 10000, steps: int = 150
    ) -> Path:
        """Generate Figure 2: Investment rate and policy analysis."""
        logger.info("Generating Figure 2: Investment Policy Analysis...")
        CheckpointManager.load_weights(self.model, checkpoint)

        # 1. Simulate & Flatten (exclude last step for S_t -> Action_t mapping)
        hist = self._simulate(n_traj, steps)
        data = {
            "k": hist.capital_history[:-1].flatten(),
            "z": hist.productivity_history[:-1].flatten(),
            "inv": hist.investment_history[:-1].flatten(),
            "rate": hist.investment_rate_history[:-1].flatten(),
        }

        # 2. Setup Bins
        bins_k = np.linspace(self.cfg.capital_min, self.cfg.capital_max, 50)
        bins_z = np.linspace(self.cfg.productivity_min, self.cfg.productivity_max, 16)
        
        # 3. Compute Averages
        avg_data = {
            "rate_k": self._bin_avg(data["k"], data["rate"], bins_k),
            "inv_k": self._bin_avg(data["k"], data["inv"], bins_k),
            "rate_z": self._bin_avg(data["z"], data["rate"], bins_z),
            "inv_z": self._bin_avg(data["z"], data["inv"], bins_z),
        }
        
        centers_k = (bins_k[:-1] + bins_k[1:]) / 2
        centers_z = (bins_z[:-1] + bins_z[1:]) / 2

        # 4. Plot Configuration
        plot_cfg = [
            (0, 0, centers_k, avg_data["rate_k"], "forestgreen", "o", 
             "Inv. Rate vs Capital", r"$K_t$", r"$I_t/K_t$"),
            (0, 1, centers_z, avg_data["rate_z"], "purple", "s", 
             "Inv. Rate vs Productivity", r"$Z_t$", r"$I_t/K_t$"),
            (1, 0, centers_k, avg_data["inv_k"], "darkorange", "o", 
             "Investment vs Capital", r"$K_t$", r"$I_t$"),
            (1, 1, centers_z, avg_data["inv_z"], "firebrick", "s", 
             "Investment vs Productivity", r"$Z_t$", r"$I_t$"),
        ]

        apply_plot_style(PlotStyle.SEABORN)
        fig, axes = plt.subplots(2, 2, figsize=(14, 12), constrained_layout=True)

        for r, c, x, y, color, marker, title, xl, yl in plot_cfg:
            self._draw_curve(axes[r, c], x, y, color, marker, title, xl, yl)

        return save_figure(fig, self.output_dir, "policy_function.png", self.dpi)

    # --- Plotting Helpers ---

    def _draw_heatmap(self, ax, data, steps, k_min, k_max, rate_min, rate_max):
        """Draws the capital distribution evolution heatmap."""
        bins = 80
        bin_edges = np.linspace(k_min, k_max, bins + 1)
        
        # Fast 2D histogram
        hist_data = np.array([
            np.histogram(data[t], bins=bin_edges, density=True)[0] 
            for t in range(steps)
        ])

        vmax = np.percentile(hist_data[hist_data > 0], 95) if hist_data.any() else 1
        
        im = ax.imshow(
            hist_data.T, aspect="auto", origin="lower", cmap="viridis",
            extent=[0, steps, k_min, k_max], vmin=0, vmax=vmax
        )
        plt.colorbar(im, ax=ax).set_label("Density", fontsize=11)

        # Annotations
        props = dict(facecolor="crimson", alpha=0.7, edgecolor="none", pad=2)
        for y, label, rate, offset in [(k_max, "Max", rate_max, -0.08), (k_min, "Min", rate_min, 0.02)]:
            ax.axhline(y, color="red", ls="--", lw=1.5, alpha=0.9)
            ax.text(steps * 0.02, y + (k_max - k_min) * offset, 
                    f"{label}: {y:.2f}\nHit: {rate:.2f}%", 
                    color="white", fontsize=9, bbox=props)

        ax.set_title("Evolution of Capital Distribution", fontsize=14, fontweight="bold")
        ax.set_ylabel("Capital (K)")
        ax.set_xlabel("Time (t)")

    def _draw_snapshots(self, ax, data, steps, k_min, k_max):
        """Draws density snapshots at t=0, t=mid, and t=end."""
        x_grid = np.linspace(k_min * 0.5, k_max * 1.1, 300)
        
        # Import transforms to help position text at the top of the plot
        import matplotlib.transforms as transforms
        trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)

        configs = [
            (0, "Initial", "gray", True, "-"),
            (steps // 2, f"t={steps//2}", "navy", False, "--"),
            (-1, "Stationary", "crimson", True, "-")
        ]

        for t_idx, label, color, fill, ls in configs:
            subset = data[t_idx]
            self._plot_density(ax, subset, x_grid, color, label, fill, ls)

            # --- MODIFICATION START: Add Mean Annotations ---
            if t_idx in [0, -1]:
                mu = np.mean(subset)
                
                # 1. Add a vertical dotted line at the mean
                ax.axvline(mu, color=color, linestyle=":", linewidth=2, alpha=0.8)
                
                # 2. Add text annotation
                # We stagger the Y position (0.95 vs 0.88) to prevent overlap if means are close
                y_pos = 0.95 if t_idx == 0 else 0.88
                
                ax.text(
                    mu, y_pos, 
                    f"Mean: {mu:.2f}", 
                    transform=trans,  # x is data coords, y is relative axis coords (0-1)
                    color=color, 
                    fontweight="bold", 
                    ha="center", 
                    fontsize=10,
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1.5)
                )
            # --- MODIFICATION END ---

        ax.axvline(k_min, color="red", ls="--", alpha=0.8)
        ax.axvline(k_max, color="red", ls="--", alpha=0.8)
        ax.set_title("Convergence to Stationary Distribution", fontsize=14, fontweight="bold")
        ax.legend()

    def _draw_curve(self, ax, x, y, color, marker, title, xlabel, ylabel):
        """Standardized curve plotter."""
        mask = ~np.isnan(y)
        ax.plot(x[mask], y[mask], color=color, lw=2, marker=marker, 
                ms=4 if marker=='o' else 6, alpha=0.8)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    @staticmethod
    def _plot_density(ax, data, x_grid, color, label, fill, ls):
        """Robust density plotter (KDE with fallback)."""
        if np.std(data) < 1e-10:
            ax.axvline(np.mean(data), color=color, ls=ls, label=f"{label} (point)")
            return

        try:
            y = stats.gaussian_kde(data)(x_grid)
            if fill:
                ax.fill_between(x_grid, y, alpha=0.3, color=color, label=label)
            ax.plot(x_grid, y, color=color, ls=ls, lw=1.5, label=label if not fill else "")
        except np.linalg.LinAlgError:
            # Fallback to histogram
            counts, edges = np.histogram(data, bins=50, density=True)
            centers = (edges[:-1] + edges[1:]) / 2
            ax.step(centers, counts, where="mid", color=color, ls=ls, label=label)

    @staticmethod
    def _bin_avg(x: FloatArray, y: FloatArray, bins: FloatArray) -> FloatArray:
        """Compute mean of y for each bin in x."""
        # Digitizing is faster than looping manually
        indices = np.digitize(x, bins) - 1
        n_bins = len(bins) - 1
        
        # Use bincount for fast summation and counting
        # Filter out-of-bound indices just in case
        mask = (indices >= 0) & (indices < n_bins)
        valid_idx = indices[mask]
        valid_y = y[mask]
        
        sums = np.bincount(valid_idx, weights=valid_y, minlength=n_bins)
        counts = np.bincount(valid_idx, minlength=n_bins)
        
        with np.errstate(invalid='ignore'):
            return sums / counts


class Pipeline:
    """Orchestrates the policy simulation."""

    def __init__(self, paths: PathConfig = None, cfg: EvaluationConfig = None):
        self.paths = paths or PathConfig()
        self.cfg = cfg or EvaluationConfig()

    def run(self):
        logger.info("=== Starting Policy Simulation Pipeline ===")
        
        if missing := self.paths.validate():
            logger.error(f"Missing files: {missing}")
            return

        self.paths.ensure_output_dirs()
        model = ModelFactory(self.paths).create_model()
        
        ckpt_mgr = CheckpointManager(self.paths.checkpoints, self.cfg.checkpoint_epoch_filter)
        checkpoints = ckpt_mgr.discover_checkpoints()

        if not checkpoints:
            logger.error("No checkpoints found.")
            return

        analyzer = PolicyAnalyzer(model, self.paths.figures, self.cfg.figure_dpi)
        latest = checkpoints[-1]

        analyzer.plot_stationary_distribution(
            latest, self.cfg.simulation_trajectories, self.cfg.simulation_timesteps
        )
        analyzer.plot_policy_functions(
            latest, self.cfg.simulation_trajectories, self.cfg.simulation_timesteps
        )
        
        logger.info(f"=== Simulation Complete. Figures at: {self.paths.figures} ===")


if __name__ == "__main__":
    try:
        Pipeline().run()
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        sys.exit(1)