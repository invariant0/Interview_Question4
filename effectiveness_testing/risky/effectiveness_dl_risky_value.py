"""
Test script to evaluate the effectiveness of trained DL solution vs VFI matrix.

This script:
1. Loads the VFI solution matrix and the trained DL model checkpoint
2. Computes and plots metrics across all saved checkpoints:
   - MAE evolution
   - MAPE in non-default region evolution
   - Default region identification accuracy evolution
   - Bellman residual evolution
3. Generates 3D comparison plots for value function and default probability
   under low, medium, and high productivity shocks

Supports both 'risky' and 'risky_upgrade' model architectures.
"""

import argparse
import os
import sys
import glob
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List, Tuple, Optional
import json

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from econ_models.config.economic_params import EconomicParams
from econ_models.config.dl_config import DeepLearningConfig, load_dl_config
from econ_models.core.nets import NeuralNetFactory
from econ_models.core.standardize import StateSpaceNormalizer
from econ_models.core.types import TENSORFLOW_DTYPE
from econ_models.core.sampling.transitions import TransitionFunctions
from econ_models.core.sampling.state_sampler import StateSampler
from econ_models.core.sampling.candidate_sampler import CandidateSampler
from econ_models.econ import (
    ProductionFunctions,
    AdjustmentCostCalculator,
    CashFlowCalculator,
    BondPricingCalculator
)

tfd = tfp.distributions


def load_json_file(filename: str) -> Dict:
    """Load a JSON file."""
    with open(filename, 'r') as f:
        return json.load(f)


def load_economic_params(filename: str) -> EconomicParams:
    """Load economic parameters from JSON."""
    data = load_json_file(filename)
    return EconomicParams(**data)


def load_vfi_results(filename: str) -> Dict[str, np.ndarray]:
    """Load VFI results from npz file."""
    data = np.load(filename)
    return {key: data[key] for key in data.files}


def build_value_network(config: DeepLearningConfig) -> tf.keras.Model:
    """Build the value network architecture."""
    return NeuralNetFactory.build_mlp(
        input_dim=3,
        output_dim=1,
        config=config,
        output_activation="relu",
        scale_factor=config.value_scale_factor,
        name="RiskyValueNet"
    )


def find_checkpoint_files(checkpoint_dir: str) -> List[Tuple[int, str]]:
    """
    Find all checkpoint files and return sorted by epoch.
    
    Returns:
        List of (epoch, filepath) tuples sorted by epoch.
    """
    # Note: Both risky and risky_upgrade use the same naming convention for value nets
    pattern = os.path.join(checkpoint_dir, "risky_value_net_*.weights.h5")
    files = glob.glob(pattern)
    
    epoch_files = []
    for f in files:
        basename = os.path.basename(f)
        try:
            epoch_str = basename.replace("risky_value_net_", "").replace(".weights.h5", "")
            epoch = int(epoch_str)
            epoch_files.append((epoch, f))
        except ValueError:
            continue
    
    return sorted(epoch_files, key=lambda x: x[0])


def evaluate_model_on_grid(
    model: tf.keras.Model,
    normalizer: StateSpaceNormalizer,
    k_grid: np.ndarray,
    b_grid: np.ndarray,
    z_grid: np.ndarray
) -> np.ndarray:
    """
    Evaluate the DL model on the VFI grid.
    """
    n_k, n_b, n_z = len(k_grid), len(b_grid), len(z_grid)
    
    K, B, Z = np.meshgrid(k_grid, b_grid, z_grid, indexing='ij')
    
    k_flat = K.flatten().reshape(-1, 1)
    b_flat = B.flatten().reshape(-1, 1)
    z_flat = Z.flatten().reshape(-1, 1)
    
    k_tensor = tf.constant(k_flat, dtype=TENSORFLOW_DTYPE)
    b_tensor = tf.constant(b_flat, dtype=TENSORFLOW_DTYPE)
    z_tensor = tf.constant(z_flat, dtype=TENSORFLOW_DTYPE)
    
    k_norm = normalizer.normalize_capital(k_tensor)
    b_norm = normalizer.normalize_debt(b_tensor)
    z_norm = normalizer.normalize_productivity(z_tensor)
    
    inputs = tf.concat([k_norm, b_norm, z_norm], axis=1)
    predictions = model(inputs, training=False).numpy()
    
    return predictions.reshape(n_k, n_b, n_z)


def evaluate_model_on_2d_grid(
    model: tf.keras.Model,
    normalizer: StateSpaceNormalizer,
    k_grid: np.ndarray,
    b_grid: np.ndarray,
    z_value: float
) -> np.ndarray:
    """
    Evaluate the DL model on a 2D grid (K, B) for a fixed Z value.
    """
    n_k, n_b = len(k_grid), len(b_grid)
    
    K, B = np.meshgrid(k_grid, b_grid, indexing='ij')
    Z = np.full_like(K, z_value)
    
    k_flat = K.flatten().reshape(-1, 1)
    b_flat = B.flatten().reshape(-1, 1)
    z_flat = Z.flatten().reshape(-1, 1)
    
    k_tensor = tf.constant(k_flat, dtype=TENSORFLOW_DTYPE)
    b_tensor = tf.constant(b_flat, dtype=TENSORFLOW_DTYPE)
    z_tensor = tf.constant(z_flat, dtype=TENSORFLOW_DTYPE)
    
    k_norm = normalizer.normalize_capital(k_tensor)
    b_norm = normalizer.normalize_debt(b_tensor)
    z_norm = normalizer.normalize_productivity(z_tensor)
    
    inputs = tf.concat([k_norm, b_norm, z_norm], axis=1)
    predictions = model(inputs, training=False).numpy()
    
    return predictions.reshape(n_k, n_b)


def compute_mae(vfi_values: np.ndarray, dl_values: np.ndarray) -> float:
    """Compute Mean Absolute Error between VFI and DL predictions."""
    return float(np.mean(np.abs(vfi_values - dl_values)))


def compute_mape_non_default(
    vfi_values: np.ndarray, 
    dl_values: np.ndarray, 
    default_threshold: float,
    epsilon: float = 1e-8
) -> float:
    """
    Compute Mean Absolute Percentage Error in non-default region.
    Non-default region is defined as where VFI value > default_threshold.
    """
    non_default_mask = vfi_values > default_threshold
    
    if not np.any(non_default_mask):
        return 0.0
    
    vfi_non_default = vfi_values[non_default_mask]
    dl_non_default = dl_values[non_default_mask]
    
    safe_vfi = np.maximum(np.abs(vfi_non_default), epsilon)
    mape = float(np.mean(np.abs(vfi_non_default - dl_non_default) / safe_vfi) * 100)
    
    return mape


def compute_default_region_f1(
    vfi_values: np.ndarray, 
    dl_values: np.ndarray, 
    default_threshold: float
) -> float:
    """
    Compute F1 score for default region identification.
    Default region is defined as where value <= default_threshold.
    """
    # Binary classification: default (1) vs non-default (0)
    vfi_default = (vfi_values <= default_threshold).astype(int)
    dl_default = (dl_values <= default_threshold).astype(int)
    
    # True positives: both VFI and DL identify as default
    tp = np.sum((vfi_default == 1) & (dl_default == 1))
    # False positives: DL says default but VFI says non-default
    fp = np.sum((vfi_default == 0) & (dl_default == 1))
    # False negatives: VFI says default but DL says non-default
    fn = np.sum((vfi_default == 1) & (dl_default == 0))
    
    # Precision: of all DL-predicted defaults, how many are true defaults
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    # Recall: of all true defaults, how many did DL identify
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    
    # F1 score: harmonic mean of precision and recall
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    
    return float(f1)


def determine_default_threshold(vfi_values: np.ndarray) -> float:
    """
    Automatically determine a reasonable default threshold.
    """
    vfi_min = vfi_values.min()
    vfi_max = vfi_values.max()
    vfi_range = vfi_max - vfi_min
    
    # Use the larger of: 5th percentile or 1% of the range above minimum
    percentile_threshold = np.percentile(vfi_values, 5)
    range_threshold = vfi_min + 0.01 * vfi_range
    
    # Also consider near-zero values
    near_zero_threshold = 0.01 * vfi_max
    
    threshold = max(percentile_threshold, range_threshold, near_zero_threshold)
    
    return float(threshold)


def compute_default_probability_vfi(
    vfi_values: np.ndarray,
    default_threshold: float
) -> np.ndarray:
    """Compute default probability from VFI values."""
    return (vfi_values <= default_threshold).astype(float)


def compute_default_probability_dl(
    dl_values: np.ndarray,
    default_threshold: float
) -> np.ndarray:
    """Compute default probability from DL values."""
    return (dl_values <= default_threshold).astype(float)


def plot_3d_value_and_default_comparison(
    model: tf.keras.Model,
    normalizer: StateSpaceNormalizer,
    V_vfi: np.ndarray,
    k_grid: np.ndarray,
    b_grid: np.ndarray,
    z_grid: np.ndarray,
    default_threshold: float,
    model_name: str,
    output_dir: str = "./test_results"
) -> plt.Figure:
    """
    Plot 3D comparison of value function and default probability.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Select low, medium, and high productivity indices
    n_z = len(z_grid)
    z_indices = {
        'low': 0,
        'medium': n_z // 2,
        'high': n_z - 1
    }
    z_labels = {
        'low': f'Low Shock (Z={z_grid[z_indices["low"]]:.3f})',
        'medium': f'Medium Shock (Z={z_grid[z_indices["medium"]]:.3f})',
        'high': f'High Shock (Z={z_grid[z_indices["high"]]:.3f})'
    }
    
    # Create meshgrid for 3D plotting
    K_mesh, B_mesh = np.meshgrid(k_grid, b_grid, indexing='ij')
    
    # Create figure with 2 rows x 3 columns
    fig = plt.figure(figsize=(18, 12))
    
    shock_levels = ['low', 'medium', 'high']
    
    # Define vivid color schemes
    from matplotlib.colors import LinearSegmentedColormap
    
    # Custom colormaps for more vivid distinction
    vfi_colors = ['#004d66', '#0099cc', '#00ccff', '#66e0ff']  # Cyan/Teal gradient
    dl_colors = ['#cc3300', '#ff6600', '#ff9933', '#ffcc66']   # Orange/Warm gradient
    
    vfi_cmap = LinearSegmentedColormap.from_list('vfi_cyan', vfi_colors)
    dl_cmap = LinearSegmentedColormap.from_list('dl_orange', dl_colors)
    
    for col_idx, shock in enumerate(shock_levels):
        z_idx = z_indices[shock]
        z_val = z_grid[z_idx]
        
        # Extract VFI slice for this Z
        V_vfi_slice = V_vfi[:, :, z_idx]
        
        # Evaluate DL model for this Z
        V_dl_slice = evaluate_model_on_2d_grid(model, normalizer, k_grid, b_grid, z_val)
        
        # Compute default probabilities
        default_prob_vfi = compute_default_probability_vfi(V_vfi_slice, default_threshold)
        default_prob_dl = compute_default_probability_dl(V_dl_slice, default_threshold)
        
        # --- Top Row: Value Function Comparison ---
        ax_top = fig.add_subplot(2, 3, col_idx + 1, projection='3d')
        
        # Plot VFI surface (Cyan/Teal)
        surf_vfi = ax_top.plot_surface(
            K_mesh, B_mesh, V_vfi_slice,
            alpha=0.7, cmap=vfi_cmap, edgecolor='darkblue',
            linewidth=0.1, antialiased=True,
            label='VFI'
        )
        
        # Plot DL surface (Orange/Warm)
        surf_dl = ax_top.plot_surface(
            K_mesh, B_mesh, V_dl_slice,
            alpha=0.5, cmap=dl_cmap, edgecolor='darkred',
            linewidth=0.1, antialiased=True,
            label=f'DL ({model_name})'
        )
        
        ax_top.set_xlabel('Capital (K)', fontsize=10, labelpad=10)
        ax_top.set_ylabel('Debt (B)', fontsize=10, labelpad=10)
        ax_top.set_zlabel('Value V(K,B)', fontsize=10, labelpad=10)
        ax_top.set_title(f'Value Function\n{z_labels[shock]}', fontsize=12, fontweight='bold')
        
        # Add legend proxy with vivid colors
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#0099cc', alpha=0.8, edgecolor='darkblue', label='VFI'),
            Patch(facecolor='#ff6600', alpha=0.8, edgecolor='darkred', label=f'DL ({model_name})')
        ]
        ax_top.legend(handles=legend_elements, loc='upper left', fontsize=9)
        
        # Adjusted view angle - rotated more to the side for better visibility
        ax_top.view_init(elev=15, azim=45)
        
        # --- Bottom Row: Default Probability Comparison ---
        ax_bottom = fig.add_subplot(2, 3, col_idx + 4, projection='3d')
        
        # For default probability, use solid vivid colors
        vfi_color_solid = '#00b3b3'  # Teal
        dl_color_solid = '#ff6b35'   # Vivid Orange
        
        # Plot VFI default probability surface
        surf_def_vfi = ax_bottom.plot_surface(
            K_mesh, B_mesh, default_prob_vfi,
            alpha=0.7, color=vfi_color_solid, edgecolor='#006666',
            linewidth=0.2, antialiased=True,
            label='VFI'
        )
        
        # Plot DL default probability surface
        surf_def_dl = ax_bottom.plot_surface(
            K_mesh, B_mesh, default_prob_dl,
            alpha=0.5, color=dl_color_solid, edgecolor='#cc4400',
            linewidth=0.2, antialiased=True,
            label=f'DL ({model_name})'
        )
        
        ax_bottom.set_xlabel('Capital (K)', fontsize=10, labelpad=10)
        ax_bottom.set_ylabel('Debt (B)', fontsize=10, labelpad=10)
        ax_bottom.set_zlabel('Default Prob.', fontsize=10, labelpad=10)
        ax_bottom.set_title(f'Default Probability\n{z_labels[shock]}', fontsize=12, fontweight='bold')
        ax_bottom.set_zlim(-0.1, 1.1)
        
        # Add legend proxy with matching colors
        legend_elements_default = [
            Patch(facecolor=vfi_color_solid, alpha=0.8, edgecolor='#006666', label='VFI'),
            Patch(facecolor=dl_color_solid, alpha=0.8, edgecolor='#cc4400', label=f'DL ({model_name})')
        ]
        ax_bottom.legend(handles=legend_elements_default, loc='upper left', fontsize=9)
        
        # Adjusted view angle - rotated more to the side for better visibility
        ax_bottom.view_init(elev=15, azim=60)
    
    plt.tight_layout()
    
    # Add overall title
    fig.suptitle(
        f'VFI vs Deep Learning ({model_name}): Value Function and Default Probability Comparison',
        fontsize=14, fontweight='bold', y=1.02
    )
    
    save_path = os.path.join(output_dir, "value_default_3d_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved 3D comparison plot to {save_path}")
    
    return fig

class BellmanResidualCalculator:
    """
    Calculator for Bellman residuals using the risky debt model structure.
    """
    
    def __init__(
        self,
        params: EconomicParams,
        config: DeepLearningConfig,
        normalizer: StateSpaceNormalizer
    ):
        self.params = params
        self.config = config
        self.normalizer = normalizer
        
        # Build shock distribution
        self.shock_dist = tfd.Normal(
            loc=tf.cast(0.0, TENSORFLOW_DTYPE),
            scale=tf.cast(params.productivity_std_dev, TENSORFLOW_DTYPE)
        )
    
    def sample_evaluation_states(
        self,
        batch_size: int = 500,
        seed: Optional[int] = 42
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        if seed is not None:
            tf.random.set_seed(seed)
        
        # Sample states at full progress (entire domain)
        progress = tf.constant(1.0, dtype=TENSORFLOW_DTYPE)
        k, b, z = StateSampler.sample_states(
            batch_size=batch_size,
            config=self.config,
            include_debt=True,
            progress=progress
        )
        
        return k, b, z
    
    def estimate_bond_price(
        self,
        model: tf.keras.Model,
        k_prime_cand: tf.Tensor,
        b_prime_cand: tf.Tensor,
        z_curr: tf.Tensor,
        n_samples: int = 30
    ) -> tf.Tensor:
        batch_size = tf.shape(z_curr)[0]
        n_cand = tf.shape(k_prime_cand)[1]
        
        # Sample future shocks
        eps = self.shock_dist.sample(sample_shape=(batch_size, n_cand, n_samples))
        z_curr_bc = tf.expand_dims(
            tf.broadcast_to(z_curr, (batch_size, n_cand)), -1
        )
        z_prime = TransitionFunctions.log_ar1_transition(
            z_curr_bc, self.params.productivity_persistence, eps
        )
        
        # Broadcast candidates
        k_prime_bc = tf.broadcast_to(
            tf.expand_dims(k_prime_cand, -1), (batch_size, n_cand, n_samples)
        )
        b_prime_bc = tf.broadcast_to(
            tf.expand_dims(b_prime_cand, -1), (batch_size, n_cand, n_samples)
        )
        
        # Evaluate value at (k', b', z') for default check
        flat_shape = (batch_size * n_cand * n_samples, 1)
        inputs_eval = tf.concat([
            self.normalizer.normalize_capital(tf.reshape(k_prime_bc, flat_shape)),
            self.normalizer.normalize_debt(tf.reshape(b_prime_bc, flat_shape)),
            self.normalizer.normalize_productivity(tf.reshape(z_prime, flat_shape))
        ], axis=1)
        
        v_prime_eval = model(inputs_eval, training=False)
        is_default = tf.cast(
            v_prime_eval <= self.config.epsilon_value_default,
            TENSORFLOW_DTYPE
        )
        
        # Calculate payoff
        profit = (
            (1.0 - self.params.corporate_tax_rate)
            * ProductionFunctions.cobb_douglas(
                tf.reshape(k_prime_bc, flat_shape),
                tf.reshape(z_prime, flat_shape),
                self.params
            )
        )
        recovery = BondPricingCalculator.recovery_value(
            profit, tf.reshape(k_prime_bc, flat_shape), self.params
        )
        payoff = BondPricingCalculator.bond_payoff(
            recovery, tf.reshape(b_prime_bc, flat_shape), is_default
        )
        
        # Average over samples
        expected_payoff = tf.reduce_mean(
            tf.reshape(payoff, (batch_size, n_cand, n_samples)), axis=2
        )
        
        return BondPricingCalculator.risk_neutral_price(
            expected_payoff,
            b_prime_cand,
            self.params.risk_free_rate,
            self.config.epsilon_debt,
            self.config.min_q_price
        )
    
    def compute_bellman_residual(
        self,
        model: tf.keras.Model,
        k: tf.Tensor,
        b: tf.Tensor,
        z: tf.Tensor,
        n_candidates: int = 100,
        n_mc_samples: int = 30
    ) -> tf.Tensor:
        batch_size = tf.shape(k)[0]
        
        # Generate candidates for (k', b') at full progress
        progress = tf.constant(1.0, dtype=TENSORFLOW_DTYPE)
        k_cand, b_cand = CandidateSampler.sample_candidate(
            batch_size, n_candidates, k, b,
            self.config, progress=progress
        )
        n_actual = tf.shape(k_cand)[1]
        
        # Estimate bond prices for all candidates
        q_cand = self.estimate_bond_price(model, k_cand, b_cand, z)
        
        # Calculate current period revenue
        revenue = (
            (1.0 - self.params.corporate_tax_rate)
            * ProductionFunctions.cobb_douglas(k, z, self.params)
        )
        
        # Broadcast current states
        k_curr_bc = tf.broadcast_to(k, (batch_size, n_actual))
        b_curr_bc = tf.broadcast_to(b, (batch_size, n_actual))
        
        # Calculate investment and adjustment costs
        investment = ProductionFunctions.calculate_investment(
            k_curr_bc, k_cand, self.params
        )
        adj_cost, _ = AdjustmentCostCalculator.calculate(
            investment, k_curr_bc, self.params
        )
        
        # Calculate dividends for each candidate
        dividend = CashFlowCalculator.risky_cash_flow(
            tf.broadcast_to(revenue, (batch_size, n_actual)),
            investment, adj_cost,
            b_curr_bc, b_cand, q_cand, self.params
        )
        
        # Estimate expected continuation value via Monte Carlo
        eps_samples = self.shock_dist.sample(
            sample_shape=(batch_size, n_actual, n_mc_samples)
        )
        
        z_curr_expanded = tf.expand_dims(
            tf.broadcast_to(z, (batch_size, n_actual)), -1
        )
        z_prime_samples = TransitionFunctions.log_ar1_transition(
            z_curr_expanded, self.params.productivity_persistence, eps_samples
        )
        
        k_cand_expanded = tf.broadcast_to(
            tf.expand_dims(k_cand, -1), (batch_size, n_actual, n_mc_samples)
        )
        b_cand_expanded = tf.broadcast_to(
            tf.expand_dims(b_cand, -1), (batch_size, n_actual, n_mc_samples)
        )
        
        flat_dim = batch_size * n_actual * n_mc_samples
        inputs_next = tf.concat([
            self.normalizer.normalize_capital(
                tf.reshape(k_cand_expanded, (flat_dim, 1))
            ),
            self.normalizer.normalize_debt(
                tf.reshape(b_cand_expanded, (flat_dim, 1))
            ),
            self.normalizer.normalize_productivity(
                tf.reshape(z_prime_samples, (flat_dim, 1))
            )
        ], axis=1)
        
        v_prime_flat = model(inputs_next, training=False)
        v_prime_samples = tf.reshape(
            v_prime_flat, (batch_size, n_actual, n_mc_samples)
        )
        expected_v_prime = tf.reduce_mean(v_prime_samples, axis=2)
        
        # Calculate RHS of Bellman equation for each candidate
        beta = tf.cast(self.params.discount_factor, TENSORFLOW_DTYPE)
        rhs_cand = dividend + beta * expected_v_prime
        
        # Take maximum over candidates (optimal policy)
        max_rhs = tf.reduce_max(rhs_cand, axis=1)
        
        # Compute current value
        inputs_curr = tf.concat([
            self.normalizer.normalize_capital(k),
            self.normalizer.normalize_debt(b),
            self.normalizer.normalize_productivity(z)
        ], axis=1)
        v_curr = tf.squeeze(model(inputs_curr, training=False), axis=1)
        
        # Bellman residual: V(s) - max_{a} { r(s,a) + beta * E[V(s')] }
        residual = v_curr - max_rhs
        
        return residual


def compute_bellman_residual_stats(
    model: tf.keras.Model,
    calculator: BellmanResidualCalculator,
    k_eval: tf.Tensor,
    b_eval: tf.Tensor,
    z_eval: tf.Tensor,
    n_candidates: int = 100,
    n_mc_samples: int = 30
) -> Dict[str, float]:
    """Compute Bellman residual statistics for a batch of evaluation states."""
    residuals = calculator.compute_bellman_residual(
        model, k_eval, b_eval, z_eval,
        n_candidates=n_candidates,
        n_mc_samples=n_mc_samples
    )
    
    residuals_np = residuals.numpy()
    
    return {
        "mean_abs_residual": float(np.mean(np.abs(residuals_np))),
        "max_abs_residual": float(np.max(np.abs(residuals_np))),
        "std_residual": float(np.std(residuals_np)),
        "mean_residual": float(np.mean(residuals_np))
    }


def plot_training_metrics(
    epochs: List[int],
    maes: List[float],
    mapes_non_default: List[float],
    f1_scores: List[float],
    bellman_residuals: List[float],
    model_name: str,
    output_dir: str = "./test_results"
) -> plt.Figure:
    """
    Plot four metrics in a single figure with four subplots.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Training Metrics Evolution: {model_name}", fontsize=16, y=0.98)
    
    # Plot 1: MAE Evolution
    ax1 = axes[0, 0]
    ax1.plot(epochs, maes, 'b-', linewidth=2, marker='o', markersize=3, alpha=0.7)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('MAE', fontsize=12)
    ax1.set_title('MAE Evolution', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Annotate minimum MAE
    min_idx = np.argmin(maes)
    ax1.scatter([epochs[min_idx]], [maes[min_idx]], color='red', s=100, zorder=5, marker='*')
    ax1.annotate(
        f'Min: {maes[min_idx]:.4f}\n(Epoch {epochs[min_idx]})',
        xy=(epochs[min_idx], maes[min_idx]),
        xytext=(0.6, 0.85),
        textcoords='axes fraction',
        fontsize=9,
        color='red',
        arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='red', alpha=0.8)
    )
    
    # Plot 2: MAPE in Non-Default Region Evolution
    ax2 = axes[0, 1]
    ax2.plot(epochs, mapes_non_default, 'g-', linewidth=2, marker='s', markersize=3, alpha=0.7)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('MAPE (%)', fontsize=12)
    ax2.set_title('MAPE in Non-Default Region', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # Annotate minimum MAPE
    min_idx_mape = np.argmin(mapes_non_default)
    ax2.scatter([epochs[min_idx_mape]], [mapes_non_default[min_idx_mape]], 
                color='darkgreen', s=100, zorder=5, marker='*')
    ax2.annotate(
        f'Min: {mapes_non_default[min_idx_mape]:.2f}%\n(Epoch {epochs[min_idx_mape]})',
        xy=(epochs[min_idx_mape], mapes_non_default[min_idx_mape]),
        xytext=(0.6, 0.85),
        textcoords='axes fraction',
        fontsize=9,
        color='darkgreen',
        arrowprops=dict(arrowstyle='->', color='darkgreen', alpha=0.7),
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='darkgreen', alpha=0.8)
    )
    
    # Plot 3: Default Region Identification F1 Score Evolution
    ax3 = axes[1, 0]
    ax3.plot(epochs, f1_scores, 'r-', linewidth=2, marker='^', markersize=3, alpha=0.7)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('F1 Score', fontsize=12)
    ax3.set_title('Default Region Identification\n(F1 Score)', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1.05)
    
    # Annotate maximum F1 score
    max_idx_f1 = np.argmax(f1_scores)
    ax3.scatter([epochs[max_idx_f1]], [f1_scores[max_idx_f1]], 
                color='darkred', s=100, zorder=5, marker='*')
    ax3.annotate(
        f'Max: {f1_scores[max_idx_f1]:.4f}\n(Epoch {epochs[max_idx_f1]})',
        xy=(epochs[max_idx_f1], f1_scores[max_idx_f1]),
        xytext=(0.6, 0.15),
        textcoords='axes fraction',
        fontsize=9,
        color='darkred',
        arrowprops=dict(arrowstyle='->', color='darkred', alpha=0.7),
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='darkred', alpha=0.8)
    )
    
    # Add horizontal line at F1=1.0 (perfect)
    ax3.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Perfect Score')
    
    # Plot 4: Bellman Residual Evolution
    ax4 = axes[1, 1]
    ax4.plot(epochs, bellman_residuals, 'm-', linewidth=2, marker='d', markersize=3, alpha=0.7)
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Mean Absolute Bellman Residual', fontsize=12)
    ax4.set_title('Bellman Residual Evolution', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    
    # Annotate minimum Bellman residual
    min_idx_bellman = np.argmin(bellman_residuals)
    ax4.scatter([epochs[min_idx_bellman]], [bellman_residuals[min_idx_bellman]], 
                color='purple', s=100, zorder=5, marker='*')
    ax4.annotate(
        f'Min: {bellman_residuals[min_idx_bellman]:.4f}\n(Epoch {epochs[min_idx_bellman]})',
        xy=(epochs[min_idx_bellman], bellman_residuals[min_idx_bellman]),
        xytext=(0.6, 0.85),
        textcoords='axes fraction',
        fontsize=9,
        color='purple',
        arrowprops=dict(arrowstyle='->', color='purple', alpha=0.7),
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='purple', alpha=0.8)
    )
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    save_path = os.path.join(output_dir, "training_metrics_evolution.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved metrics plot to {save_path}")
    
    return fig


def to_python_float(val) -> float:
    """Convert numpy/tensorflow scalar to Python float for JSON serialization."""
    if hasattr(val, 'numpy'):
        return float(val.numpy())
    elif hasattr(val, 'item'):
        return float(val.item())
    else:
        return float(val)


def main():
    """Main evaluation script."""
    parser = argparse.ArgumentParser(
        description="Evaluate Economic Deep Learning Models against VFI"
    )
    parser.add_argument(
        '--model',
        type=str,
        default='risky',
        choices=['risky', 'risky_upgrade'],
        help="Model architecture to evaluate."
    )
    args = parser.parse_args()

    print("=" * 60)
    print(f"Risky Model ({args.model}): DL vs VFI Comparison Test")
    print("=" * 60)
    
    # Configuration paths
    vfi_results_path = "./ground_truth/risky_debt_model_vfi_results.npz"
    bounds_path = "./hyperparam/autogen/bounds_risky.json"
    econ_params_path = "./hyperparam/prefixed/econ_params_risky.json"
    dl_params_path = "./hyperparam/prefixed/dl_params.json"
    
    # Set directories based on model type
    if args.model == 'risky':
        checkpoint_dir = "./checkpoints/risky"
        output_dir = "./results/result_effectiveness_dl_risky"
    elif args.model == 'risky_upgrade':
        checkpoint_dir = "./checkpoints/risky_upgrade"
        output_dir = "./results/result_effectiveness_dl_risky_upgrade"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Load VFI Results
    print("\n[1] Loading VFI Results...")
    if not os.path.exists(vfi_results_path):
        print(f"ERROR: VFI results file not found at {vfi_results_path}")
        print("Please run 'python solve_vfi.py --model risky --auto-bounds' first.")
        return
    
    vfi_data = load_vfi_results(vfi_results_path)
    V_vfi = vfi_data['V']
    k_grid = vfi_data['K']
    b_grid = vfi_data['B']
    z_grid = vfi_data['Z']
    
    print(f"  VFI Value Function shape: {V_vfi.shape}")
    print(f"  K grid: [{k_grid.min():.4f}, {k_grid.max():.4f}], n={len(k_grid)}")
    print(f"  B grid: [{b_grid.min():.4f}, {b_grid.max():.4f}], n={len(b_grid)}")
    print(f"  Z grid: [{z_grid.min():.4f}, {z_grid.max():.4f}], n={len(z_grid)}")
    
    # Determine default threshold
    default_threshold = determine_default_threshold(V_vfi)
    n_default_states = np.sum(V_vfi <= default_threshold)
    total_states = V_vfi.size
    print(f"\n  Default threshold: {default_threshold:.6f}")
    print(f"  Default states: {n_default_states} / {total_states} ({100*n_default_states/total_states:.2f}%)")
    
    # Step 2: Load Economic Parameters and DL Config
    print("\n[2] Loading Configurations...")
    
    if not os.path.exists(econ_params_path):
        print(f"ERROR: Economic params file not found at {econ_params_path}")
        return
    
    econ_params = load_economic_params(econ_params_path)
    print(f"  Economic params loaded from {econ_params_path}")
    
    # Load DL config with the correct model tag
    dl_config = load_dl_config(dl_params_path, args.model)
    
    if os.path.exists(bounds_path):
        bounds_data = load_json_file(bounds_path)
        bounds = bounds_data['bounds']
        dl_config.capital_min = bounds['k_min']
        dl_config.capital_max = bounds['k_max']
        dl_config.debt_min = bounds['b_min']
        dl_config.debt_max = bounds['b_max']
        dl_config.productivity_min = bounds['z_min']
        dl_config.productivity_max = bounds['z_max']
        print(f"  Bounds loaded from {bounds_path}")
    else:
        print(f"  WARNING: Bounds file not found, using default DL config bounds")
    dl_config.update_value_scale(econ_params)
    
    # Step 3: Build Model Architecture and Bellman Calculator
    print("\n[3] Building Value Network and Bellman Calculator...")
    normalizer = StateSpaceNormalizer(dl_config)
    value_net = build_value_network(dl_config)
    
    # Initialize Bellman residual calculator
    bellman_calculator = BellmanResidualCalculator(econ_params, dl_config, normalizer)
    
    # Sample fixed evaluation states for Bellman residual (for consistency across checkpoints)
    print("  Sampling evaluation states for Bellman residual...")
    bellman_batch_size = 200  # Reduced for computational efficiency
    k_eval, b_eval, z_eval = bellman_calculator.sample_evaluation_states(
        batch_size=bellman_batch_size, seed=42
    )
    print(f"  Sampled {bellman_batch_size} states for Bellman residual evaluation")
    
    # Step 4: Find Checkpoint Files
    print(f"\n[4] Scanning for Checkpoints in {checkpoint_dir}...")
    if not os.path.exists(checkpoint_dir):
        print(f"ERROR: Checkpoint directory not found: {checkpoint_dir}")
        return

    checkpoint_files = find_checkpoint_files(checkpoint_dir)
    
    if not checkpoint_files:
        print(f"ERROR: No checkpoint files found in {checkpoint_dir}")
        print(f"Please train the model first using 'python train_dl.py --model {args.model}'")
        return
    
    print(f"  Found {len(checkpoint_files)} checkpoints")
    print(f"  Epochs: {[e for e, _ in checkpoint_files[:5]]}...{[e for e, _ in checkpoint_files[-3:]]}")
    
    # Step 5: Evaluate All Checkpoints
    print("\n[5] Evaluating All Checkpoints...")
    epochs: List[int] = []
    maes: List[float] = []
    mapes_non_default: List[float] = []
    f1_scores: List[float] = []
    bellman_residuals: List[float] = []
    
    for i, (epoch, filepath) in enumerate(checkpoint_files):
        try:
            value_net.load_weights(filepath)
            
            # Compute VFI comparison metrics
            predictions = evaluate_model_on_grid(value_net, normalizer, k_grid, b_grid, z_grid)
            
            mae = compute_mae(V_vfi, predictions)
            mape_nd = compute_mape_non_default(V_vfi, predictions, default_threshold)
            f1 = compute_default_region_f1(V_vfi, predictions, default_threshold)
            
            # Compute Bellman residual
            bellman_stats = compute_bellman_residual_stats(
                value_net, bellman_calculator,
                k_eval, b_eval, z_eval,
                n_candidates=100,
                n_mc_samples=30
            )
            bellman_res = bellman_stats["mean_abs_residual"]
            
            epochs.append(epoch)
            maes.append(mae)
            mapes_non_default.append(mape_nd)
            f1_scores.append(f1)
            bellman_residuals.append(bellman_res)
            
            if i % 10 == 0 or i == len(checkpoint_files) - 1:
                print(f"  Epoch {epoch:5d}: MAE={mae:.6f}, MAPE(non-def)={mape_nd:.2f}%, "
                      f"F1={f1:.4f}, Bellman={bellman_res:.6f}")
                
        except Exception as e:
            print(f"  WARNING: Failed to load checkpoint {filepath}: {e}")
            continue
    
    if len(epochs) == 0:
        print("ERROR: Could not evaluate any checkpoints")
        return
    
    # Step 6: Generate Visualization - Training Metrics
    print("\n[6] Generating Metrics Plot...")
    fig = plot_training_metrics(
        epochs, maes, mapes_non_default, f1_scores, bellman_residuals, args.model, output_dir
    )
    plt.close(fig)
    
    # Step 7: Generate 3D Value Function and Default Probability Comparison
    print("\n[7] Generating 3D Value Function and Default Probability Comparison...")
    
    # Load the best checkpoint based on MAE for the 3D comparison
    best_mae_idx = np.argmin(maes)
    best_epoch = epochs[best_mae_idx]
    best_checkpoint_path = None
    for epoch, filepath in checkpoint_files:
        if epoch == best_epoch:
            best_checkpoint_path = filepath
            break
    
    if best_checkpoint_path is not None:
        print(f"  Loading best checkpoint (Epoch {best_epoch}) for 3D visualization...")
        value_net.load_weights(best_checkpoint_path)
        
        fig_3d = plot_3d_value_and_default_comparison(
            value_net, normalizer, V_vfi,
            k_grid, b_grid, z_grid,
            default_threshold, args.model, output_dir
        )
        plt.close(fig_3d)
    else:
        print("  WARNING: Could not find best checkpoint for 3D visualization")
    
    # Step 8: Print Summary Statistics
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    
    print(f"\n  Final Checkpoint (Epoch {epochs[-1]}):")
    print(f"    MAE:                {maes[-1]:.6f}")
    print(f"    MAPE (non-default): {mapes_non_default[-1]:.2f}%")
    print(f"    F1 Score:           {f1_scores[-1]:.4f}")
    print(f"    Bellman Residual:   {bellman_residuals[-1]:.6f}")
    
    best_mae_idx = np.argmin(maes)
    best_mape_idx = np.argmin(mapes_non_default)
    best_f1_idx = np.argmax(f1_scores)
    best_bellman_idx = np.argmin(bellman_residuals)
    
    print(f"\n  Best MAE (Epoch {epochs[best_mae_idx]}):")
    print(f"    MAE: {maes[best_mae_idx]:.6f}")
    
    print(f"\n  Best MAPE non-default (Epoch {epochs[best_mape_idx]}):")
    print(f"    MAPE: {mapes_non_default[best_mape_idx]:.2f}%")
    
    print(f"\n  Best F1 Score (Epoch {epochs[best_f1_idx]}):")
    print(f"    F1: {f1_scores[best_f1_idx]:.4f}")
    
    print(f"\n  Best Bellman Residual (Epoch {epochs[best_bellman_idx]}):")
    print(f"    Bellman: {bellman_residuals[best_bellman_idx]:.6f}")
    
    print(f"\n  Default Threshold: {default_threshold:.6f}")
    print(f"  Default Region: {100*n_default_states/total_states:.2f}% of state space")
    
    print(f"\n  Results saved to: {output_dir}")
    print("=" * 60)
    
    # Save numerical results to JSON
    results_summary = {
        "model_type": args.model,
        "default_threshold": to_python_float(default_threshold),
        "default_region_percentage": to_python_float(100*n_default_states/total_states),
        "bellman_evaluation_batch_size": bellman_batch_size,
        "final_epoch": int(epochs[-1]),
        "final_metrics": {
            "mae": to_python_float(maes[-1]),
            "mape_non_default": to_python_float(mapes_non_default[-1]),
            "f1_score": to_python_float(f1_scores[-1]),
            "bellman_residual": to_python_float(bellman_residuals[-1])
        },
        "best_metrics": {
            "mae": {"value": to_python_float(maes[best_mae_idx]), "epoch": int(epochs[best_mae_idx])},
            "mape_non_default": {"value": to_python_float(mapes_non_default[best_mape_idx]), "epoch": int(epochs[best_mape_idx])},
            "f1_score": {"value": to_python_float(f1_scores[best_f1_idx]), "epoch": int(epochs[best_f1_idx])},
            "bellman_residual": {"value": to_python_float(bellman_residuals[best_bellman_idx]), "epoch": int(epochs[best_bellman_idx])}
        },
        "epochs": [int(e) for e in epochs],
        "maes": [to_python_float(m) for m in maes],
        "mapes_non_default": [to_python_float(m) for m in mapes_non_default],
        "f1_scores": [to_python_float(f) for f in f1_scores],
        "bellman_residuals": [to_python_float(b) for b in bellman_residuals]
    }
    
    summary_path = os.path.join(output_dir, "evaluation_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nNumerical results saved to: {summary_path}")


if __name__ == "__main__":
    main()