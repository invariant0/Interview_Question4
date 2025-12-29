"""
Deep Learning RBC Model Evaluation Script.

This module evaluates the performance of a Deep Learning-based Real Business Cycle (RBC)
model against a Value Function Iteration (VFI) benchmark. It generates three figures:
1. Error evolution (RMSE, Bellman, Euler) over training epochs.
2. A 2x2 comparative analysis of Value Functions and Error Distributions (MAE/MAPE).
3. Stochastic simulation of trajectories compared to theoretical steady states.

Standards:
    - PEP 8 (Style Guide for Python Code)
    - PEP 257 (Docstring Conventions)
    - Clean Code principles (Modularity, Clarity)

Author: AI Assistant
Date: 2025-12-26
"""

import os
import glob
import re
import sys
import json
import logging
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# --- Import Project Modules ---
# Assuming these modules exist in the user's environment
try:
    from econ_models.config.economic_params import EconomicParams
    from econ_models.config.dl_config import DeepLearningConfig
    from econ_models.dl.basic import BasicModelDL
    from econ_models.core.types import TENSORFLOW_DTYPE
    from econ_models.econ import EconomicLogic
    from econ_models.core.math import MathUtils
except ImportError as e:
    print(f"Critical Error: Failed to import project modules. {e}")
    sys.exit(1)

# --- Configuration & Constants ---
LOG_LEVEL = logging.INFO
logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PATHS = {
    "econ_params": "./hyperparam/prefixed/econ_params_basic.json",
    "dl_params": "./hyperparam/prefixed/dl_params.json",
    "bounds": "./hyperparam/autogen/bounds_basic.json",
    "vfi_results": "./ground_truth/basic_model_vfi_results.npz",
    "checkpoints": "checkpoints/basic",
    "figures": "./result_effectiveness_dl_basic"
}

# Ensure figure directory exists
os.makedirs(PATHS["figures"], exist_ok=True)

tfd = tfp.distributions


def load_json_file(filepath: str) -> Dict[str, Any]:
    """
    Load and parse a JSON file safely.

    Args:
        filepath: Path to the JSON file.

    Returns:
        A dictionary containing the JSON data.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    if not os.path.exists(filepath):
        logger.error(f"File not found: {filepath}")
        sys.exit(1)
    
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON from {filepath}: {e}")
        sys.exit(1)


def setup_model() -> BasicModelDL:
    """
    Initialize the Deep Learning model with configuration from files.

    Returns:
        An initialized BasicModelDL instance with weights ready to be loaded.
    """
    logger.info("Initializing Model Configuration...")

    # 1. Load Economic Parameters
    econ_data = load_json_file(PATHS["econ_params"])
    econ_params = EconomicParams(**econ_data)

    # 2. Load Deep Learning Configuration
    dl_data_raw = load_json_file(PATHS["dl_params"])
    # Handle nested config structure if present
    dl_config_dict = dl_data_raw.get('basic', dl_data_raw)
    
    # Ensure hidden layers are tuples (immutable)
    if 'hidden_layers' in dl_config_dict:
        dl_config_dict['hidden_layers'] = tuple(dl_config_dict['hidden_layers'])
    
    dl_config = DeepLearningConfig(**dl_config_dict)

    # 3. Load State Space Bounds
    bounds_data = load_json_file(PATHS["bounds"])
    bounds = bounds_data.get('bounds', {})
    
    dl_config.capital_min = bounds.get('k_min', 0.1)
    dl_config.capital_max = bounds.get('k_max', 10.0)
    dl_config.productivity_min = bounds.get('z_min', 0.5)
    dl_config.productivity_max = bounds.get('z_max', 1.5)

    logger.info(f"Bounds Set: K=[{dl_config.capital_min:.2f}, {dl_config.capital_max:.2f}], "
                f"Z=[{dl_config.productivity_min:.2f}, {dl_config.productivity_max:.2f}]")

    # 4. Instantiate Model
    model = BasicModelDL(econ_params, dl_config)

    # 5. Build Graph (Dummy Pass)
    # This ensures variables are created before loading weights
    dummy_k = tf.constant([[1.0]], dtype=TENSORFLOW_DTYPE)
    dummy_z = tf.constant([[1.0]], dtype=TENSORFLOW_DTYPE)
    _ = model.compute_loss(dummy_k, dummy_z)

    return model


def get_checkpoints() -> List[Tuple[int, str, str]]:
    """
    Retrieve and sort model checkpoints by epoch.

    Returns:
        A list of tuples: (epoch, value_net_path, policy_net_path).
    """
    if not os.path.exists(PATHS["checkpoints"]):
        logger.error(f"Checkpoint directory {PATHS['checkpoints']} does not exist.")
        sys.exit(1)

    # Pattern matches: basic_value_net_{EPOCH}.weights.h5
    val_pattern = os.path.join(PATHS["checkpoints"], "basic_value_net_*.weights.h5")
    files = glob.glob(val_pattern)
    
    checkpoints = []
    for f_path in files:
        # Extract epoch number using regex
        match = re.search(r"basic_value_net_(\d+)\.weights\.h5", f_path)
        if match:
            epoch = int(match.group(1))
            # Construct corresponding policy path
            pol_path = f_path.replace("value_net", "policy_net")
            
            if os.path.exists(pol_path):
                checkpoints.append((epoch, f_path, pol_path))
    
    # Sort by epoch ascending
    checkpoints.sort(key=lambda x: x[0])
    checkpoints = [x for x in checkpoints if x[0]%20==0]  # Keep only even epochs
    
    if not checkpoints:
        logger.warning("No valid checkpoints found.")
    
    return checkpoints


@tf.function(reduce_retracing=True)
def compute_residuals_vectorized(
    model: BasicModelDL, 
    k_batch: tf.Tensor, 
    z_batch: tf.Tensor, 
    n_samples: int = 100
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Computes Bellman and Euler residuals using vectorized Monte Carlo integration.
    
    Optimization:
    - Eliminates Python loops using tensor tiling/repeating.
    - Performs a single forward pass of the Value Network for all MC samples.
    - Computes gradients for Euler equation in parallel.
    
    Args:
        model: The trained Deep Learning model.
        k_batch: Tensor of capital states, shape (Batch_Size, 1).
        z_batch: Tensor of productivity states, shape (Batch_Size, 1).
        n_samples: Number of Monte Carlo samples for expectation.

    Returns:
        Tuple of (bellman_residuals, euler_residuals) with shape (Batch_Size, 1).
    """
    # --- 1. Current Period Processing (Batch_Size, 1) ---
    # Normalize current state
    k_norm = model.normalizer.normalize_capital(k_batch)
    z_norm = model.normalizer.normalize_productivity(z_batch)
    inputs_curr = tf.concat([k_norm, z_norm], axis=1)

    # Get current Value and Policy
    v_curr = model.value_net(inputs_curr)
    investment_rate = model.policy_net(inputs_curr)
    investment = investment_rate * k_batch

    # Compute current economics
    profit = EconomicLogic.production_function(k_batch, z_batch, model.params)
    adj_cost, marginal_adj_cost = EconomicLogic.adjustment_costs(investment, k_batch, model.params)
    cash_flow = EconomicLogic.calculate_cash_flow(profit, investment, adj_cost)

    # Calculate deterministic next capital K'
    k_prime = investment + (1.0 - model.params.depreciation_rate) * k_batch

    # --- 2. Vectorized Monte Carlo Expansion ---
    # We expand the batch from (B, 1) to (B * N, 1) to process all samples in parallel.
    # tf.repeat interleave repeats: [k1, k1, k2, k2...]
    
    batch_size = tf.shape(k_batch)[0]
    total_samples = batch_size * n_samples

    # Expand K' (Next period capital is deterministic given current state)
    k_prime_expanded = tf.repeat(k_prime, repeats=n_samples, axis=0)  # Shape: (B*N, 1)
    
    # Expand Z (Current Z needed for transition logic)
    z_curr_expanded = tf.repeat(z_batch, repeats=n_samples, axis=0)   # Shape: (B*N, 1)

    # Sample Shocks using TFP (Vectorized sampling)
    # Shape: (B*N, 1)
    eps_vectorized = model.shock_dist.sample(sample_shape=(total_samples, 1))

    # Compute Z' (Next period productivity)
    z_prime_expanded = MathUtils.log_ar1_transition(
        z_curr_expanded, 
        model.params.productivity_persistence, 
        eps_vectorized
    )

    # --- 3. Normalization & Gradient Computation (Super-Batch) ---
    
    # Normalize expanded inputs
    # Note: Gradients are required for Euler equation (dV/dK')
    with tf.GradientTape() as tape:
        k_prime_norm_exp = model.normalizer.normalize_capital(k_prime_expanded)
        tape.watch(k_prime_norm_exp)
        
        z_prime_norm_exp = model.normalizer.normalize_productivity(z_prime_expanded)
        
        # Concatenate inputs for the network
        inputs_next_expanded = tf.concat([k_prime_norm_exp, z_prime_norm_exp], axis=1)
        
        # SINGLE Network Pass for all (B*N) points
        v_next_expanded = model.value_net(inputs_next_expanded)

    # Compute Gradients dV/dK_norm for the whole super-batch
    dv_dk_norm_expanded = tape.gradient(v_next_expanded, k_prime_norm_exp)
    
    # Convert to dV/dK (Chain rule with normalizer scaling)
    # dV/dK = (dV/dK_norm) * (1 / range)
    dv_dk_expanded = dv_dk_norm_expanded / model.normalizer.k_range

    # --- 4. Aggregation (Reshape & Average) ---
    
    # Reshape from (B*N, 1) -> (B, N) to average across samples
    v_next_reshaped = tf.reshape(v_next_expanded, (batch_size, n_samples))
    dv_dk_reshaped = tf.reshape(dv_dk_expanded, (batch_size, n_samples))

    # Compute Expectations E[.]
    v_next_expected = tf.reduce_mean(v_next_reshaped, axis=1, keepdims=True) # Shape: (B, 1)
    dv_dk_expected = tf.reduce_mean(dv_dk_reshaped, axis=1, keepdims=True)   # Shape: (B, 1)

    # --- 5. Residual Calculation ---

    # Bellman Residual: V - (Reward + beta * E[V'])
    bellman_target = cash_flow + model.params.discount_factor * v_next_expected
    bellman_resid = v_curr - bellman_target

    # Euler Residual (Unit-free)
    # 1 - (beta * E[dV/dK']) / (1 + MarginalCost)
    mc_investment = 1.0 + marginal_adj_cost
    euler_fraction = (model.params.discount_factor * dv_dk_expected) / (mc_investment + 1e-8)
    euler_resid = 1.0 - euler_fraction

    return bellman_resid, euler_resid


def plot_error_evolution(model: BasicModelDL, checkpoints: List[Tuple], vfi_data: Optional[Dict]) -> None:
    """
    Figure 1: Plot the evolution of RMSE, Bellman Residuals, and Euler Residuals.
    """
    logger.info("Generating Figure 1: Error Evolution...")
    
    epochs = []
    metrics = {"vfi_rmse": [], "bellman_rmse": [], "euler_rmse": []}
    
    # Sample states for residual calculation
    k_sample, z_sample = MathUtils.sample_states(
        500, model.config, include_debt=False, progress=tf.constant(1.0, TENSORFLOW_DTYPE)
    )

    # Prepare VFI data if available
    vfi_flat_k, vfi_flat_z, vfi_flat_v = None, None, None
    if vfi_data:
        K_mesh, Z_mesh = np.meshgrid(vfi_data['K'], vfi_data['Z'], indexing='ij')
        vfi_flat_k = tf.constant(K_mesh.flatten().reshape(-1, 1), dtype=TENSORFLOW_DTYPE)
        vfi_flat_z = tf.constant(Z_mesh.flatten().reshape(-1, 1), dtype=TENSORFLOW_DTYPE)
        vfi_flat_v = vfi_data['V'].flatten()
        # Filter NaNs
        mask = ~np.isnan(vfi_flat_v)
        vfi_flat_k = tf.boolean_mask(vfi_flat_k, mask)
        vfi_flat_z = tf.boolean_mask(vfi_flat_z, mask)
        vfi_flat_v = vfi_flat_v[mask]

    for epoch, val_path, pol_path in checkpoints:
        try:
            model.value_net.load_weights(val_path)
            model.policy_net.load_weights(pol_path)
        except Exception as e:
            logger.warning(f"Skipping corrupt checkpoint epoch {epoch}: {e}")
            continue

        epochs.append(epoch)

        # 1. VFI RMSE
        if vfi_flat_v is not None:
            k_n = model.normalizer.normalize_capital(vfi_flat_k)
            z_n = model.normalizer.normalize_productivity(vfi_flat_z)
            v_pred = model.value_net(tf.concat([k_n, z_n], axis=1)).numpy().flatten()
            metrics["vfi_rmse"].append(np.sqrt(np.mean((v_pred - vfi_flat_v)**2)))
        else:
            metrics["vfi_rmse"].append(np.nan)

        # 2. Residuals
        b_res, e_res = compute_residuals_vectorized(model, k_sample, z_sample, n_samples=10000)
        metrics["bellman_rmse"].append(np.sqrt(np.mean(b_res.numpy()**2)))
        metrics["euler_rmse"].append(np.sqrt(np.mean(e_res.numpy()**2)))
        
        if epoch % 100 == 0:
            logger.info(f"Processed Epoch {epoch}")

    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    plt.style.use('seaborn-v0_8-whitegrid')

    configs = [
        ("vfi_rmse", "Value RMSE vs VFI", "blue", "o"),
        ("bellman_rmse", "Bellman Residual RMSE", "red", "s"),
        ("euler_rmse", "Euler Residual RMSE", "green", "^")
    ]

    for ax, (key, title, color, marker) in zip(axes, configs):
        data = metrics[key]
        if np.all(np.isnan(data)):
            ax.text(0.5, 0.5, "Data Unavailable", ha='center', transform=ax.transAxes)
        else:
            ax.semilogy(epochs, data, color=color, marker=marker, markersize=3, linewidth=1.5, alpha=0.8)
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Error (Log Scale)", fontsize=12)
        ax.grid(True, which="both", ls="-", alpha=0.2)

    plt.suptitle("Figure 1: Training Convergence and Error Dynamics", fontsize=16, y=1.05)
    plt.tight_layout()
    save_path = os.path.join(PATHS["figures"], "figure1_error_evolution.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved {save_path}")


def plot_comparative_analysis(model: BasicModelDL, last_checkpoint: Tuple, vfi_data: Dict) -> None:
    """
    Figure 2: 2x2 Subplot of Value Functions and Error Metrics.
    
    Layout:
    [1,1] Deep Learning Value Surface
    [1,2] VFI Value Surface
    [2,1] Absolute Error Surface (Annotated with MAE)
    [2,2] Percentage Error Surface (Annotated with MAPE)
    """
    logger.info("Generating Figure 2: Comparative Analysis (2x2)...")

    if not vfi_data:
        logger.error("VFI data is required for Figure 2. Skipping.")
        return

    # Load Weights
    _, val_path, pol_path = last_checkpoint
    model.value_net.load_weights(val_path)
    
    # Prepare Grid
    K_mesh, Z_mesh = np.meshgrid(vfi_data['K'], vfi_data['Z'], indexing='ij')
    v_vfi = vfi_data['V']
    
    # Predict DL Values
    k_flat = K_mesh.flatten().reshape(-1, 1)
    z_flat = Z_mesh.flatten().reshape(-1, 1)
    
    k_tf = tf.constant(k_flat, dtype=TENSORFLOW_DTYPE)
    z_tf = tf.constant(z_flat, dtype=TENSORFLOW_DTYPE)
    
    k_norm = model.normalizer.normalize_capital(k_tf)
    z_norm = model.normalizer.normalize_productivity(z_tf)
    
    v_dl_flat = model.value_net(tf.concat([k_norm, z_norm], axis=1)).numpy().flatten()
    v_dl = v_dl_flat.reshape(K_mesh.shape)

    # Calculate Errors
    abs_error = np.abs(v_dl - v_vfi)
    # Avoid division by zero for percentage error
    safe_vfi = np.where(np.abs(v_vfi) < 1e-6, 1e-6, v_vfi) 
    pct_error = (abs_error / np.abs(safe_vfi)) * 100.0

    # Scalar Metrics
    mae = np.nanmean(abs_error)
    mape = np.nanmean(pct_error)

    # --- Visualization ---
    fig = plt.figure(figsize=(16, 12))
    plt.style.use('seaborn-v0_8-whitegrid')

    # Helper for 3D plotting
    def plot_surface(ax, X, Y, Z, title, cmap, z_label):
        surf = ax.plot_surface(X, Y, Z, cmap=cmap, edgecolor='none', alpha=0.9, antialiased=True)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
        ax.set_xlabel('Capital (K)', fontsize=10, labelpad=5)
        ax.set_ylabel('Productivity (Z)', fontsize=10, labelpad=5)
        ax.set_zlabel(z_label, fontsize=10, labelpad=5)
        ax.view_init(elev=30, azim=-60)
        # Add colorbar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)
        return surf

    # 1. Deep Learning Solution
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    plot_surface(ax1, K_mesh, Z_mesh, v_dl, "Deep Learning Solution", cm.viridis, "Value V(K,Z)")

    # 2. VFI Benchmark
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    plot_surface(ax2, K_mesh, Z_mesh, v_vfi, "VFI Benchmark", cm.viridis, "Value V(K,Z)")

    # 3. Absolute Error Distribution (MAE)
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    plot_surface(ax3, K_mesh, Z_mesh, abs_error, "Absolute Error Distribution", cm.inferno, "|Error|")
    
    # Annotation for MAE
    mae_text = (
        r"$\mathbf{MAE} = \frac{1}{N} \sum |V_{DL} - V_{VFI}|$" + "\n" +
        f"Value: {mae:.5f}"
    )
    ax3.text2D(0.05, 0.95, mae_text, transform=ax3.transAxes, fontsize=12, 
               bbox=dict(boxstyle="round", facecolor='white', alpha=0.9))

    # 4. Percentage Error Distribution (MAPE)
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    plot_surface(ax4, K_mesh, Z_mesh, pct_error, "Percentage Error Distribution", cm.cividis, "Error (%)")
    
    # Annotation for MAPE
    mape_text = (
        r"$\mathbf{MAPE} = \frac{1}{N} \sum \left| \frac{V_{DL} - V_{VFI}}{V_{VFI}} \right| \times 100$" + "\n" +
        f"Value: {mape:.4f}%"
    )
    ax4.text2D(0.05, 0.95, mape_text, transform=ax4.transAxes, fontsize=12,
               bbox=dict(boxstyle="round", facecolor='white', alpha=0.9))

    plt.suptitle("Figure 2: Model Accuracy Analysis (DL vs VFI)", fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    save_path = os.path.join(PATHS["figures"], "figure2_comparative_analysis.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved {save_path}")


def plot_simulation(model: BasicModelDL, last_checkpoint: Tuple, vfi_data: Optional[Dict], 
                    n_traj: int = 500, t_steps: int = 1000) -> None:
    """
    Figure 3: Stochastic Simulation of the economy from Steady State.
    """
    logger.info("Generating Figure 3: Stochastic Simulation...")

    # Load Weights
    _, val_path, pol_path = last_checkpoint
    model.value_net.load_weights(val_path)
    model.policy_net.load_weights(pol_path)

    # Initial State (Steady State approx)
    k_ss = model.config.capital_steady_state if model.config.capital_steady_state else 1.0
    z_ss = 1.0
    
    # Initialize Batches
    k_curr = tf.fill([n_traj, 1], tf.cast(k_ss, TENSORFLOW_DTYPE))
    z_curr = tf.fill([n_traj, 1], tf.cast(z_ss, TENSORFLOW_DTYPE))

    # Storage
    k_hist = np.zeros((t_steps, n_traj))
    rewards = np.zeros((t_steps, n_traj))

    # Simulation Loop
    for t in range(t_steps):
        # Normalize
        k_n = model.normalizer.normalize_capital(k_curr)
        z_n = model.normalizer.normalize_productivity(z_curr)
        
        # Policy
        inv_rate = model.policy_net(tf.concat([k_n, z_n], axis=1))
        inv = inv_rate * k_curr
        
        # Rewards
        profit = EconomicLogic.production_function(k_curr, z_curr, model.params)
        adj_cost, _ = EconomicLogic.adjustment_costs(inv, k_curr, model.params)
        cf = EconomicLogic.calculate_cash_flow(profit, inv, adj_cost)
        
        k_hist[t] = k_curr.numpy().flatten()
        rewards[t] = cf.numpy().flatten()
        
        # Transition
        k_next = inv + (1.0 - model.params.depreciation_rate) * k_curr
        eps = model.shock_dist.sample(sample_shape=(n_traj, 1))
        z_next = MathUtils.log_ar1_transition(z_curr, model.params.productivity_persistence, eps)
        
        # Clip
        k_curr = tf.clip_by_value(k_next, model.config.capital_min, model.config.capital_max)
        z_curr = tf.clip_by_value(z_next, model.config.productivity_min, model.config.productivity_max)

    # Discounted Sum
    beta = model.params.discount_factor
    discounts = np.array([beta**t for t in range(t_steps)])
    cum_rewards = np.sum(rewards * discounts[:, None], axis=0)
    
    # Infinite Horizon Correction
    term_val = rewards[-1] / (1.0 - beta) * (beta**t_steps)
    total_val = cum_rewards + term_val

    # --- Visualization ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    plt.style.use('seaborn-v0_8-whitegrid')

    # 1. Trajectories
    ax1 = axes[0]
    time = np.arange(t_steps)
    ax1.plot(time, k_hist, color='steelblue', alpha=0.1) # Individual lines
    ax1.plot(time, np.mean(k_hist, axis=1), color='darkred', linewidth=2.5, label='Mean Trajectory')
    ax1.axhline(k_ss, color='black', linestyle='--', label='Steady State')
    
    ax1.set_title(f"Capital Accumulation ({n_traj} Sims)", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Time Period", fontsize=12)
    ax1.set_ylabel("Capital Stock", fontsize=12)
    ax1.legend()

    # 2. Value Distribution
    ax2 = axes[1]
    ax2.hist(total_val, bins=30, density=True, alpha=0.7, color='forestgreen', edgecolor='black')
    
    mean_sim = np.mean(total_val)
    ax2.axvline(mean_sim, color='darkred', linewidth=2, label=f'Simulated Mean: {mean_sim:.2f}')
    
    # Compare with VFI if available
    if vfi_data:
        # Find VFI value closest to SS
        k_grid, z_grid, v_grid = vfi_data['K'], vfi_data['Z'], vfi_data['V']
        k_idx = np.abs(k_grid - k_ss).argmin()
        z_idx = np.abs(z_grid - z_ss).argmin()
        vfi_ss_val = v_grid[k_idx, z_idx]
        ax2.axvline(vfi_ss_val, color='blue', linestyle='--', linewidth=2, label=f'VFI Theoretical: {vfi_ss_val:.2f}')
    
    ax2.set_title("Distribution of Lifetime Rewards", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Discounted Value", fontsize=12)
    ax2.legend()

    plt.suptitle("Figure 3: Dynamic Simulation Analysis", fontsize=16, y=0.98)
    plt.tight_layout()
    
    save_path = os.path.join(PATHS["figures"], "figure3_simulation.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved {save_path}")


def main():
    """Main execution entry point."""
    logger.info("Starting Basic RBC Model Evaluation")
    
    # 1. Setup
    model = setup_model()
    checkpoints = get_checkpoints()
    
    if not checkpoints:
        logger.error("No checkpoints found. Train the model first.")
        return

    # 2. Load VFI Data (Optional but recommended)
    vfi_data = None
    if os.path.exists(PATHS["vfi_results"]):
        logger.info("Loading VFI benchmark results...")
        vfi_data = np.load(PATHS["vfi_results"])
    else:
        logger.warning("VFI results not found. Some comparisons will be skipped.")

    # 3. Generate Figures
    last_ckpt = checkpoints[-1]
    
    plot_error_evolution(model, checkpoints, vfi_data)
    plot_comparative_analysis(model, last_ckpt, vfi_data)
    plot_simulation(model, last_ckpt, vfi_data)

    logger.info("Evaluation Complete. Check output directory.")

if __name__ == "__main__":
    main()