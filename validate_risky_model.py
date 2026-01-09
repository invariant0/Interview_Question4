import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from scipy.interpolate import RegularGridInterpolator
from typing import Tuple, List, Dict, Optional

# --- Import provided modules ---
from econ_models.config.economic_params import EconomicParams, load_economic_params
from econ_models.config.vfi_config import GridConfig, load_grid_config
from econ_models.vfi.risky_debt import RiskyDebtModelVFI
from econ_models.vfi.simulation.simulator import Simulator, SimulationHistory
from econ_models.econ import ProductionFunctions
from econ_models.core.types import TENSORFLOW_DTYPE
from econ_models.io.file_utils import load_json_file

# --- Configuration & Styling ---
PARAMS_PATH = "./hyperparam/prefixed/econ_params_risky.json"
CONFIG_PATH = "./hyperparam/prefixed/vfi_params.json"
BOUNDS_PATH = "./hyperparam/autogen/bounds_risky.json"
SAVE_DIR = "./results/validate_risky_model/"
os.makedirs(SAVE_DIR, exist_ok=True)
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300


# --- Helper Functions ---

def convert_history_to_values(
    history: SimulationHistory,
    model: RiskyDebtModelVFI
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert index-based simulation history to actual grid values.
    Maps -1 indices (defaults) to np.nan.
    """
    k_grid = model.k_grid.numpy()
    b_grid = model.b_grid.numpy()
    z_grid = model.z_grid.numpy()
    
    k_indices = history.trajectories["k_idx"].T  # (n_steps, n_batches)
    b_indices = history.trajectories["b_idx"].T
    z_indices = history.trajectories["z_idx"].T
    
    # Initialize with NaNs
    k_values = np.full(k_indices.shape, np.nan)
    b_values = np.full(b_indices.shape, np.nan)
    z_values = z_grid[z_indices] # Z usually doesn't default, but if needed handle similarly
    
    # Create masks for valid (non-default) indices
    valid_k = k_indices != -1
    valid_b = b_indices != -1
    
    # Map only valid indices
    k_values[valid_k] = k_grid[k_indices[valid_k]]
    b_values[valid_b] = b_grid[b_indices[valid_b]]
    return k_values, b_values, z_values


def compare_grids_3d(
    prev_k: np.ndarray,
    prev_b: np.ndarray,
    prev_v: np.ndarray,
    curr_k: np.ndarray,
    curr_b: np.ndarray,
    curr_v: np.ndarray,
    prev_default: Optional[np.ndarray] = None,
    curr_default: Optional[np.ndarray] = None
) -> Tuple[float, float, float]:
    """
    Calculates the Sup-Norm, Mean Absolute Error, and Max Absolute Error in non-default regions
    between two value functions on different 3D grids.
    
    Args:
        prev_k: Previous capital grid.
        prev_b: Previous debt grid.
        prev_v: Previous value function (n_k, n_b, n_z).
        curr_k: Current capital grid.
        curr_b: Current debt grid.
        curr_v: Current value function (n_k, n_b, n_z).
        prev_default: Previous default indicator (n_k, n_b, n_z), binary. Optional.
        curr_default: Current default indicator (n_k, n_b, n_z), binary. Optional.
        
    Returns:
        Tuple of (sup_norm, mean_absolute_error, max_abs_error_non_default) across all productivity states.
    """
    n_z = prev_v.shape[2]
    global_max_diff = 0.0
    global_max_diff_non_default = 0.0
    all_diffs = []
    
    for z in range(n_z):
        # Create interpolator for previous grid
        interp_func = RegularGridInterpolator(
            (prev_k, prev_b),
            prev_v[:, :, z],
            method='linear',
            bounds_error=False,
            fill_value=None
        )
        
        # Create meshgrid for current grid points
        K_curr, B_curr = np.meshgrid(curr_k, curr_b, indexing='ij')
        points = np.column_stack([K_curr.ravel(), B_curr.ravel()])
        
        # Interpolate and compare
        v_prev_interp = interp_func(points).reshape(len(curr_k), len(curr_b))
        abs_diff = np.abs(curr_v[:, :, z] - v_prev_interp)
        
        # Sup-norm (all regions)
        diff = np.nanmax(abs_diff)
        global_max_diff = max(global_max_diff, diff)
        
        # Collect all differences for MAE
        valid_diffs = abs_diff[~np.isnan(abs_diff)]
        all_diffs.extend(valid_diffs.flatten())
        
        # Max absolute error in non-default regions
        if prev_default is not None and curr_default is not None:
            # Interpolate previous default to current grid
            interp_default = RegularGridInterpolator(
                (prev_k, prev_b),
                prev_default[:, :, z].astype(float),
                method='nearest',
                bounds_error=False,
                fill_value=1.0  # Treat out-of-bounds as default
            )
            prev_default_interp = interp_default(points).reshape(len(curr_k), len(curr_b))
            prev_default_binary = (prev_default_interp > 0.5)
            curr_default_binary = curr_default[:, :, z] > 0.5
            
            # Non-default mask: both previous and current are non-default
            non_default_mask = ~prev_default_binary & ~curr_default_binary
            
            if np.any(non_default_mask):
                abs_diff_non_default = abs_diff.copy()
                abs_diff_non_default[~non_default_mask] = np.nan
                max_diff_non_default = np.nanmax(abs_diff_non_default)
                if not np.isnan(max_diff_non_default):
                    global_max_diff_non_default = max(global_max_diff_non_default, max_diff_non_default)
    
    mae = np.mean(all_diffs) if all_diffs else 0.0
    return global_max_diff, mae, global_max_diff_non_default


# --- Plotting Functions ---

def generate_figure_1_convergence(
    grid_sizes: List[Tuple[int, int]],
    sup_norm_errors: List[Optional[float]],
    mae_errors: List[Optional[float]],
    max_abs_non_default_errors: List[Optional[float]],
    chosen_grid_size: Tuple[int, int] = (80, 80)
):
    """
    Generates Figure 1: Convergence Analysis with Sup-Norm, MAE, and Max Abs Error in Non-Default Regions.
    
    Left panel: Max Absolute Difference (Sup-Norm) vs Grid Size
    Middle panel: Mean Absolute Error (MAE) vs Grid Size
    Right panel: Max Absolute Error in Non-Default Regions vs Grid Size
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    valid_indices = [i for i, e in enumerate(sup_norm_errors) if e is not None]
    x_labels = [f"{grid_sizes[i][0]}x{grid_sizes[i][1]}" for i in valid_indices]
    x_vals = list(range(len(x_labels)))
    sup_norm_vals = [sup_norm_errors[i] for i in valid_indices]
    mae_vals = [mae_errors[i] for i in valid_indices]
    max_abs_non_default_vals = [max_abs_non_default_errors[i] for i in valid_indices]
    
    chosen_label = f"{chosen_grid_size[0]}x{chosen_grid_size[1]}"
    
    # --- Left Panel: Sup-Norm ---
    ax1 = axes[0]
    ax1.plot(x_vals, sup_norm_vals, marker='o', linestyle='-', linewidth=2, 
             color='#2c3e50', label='Sup-Norm')
    
    if chosen_label in x_labels:
        idx = x_labels.index(chosen_label)
        ax1.scatter([idx], [sup_norm_vals[idx]], color='crimson', s=100, zorder=5)
        ax1.annotate(
            f'Selected\n({chosen_label})', 
            xy=(idx, sup_norm_vals[idx]), 
            xytext=(idx + 0.3, sup_norm_vals[idx] + (max(sup_norm_vals) - min(sup_norm_vals)) * 0.2),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
            horizontalalignment='left', fontsize=10, fontweight='bold'
        )
    
    ax1.set_xticks(x_vals)
    ax1.set_xticklabels(x_labels)
    ax1.set_xlabel("Grid Size ($N_{capital} \\times N_{debt}$)", fontsize=12)
    ax1.set_ylabel("Max Absolute Difference (Sup-Norm)", fontsize=12)
    ax1.set_title("Sup-Norm Convergence\n(All Regions)", fontsize=13, pad=10)
    ax1.legend(loc='upper right')
    sns.despine(ax=ax1)
    
    # --- Middle Panel: MAE ---
    ax2 = axes[1]
    ax2.plot(x_vals, mae_vals, marker='o', linestyle='-', linewidth=2, 
             color='#8e44ad', label='Mean Absolute Error')
    
    if chosen_label in x_labels:
        idx = x_labels.index(chosen_label)
        ax2.scatter([idx], [mae_vals[idx]], color='crimson', s=100, zorder=5)
        ax2.annotate(
            f'Selected\n({chosen_label})', 
            xy=(idx, mae_vals[idx]), 
            xytext=(idx + 0.3, mae_vals[idx] + (max(mae_vals) - min(mae_vals)) * 0.2),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
            horizontalalignment='left', fontsize=10, fontweight='bold'
        )
    
    ax2.set_xticks(x_vals)
    ax2.set_xticklabels(x_labels)
    ax2.set_xlabel("Grid Size ($N_{capital} \\times N_{debt}$)", fontsize=12)
    ax2.set_ylabel("Mean Absolute Error (MAE)", fontsize=12)
    ax2.set_title("Mean Absolute Error Convergence\n(All Regions)", fontsize=13, pad=10)
    ax2.legend(loc='upper right')
    sns.despine(ax=ax2)
    
    # --- Right Panel: Max Abs Error in Non-Default Regions ---
    ax3 = axes[2]
    ax3.plot(x_vals, max_abs_non_default_vals, marker='s', linestyle='-', linewidth=2, 
             color='#27ae60', label='Max Abs. Error (Non-Default)')
    
    if chosen_label in x_labels:
        idx = x_labels.index(chosen_label)
        ax3.scatter([idx], [max_abs_non_default_vals[idx]], color='crimson', s=100, zorder=5)
        y_range = max(max_abs_non_default_vals) - min(max_abs_non_default_vals)
        if y_range == 0:
            y_range = max(max_abs_non_default_vals) * 0.1 if max(max_abs_non_default_vals) > 0 else 0.1
        ax3.annotate(
            f'Selected\n({chosen_label})', 
            xy=(idx, max_abs_non_default_vals[idx]), 
            xytext=(idx + 0.3, max_abs_non_default_vals[idx] + y_range * 0.2),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
            horizontalalignment='left', fontsize=10, fontweight='bold'
        )
    
    ax3.set_xticks(x_vals)
    ax3.set_xticklabels(x_labels)
    ax3.set_xlabel("Grid Size ($N_{capital} \\times N_{debt}$)", fontsize=12)
    ax3.set_ylabel("Max Absolute Error", fontsize=12)
    ax3.set_title("Max Absolute Error Convergence\n(Non-Default Regions Only)", fontsize=13, pad=10)
    ax3.legend(loc='upper right')
    sns.despine(ax=ax3)
    
    fig.suptitle("Value Function Convergence (Risky Debt Model)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "figure_1_risky_convergence.png"), bbox_inches='tight')
    print(f"-> Figure 1 saved to {os.path.join(SAVE_DIR, 'figure_1_risky_convergence.png')}")
    plt.close()


def generate_figure_2_batch_distribution(
    history: SimulationHistory,
    model: RiskyDebtModelVFI,
    sim_stats: Dict[str, float],
    k_bounds: Tuple[float, float],
    b_bounds: Tuple[float, float]
):
    """Generates Figure 2: Evolution of Capital and Debt Distributions over time."""
    k_grid_min, k_grid_max = k_bounds
    b_grid_min, b_grid_max = b_bounds
    
    k_history, b_history, _ = convert_history_to_values(history, model)
    time_steps = history.n_steps

    k_min_rate = sim_stats["k_min"] * 100
    k_max_rate = sim_stats["k_max"] * 100
    b_min_rate = sim_stats["b_min"] * 100
    b_max_rate = sim_stats["b_max"] * 100
    default_rate = sim_stats.get("default_rate", 0.0) * 100
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), constrained_layout=True)
    
    # --- Capital Heatmap ---
    ax_k_heat = axes[0, 0]
    bins_k = 50
    bin_edges_k = np.linspace(k_grid_min, k_grid_max, bins_k + 1)
    
    # Filter NaNs for histogram
    hist_data_k = []
    for t in range(time_steps):
        step_data = k_history[t, :]
        valid_data = step_data[~np.isnan(step_data)]
        if len(valid_data) > 0:
            hist, _ = np.histogram(valid_data, bins=bin_edges_k, density=True)
        else:
            hist = np.zeros(bins_k)
        hist_data_k.append(hist)
    hist_data_k = np.array(hist_data_k)
    
    im_k = ax_k_heat.imshow(
        hist_data_k.T, aspect='auto', origin='lower', cmap='viridis',
        extent=[0, time_steps, k_grid_min, k_grid_max]
    )
    plt.colorbar(im_k, ax=ax_k_heat).set_label('Density (Active Firms)')
    
    # Add Default Rate Annotation
    ax_k_heat.text(
        time_steps * 0.02, k_grid_max * 0.9,
        f"Default Rate: {default_rate:.2f}%",
        color='white', fontweight='bold', fontsize=10,
        bbox=dict(facecolor='black', alpha=0.6, edgecolor='none')
    )
    
    ax_k_heat.set_title("Capital Distribution (Active Firms)", fontsize=12)
    ax_k_heat.set_ylabel("Capital ($K$)")
    ax_k_heat.set_xlabel("Time ($t$)")
    
    # --- Debt Heatmap ---
    ax_b_heat = axes[0, 1]
    bins_b = 50
    bin_edges_b = np.linspace(b_grid_min, b_grid_max, bins_b + 1)
    
    hist_data_b = []
    for t in range(time_steps):
        step_data = b_history[t, :]
        valid_data = step_data[~np.isnan(step_data)]
        if len(valid_data) > 0:
            hist, _ = np.histogram(valid_data, bins=bin_edges_b, density=True)
        else:
            hist = np.zeros(bins_b)
        hist_data_b.append(hist)
    hist_data_b = np.array(hist_data_b)
    
    im_b = ax_b_heat.imshow(
        hist_data_b.T, aspect='auto', origin='lower', cmap='plasma',
        extent=[0, time_steps, b_grid_min, b_grid_max]
    )
    plt.colorbar(im_b, ax=ax_b_heat).set_label('Density (Active Firms)')
    ax_b_heat.set_title("Debt Distribution (Active Firms)", fontsize=12)
    ax_b_heat.set_ylabel("Debt ($B$)")
    ax_b_heat.set_xlabel("Time ($t$)")
    
    # --- Capital KDE ---
    ax_k_dist = axes[1, 0]
    # Helper to safely plot KDE ignoring NaNs
    def safe_kde(ax, data, color, label, linestyle="-", fill=False):
        valid = data[~np.isnan(data)]
        if len(valid) > 10: # Need enough points for KDE
            sns.kdeplot(valid, ax=ax, fill=fill, color=color, alpha=0.3 if fill else 1.0, 
                        linestyle=linestyle, label=label)
            
    safe_kde(ax_k_dist, k_history[0, :], "gray", "Initial", fill=True)
    safe_kde(ax_k_dist, k_history[time_steps // 2, :], "blue", f"t={time_steps // 2}", linestyle="--")
    safe_kde(ax_k_dist, k_history[-1, :], "crimson", f"Stationary (t={time_steps})", fill=True)
    
    ax_k_dist.set_title("Capital Density (Active)", fontsize=12)
    ax_k_dist.legend(loc='upper right')
    
    # --- Debt KDE ---
    ax_b_dist = axes[1, 1]
    safe_kde(ax_b_dist, b_history[0, :], "gray", "Initial", fill=True)
    safe_kde(ax_b_dist, b_history[time_steps // 2, :], "purple", f"t={time_steps // 2}", linestyle="--")
    safe_kde(ax_b_dist, b_history[-1, :], "darkorange", f"Stationary (t={time_steps})", fill=True)
    
    ax_b_dist.set_title("Debt Density (Active)", fontsize=12)
    ax_b_dist.legend(loc='upper right')
    
    fig.suptitle(f"Batch Simulation (N={history.n_batches}) - Excluding Defaults", fontsize=14)
    plt.savefig(os.path.join(SAVE_DIR, "figure_2_risky_batch_distribution.png"))
    print(f"-> Figure 2 saved to {os.path.join(SAVE_DIR, 'figure_2_risky_batch_distribution.png')}")
    plt.close()


def generate_figure_3_value_surface(model: RiskyDebtModelVFI, v_star: np.ndarray):
    """
    Generates Figure 3: 3D Value Function Surfaces.
    
    Top row: V(K, Z) for Low, Medium, High debt levels.
    Bottom row: V(K, B) for Low, Medium, High productivity levels.
    """
    k_grid = model.k_grid.numpy()
    b_grid = model.b_grid.numpy()
    z_grid = model.z_grid.numpy()
    
    n_b = len(b_grid)
    n_z = len(z_grid)
    
    # Select low, medium, high indices
    b_indices = [0, n_b // 2, n_b - 1]
    z_indices = [0, n_z // 2, n_z - 1]
    
    b_labels = ["Low Debt", "Medium Debt", "High Debt"]
    z_labels = ["Low Productivity", "Medium Productivity", "High Productivity"]
    
    fig = plt.figure(figsize=(18, 12))
    
    # --- Top Row: V(K, Z) at different debt levels ---
    for col, (b_idx, b_label) in enumerate(zip(b_indices, b_labels)):
        ax = fig.add_subplot(2, 3, col + 1, projection='3d')
        
        Z_mesh, K_mesh = np.meshgrid(z_grid, k_grid)
        V_slice = v_star[:, b_idx, :]  # Shape: (n_k, n_z)
        
        surf = ax.plot_surface(K_mesh, Z_mesh, V_slice, cmap=cm.coolwarm, 
                               linewidth=0, antialiased=True, alpha=0.9)
        
        ax.set_title(f"V(K, Z) | {b_label}\n(B = {b_grid[b_idx]:.2f})", fontsize=11, pad=10)
        ax.set_xlabel("Capital ($K$)", fontsize=9)
        ax.set_ylabel("Productivity ($Z$)", fontsize=9)
        ax.set_zlabel("Value ($V$)", fontsize=9)
        ax.tick_params(labelsize=8)
    
    # --- Bottom Row: V(K, B) at different productivity levels ---
    for col, (z_idx, z_label) in enumerate(zip(z_indices, z_labels)):
        ax = fig.add_subplot(2, 3, col + 4, projection='3d')
        
        B_mesh, K_mesh = np.meshgrid(b_grid, k_grid)
        V_slice = v_star[:, :, z_idx]  # Shape: (n_k, n_b)
        
        surf = ax.plot_surface(K_mesh, B_mesh, V_slice, cmap=cm.viridis, 
                               linewidth=0, antialiased=True, alpha=0.9)
        
        ax.set_title(f"V(K, B) | {z_label}\n(Z = {z_grid[z_idx]:.2f})", fontsize=11, pad=10)
        ax.set_xlabel("Capital ($K$)", fontsize=9)
        ax.set_ylabel("Debt ($B$)", fontsize=9)
        ax.set_zlabel("Value ($V$)", fontsize=9)
        ax.tick_params(labelsize=8)
    
    fig.suptitle("Value Function Surfaces (Risky Debt Model)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "figure_3_risky_value_surface.png"), bbox_inches='tight')
    print(f"-> Figure 3 saved to {os.path.join(SAVE_DIR, 'figure_3_risky_value_surface.png')}")
    plt.close()


def generate_figure_4_investment_debt_policies(
    history: SimulationHistory,
    model: RiskyDebtModelVFI,
):
    # ... (Grid retrieval same as before) ...
    k_grid = model.k_grid.numpy()
    b_grid = model.b_grid.numpy()
    z_grid = model.z_grid.numpy()
    
    n_k, n_b, n_z = len(k_grid), len(b_grid), len(z_grid)
    
    # Get indices directly to filter -1
    k_idx_traj = history.trajectories["k_idx"]
    b_idx_traj = history.trajectories["b_idx"]
    z_idx_traj = history.trajectories["z_idx"]
    
    # Convert to values (with NaNs for defaults)
    k_values, b_values, z_values = convert_history_to_values(history, model)
    
    # Calculate Investment and Next Debt
    delta = model.params.depreciation_rate
    
    # We look at transitions from t to t+1
    # If an agent defaults at t (k_idx = -1) or t+1, we skip
    k_current = k_values.T[:, :-1]
    k_next = k_values.T[:, 1:]
    b_next = b_values.T[:, 1:]
    
    # Investment is only valid if both current and next K are valid numbers
    investment = k_next - (1 - delta) * k_current
    
    # Flatten everything
    k_idx_flat = k_idx_traj[:, :-1].flatten()
    b_idx_flat = b_idx_traj[:, :-1].flatten()
    z_idx_flat = z_idx_traj[:, :-1].flatten()
    
    investment_flat = investment.flatten()
    b_next_flat = b_next.flatten()
    
    # --- Compute Averages ---
    
    # Initialize accumulators
    inv_by_k = np.zeros(n_k)
    debt_by_k = np.zeros(n_k)
    count_by_k = np.zeros(n_k)
    
    # Loop through flattened data
    for i in range(len(k_idx_flat)):
        k_i = k_idx_flat[i]
        
        # SKIP if current state is default (-1) or if data is NaN
        if k_i == -1 or np.isnan(investment_flat[i]) or np.isnan(b_next_flat[i]):
            continue
            
        inv_by_k[k_i] += investment_flat[i]
        debt_by_k[k_i] += b_next_flat[i]
        count_by_k[k_i] += 1
        
    # Normalize
    mask_k = count_by_k > 0
    inv_by_k[mask_k] /= count_by_k[mask_k]
    debt_by_k[mask_k] /= count_by_k[mask_k]
    inv_by_k[~mask_k] = np.nan
    debt_by_k[~mask_k] = np.nan

    # Repeat logic for B
    inv_by_b = np.zeros(n_b)
    debt_by_b = np.zeros(n_b)
    count_by_b = np.zeros(n_b)
    
    for i in range(len(b_idx_flat)):
        b_i = b_idx_flat[i]
        if b_i == -1 or np.isnan(investment_flat[i]) or np.isnan(b_next_flat[i]):
            continue
        inv_by_b[b_i] += investment_flat[i]
        debt_by_b[b_i] += b_next_flat[i]
        count_by_b[b_i] += 1
        
    mask_b = count_by_b > 0
    inv_by_b[mask_b] /= count_by_b[mask_b]
    debt_by_b[mask_b] /= count_by_b[mask_b]
    inv_by_b[~mask_b] = np.nan
    debt_by_b[~mask_b] = np.nan

    # Repeat logic for Z
    inv_by_z = np.zeros(n_z)
    debt_by_z = np.zeros(n_z)
    count_by_z = np.zeros(n_z)
    
    for i in range(len(z_idx_flat)):
        z_i = z_idx_flat[i]
        # Z usually doesn't have -1, but check investment validity
        if np.isnan(investment_flat[i]) or np.isnan(b_next_flat[i]):
            continue
        inv_by_z[z_i] += investment_flat[i]
        debt_by_z[z_i] += b_next_flat[i]
        count_by_z[z_i] += 1

    mask_z = count_by_z > 0
    inv_by_z[mask_z] /= count_by_z[mask_z]
    debt_by_z[mask_z] /= count_by_z[mask_z]
    inv_by_z[~mask_z] = np.nan
    debt_by_z[~mask_z] = np.nan

    # --- Create Figure ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Color scheme
    inv_color = '#2980b9'  # Blue for investment
    debt_color = '#c0392b'  # Red for debt
    
    # --- Top Row: Investment ---
    
    # Investment vs K
    ax = axes[0, 0]
    valid_k = mask_k
    ax.plot(k_grid[valid_k], inv_by_k[valid_k], linewidth=2.5, color=inv_color, marker='o', markersize=4)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.fill_between(k_grid[valid_k], 0, inv_by_k[valid_k], alpha=0.2, color=inv_color)
    ax.set_xlabel("Capital ($K$)", fontsize=12)
    ax.set_ylabel("Average Investment ($I$)", fontsize=12)
    ax.set_title("Investment vs Capital", fontsize=13, pad=10)
    sns.despine(ax=ax)
    
    # Investment vs B
    ax = axes[0, 1]
    valid_b = mask_b & (b_grid >= 20) & (b_grid <= 30)
    ax.plot(b_grid[valid_b], inv_by_b[valid_b], linewidth=2.5, color=inv_color, marker='o', markersize=4)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.fill_between(b_grid[valid_b], 0, inv_by_b[valid_b], alpha=0.2, color=inv_color)
    ax.set_xlabel("Debt ($B$)", fontsize=12)
    ax.set_ylabel("Average Investment ($I$)", fontsize=12)
    ax.set_title("Investment vs Debt", fontsize=13, pad=10)
    sns.despine(ax=ax)
    
    # Investment vs Z
    ax = axes[0, 2]
    valid_z = mask_z
    ax.plot(z_grid[valid_z], inv_by_z[valid_z], linewidth=2.5, color=inv_color, marker='o', markersize=4)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.fill_between(z_grid[valid_z], 0, inv_by_z[valid_z], alpha=0.2, color=inv_color)
    ax.set_xlabel("Productivity ($Z$)", fontsize=12)
    ax.set_ylabel("Average Investment ($I$)", fontsize=12)
    ax.set_title("Investment vs Productivity", fontsize=13, pad=10)
    sns.despine(ax=ax)
    
    # --- Bottom Row: Next-Period Debt ---
    
    # Next Debt vs K
    ax = axes[1, 0]
    ax.plot(k_grid[valid_k], debt_by_k[valid_k], linewidth=2.5, color=debt_color, marker='s', markersize=4)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.fill_between(k_grid[valid_k], 0, debt_by_k[valid_k], alpha=0.2, color=debt_color)
    ax.set_xlabel("Capital ($K$)", fontsize=12)
    ax.set_ylabel("Average Next-Period Debt ($B'$)", fontsize=12)
    ax.set_title("Next-Period Debt vs Capital", fontsize=13, pad=10)
    sns.despine(ax=ax)
    
    # Next Debt vs B
    ax = axes[1, 1]
    valid_b_debt = mask_b & (b_grid >= 20) & (b_grid <= 30)
    ax.plot(b_grid[valid_b_debt], debt_by_b[valid_b_debt], linewidth=2.5, color=debt_color, marker='s', markersize=4)
    # Add 45-degree line for reference
    b_range = b_grid[valid_b_debt]
    ax.plot(b_range, b_range, linestyle='--', color='gray', linewidth=1.5, alpha=0.7, label='45° line')
    ax.fill_between(b_grid[valid_b_debt], 0, debt_by_b[valid_b_debt], alpha=0.2, color=debt_color)
    ax.set_xlabel("Debt ($B$)", fontsize=12)
    ax.set_ylabel("Average Next-Period Debt ($B'$)", fontsize=12)
    ax.set_title("Next-Period Debt vs Current Debt", fontsize=13, pad=10)
    ax.legend(loc='upper left', fontsize=10)
    sns.despine(ax=ax)
    
    # Next Debt vs Z
    ax = axes[1, 2]
    ax.plot(z_grid[valid_z], debt_by_z[valid_z], linewidth=2.5, color=debt_color, marker='s', markersize=4)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.fill_between(z_grid[valid_z], 0, debt_by_z[valid_z], alpha=0.2, color=debt_color)
    ax.set_xlabel("Productivity ($Z$)", fontsize=12)
    ax.set_ylabel("Average Next-Period Debt ($B'$)", fontsize=12)
    ax.set_title("Next-Period Debt vs Productivity", fontsize=13, pad=10)
    sns.despine(ax=ax)
    
    fig.suptitle("Policy Functions: Investment and Next-Period Debt\n"
                 "(Averaged from Simulation Data)", 
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "figure_4_investment_debt_policies.png"), bbox_inches='tight')
    print(f"-> Figure 4 saved to {os.path.join(SAVE_DIR, 'figure_4_investment_debt_policies.png')}")
    plt.close()

# --- Main Execution Logic ---

def main():
    print("Starting Risky Debt Model Analysis...")
    
    # Load parameters using existing infrastructure
    params = load_economic_params(PARAMS_PATH)
    base_config = load_grid_config(CONFIG_PATH, "risky")
    
    print("\n[Phase 1] Loading Pre-saved Grid Bounds...")
    bounds_data = load_json_file(BOUNDS_PATH)
    k_min = bounds_data['bounds']['k_min']
    k_max = bounds_data['bounds']['k_max']
    b_min = bounds_data['bounds']['b_min']
    b_max = bounds_data['bounds']['b_max']
    print(f"  Bounds loaded: K in [{k_min:.2f}, {k_max:.2f}], B in [{b_min:.2f}, {b_max:.2f}]")

    # Grid Convergence Test
    print("\n[Phase 2] Running Grid Convergence Test (Figure 1)...")
    
    test_grid_sizes = [(50, 50), (60, 60), (70, 70), (80, 80), (90, 90)]
    # test_grid_sizes = [(50, 50), (90, 90)]
    sup_norm_errors: List[Optional[float]] = []
    mae_errors: List[Optional[float]] = []
    max_abs_non_default_errors: List[Optional[float]] = []
    prev_v, prev_k, prev_b, prev_default = None, None, None, None
    optimal_model, optimal_res = None, None
    TARGET_GRID_SIZE = (90, 90)
    
    for n_k, n_b in test_grid_sizes:
        print(f"  Solving for Grid Size: {n_k} x {n_b}...")
        
        current_config = GridConfig(
            n_capital=n_k,
            n_productivity=base_config.n_productivity,
            n_debt=n_b,
            tauchen_width=base_config.tauchen_width,
            tol_vfi=base_config.tol_vfi,
            max_iter_vfi=base_config.max_iter_vfi,
            max_outer=base_config.max_outer,
            tol_outer=base_config.tol_outer,
            relax_q=base_config.relax_q,
            v_default_eps=base_config.v_default_eps,
            b_eps=base_config.b_eps,
            q_min=base_config.q_min,
            collateral_violation_penalty=base_config.collateral_violation_penalty
        )
        
        model = RiskyDebtModelVFI(
            params, 
            current_config, 
            k_bounds=(k_min, k_max),
            b_bounds=(b_min, b_max)
        )
        res = model.solve()
        
        if (n_k, n_b) == TARGET_GRID_SIZE:
            optimal_model, optimal_res = model, res
        
        curr_v = res['V']
        curr_k = res['K']
        curr_b = res['B']
        
        # Extract default indicator from the model result
        if 'default' in res:
            curr_default = res['default']
        else:
            # Create default indicator based on bond price (Q = 0 indicates default region)
            curr_q = res.get('Q', np.ones((len(curr_k), len(curr_b), base_config.n_productivity)))
            curr_default = (curr_q < 0.01).astype(int)
        
        if prev_v is None:
            sup_norm_errors.append(None)
            mae_errors.append(None)
            max_abs_non_default_errors.append(None)
        else:
            sup_norm, mae, max_abs_non_default = compare_grids_3d(
                prev_k, prev_b, prev_v, 
                curr_k, curr_b, curr_v,
                prev_default, curr_default
            )
            sup_norm_errors.append(sup_norm)
            mae_errors.append(mae)
            max_abs_non_default_errors.append(max_abs_non_default)
            print(f"    -> Sup-Norm vs previous: {sup_norm:.6e}")
            print(f"    -> MAE vs previous: {mae:.6e}")
            print(f"    -> Max Abs Error (Non-Default) vs previous: {max_abs_non_default:.6e}")
            
        prev_v, prev_k, prev_b, prev_default = curr_v, curr_k, curr_b, curr_default
        
    generate_figure_1_convergence(
        test_grid_sizes, 
        sup_norm_errors, 
        mae_errors, 
        max_abs_non_default_errors,
        chosen_grid_size=TARGET_GRID_SIZE
    )

    # Batch Simulation using model.simulate()
    print(f"\n[Phase 3] Running Batch Simulation using {TARGET_GRID_SIZE[0]}x{TARGET_GRID_SIZE[1]} grid (Figure 2)...")
    
    v_star = optimal_res['V']
    q_star = optimal_res['Q']
    BATCH_SIZE = 10000
    T_SIM = 50
    SEED = 42
    
    print(f"  ... Simulating batch of {BATCH_SIZE} agents for {T_SIM} periods.")
    history, sim_stats = optimal_model.simulate(
        value_function=v_star,
        q_sched=q_star,
        n_steps=T_SIM,
        n_batches=BATCH_SIZE,
        seed=SEED
    )
    
    print(f"  ... Simulation complete. Total observations: {sim_stats['total_observations']}")
    print(f"      K min boundary hit rate: {sim_stats['k_min']*100:.2f}%")
    print(f"      K max boundary hit rate: {sim_stats['k_max']*100:.2f}%")
    print(f"      B min boundary hit rate: {sim_stats['b_min']*100:.2f}%")
    print(f"      B max boundary hit rate: {sim_stats['b_max']*100:.2f}%")
    
    generate_figure_2_batch_distribution(
        history, 
        optimal_model, 
        sim_stats, 
        k_bounds=(k_min, k_max),
        b_bounds=(b_min, b_max)
    )
    
    # Value Function Analysis
    print("\n[Phase 4] Generating Value Function Surface Analysis (Figure 3)...")
    generate_figure_3_value_surface(optimal_model, v_star)
    
    print("\n[Phase 5] Generating Investment and Debt Policy Analysis (Figure 4)...")
    generate_figure_4_investment_debt_policies(history, optimal_model)
    
    # print("\n[Phase 6] Generating Investment Rate and Debt Ratio Policy Analysis (Figure 5)...")
    # generate_figure_5_investment_rate_debt_ratio_policies(history, optimal_model)
    # print("\nRisky Debt Model Analysis Complete.")


if __name__ == "__main__":
    main()