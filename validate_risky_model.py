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
    
    Args:
        history: SimulationHistory object from Simulator.
        model: RiskyDebtModelVFI instance with grid definitions.
        
    Returns:
        Tuple of (k_values, b_values, z_values) arrays with shape (n_steps, n_batches).
    """
    k_grid = model.k_grid.numpy()
    b_grid = model.b_grid.numpy()
    z_grid = model.z_grid.numpy()
    
    # History trajectories have shape (n_batches, n_steps), transpose for consistency
    k_indices = history.trajectories["k_idx"].T  # Now (n_steps, n_batches)
    b_indices = history.trajectories["b_idx"].T
    z_indices = history.trajectories["z_idx"].T
    
    k_values = k_grid[k_indices]
    b_values = b_grid[b_indices]
    z_values = z_grid[z_indices]
    
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
    
    # Convert indices to actual values
    k_history, b_history, _ = convert_history_to_values(history, model)
    time_steps = history.n_steps

    # Extract statistics from Simulator
    k_min_rate = sim_stats["k_min"] * 100
    k_max_rate = sim_stats["k_max"] * 100
    b_min_rate = sim_stats["b_min"] * 100
    b_max_rate = sim_stats["b_max"] * 100
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), constrained_layout=True)
    
    # --- Capital Heatmap ---
    ax_k_heat = axes[0, 0]
    bins_k = 50
    bin_edges_k = np.linspace(k_grid_min, k_grid_max, bins_k + 1)
    hist_data_k = np.array([np.histogram(k_history[t, :], bins=bin_edges_k, density=True)[0] 
                            for t in range(time_steps)])
    
    im_k = ax_k_heat.imshow(
        hist_data_k.T, aspect='auto', origin='lower', cmap='viridis',
        extent=[0, time_steps, k_grid_min, k_grid_max]
    )
    plt.colorbar(im_k, ax=ax_k_heat).set_label('Density')
    
    for y_val, label, rate in [(k_grid_min, "Min", k_min_rate), (k_grid_max, "Max", k_max_rate)]:
        ax_k_heat.axhline(y_val, color='red', linestyle='--', linewidth=2, alpha=0.8)
        y_offset = 0.05 if y_val == k_grid_min else -0.12
        ax_k_heat.text(
            time_steps * 0.02, y_val + (k_grid_max - k_grid_min) * y_offset,
            f"{label}: {y_val:.2f}\nHit: {rate:.2f}%",
            color='white', fontweight='bold', fontsize=9,
            bbox=dict(facecolor='red', alpha=0.5, edgecolor='none')
        )
    
    ax_k_heat.set_title("Capital Distribution Evolution", fontsize=12)
    ax_k_heat.set_ylabel("Capital ($K$)")
    ax_k_heat.set_xlabel("Time ($t$)")
    
    # --- Debt Heatmap ---
    ax_b_heat = axes[0, 1]
    bins_b = 50
    bin_edges_b = np.linspace(b_grid_min, b_grid_max, bins_b + 1)
    hist_data_b = np.array([np.histogram(b_history[t, :], bins=bin_edges_b, density=True)[0] 
                            for t in range(time_steps)])
    
    im_b = ax_b_heat.imshow(
        hist_data_b.T, aspect='auto', origin='lower', cmap='plasma',
        extent=[0, time_steps, b_grid_min, b_grid_max]
    )
    plt.colorbar(im_b, ax=ax_b_heat).set_label('Density')
    
    for y_val, label, rate in [(b_grid_min, "Min", b_min_rate), (b_grid_max, "Max", b_max_rate)]:
        ax_b_heat.axhline(y_val, color='cyan', linestyle='--', linewidth=2, alpha=0.8)
        y_offset = 0.05 if y_val == b_grid_min else -0.12
        ax_b_heat.text(
            time_steps * 0.02, y_val + (b_grid_max - b_grid_min) * y_offset,
            f"{label}: {y_val:.2f}\nHit: {rate:.2f}%",
            color='white', fontweight='bold', fontsize=9,
            bbox=dict(facecolor='cyan', alpha=0.5, edgecolor='none')
        )
    
    ax_b_heat.set_title("Debt Distribution Evolution", fontsize=12)
    ax_b_heat.set_ylabel("Debt ($B$)")
    ax_b_heat.set_xlabel("Time ($t$)")
    
    # --- Capital KDE ---
    ax_k_dist = axes[1, 0]
    sns.kdeplot(k_history[0, :], ax=ax_k_dist, fill=True, color="gray", alpha=0.3, label="Initial")
    sns.kdeplot(k_history[time_steps // 2, :], ax=ax_k_dist, color="blue", linestyle="--", 
                label=f"t={time_steps // 2}")
    sns.kdeplot(k_history[-1, :], ax=ax_k_dist, fill=True, color="crimson", alpha=0.4, 
                label=f"Stationary (t={time_steps})")
    
    ax_k_dist.axvline(k_grid_min, color='red', linestyle='--', linewidth=2, alpha=0.8)
    ax_k_dist.axvline(k_grid_max, color='red', linestyle='--', linewidth=2, alpha=0.8)
    ax_k_dist.set_title("Capital: Convergence to Stationary", fontsize=12)
    ax_k_dist.set_xlabel("Capital ($K$)")
    ax_k_dist.set_ylabel("Density")
    ax_k_dist.legend(loc='upper right')
    
    # --- Debt KDE ---
    ax_b_dist = axes[1, 1]
    sns.kdeplot(b_history[0, :], ax=ax_b_dist, fill=True, color="gray", alpha=0.3, label="Initial")
    sns.kdeplot(b_history[time_steps // 2, :], ax=ax_b_dist, color="purple", linestyle="--", 
                label=f"t={time_steps // 2}")
    sns.kdeplot(b_history[-1, :], ax=ax_b_dist, fill=True, color="darkorange", alpha=0.4, 
                label=f"Stationary (t={time_steps})")
    
    ax_b_dist.axvline(b_grid_min, color='cyan', linestyle='--', linewidth=2, alpha=0.8)
    ax_b_dist.axvline(b_grid_max, color='cyan', linestyle='--', linewidth=2, alpha=0.8)
    ax_b_dist.set_title("Debt: Convergence to Stationary", fontsize=12)
    ax_b_dist.set_xlabel("Debt ($B$)")
    ax_b_dist.set_ylabel("Density")
    ax_b_dist.legend(loc='upper right')
    
    fig.suptitle(f"Batch Simulation & Grid Sufficiency (N={history.n_batches} agents)", 
                 fontsize=14)
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


def generate_figure_4_bond_price(
    model: RiskyDebtModelVFI,
    q_star: np.ndarray
):
    """
    Generates Figure 4: Bond Price Q(K, B', Z) with respect to Debt Level.
    
    Shows how the bond price varies with next-period debt (B') for different
    capital levels and productivity shocks.
    
    Top row: Q vs B' for different capital levels (Low, Medium, High K) at medium Z.
    Bottom row: Q vs B' for different productivity levels (Low, Medium, High Z) at medium K.
    
    Args:
        model: RiskyDebtModelVFI instance.
        q_star: Bond price schedule with shape (n_k, n_b, n_z).
    """
    k_grid = model.k_grid.numpy()
    b_grid = model.b_grid.numpy()
    z_grid = model.z_grid.numpy()
    
    n_k = len(k_grid)
    n_b = len(b_grid)
    n_z = len(z_grid)
    
    # Select low, medium, high indices
    k_indices = [0, n_k // 4, n_k // 2, 3 * n_k // 4, n_k - 1]
    z_indices = [0, n_z // 4, n_z // 2, 3 * n_z // 4, n_z - 1]
    
    k_mid_idx = n_k // 2
    z_mid_idx = n_z // 2
    
    # Color palettes
    k_colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(k_indices)))
    z_colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(z_indices)))
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # --- Top Row: Q vs B' for different capital levels at different Z levels ---
    z_plot_indices = [0, n_z // 2, n_z - 1]
    z_plot_labels = ["Low Productivity", "Medium Productivity", "High Productivity"]
    
    for col, (z_idx, z_label) in enumerate(zip(z_plot_indices, z_plot_labels)):
        ax = axes[0, col]
        
        for i, k_idx in enumerate(k_indices):
            q_slice = q_star[k_idx, :, z_idx]
            label = f"K = {k_grid[k_idx]:.2f}"
            ax.plot(b_grid, q_slice, linewidth=2, color=k_colors[i], label=label)
        
        ax.set_xlabel("Next-Period Debt ($B'$)", fontsize=11)
        ax.set_ylabel("Bond Price ($Q$)", fontsize=11)
        ax.set_title(f"Bond Price vs Debt | {z_label}\n(Z = {z_grid[z_idx]:.3f})", fontsize=12, pad=10)
        ax.legend(loc='upper right', fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.axhline(y=model.params.discount_factor, color='gray', linestyle='--', linewidth=1, alpha=0.7)
        ax.text(b_grid[0], model.params.discount_factor + 0.02, f'β = {model.params.discount_factor:.3f}', 
                fontsize=9, color='gray')
        sns.despine(ax=ax)
    
    # --- Bottom Row: Q vs B' for different productivity levels at different K levels ---
    k_plot_indices = [0, n_k // 2, n_k - 1]
    k_plot_labels = ["Low Capital", "Medium Capital", "High Capital"]
    
    for col, (k_idx, k_label) in enumerate(zip(k_plot_indices, k_plot_labels)):
        ax = axes[1, col]
        
        for i, z_idx in enumerate(z_indices):
            q_slice = q_star[k_idx, :, z_idx]
            label = f"Z = {z_grid[z_idx]:.3f}"
            ax.plot(b_grid, q_slice, linewidth=2, color=z_colors[i], label=label)
        
        ax.set_xlabel("Next-Period Debt ($B'$)", fontsize=11)
        ax.set_ylabel("Bond Price ($Q$)", fontsize=11)
        ax.set_title(f"Bond Price vs Debt | {k_label}\n(K = {k_grid[k_idx]:.2f})", fontsize=12, pad=10)
        ax.legend(loc='upper right', fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.axhline(y=model.params.discount_factor, color='gray', linestyle='--', linewidth=1, alpha=0.7)
        ax.text(b_grid[0], model.params.discount_factor + 0.02, f'β = {model.params.discount_factor:.3f}', 
                fontsize=9, color='gray')
        sns.despine(ax=ax)
    
    fig.suptitle("Bond Price Schedule Q(K, B', Z)\n"
                 "Top: Varying Capital Levels | Bottom: Varying Productivity Levels", 
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "figure_4_risky_bond_price.png"), bbox_inches='tight')
    print(f"-> Figure 4 saved to {os.path.join(SAVE_DIR, 'figure_4_risky_bond_price.png')}")
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
    q_star = optimal_res.get('Q', None)
    BATCH_SIZE = 10000
    T_SIM = 500
    SEED = 42
    
    print(f"  ... Simulating batch of {BATCH_SIZE} agents for {T_SIM} periods.")
    history, sim_stats = optimal_model.simulate(
        value_function=v_star,
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
    
    # Bond Price Analysis
    print("\n[Phase 5] Generating Bond Price Analysis (Figure 4)...")
    if q_star is not None:
        generate_figure_4_bond_price(optimal_model, q_star)
    else:
        print("  Warning: Bond price schedule (Q) not found in model results. Skipping Figure 4.")
    
    print("\nRisky Debt Model Analysis Complete.")


if __name__ == "__main__":
    main()