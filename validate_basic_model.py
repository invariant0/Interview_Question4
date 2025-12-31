import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from scipy.interpolate import interp1d
from typing import Tuple, List, Dict

# --- Import provided modules ---
from econ_models.config.economic_params import EconomicParams, load_economic_params
from econ_models.config.vfi_config import GridConfig, load_grid_config
from econ_models.vfi.basic import BasicModelVFI
from econ_models.vfi.simulation.simulator import Simulator, SimulationHistory
from econ_models.econ import ProductionFunctions
from econ_models.core.types import TENSORFLOW_DTYPE
from econ_models.io.file_utils import load_json_file

# --- Configuration & Styling ---
PARAMS_PATH = "./hyperparam/prefixed/econ_params_basic.json"
CONFIG_PATH = "./hyperparam/prefixed/vfi_params.json"
BOUNDS_PATH = "./hyperparam/autogen/bounds_basic.json"
SAVE_DIR = "./results/validate_basic_model/"
os.makedirs(SAVE_DIR, exist_ok=True)
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300


# --- Helper Functions ---

def convert_history_to_values(
    history: SimulationHistory,
    model: BasicModelVFI
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert index-based simulation history to actual grid values.
    
    Args:
        history: SimulationHistory object from Simulator.
        model: BasicModelVFI instance with grid definitions.
        
    Returns:
        Tuple of (k_values, z_values) arrays with shape (n_steps, n_batches).
    """
    k_grid = model.k_grid.numpy()
    z_grid = model.z_grid.numpy()
    
    # History trajectories have shape (n_batches, n_steps), transpose for consistency
    k_indices = history.trajectories["k_idx"].T  # Now (n_steps, n_batches)
    z_indices = history.trajectories["z_idx"].T
    
    k_values = k_grid[k_indices]
    z_values = z_grid[z_indices]
    
    return k_values, z_values


def compare_grids(
    prev_k: np.ndarray, 
    prev_v: np.ndarray, 
    curr_k: np.ndarray, 
    curr_v: np.ndarray
) -> float:
    """Calculates the Sup-Norm between two value functions on different grids."""
    n_z = prev_v.shape[1]
    global_max_diff = 0.0
    
    for z in range(n_z):
        interp_func = interp1d(prev_k, prev_v[:, z], kind='linear', fill_value="extrapolate")
        v_prev_interp = interp_func(curr_k)
        diff = np.max(np.abs(curr_v[:, z] - v_prev_interp))
        global_max_diff = max(global_max_diff, diff)
            
    return global_max_diff


# --- Plotting Functions ---

def generate_figure_1_convergence(
    grid_sizes: List[int], 
    errors: List[float],
    chosen_grid_size: int = 150
):
    """Generates Figure 1: Max Absolute Difference (Sup-Norm) vs Grid Size."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    valid_indices = [i for i, e in enumerate(errors) if e is not None]
    x_vals = [grid_sizes[i] for i in valid_indices]
    y_vals = [errors[i] for i in valid_indices]
    
    ax.plot(x_vals, y_vals, marker='o', linestyle='-', linewidth=2, color='#2c3e50', label='Max Abs. Difference (Sup-Norm)')
    
    if chosen_grid_size in x_vals:
        idx = x_vals.index(chosen_grid_size)
        val = y_vals[idx]
        ax.scatter([chosen_grid_size], [val], color='crimson', s=100, zorder=5, label='Selected Grid Size')
        ax.annotate(
            f'Current Setting\n(N={chosen_grid_size})', 
            xy=(chosen_grid_size, val), 
            xytext=(chosen_grid_size, val + (max(y_vals) - min(y_vals))*0.2),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
            horizontalalignment='center', fontsize=11, fontweight='bold'
        )

    ax.set_title("Value Function Convergence Sensitivity", fontsize=14, pad=15)
    ax.set_xlabel("Grid Size ($N_{capital}$)", fontsize=12)
    ax.set_ylabel("Max Absolute Difference (Sup-Norm)", fontsize=12)
    ax.legend(loc='upper right')
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "figure_1_convergence.png"))
    print(f"-> Figure 1 saved to {os.path.join(SAVE_DIR, 'figure_1_convergence.png')}")
    plt.close()


def generate_figure_2_batch_distribution(
    history: SimulationHistory,
    model: BasicModelVFI,
    sim_stats: Dict[str, float],
    grid_bounds: Tuple[float, float]
):
    """Generates Figure 2: Evolution of the Capital Distribution over time."""
    k_grid_min, k_grid_max = grid_bounds
    
    # Convert indices to actual capital values
    k_history, _ = convert_history_to_values(history, model)
    time_steps = history.n_steps
    
    # Use statistics from Simulator
    rate_min = sim_stats["min_hit_pct"] * 100
    rate_max = sim_stats["max_hit_pct"] * 100
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 12), constrained_layout=True)
    
    ax_heat = axes[0]
    bins = 50
    bin_edges = np.linspace(k_grid_min, k_grid_max, bins + 1)
    hist_data = np.array([np.histogram(k_history[t, :], bins=bin_edges, density=True)[0] 
                          for t in range(time_steps)])

    im = ax_heat.imshow(
        hist_data.T, aspect='auto', origin='lower', cmap='viridis',
        extent=[0, time_steps, k_grid_min, k_grid_max]
    )
    
    plt.colorbar(im, ax=ax_heat).set_label('Density')
    
    for y_val, label, rate in [(k_grid_min, "Min", rate_min), (k_grid_max, "Max", rate_max)]:
        ax_heat.axhline(y_val, color='red', linestyle='--', linewidth=2, alpha=0.8)
        y_offset = 0.02 if y_val == k_grid_min else -0.08
        ax_heat.text(
            time_steps * 0.02, y_val + (k_grid_max - k_grid_min) * y_offset,
            f"{label}: {y_val:.2f}\nHit Rate: {rate:.2f}%",
            color='white', fontweight='bold', fontsize=10,
            bbox=dict(facecolor='red', alpha=0.5, edgecolor='none')
        )
    
    ax_heat.set_title("Evolution of Capital Distribution (Batch Simulation)", fontsize=14)
    ax_heat.set_ylabel("Capital ($K$)")
    ax_heat.set_xlabel("Time ($t$)")
    
    ax_dist = axes[1]
    sns.kdeplot(k_history[0, :], ax=ax_dist, fill=True, color="gray", alpha=0.3, label="Initial Distribution")
    sns.kdeplot(k_history[time_steps // 2, :], ax=ax_dist, color="blue", linestyle="--", label=f"t={time_steps // 2}")
    sns.kdeplot(k_history[-1, :], ax=ax_dist, fill=True, color="crimson", alpha=0.4, label=f"Stationary (t={time_steps})")
    
    ax_dist.axvline(k_grid_min, color='red', linestyle='--', linewidth=2, alpha=0.8)
    ax_dist.axvline(k_grid_max, color='red', linestyle='--', linewidth=2, alpha=0.8)
    ax_dist.set_title("Convergence to Stationary Distribution", fontsize=14)
    ax_dist.set_xlabel("Capital ($K$)")
    ax_dist.set_ylabel("Density")
    ax_dist.legend(loc='upper right')
            
    fig.suptitle(f"Batch Simulation & Grid Sufficiency (N={history.n_batches})", fontsize=16)
    plt.savefig(os.path.join(SAVE_DIR, "figure_2_batch_distribution.png"))
    print(f"-> Figure 2 saved to {os.path.join(SAVE_DIR, 'figure_2_batch_distribution.png')}")
    plt.close()


def generate_figure_3_value_surface(model: BasicModelVFI, v_star: np.ndarray):
    """Generates Figure 3: 3D Surface and slices of Value Function."""
    k_grid = model.k_grid.numpy()
    z_grid = model.z_grid.numpy()
    
    fig = plt.figure(figsize=(18, 6))
    
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    Z_mesh, K_mesh = np.meshgrid(z_grid, k_grid)
    surf = ax1.plot_surface(K_mesh, Z_mesh, v_star, cmap=cm.coolwarm, linewidth=0, antialiased=False, alpha=0.9)
    ax1.set_title("3D Value Function Surface", fontsize=14, pad=10)
    ax1.set_xlabel("Capital ($K$)")
    ax1.set_ylabel("Productivity ($Z$)")
    ax1.set_zlabel("Value ($V$)")
    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=10, pad=0.1)

    for idx, (ax_idx, shock_idx, color, label) in enumerate([
        (2, 0, 'darkblue', 'Low'),
        (3, len(z_grid) - 1, 'darkred', 'High')
    ]):
        ax = fig.add_subplot(1, 3, ax_idx)
        v_slice = v_star[:, shock_idx]
        ax.plot(k_grid, v_slice, color=color, linewidth=2.5)
        ax.fill_between(k_grid, v_slice, min(v_slice), color=color, alpha=0.1)
        ax.set_title(f"Value Function Slice: {label} Shock (Z={z_grid[shock_idx]:.2f})", fontsize=14)
        ax.set_xlabel("Capital ($K$)")
        ax.set_ylabel("Value ($V$)")
    
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "figure_3_value_surface.png"))
    print(f"-> Figure 3 saved to {os.path.join(SAVE_DIR, 'figure_3_value_surface.png')}")
    plt.close()


def generate_figure_4_investment_rates(
    model: BasicModelVFI,
    history: SimulationHistory
):
    """Generates Figure 4: Investment Analysis using ProductionFunctions module."""
    k_grid = model.k_grid.numpy()
    z_grid = model.z_grid.numpy()
    
    # Convert history to values
    k_history, _ = convert_history_to_values(history, model)
    z_idx_history = history.trajectories["z_idx"].T  # (n_steps, n_batches)
    
    k_current = k_history[:-1, :]
    k_next = k_history[1:, :]
    z_current_indices = z_idx_history[:-1, :]
    
    # Use existing ProductionFunctions.calculate_investment
    k_curr_tf = tf.constant(k_current, dtype=TENSORFLOW_DTYPE)
    k_next_tf = tf.constant(k_next, dtype=TENSORFLOW_DTYPE)
    investment = ProductionFunctions.calculate_investment(k_curr_tf, k_next_tf, model.params).numpy()
    
    with np.errstate(divide='ignore', invalid='ignore'):
        inv_rate = investment / k_current
        inv_rate[k_current == 0] = 0
        
    k_flat, z_idx_flat = k_current.flatten(), z_current_indices.flatten()
    rate_flat, inv_flat = inv_rate.flatten(), investment.flatten()
    
    def compute_avg_by_grid(grid_vals, flat_vals, target_flat, is_index=False):
        result = []
        for i, val in enumerate(grid_vals):
            mask = (flat_vals == i) if is_index else np.isclose(flat_vals, val)
            result.append(np.mean(target_flat[mask]) if np.any(mask) else np.nan)
        return result
    
    avg_rate_by_k = compute_avg_by_grid(k_grid, k_flat, rate_flat)
    avg_inv_by_k = compute_avg_by_grid(k_grid, k_flat, inv_flat)
    avg_rate_by_z = compute_avg_by_grid(z_grid, z_idx_flat, rate_flat, is_index=True)
    avg_inv_by_z = compute_avg_by_grid(z_grid, z_idx_flat, inv_flat, is_index=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    plot_configs = [
        (axes[0, 0], k_grid, avg_rate_by_k, 'forestgreen', 'o', "Avg. Investment Rate vs. Current Capital", "Current Capital ($K_t$)", "Investment Rate ($I_t / K_t$)"),
        (axes[0, 1], z_grid, avg_rate_by_z, 'purple', 's', "Avg. Investment Rate vs. Current Productivity", "Productivity Shock ($Z_t$)", "Investment Rate ($I_t / K_t$)"),
        (axes[1, 0], k_grid, avg_inv_by_k, 'darkorange', 'o', "Avg. Actual Investment vs. Current Capital", "Current Capital ($K_t$)", "Actual Investment ($I_t$)"),
        (axes[1, 1], z_grid, avg_inv_by_z, 'firebrick', 's', "Avg. Actual Investment vs. Current Productivity", "Productivity Shock ($Z_t$)", "Actual Investment ($I_t$)"),
    ]
    
    for ax, x, y, color, marker, title, xlabel, ylabel in plot_configs:
        ax.plot(x, y, color=color, linewidth=2.5, marker=marker, markersize=4 if marker == 'o' else 6)
        ax.set_title(title, fontsize=14)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "figure_4_investment_rates.png"))
    print(f"-> Figure 4 saved to {os.path.join(SAVE_DIR, 'figure_4_investment_rates.png')}")
    plt.close()


# --- Main Execution Logic ---

def main():
    print("Starting Economic Simulation Analysis...")
    
    # Use existing load_economic_params function
    params = load_economic_params(PARAMS_PATH)
    base_config = load_grid_config(CONFIG_PATH, "basic")
    
    print("\n[Phase 1] Loading Pre-saved Grid Bounds...")
    bounds_data = load_json_file(BOUNDS_PATH)
    k_min, k_max = bounds_data['bounds']['k_min'], bounds_data['bounds']['k_max']
    print(f"  Bounds loaded: K in [{k_min:.2f}, {k_max:.2f}]")

    # Grid Convergence Test
    print("\n[Phase 2] Running Grid Convergence Test (Figure 1)...")
    
    test_grid_sizes = [25, 50, 75, 100, 150, 200, 250, 300]
    convergence_errors = []
    prev_v, prev_k = None, None
    optimal_model, optimal_res = None, None
    TARGET_GRID_SIZE = 300
    
    for n_cap in test_grid_sizes:
        print(f"  Solving for Grid Size: {n_cap}...")
        
        current_config = GridConfig(
            n_capital=n_cap,
            n_productivity=base_config.n_productivity,
            n_debt=base_config.n_debt,
            tauchen_width=base_config.tauchen_width,
            tol_vfi=base_config.tol_vfi,
            max_iter_vfi=base_config.max_iter_vfi
        )
        
        model = BasicModelVFI(params, current_config, k_bounds=(k_min, k_max))
        res = model.solve()
        
        if n_cap == TARGET_GRID_SIZE:
            optimal_model, optimal_res = model, res
        
        curr_v, curr_k = res['V'], res['K']
        
        if prev_v is None:
            convergence_errors.append(None)
        else:
            err = compare_grids(prev_k, prev_v, curr_k, curr_v)
            convergence_errors.append(err)
            print(f"    -> Max Abs Diff (Sup-Norm) vs previous: {err:.6e}")
            
        prev_v, prev_k = curr_v, curr_k
        
    generate_figure_1_convergence(test_grid_sizes, convergence_errors, chosen_grid_size=TARGET_GRID_SIZE)

    # Batch Simulation using existing model.simulate() method
    print(f"\n[Phase 3] Running Batch Simulation using N={TARGET_GRID_SIZE} (Figure 2)...")
    
    v_star = optimal_res['V']
    BATCH_SIZE, T_SIM = 10000, 150
    SEED = 42  # For reproducibility
    
    # Use the model's built-in simulate method which leverages Simulator
    print(f"  ... Simulating batch of {BATCH_SIZE} agents for {T_SIM} periods.")
    history, sim_stats = optimal_model.simulate(
        value_function=v_star,
        n_steps=T_SIM,
        n_batches=BATCH_SIZE,
        seed=SEED
    )
    
    print(f"  ... Simulation complete. Total observations: {sim_stats['total_observations']}")
    print(f"      Min boundary hit rate: {sim_stats['min_hit_pct']*100:.2f}%")
    print(f"      Max boundary hit rate: {sim_stats['max_hit_pct']*100:.2f}%")
    
    generate_figure_2_batch_distribution(history, optimal_model, sim_stats, grid_bounds=(k_min, k_max))
    
    # Value Function Analysis
    print("\n[Phase 4] Generating Value Function Surface Analysis (Figure 3)...")
    generate_figure_3_value_surface(optimal_model, v_star)
    
    # Investment Rate Analysis
    print("\n[Phase 5] Generating Investment Rate Analysis (Figure 4)...")
    generate_figure_4_investment_rates(optimal_model, history)
    
    print("\nAnalysis Complete.")


if __name__ == "__main__":
    main()