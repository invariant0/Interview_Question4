import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d
from typing import Tuple, List, Dict, Optional

# --- Import provided modules ---
from econ_models.config.economic_params import EconomicParams
from econ_models.config.vfi_config import GridConfig, load_grid_config
from econ_models.vfi.basic import BasicModelVFI
from econ_models.vfi.bounds import BoundaryFinder
from econ_models.core.types import TENSORFLOW_DTYPE

# --- Configuration & Styling ---
PARAMS_PATH = "./hyperparam/prefixed/econ_params_basic.json"
CONFIG_PATH = "./hyperparam/prefixed/vfi_params.json"

# Set plot style for publish quality
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# --- Helper Functions ---

def load_params_from_path(path: str) -> EconomicParams:
    """
    Loads EconomicParams from a specific JSON path with error handling.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Parameter file not found: {path}")
    with open(path, 'r') as f:
        data = json.load(f)
    return EconomicParams(**data)

def simulate_trajectory(
    model: BasicModelVFI, 
    v_star: np.ndarray, 
    start_k_idx: int, 
    start_z_idx: int, 
    t_steps: int = 200
) -> np.ndarray:
    """
    Simulates a single capital trajectory starting from a specific state.
    """
    v_tensor = tf.constant(v_star, dtype=TENSORFLOW_DTYPE)
    
    # Get policy function (indices of next capital)
    policy_map = model.get_policy_indices(v_tensor).numpy()
    
    # Transition matrix for productivity
    p_matrix_np = model.P.numpy()
    k_grid = model.k_grid.numpy()
    
    current_k_idx = start_k_idx
    current_z_idx = start_z_idx
    
    k_history = np.zeros(t_steps)
    
    for t in range(t_steps):
        # Record current capital
        k_history[t] = k_grid[current_k_idx]
        
        # Determine next state
        next_k_idx = policy_map[current_k_idx, current_z_idx]
        
        # Stochastic transition for Z
        next_z_idx = np.random.choice(
            model.config.n_productivity, 
            p=p_matrix_np[current_z_idx]
        )
        
        # Update state
        current_k_idx = next_k_idx
        current_z_idx = next_z_idx
        
    return k_history

def compare_grids(
    prev_k: np.ndarray, 
    prev_v: np.ndarray, 
    curr_k: np.ndarray, 
    curr_v: np.ndarray
) -> float:
    """
    Calculates the Mean Absolute Difference (MAD) between two value functions
    on different grids by interpolating the previous result onto the current grid.
    """
    n_z = prev_v.shape[1]
    max_diff = 0.0
    
    for z in range(n_z):
        # Interpolate previous V(:, z) onto current K grid
        interp_func = interp1d(
            prev_k, prev_v[:, z], 
            kind='linear', 
            fill_value="extrapolate"
        )
        v_prev_interp = interp_func(curr_k)
        
        # Calculate Mean Absolute Difference for this Z state
        diff = np.mean(np.abs(curr_v[:, z] - v_prev_interp))
        if diff > max_diff:
            max_diff = diff
            
    return max_diff

# --- Plotting Functions ---

def generate_figure_1_convergence(
    grid_sizes: List[int], 
    errors: List[float],
    chosen_grid_size: int = 150
):
    """
    Generates Figure 1: Mean absolute difference vs Grid Size.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Filter out None values (first iteration)
    valid_indices = [i for i, e in enumerate(errors) if e is not None]
    x_vals = [grid_sizes[i] for i in valid_indices]
    y_vals = [errors[i] for i in valid_indices]
    
    # Plot line
    ax.plot(x_vals, y_vals, marker='o', linestyle='-', linewidth=2, color='#2c3e50', label='Mean Abs. Difference')
    
    # Annotate the chosen grid size (150)
    if chosen_grid_size in x_vals:
        idx = x_vals.index(chosen_grid_size)
        val = y_vals[idx]
        
        # Draw a distinct point
        ax.scatter([chosen_grid_size], [val], color='crimson', s=100, zorder=5, label='Selected Grid Size')
        
        # Add annotation arrow
        ax.annotate(
            f'Current Setting\n(N={chosen_grid_size})', 
            xy=(chosen_grid_size, val), 
            xytext=(chosen_grid_size, val + (max(y_vals) - min(y_vals))*0.2), # Shift text up
            arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
            horizontalalignment='center',
            fontsize=11,
            fontweight='bold'
        )

    ax.set_title("Figure 1: Value Function Convergence Sensitivity", fontsize=14, pad=15)
    ax.set_xlabel("Grid Size ($N_{capital}$)", fontsize=12)
    ax.set_ylabel("Mean Absolute Difference (Linear Scale)", fontsize=12)
    
    # Ensure linear scale is used (default)
    ax.set_yscale('linear')
    
    ax.legend(loc='upper right')
    sns.despine()
    plt.tight_layout()
    plt.savefig("figure_1_convergence.png")
    print("-> Figure 1 saved to figure_1_convergence.png")
    plt.close()

def generate_figure_2_scenarios(
    scenarios_data: Dict[str, np.ndarray],
    time_steps: int
):
    """
    Generates Figure 2: 5 rows of scenarios (Time Series + Distribution).
    """
    scenario_names = list(scenarios_data.keys())
    n_scenarios = len(scenario_names)
    
    fig, axes = plt.subplots(n_scenarios, 2, figsize=(12, 3 * n_scenarios), constrained_layout=True)
    
    colors = sns.color_palette("husl", n_scenarios)
    
    for i, name in enumerate(scenario_names):
        data = scenarios_data[name]
        color = colors[i]
        
        # Left Column: Time Series
        ax_ts = axes[i, 0]
        ax_ts.plot(range(time_steps), data, color=color, linewidth=1.5)
        ax_ts.set_title(f"{name}: Capital Sequence", fontsize=11, fontweight='bold')
        ax_ts.set_ylabel("Capital ($K$)")
        ax_ts.set_xlim(0, time_steps)
        if i == n_scenarios - 1:
            ax_ts.set_xlabel("Time ($t$)")
        
        # Right Column: Distribution (Histogram/KDE)
        ax_dist = axes[i, 1]
        sns.histplot(data, ax=ax_dist, color=color, kde=True, stat="density", edgecolor=None, alpha=0.6)
        ax_dist.set_title(f"{name}: Capital Distribution", fontsize=11, fontweight='bold')
        ax_dist.set_ylabel("Density")
        if i == n_scenarios - 1:
            ax_dist.set_xlabel("Capital ($K$)")
            
    fig.suptitle("Figure 2: Simulation Scenarios & Capital Distribution (N=150)", fontsize=16)
    
    plt.savefig("figure_2_scenarios.png")
    print("-> Figure 2 saved to figure_2_scenarios.png")
    plt.close()

# --- Main Execution Logic ---

def main():
    print("Starting Economic Simulation Analysis...")
    
    # 1. Load Configuration
    params = load_params_from_path(PARAMS_PATH)
    base_config = load_grid_config(CONFIG_PATH, "basic")
    
    # 2. Find Optimal Bounds (Pre-requisite)
    print("\n[Phase 1] Determining Optimal Grid Bounds...")
    finder = BoundaryFinder(params, base_config)
    result_dic = finder.find_basic_bounds()
    k_min, k_max = result_dic['k_bounds_original']
    print(f"  Bounds determined: K in [{k_min:.2f}, {k_max:.2f}]")

    # ==========================================
    # PART 1: Grid Convergence (Figure 1)
    # ==========================================
    print("\n[Phase 2] Running Grid Convergence Test (Figure 1)...")
    
    # Modified Grid Sizes
    test_grid_sizes = [25, 50, 75, 100, 150, 200, 250, 300]
    convergence_errors = []
    
    prev_v = None
    prev_k = None
    
    # We need to store the model corresponding to N=150 for Part 2
    optimal_model = None
    optimal_res = None
    TARGET_GRID_SIZE = 200
    
    for n_cap in test_grid_sizes:
        print(f"  Solving for Grid Size: {n_cap}...")
        
        # Update config with new grid size
        current_config = GridConfig(
            n_capital=n_cap,
            n_productivity=base_config.n_productivity,
            n_debt=base_config.n_debt,
            tauchen_width=base_config.tauchen_width,
            tol_vfi=base_config.tol_vfi,
            max_iter_vfi=base_config.max_iter_vfi
        )
        
        # Solve
        model = BasicModelVFI(params, current_config, k_bounds=(k_min, k_max))
        res = model.solve()
        
        # Save the optimal model (N=150) for later use
        if n_cap == TARGET_GRID_SIZE:
            optimal_model = model
            optimal_res = res
        
        curr_v = res['V']
        curr_k = res['K']
        
        if prev_v is None:
            convergence_errors.append(None) # First run has no baseline
        else:
            # Calculate MAD vs previous run
            err = compare_grids(prev_k, prev_v, curr_k, curr_v)
            convergence_errors.append(err)
            print(f"    -> MAD vs previous: {err:.6e}")
            
        prev_v = curr_v
        prev_k = curr_k
        
    generate_figure_1_convergence(test_grid_sizes, convergence_errors, chosen_grid_size=TARGET_GRID_SIZE)

    # ==========================================
    # PART 2: Scenario Simulation (Figure 2)
    # ==========================================
    print(f"\n[Phase 3] Running Scenarios using N={TARGET_GRID_SIZE} (Figure 2)...")
    
    if optimal_model is None:
        raise ValueError(f"Target grid size {TARGET_GRID_SIZE} was not found in the test list.")
        
    v_star = optimal_res['V']
    
    # Define Indices based on the N=150 model
    idx_k_low = 5  # Slightly above absolute min
    idx_k_high = optimal_model.config.n_capital - 5
    idx_k_mid = optimal_model.config.n_capital // 2
    
    # Assuming productivity is sorted (Standard Tauchen implementation)
    idx_z_low = 0
    idx_z_high = optimal_model.config.n_productivity - 1
    
    # Find "Steady State" proxy
    print("  Calculating steady state proxy...")
    burn_in_path = simulate_trajectory(optimal_model, v_star, idx_k_mid, idx_z_low, t_steps=2000)
    ss_k_val = burn_in_path[-1]
    # Find index closest to this value
    idx_k_ss = (np.abs(optimal_model.k_grid.numpy() - ss_k_val)).argmin()
    
    # Simulation Parameters
    T_SIM = 500
    
    scenarios = {
        "1. Steady State Start": (idx_k_ss, idx_z_low),
        "2. High Shock, Low Capital": (idx_k_low, idx_z_high),
        "3. High Shock, High Capital": (idx_k_high, idx_z_high),
        "4. Low Shock, High Capital": (idx_k_high, idx_z_low),
        "5. Low Shock, Low Capital": (idx_k_low, idx_z_low),
    }
    
    results_data = {}
    
    for name, (k_start, z_start) in scenarios.items():
        print(f"  Simulating: {name}")
        path = simulate_trajectory(optimal_model, v_star, k_start, z_start, t_steps=T_SIM)
        results_data[name] = path
        
    generate_figure_2_scenarios(results_data, T_SIM)
    
    print("\nAnalysis Complete.")

if __name__ == "__main__":
    main()