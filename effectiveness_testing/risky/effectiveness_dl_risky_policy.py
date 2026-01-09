# effectiveness_dl_risky_policy.py

import argparse
import os
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from typing import Dict, Tuple, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from econ_models.dl.risky import RiskyModelDL, RiskyDLSimulationHistory
from econ_models.dl.risky_upgrade import RiskyModelDL_UPGRADE, RiskyDLSimulationHistory
from econ_models.config.economic_params import EconomicParams
from econ_models.config.dl_config import DeepLearningConfig, load_dl_config
from econ_models.core.types import TENSORFLOW_DTYPE
from econ_models.econ import (
    ProductionFunctions,
)
tfd = tfp.distributions

# Set plotting style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 10, 'figure.dpi': 150})

def load_json_file(filename: str) -> Dict:
    """Load a JSON file."""
    with open(filename, 'r') as f:
        return json.load(f)

def load_economic_params(filename: str) -> EconomicParams:
    """Load economic parameters from JSON."""
    data = load_json_file(filename)
    return EconomicParams(**data)

def find_latest_checkpoint(checkpoint_dir: str) -> Tuple[int, str]:
    """Find the latest checkpoint file."""
    import glob
    pattern = os.path.join(checkpoint_dir, "risky_value_net_*.weights.h5")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    
    def extract_epoch(f):
        try:
            return int(os.path.basename(f).replace("risky_value_net_", "").replace(".weights.h5", ""))
        except ValueError:
            return -1
            
    latest_file = max(files, key=extract_epoch)
    return extract_epoch(latest_file), latest_file

# -------------------------------------------------------------------------
# NEW PLOTTING FUNCTION 1: Averaged Policy Functions
# -------------------------------------------------------------------------
def plot_averaged_policy_functions(
    history: RiskyDLSimulationHistory,
    econ_params: EconomicParams,
    output_dir: str
) -> None:
    """
    Generates a 2x3 grid of policy functions averaged from simulation data.
    Row 1: Investment (I) vs K, B, Z
    Row 2: Next-Period Debt (B') vs K, B, Z
    
    Filters out defaulted observations.
    """
    print("  Generating Policy Function Plots...")
    
    # 1. Prepare Data
    # We need pairs of (State_t, Action_t). Action implies I_t and B_{t+1}.
    
    # Get active mask for time t (must be active at t to make a decision)
    # default_history is 1.0 if defaulted, 0.0 if active
    d_curr = history.default_history[:-1].flatten()
    active_mask = (d_curr == 0.0)

    k_curr = history.capital_history[:-1].flatten()[active_mask]
    b_curr = history.debt_history[:-1].flatten()[active_mask]
    z_curr = history.productivity_history[:-1].flatten()[active_mask]
    
    k_next = history.capital_history[1:].flatten()[active_mask]
    b_next = history.debt_history[1:].flatten()[active_mask] # This is B_{t+1} chosen at t
    
    # Calculate Investment
    investment = ProductionFunctions.calculate_investment(
            k_curr, k_next, econ_params
        )
    
    # Create DataFrame for easy binning
    df = pd.DataFrame({
        'K': k_curr,
        'B': b_curr,
        'Z': z_curr,
        'I': investment,
        'B_next': b_next
    })
    
    # 2. Setup Plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Policy Functions: Investment and Next-Period Debt\n(Averaged from Active Firms)', fontsize=16, y=0.98)
    
    # Helper to bin and plot
    def plot_binned(ax, x_col, y_col, x_label, y_label, n_bins=30, color='tab:blue', is_debt=False):
        if df.empty:
            return

        # Create bins
        df['bin'] = pd.cut(df[x_col], bins=n_bins)
        # Group by bin and calculate mean of X and Y
        stats = df.groupby('bin', observed=True)[[x_col, y_col]].mean()
        
        # Plot line and markers
        ax.plot(stats[x_col], stats[y_col], marker='o', markersize=4, linewidth=2, color=color)
        
        # Fill area
        ax.fill_between(stats[x_col], stats[y_col], 0, alpha=0.2, color=color)
        
        # Reference line for zero
        ax.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        
        # 45 degree line for Debt vs Debt
        if is_debt and x_col == 'B':
            min_val = min(stats[x_col].min(), stats[y_col].min())
            max_val = max(stats[x_col].max(), stats[y_col].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3, label='45° line')
            ax.legend()

        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_title(f'{y_label.split(" ")[0]} vs {x_label.split(" ")[0]}', fontsize=13)
        ax.grid(True, alpha=0.4)

    # --- Row 1: Investment ---
    plot_binned(axes[0, 0], 'K', 'I', 'Capital (K)', 'Average Investment (I)')
    plot_binned(axes[0, 1], 'B', 'I', 'Debt (B)', 'Average Investment (I)')
    plot_binned(axes[0, 2], 'Z', 'I', 'Productivity (Z)', 'Average Investment (I)')
    
    # --- Row 2: Next Period Debt ---
    color_b = 'firebrick'
    plot_binned(axes[1, 0], 'K', 'B_next', 'Capital (K)', "Average Next-Period Debt (B')", color=color_b)
    plot_binned(axes[1, 1], 'B', 'B_next', 'Debt (B)', "Average Next-Period Debt (B')", color=color_b, is_debt=True)
    plot_binned(axes[1, 2], 'Z', 'B_next', 'Productivity (Z)', "Average Next-Period Debt (B')", color=color_b)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, "policy_functions_averaged.png"), dpi=150)
    plt.close()

# -------------------------------------------------------------------------
# NEW PLOTTING FUNCTION 2: Batch Evolution (Heatmaps + Densities)
# -------------------------------------------------------------------------
def plot_batch_evolution(
    history: RiskyDLSimulationHistory,
    output_dir: str
) -> None:
    """
    Generates a 2x2 figure showing the evolution of distributions over time.
    Top Row: 2D Histogram (Time vs Value) for K and B.
    Bottom Row: KDE plots for t=0, t=25, t=50.
    """
    print("  Generating Batch Evolution Plots...")
    
    n_steps, n_batches = history.capital_history.shape
    
    # 1. Identify "Active" firms (Not defaulted)
    # Using the explicit default history from the model
    # default_history: 1.0 = Defaulted, 0.0 = Active
    active_mask = (history.default_history == 0.0)
    
    # Calculate Default Rate (percentage of firms inactive at the final step)
    final_defaults = np.sum(history.default_history[-1, :] == 1.0)
    default_rate = final_defaults / n_batches
    
    # 2. Setup Figure
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.8], hspace=0.2)
    
    ax_heat_k = fig.add_subplot(gs[0, 0])
    ax_heat_b = fig.add_subplot(gs[0, 1])
    ax_dens_k = fig.add_subplot(gs[1, 0])
    ax_dens_b = fig.add_subplot(gs[1, 1])
    
    fig.suptitle(f'Batch Simulation (N={n_batches}) - Excluding Defaults', fontsize=20, y=0.95)

    # --- Helper: Time-Series Heatmap ---
    def plot_time_heatmap(ax, data_history, mask, y_label, title, cmap='viridis'):
        # Flatten data for histogram2d: X=Time, Y=Value
        times = []
        values = []
        
        for t in range(n_steps):
            # Only select firms active at time t
            valid_vals = data_history[t, mask[t, :]]
            if len(valid_vals) > 0:
                times.append(np.full_like(valid_vals, t))
                values.append(valid_vals)
        
        if not times:
            return ax

        times_flat = np.concatenate(times)
        values_flat = np.concatenate(values)
        
        # Create 2D histogram
        h, xedges, yedges, image = ax.hist2d(
            times_flat, values_flat, 
            bins=[n_steps, 50], 
            cmap=cmap, 
            density=True
        )
        ax.set_title(title, fontsize=16)
        ax.set_xlabel('Time (t)', fontsize=14)
        ax.set_ylabel(y_label, fontsize=14)
        ax.grid(True, alpha=0.3, color='white')
        
        # Colorbar
        cbar = plt.colorbar(image, ax=ax)
        cbar.set_label('Density (Active Firms)', fontsize=14)
        
        return ax

    # Plot Heatmaps
    plot_time_heatmap(
        ax_heat_k, history.capital_history, active_mask, 
        'Capital (K)', 'Capital Distribution (Active Firms)', cmap='viridis'
    )
    # Add Default Rate Annotation to K plot
    ax_heat_k.text(
        0.02, 0.95, f'Cumulative Default Rate: {default_rate*100:.2f}%', 
        transform=ax_heat_k.transAxes, 
        color='white', fontweight='bold', fontsize=14,
        bbox=dict(facecolor='black', alpha=0.6)
    )

    plot_time_heatmap(
        ax_heat_b, history.debt_history, active_mask, 
        'Debt (B)', 'Debt Distribution (Active Firms)', cmap='plasma'
    )

    # --- Helper: Density Snapshots ---
    def plot_density_snapshots(ax, data_history, mask, x_label, title, color_fill):
        times_to_plot = [0, n_steps//2, n_steps-1]
        labels = ['Initial', f't={n_steps//2}', f'Stationary (t={n_steps})']
        colors = ['grey', 'navy', color_fill]
        styles = ['-', '--', '-']
        fills = [True, False, True]
        alphas = [0.3, 1.0, 0.4]

        for t, label, col, style, fill, alpha in zip(times_to_plot, labels, colors, styles, fills, alphas):
            valid_data = data_history[t, mask[t, :]]
            if len(valid_data) > 1:
                try:
                    sns.kdeplot(
                        valid_data, ax=ax, label=label, color=col, 
                        linestyle=style, fill=fill, alpha=alpha, linewidth=1.5
                    )
                except Exception:
                    pass # Handle cases with singular matrix if data is constant
        
        ax.set_title(title, fontsize=16)
        ax.set_xlabel(x_label, fontsize=14)
        ax.set_ylabel('Density', fontsize=14)
        ax.legend(fontsize=14)
        ax.grid(True, alpha=0.5)

    # Plot Densities
    plot_density_snapshots(
        ax_dens_k, history.capital_history, active_mask, 
        'Capital (K)', 'Capital Density (Active)', 'crimson'
    )
    plot_density_snapshots(
        ax_dens_b, history.debt_history, active_mask, 
        'Debt (B)', 'Debt Density (Active)', 'orange'
    )

    plt.savefig(os.path.join(output_dir, "batch_evolution_distribution.png"), dpi=150, bbox_inches='tight')
    plt.close()

# -------------------------------------------------------------------------
# NEW PLOTTING FUNCTION 3: Default Analysis
# -------------------------------------------------------------------------
def plot_default_analysis(
    history: RiskyDLSimulationHistory,
    output_dir: str
) -> None:
    """
    Generates a plot specifically analyzing defaults.
    1. Cumulative Default Rate over time.
    2. Histogram of Capital at the moment of default.
    3. Histogram of Productivity at the moment of default.
    """
    print("  Generating Default Analysis Plots...")
    
    n_steps, n_batches = history.default_history.shape
    
    # 1. Cumulative Default Rate
    # Sum defaults across batches for each time step, divide by N
    cum_defaults = np.mean(history.default_history, axis=1)
    
    # 2. Extract states at moment of default
    # We find indices where default switches from 0 to 1
    # Diff along time axis
    d_diff = np.diff(history.default_history, axis=0, prepend=0)
    # Where diff == 1, a default occurred at that step
    default_indices = np.where(d_diff == 1)
    
    # Extract K, B, Z at those indices
    # Note: default_indices is a tuple (time_indices, batch_indices)
    k_at_default = history.capital_history[default_indices]
    b_at_default = history.debt_history[default_indices]
    z_at_default = history.productivity_history[default_indices]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Default Analysis', fontsize=16)
    
    # Plot 1: Cumulative Rate
    axes[0].plot(range(n_steps), cum_defaults * 100, color='darkred', linewidth=2)
    axes[0].set_title('Cumulative Default Rate')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Default Rate (%)')
    axes[0].grid(True)
    
    # Plot 2: Capital at Default
    if len(k_at_default) > 0:
        sns.histplot(k_at_default, ax=axes[1], color='gray', kde=True)
        axes[1].set_title('Capital Distribution at Default')
        axes[1].set_xlabel('Capital (K)')
    else:
        axes[1].text(0.5, 0.5, "No Defaults Observed", ha='center')

    # Plot 3: Productivity at Default
    if len(z_at_default) > 0:
        sns.histplot(z_at_default, ax=axes[2], color='purple', kde=True)
        axes[2].set_title('Productivity Distribution at Default')
        axes[2].set_xlabel('Productivity (Z)')
    else:
        axes[2].text(0.5, 0.5, "No Defaults Observed", ha='center')
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "default_analysis.png"), dpi=150)
    plt.close()

# -------------------------------------------------------------------------
# EXISTING FUNCTIONS (Updated for Default Awareness)
# -------------------------------------------------------------------------

def plot_simulation_time_series(history: RiskyDLSimulationHistory, output_dir: str, n_samples: int = 3, duration: int = 100) -> None:
    actual_steps = history.capital_history.shape[0]
    duration = min(duration, actual_steps)

    # Pick first n samples
    k_hist = history.capital_history[:duration, :n_samples]
    b_hist = history.debt_history[:duration, :n_samples]
    q_hist = history.bond_price_history[:duration, :n_samples]
    z_hist = history.productivity_history[:duration, :n_samples]
    d_hist = history.default_history[:duration, :n_samples]
    
    time = np.arange(duration)
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    
    # Helper to plot with default annotation
    def plot_with_defaults(ax, data, title, ylabel):
        for i in range(n_samples):
            # If defaulted, plot differently or cut off?
            # Let's plot the full line, but mark default point
            ax.plot(time, data[:, i], label=f'Sim {i+1}')
            
            # Check if this specific trajectory defaulted
            default_idxs = np.where(d_hist[:, i] == 1.0)[0]
            if len(default_idxs) > 0:
                first_default = default_idxs[0]
                ax.axvline(first_default, color='red', linestyle=':', alpha=0.5)
                if i == 0: # Label only once
                    ax.text(first_default, ax.get_ylim()[1]*0.9, 'Default', color='red', rotation=90, verticalalignment='top')

        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if ylabel == 'Capital (K)': ax.legend()

    plot_with_defaults(axes[0], k_hist, 'Simulation Time Series: Capital', 'Capital (K)')
    plot_with_defaults(axes[1], b_hist, 'Simulation Time Series: Debt', 'Debt (B)')
    plot_with_defaults(axes[2], q_hist, 'Simulation Time Series: Bond Price', 'Bond Price (q)')
    plot_with_defaults(axes[3], z_hist, 'Simulation Time Series: Productivity', 'Productivity (Z)')
    
    axes[3].set_xlabel('Time Period')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "simulation_time_series.png"), dpi=150)
    plt.close()

def compute_moments(history: RiskyDLSimulationHistory, burn_in: int = 200) -> Dict[str, float]:
    if burn_in >= history.capital_history.shape[0]: burn_in = 0
    
    # Filter for active firms only
    active_mask = (history.default_history[burn_in:, :] == 0.0)
    
    k = history.capital_history[burn_in:, :][active_mask]
    b = history.debt_history[burn_in:, :][active_mask]
    q = history.bond_price_history[burn_in:, :][active_mask]
    
    # Avoid division by zero
    mask_nonzero = k > 1e-4
    k_safe = k[mask_nonzero]
    b_safe = b[mask_nonzero]
    
    leverage = b_safe / k_safe
    
    return {
        "capital_mean": float(np.mean(k)), "capital_std": float(np.std(k)),
        "debt_mean": float(np.mean(b)), "leverage_mean": float(np.mean(leverage)),
        "bond_price_mean": float(np.mean(q)), 
        "correlation_k_b": float(np.corrcoef(k_safe, b_safe)[0, 1]) if len(k_safe) > 1 else 0.0
    }

def find_latest_checkpoint_risky_upgrade(
    checkpoint_dir: str = "checkpoints/risky_upgrade"
) -> Tuple[int, Dict[str, str]]:
    """
    Find the latest checkpoint for risky_upgrade model.
    
    Args:
        checkpoint_dir: Directory containing checkpoint files.
        
    Returns:
        Tuple of (epoch number, dict of checkpoint paths for each network).
        
    Raises:
        FileNotFoundError: If no checkpoints are found.
    """
    path = Path(checkpoint_dir)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    
    # Find all value net checkpoints to determine available epochs
    value_net_files = list(path.glob("risky_value_net_*.weights.h5"))
    if not value_net_files:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    
    # Extract epochs from filenames
    epochs = []
    for f in value_net_files:
        # Parse epoch from filename like "risky_value_net_100.weights.h5"
        name = f.stem  # "risky_value_net_100.weights"
        parts = name.replace(".weights", "").split("_")
        try:
            epoch = int(parts[-1])
            epochs.append(epoch)
        except ValueError:
            continue
    
    if not epochs:
        raise FileNotFoundError(f"No valid checkpoint epochs found in {checkpoint_dir}")
    
    latest_epoch = max(epochs)
    
    # Build paths for all networks
    checkpoint_paths = {
        "value_net": str(path / f"risky_value_net_{latest_epoch}.weights.h5"),
        "capital_policy_net": str(path / f"risky_capital_policy_net_{latest_epoch}.weights.h5"),
        "debt_policy_net": str(path / f"risky_debt_policy_net_{latest_epoch}.weights.h5"),
    }
    
    # Check if continuous net checkpoint exists
    continuous_path = path / f"risky_continuous_net_{latest_epoch}.weights.h5"
    if continuous_path.exists():
        checkpoint_paths["continuous_net"] = str(continuous_path)
    
    # Verify all required files exist
    for net_name, net_path in checkpoint_paths.items():
        if not Path(net_path).exists():
            raise FileNotFoundError(
                f"Missing checkpoint for {net_name} at epoch {latest_epoch}: {net_path}"
            )
    
    return latest_epoch, checkpoint_paths


def load_risky_upgrade_checkpoints(
    model: "RiskyModelDL_UPGRADE",
    checkpoint_paths: Dict[str, str]
) -> None:
    """
    Load checkpoint weights into risky_upgrade model.
    
    Args:
        model: The RiskyModelDL_UPGRADE model instance.
        checkpoint_paths: Dictionary mapping network names to checkpoint file paths.
    """
    # Load value network and sync target
    model.value_net.load_weights(checkpoint_paths["value_net"])
    model.target_value_net.set_weights(model.value_net.get_weights())
    print(f"  Loaded value_net from: {checkpoint_paths['value_net']}")
    
    # Load policy networks
    model.capital_policy_net.load_weights(checkpoint_paths["capital_policy_net"])
    print(f"  Loaded capital_policy_net from: {checkpoint_paths['capital_policy_net']}")
    
    model.debt_policy_net.load_weights(checkpoint_paths["debt_policy_net"])
    print(f"  Loaded debt_policy_net from: {checkpoint_paths['debt_policy_net']}")
    
    # Load continuous net if available
    if "continuous_net" in checkpoint_paths:
        model.continuous_net.load_weights(checkpoint_paths["continuous_net"])
        print(f"  Loaded continuous_net from: {checkpoint_paths['continuous_net']}")

# -------------------------------------------------------------------------
# MAIN EXECUTION
# -------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train Economic Deep Learning Models"
    )
    parser.add_argument(
        '--model',
        type=str,
        default='risky',
        choices=['risky', 'risky_upgrade'],
        help="Model architecture to train."
    )
    args = parser.parse_args()
    print("=" * 60)
    print("Risky Model: Stationary Distribution Analysis")
    print("=" * 60)

    # Configuration paths
    econ_params_path = "./hyperparam/prefixed/econ_params_risky.json"
    dl_params_path = "./hyperparam/prefixed/dl_params.json"
    bounds_path = "./hyperparam/autogen/bounds_risky.json"
    
    # Set checkpoint and output directories based on model type
    if args.model == 'risky':
        checkpoint_dir = "./checkpoints/risky"
        output_dir = "./results/result_effectiveness_dl_risky"
        dl_config = load_dl_config(dl_params_path, "risky")
    elif args.model == 'risky_upgrade':
        checkpoint_dir = "./checkpoints/risky_upgrade"
        output_dir = "./results/result_effectiveness_dl_risky_upgrade"
        dl_config = load_dl_config(dl_params_path, "risky_upgrade")

    os.makedirs(output_dir, exist_ok=True)

    # 1. Load Parameters
    print("\n[1] Loading Configurations...")
    if not os.path.exists(econ_params_path):
        print(f"ERROR: Economic params file not found at {econ_params_path}")
        return

    econ_params = load_economic_params(econ_params_path)

    if os.path.exists(bounds_path):
        bounds_data = load_json_file(bounds_path)
        bounds = bounds_data['bounds']
        dl_config.capital_min = bounds['k_min']
        dl_config.capital_max = bounds['k_max']
        dl_config.debt_min = bounds['b_min']
        dl_config.debt_max = bounds['b_max']
        dl_config.productivity_min = bounds['z_min']
        dl_config.productivity_max = bounds['z_max']

    # 2. Initialize Model
    print("\n[2] Initializing Deep Learning Model...")
    if args.model == 'risky':
        model = RiskyModelDL(econ_params, dl_config)
    elif args.model == 'risky_upgrade':
        model = RiskyModelDL_UPGRADE(econ_params, dl_config)

    # 3. Load Checkpoint
    print("\n[3] Loading Trained Weights...")
    try:
        if args.model == 'risky':
            epoch, checkpoint_path = find_latest_checkpoint(checkpoint_dir)
            print(f"  Loading checkpoint from Epoch {epoch}: {checkpoint_path}")
            model.value_net.load_weights(checkpoint_path)
            model.target_value_net.set_weights(model.value_net.get_weights())
        elif args.model == 'risky_upgrade':
            epoch, checkpoint_paths = find_latest_checkpoint_risky_upgrade(checkpoint_dir)
            print(f"  Loading checkpoints from Epoch {epoch}:")
            load_risky_upgrade_checkpoints(model, checkpoint_paths)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return

    # 4. Run Simulation
    print("\n[4] Running Monte Carlo Simulation...")
    # Settings to match the requested plots
    n_batches = 10000  # High batch count for smooth distributions
    n_steps = 50       # Matches the time axis in the requested plot
    burn_in = 10       
    seed = 42

    print(f"  Batches: {n_batches}")
    print(f"  Steps: {n_steps}")
    
    history, stats = model.simulate(n_steps=n_steps, n_batches=n_batches, seed=seed)

    # 5. Compute Moments
    print("\n[5] Computing Economic Moments (Active Firms)...")
    moments = compute_moments(history, burn_in=burn_in)
    print(f"  Mean Capital: {moments['capital_mean']:.4f}")
    print(f"  Mean Debt:    {moments['debt_mean']:.4f}")
    print(f"  Default Rate: {stats['default_rate']*100:.2f}%")

    # 6. Generate Plots
    print("\n[6] Generating Visualizations...")
    
    # Standard Time Series with Default Annotation
    plot_simulation_time_series(history, output_dir, n_samples=3, duration=n_steps)
    
    # Averaged Policy Functions (Investment, Next Debt)
    plot_averaged_policy_functions(history, econ_params, output_dir)
    
    # Batch Evolution (Heatmaps & Densities)
    plot_batch_evolution(history, output_dir)
    
    # Default Analysis (New)
    plot_default_analysis(history, output_dir)

    print(f"\nAnalysis complete. Results saved to {output_dir}")


if __name__ == "__main__":
    main()