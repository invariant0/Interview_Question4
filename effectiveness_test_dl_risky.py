"""
Test script to evaluate the effectiveness of trained DL solution vs VFI matrix.

This script:
1. Loads the VFI solution matrix and the trained DL model checkpoint
2. Computes and plots metrics across all saved checkpoints:
   - MAE evolution
   - MAPE in non-default region evolution
   - Default region identification accuracy evolution
"""

import os
import sys
import glob
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import json

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from econ_models.config.economic_params import EconomicParams
from econ_models.config.dl_config import DeepLearningConfig, load_dl_config
from econ_models.core.nets import NeuralNetFactory
from econ_models.core.standardize import StateSpaceNormalizer
from econ_models.core.types import TENSORFLOW_DTYPE


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


def find_checkpoint_files(checkpoint_dir: str = "./checkpoints") -> List[Tuple[int, str]]:
    """
    Find all checkpoint files and return sorted by epoch.
    
    Returns:
        List of (epoch, filepath) tuples sorted by epoch.
    """
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
    F1 score balances precision and recall for identifying default states.
    
    Returns:
        F1 score (0 to 1, higher is better)
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
    
    Uses a percentile-based approach: states with very low values 
    (bottom 5% or values close to zero) are considered default.
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


def plot_training_metrics(
    epochs: List[int],
    maes: List[float],
    mapes_non_default: List[float],
    f1_scores: List[float],
    output_dir: str = "./test_results"
) -> plt.Figure:
    """
    Plot three metrics in a single figure with three subplots.
    
    1. MAE evolution
    2. MAPE in non-default region evolution
    3. Default region identification F1 score evolution
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: MAE Evolution
    ax1 = axes[0]
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
    ax2 = axes[1]
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
    ax3 = axes[2]
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
    
    plt.tight_layout()
    
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
    print("=" * 60)
    print("Risky Model: DL vs VFI Comparison Test")
    print("=" * 60)
    
    # Configuration paths
    vfi_results_path = "./ground_truth/risky_debt_model_vfi_results.npz"
    bounds_path = "./hyperparam/autogen/bounds_risky.json"
    econ_params_path = "./hyperparam/prefixed/econ_params_risky.json"
    dl_params_path = "./hyperparam/prefixed/dl_params.json"
    checkpoint_dir = "./checkpoints/risky"
    output_dir = "./result_effectiveness_dl_risky"
    
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
    
    dl_config = load_dl_config(dl_params_path, "risky")
    
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
    # Step 3: Build Model Architecture
    print("\n[3] Building Value Network...")
    normalizer = StateSpaceNormalizer(dl_config)
    value_net = build_value_network(dl_config)
    
    # Step 4: Find Checkpoint Files
    print("\n[4] Scanning for Checkpoints...")
    checkpoint_files = find_checkpoint_files(checkpoint_dir)
    
    if not checkpoint_files:
        print(f"ERROR: No checkpoint files found in {checkpoint_dir}")
        print("Please train the model first using 'python train_dl.py --model risky'")
        return
    
    print(f"  Found {len(checkpoint_files)} checkpoints")
    print(f"  Epochs: {[e for e, _ in checkpoint_files[:5]]}...{[e for e, _ in checkpoint_files[-3:]]}")
    
    # Step 5: Evaluate All Checkpoints
    print("\n[5] Evaluating All Checkpoints...")
    epochs: List[int] = []
    maes: List[float] = []
    mapes_non_default: List[float] = []
    f1_scores: List[float] = []
    
    for i, (epoch, filepath) in enumerate(checkpoint_files):
        try:
            value_net.load_weights(filepath)
            predictions = evaluate_model_on_grid(value_net, normalizer, k_grid, b_grid, z_grid)
            
            mae = compute_mae(V_vfi, predictions)
            mape_nd = compute_mape_non_default(V_vfi, predictions, default_threshold)
            f1 = compute_default_region_f1(V_vfi, predictions, default_threshold)
            
            epochs.append(epoch)
            maes.append(mae)
            mapes_non_default.append(mape_nd)
            f1_scores.append(f1)
            
            if i % 10 == 0 or i == len(checkpoint_files) - 1:
                print(f"  Epoch {epoch:5d}: MAE={mae:.6f}, MAPE(non-def)={mape_nd:.2f}%, F1={f1:.4f}")
                
        except Exception as e:
            print(f"  WARNING: Failed to load checkpoint {filepath}: {e}")
            continue
    
    if len(epochs) == 0:
        print("ERROR: Could not evaluate any checkpoints")
        return
    
    # Step 6: Generate Visualization
    print("\n[6] Generating Metrics Plot...")
    fig = plot_training_metrics(epochs, maes, mapes_non_default, f1_scores, output_dir)
    plt.close(fig)
    
    # Step 7: Print Summary Statistics
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    
    print(f"\n  Final Checkpoint (Epoch {epochs[-1]}):")
    print(f"    MAE:                {maes[-1]:.6f}")
    print(f"    MAPE (non-default): {mapes_non_default[-1]:.2f}%")
    print(f"    F1 Score:           {f1_scores[-1]:.4f}")
    
    best_mae_idx = np.argmin(maes)
    best_mape_idx = np.argmin(mapes_non_default)
    best_f1_idx = np.argmax(f1_scores)
    
    print(f"\n  Best MAE (Epoch {epochs[best_mae_idx]}):")
    print(f"    MAE: {maes[best_mae_idx]:.6f}")
    
    print(f"\n  Best MAPE non-default (Epoch {epochs[best_mape_idx]}):")
    print(f"    MAPE: {mapes_non_default[best_mape_idx]:.2f}%")
    
    print(f"\n  Best F1 Score (Epoch {epochs[best_f1_idx]}):")
    print(f"    F1: {f1_scores[best_f1_idx]:.4f}")
    
    print(f"\n  Default Threshold: {default_threshold:.6f}")
    print(f"  Default Region: {100*n_default_states/total_states:.2f}% of state space")
    
    print(f"\n  Results saved to: {output_dir}")
    print("=" * 60)
    
    # Save numerical results to JSON
    results_summary = {
        "default_threshold": to_python_float(default_threshold),
        "default_region_percentage": to_python_float(100*n_default_states/total_states),
        "final_epoch": int(epochs[-1]),
        "final_metrics": {
            "mae": to_python_float(maes[-1]),
            "mape_non_default": to_python_float(mapes_non_default[-1]),
            "f1_score": to_python_float(f1_scores[-1])
        },
        "best_metrics": {
            "mae": {"value": to_python_float(maes[best_mae_idx]), "epoch": int(epochs[best_mae_idx])},
            "mape_non_default": {"value": to_python_float(mapes_non_default[best_mape_idx]), "epoch": int(epochs[best_mape_idx])},
            "f1_score": {"value": to_python_float(f1_scores[best_f1_idx]), "epoch": int(epochs[best_f1_idx])}
        },
        "epochs": [int(e) for e in epochs],
        "maes": [to_python_float(m) for m in maes],
        "mapes_non_default": [to_python_float(m) for m in mapes_non_default],
        "f1_scores": [to_python_float(f) for f in f1_scores]
    }
    
    summary_path = os.path.join(output_dir, "evaluation_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nNumerical results saved to: {summary_path}")


if __name__ == "__main__":
    main()