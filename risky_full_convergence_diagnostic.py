#!/usr/bin/env python3
"""
Full Convergence Diagnostic — Risky Debt Model
═══════════════════════════════════════════════

Standalone script that runs the complete convergence study end-to-end:

  1. Solves the VFI across a 10×10 grid of (N_k, N_d) ∈ {50,100,…,500}²
     (uses cached .npz results when available)
  2. Simulates each solution with shared synthetic data
  3. Computes 12 target moments per grid configuration
  4. Generates two publication-quality convergence visualisations
  5. Prints convergence diagnostics

No prerequisite scripts need to be run first.

Plots produced
──────────────
01_moment_convergence.png   — running mean ± σ/2σ, high-res & grand means
02_diagonal_convergence.png — N_k = N_d path with ±2 % safety band

RTX 5090 (sm_120) compatible:
  - Single GPU, sequential solves
  - XLA-only kernels (jit_compile=True)
  - Optimised chunk sizes from benchmark study
"""

import dataclasses
import logging
import math

from src.econ_models.vfi.grids.grid_utils import compute_optimal_chunks
import os
import shutil
import sys
import time
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

# ── TF / project imports (GPU configured in main) ───────────────────────
from src.econ_models.config.economic_params import EconomicParams
from src.econ_models.config.bond_config import BondsConfig
from src.econ_models.io.file_utils import load_json_file

from risky_common import (
    BASE_DIR,
    N_PRODUCTIVITY,
    econ_tag,
    get_econ_params_path_by_list,
    get_bounds_path_by_list,
    get_vfi_cache_path,
    to_python_float,
    apply_burn_in,
)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════════

# NOTE: this econ_list differs from DEFAULT_ECON_LIST (second entry is different)
econ_list = [
    [0.6, 0.17, 1.0, 0.02, 0.1, 0.08],
    [0.6, 0.17, 1.0, 0.02, 0.03, 0.01],
]
econ_id = 0

CONFIG_PARAMS_FILE = os.path.join(BASE_DIR, "hyperparam/prefixed/vfi_params.json")
ECON_PARAMS_FILE_RISKY = get_econ_params_path_by_list(econ_list[econ_id])
BOUNDARY_RISKY = get_bounds_path_by_list(econ_list[econ_id])

RESULTS_DIR = './results/golden_vfi_risky'
SAVE_DIR = './results/golden_vfi_risky/convergence_analysis'

# ── Moment keys & labels ────────────────────────────────────────────────
MOMENT_KEYS = [
    'mean_K', 'mean_B', 'mean_leverage', 'mean_investment_rate',
    'std_K', 'std_B', 'std_leverage', 'std_output',
    'autocorr_investment_rate', 'autocorr_leverage', 'autocorr_output', 'autocorr_issuance',
    'issuance_freq', 'investment_freq', 'inaction_rate',
]

MOMENT_LABELS = {
    'mean_K': 'Mean Capital',
    'mean_B': 'Mean Debt',
    'mean_leverage': 'Mean Leverage',
    'mean_investment_rate': 'Mean Inv. Rate',
    'std_K': 'Std Capital',
    'std_B': 'Std Debt',
    'std_leverage': 'Std Leverage',
    'std_output': 'Std Output',
    'autocorr_investment_rate': 'Autocorr(Inv Rate)',
    'autocorr_leverage': 'Autocorr(Leverage)',
    'autocorr_output': 'Autocorr(Output)',
    'autocorr_issuance': 'Autocorr(Eq. Issuance)',
    'issuance_freq': 'Eq. Issuance Freq.',
    'investment_freq': 'Investment Freq.',
    'inaction_rate': 'Inaction Rate',
}

# ── Matplotlib defaults ─────────────────────────────────────────────────
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 180,
    'savefig.bbox': 'tight',
})


# ═══════════════════════════════════════════════════════════════════════════
#  Data Setup
# ═══════════════════════════════════════════════════════════════════════════

def setup_shared_data() -> Dict[str, Any]:
    """
    Create shared synthetic data for all convergence tests.

    Data generation runs on CPU to avoid sm_120 eager-op crashes
    (tfp.distributions.gamma uses non-XLA kernels).
    """
    import tensorflow as tf
    from src.econ_models.simulator.synthetic_data_gen import synthetic_data_generator

    BOUNDS = load_json_file(BOUNDARY_RISKY)
    ECON_PARAMS = EconomicParams(**load_json_file(ECON_PARAMS_FILE_RISKY))

    sample_bonds_config = BondsConfig.validate_and_load(
        bounds_file=BOUNDARY_RISKY,
        current_params=ECON_PARAMS,
    )

    data_gen = synthetic_data_generator(
        econ_params_benchmark=ECON_PARAMS,
        sample_bonds_config=sample_bonds_config,
        batch_size=10000,
        T_periods=1000,
        include_debt=True,
    )

    with tf.device('/CPU:0'):
        initial_states, shock_sequence = data_gen.gen()

    initial_states_np = tuple(
        s.numpy() if hasattr(s, 'numpy') else np.array(s)
        for s in initial_states
    )
    shock_sequence_np = (
        shock_sequence.numpy() if hasattr(shock_sequence, 'numpy')
        else np.array(shock_sequence)
    )

    return {
        'initial_states_np': initial_states_np,
        'shock_sequence_np': shock_sequence_np,
        'econ_params': ECON_PARAMS,
        'bounds': BOUNDS,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  VFI Solve & Simulate
# ═══════════════════════════════════════════════════════════════════════════

def solve_risky_model(
    econ_params: EconomicParams,
    bounds: Dict,
    n_capital: int,
    n_productivity: int,
    n_debt: int,
    k_chunk_size: int,
    b_chunk_size: int,
    kp_chunk_size: int,
    bp_chunk_size: int,
) -> Dict:
    """Solve VFI with specified grid resolution on the configured GPU."""
    from src.econ_models.config.vfi_config import load_grid_config
    from src.econ_models.vfi.risky import RiskyDebtModelVFI

    config = load_grid_config(CONFIG_PARAMS_FILE, 'risky')
    config = dataclasses.replace(
        config,
        n_capital=n_capital,
        n_productivity=n_productivity,
        n_debt=n_debt,
    )

    model = RiskyDebtModelVFI(
        econ_params,
        config,
        k_bounds=(bounds['bounds']['k_min'], bounds['bounds']['k_max']),
        b_bounds=(bounds['bounds']['b_min'], bounds['bounds']['b_max']),
        k_chunk_size=k_chunk_size,
        b_chunk_size=b_chunk_size,
        kp_chunk_size=kp_chunk_size,
        bp_chunk_size=bp_chunk_size,
    )
    return model.solve()


def run_simulation(
    initial_states_np: Tuple[np.ndarray, ...],
    shock_sequence_np: np.ndarray,
    vfi_results: Dict,
    econ_params: EconomicParams,
) -> Dict[str, np.ndarray]:
    """Run simulation using CPU NumPy path."""
    from src.econ_models.simulator.vfi.risky import VFISimulator_risky

    simulator = VFISimulator_risky(econ_params)
    simulator.load_solved_vfi_solution(vfi_results)

    return simulator.simulate(initial_states_np, shock_sequence_np)


# ═══════════════════════════════════════════════════════════════════════════
#  Post-processing helpers
# ═══════════════════════════════════════════════════════════════════════════


def compute_all_moments(
    simulation_results: Dict[str, np.ndarray],
    econ_params: EconomicParams,
) -> Dict[str, float]:
    """Compute all 12 target moments from simulation results."""
    import tensorflow as tf
    from src.econ_models.moment_calculator.compute_derived_quantities import compute_all_derived_quantities
    from src.econ_models.moment_calculator.compute_mean import compute_global_mean
    from src.econ_models.moment_calculator.compute_std import compute_global_std
    from src.econ_models.moment_calculator.compute_autocorrelation import compute_autocorrelation_lags_1_to_5
    from src.econ_models.moment_calculator.compute_inaction_rate import compute_inaction_rate

    with tf.device('/CPU:0'):
        derived = compute_all_derived_quantities(
            simulation_results,
            econ_params.depreciation_rate,
            econ_params.capital_share,
            include_debt=True,
        )

        leverage = derived['leverage']

        return {
            # Row 1 — Means
            'mean_K': to_python_float(compute_global_mean(derived['capital'])),
            'mean_B': to_python_float(compute_global_mean(simulation_results['B_curr'])),
            'mean_investment_rate': to_python_float(compute_global_mean(derived['investment_rate'])),
            'mean_leverage': to_python_float(compute_global_mean(leverage)),
            # Row 2 — Standard deviations
            'std_K': to_python_float(compute_global_std(derived['capital'])),
            'std_B': to_python_float(compute_global_std(simulation_results['B_curr'])),
            'std_output': to_python_float(compute_global_std(derived['output'])),
            'std_leverage': to_python_float(compute_global_std(leverage)),
            # Row 3 — Autocorrelations
            'autocorr_investment_rate': to_python_float(
                compute_autocorrelation_lags_1_to_5(derived['investment_rate'])['lag_1']
            ),
            'autocorr_leverage': to_python_float(
                compute_autocorrelation_lags_1_to_5(leverage)['lag_1']
            ),
            'autocorr_output': to_python_float(
                compute_autocorrelation_lags_1_to_5(derived['output'])['lag_1']
            ),
            'autocorr_issuance': to_python_float(
                compute_autocorrelation_lags_1_to_5(derived['equity_issuance_rate'])['lag_1']
            ),
            # Row 4 — Frequencies & inaction
            'issuance_freq': to_python_float(compute_global_mean(derived['issuance_binary'])),
            'investment_freq': to_python_float(compute_global_mean(derived['investment_binary'])),
            'inaction_rate': to_python_float(
                compute_inaction_rate(derived['investment_rate'], -0.001, 0.001)
            ),
        }


# ═══════════════════════════════════════════════════════════════════════════
#  Single-Job Runner
# ═══════════════════════════════════════════════════════════════════════════

def run_single_job(
    job_idx: int,
    n_k: int,
    n_d: int,
    n_productivity: int,
    burn_in_periods: int,
    initial_states_np: Tuple[np.ndarray, ...],
    shock_sequence_np: np.ndarray,
    econ_params: EconomicParams,
    bounds: Dict,
) -> Dict[str, Any]:
    """Run a single (solve + simulate + moments) job, with caching."""
    from src.econ_models.io.artifacts import save_vfi_results

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Job {job_idx}: (n_k, n_d) = ({n_k}, {n_d})")
    logger.info('=' * 60)

    k_chunk, b_chunk, kp_chunk, bp_chunk = compute_optimal_chunks(
        n_k, n_d, n_z=n_productivity
    )

    cache_path = get_vfi_cache_path(econ_tag(econ_list[econ_id]), n_k, n_d)

    # -- Solve (or load cached) ----------------------------------------
    t0 = time.time()
    if os.path.exists(cache_path):
        logger.info(f"  Loading cached VFI from {cache_path}")
        vfi_results = np.load(cache_path, allow_pickle=True)
    else:
        logger.info("  Solving VFI ...")
        vfi_results = solve_risky_model(
            econ_params, bounds,
            n_capital=n_k, n_productivity=n_productivity, n_debt=n_d,
            k_chunk_size=k_chunk, b_chunk_size=b_chunk,
            kp_chunk_size=kp_chunk, bp_chunk_size=bp_chunk,
        )
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        save_vfi_results(vfi_results, cache_path)
        logger.info(f"  VFI cached to {cache_path}")
    solve_time = time.time() - t0

    # -- Simulate ------------------------------------------------------
    logger.info("  Simulating ...")
    t1 = time.time()
    sim_results = run_simulation(
        initial_states_np, shock_sequence_np, vfi_results, econ_params,
    )
    sim_time = time.time() - t1

    sim_stationary = apply_burn_in(sim_results, burn_in_periods)

    # -- Moments -------------------------------------------------------
    moments = compute_all_moments(sim_stationary, econ_params)

    logger.info(
        f"  Done ({n_k},{n_d}): solve={solve_time:.1f}s, sim={sim_time:.1f}s | "
        f"inv_rate={moments['mean_investment_rate']:.4f}, "
        f"leverage={moments['mean_leverage']:.4f}"
    )

    return {
        'job_idx': job_idx,
        'n_k': n_k,
        'n_d': n_d,
        'moments': moments,
        'solve_time': solve_time,
        'sim_time': sim_time,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Visualisation helpers
# ═══════════════════════════════════════════════════════════════════════════

def build_2d_grid(n_k_arr, n_d_arr, values):
    """Reshape flat arrays into a 2-D matrix indexed by unique (n_k, n_d)."""
    k_unique = np.sort(np.unique(n_k_arr))
    d_unique = np.sort(np.unique(n_d_arr))
    matrix = np.full((len(k_unique), len(d_unique)), np.nan)

    k_idx = {v: i for i, v in enumerate(k_unique)}
    d_idx = {v: i for i, v in enumerate(d_unique)}

    for nk, nd, val in zip(n_k_arr, n_d_arr, values):
        matrix[k_idx[nk], d_idx[nd]] = val

    return k_unique, d_unique, matrix


# ═══════════════════════════════════════════════════════════════════════════
#  Plot 1 — Moment Convergence
# ═══════════════════════════════════════════════════════════════════════════

def plot_moment_convergence(n_k_arr, n_d_arr, moments, save_dir):
    """
    Convergence plot sorted by total grid points (N_k × N_d).
    Shows running mean ± σ/2σ envelopes, high-res reference mean,
    and grand mean.
    """
    total_points = n_k_arr * n_d_arr
    sort_idx = np.argsort(total_points)
    sorted_total_all = total_points[sort_idx]
    sorted_nk_all = n_k_arr[sort_idx]
    sorted_nd_all = n_d_arr[sort_idx]

    # Informative tick positions
    n_all = len(sorted_total_all)
    tick_idx = np.linspace(0, n_all - 1, min(8, n_all), dtype=int)
    tick_pos = sorted_total_all[tick_idx].astype(float)
    tick_lbl = [
        f'{sorted_total_all[i] // 1000:.0f}k\n({sorted_nk_all[i]},{sorted_nd_all[i]})'
        for i in tick_idx
    ]

    fig, axes = plt.subplots(4, 4, figsize=(24, 18))
    axes = axes.flatten()

    for idx, key in enumerate(MOMENT_KEYS):
        ax = axes[idx]
        sorted_vals = moments[key][sort_idx]
        n = len(sorted_vals)
        x = sorted_total_all.astype(float)

        # High-res reference mean
        large_mask = (n_k_arr >= 200) & (n_d_arr >= 200)
        if large_mask.sum() >= 5:
            refined_mean = np.mean(moments[key][large_mask])
            refined_std = np.std(moments[key][large_mask])
        else:
            refined_mean = np.mean(moments[key])
            refined_std = np.std(moments[key])

        # Edge-aware running mean & std
        window = max(7, n // 8)
        pad = window // 2
        running_mean = np.array([
            np.mean(sorted_vals[max(0, i - pad):i + pad + 1])
            for i in range(n)
        ])
        running_std = np.array([
            np.std(sorted_vals[max(0, i - pad):i + pad + 1])
            for i in range(n)
        ])

        ax.scatter(x, sorted_vals, c=sorted_total_all, s=18, alpha=0.6,
                   cmap='viridis', zorder=3, edgecolors='none')
        ax.plot(x, running_mean, color='#e74c3c', linewidth=2.5,
                label='Running mean', zorder=4)
        ax.fill_between(x, running_mean - running_std, running_mean + running_std,
                        alpha=0.15, color='#e74c3c', zorder=1, label='±1σ')
        ax.fill_between(x, running_mean - 2 * running_std, running_mean + 2 * running_std,
                        alpha=0.07, color='#e74c3c', zorder=0, label='±2σ')
        ax.axhline(refined_mean, color='#2ecc71', linestyle='-', linewidth=2,
                   alpha=0.8, label=f'High-res mean = {refined_mean:.4f}')
        ax.axhspan(refined_mean - refined_std, refined_mean + refined_std,
                   alpha=0.1, color='#2ecc71', zorder=0)

        grand_mean = np.mean(moments[key])
        ax.axhline(grand_mean, color='black', linestyle='--', linewidth=1,
                   alpha=0.6, label=f'Grand mean = {grand_mean:.4f}')

        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_lbl, fontsize=6)
        ax.set_xlabel('Total Grid Points  N_k × N_d\n(N_k, N_d)', fontsize=8)
        ax.set_ylabel('Value', fontsize=9)
        ax.set_title(MOMENT_LABELS[key], fontweight='bold')
        ax.legend(fontsize=6, loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.15)

    for i in range(len(MOMENT_KEYS), len(axes)):
        axes[i].axis('off')

    fig.suptitle(
        'Moment Convergence — Sorted by Total Grid Points (N_k × N_d)\n'
        'Red = running mean ± σ;  Green = high-res mean (N_k,N_d ≥ 200);  '
        'Black dashed = grand mean',
        fontsize=13, fontweight='bold', y=1.03,
    )
    plt.tight_layout()
    path = os.path.join(save_dir, '01_moment_convergence.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════════════════
#  Plot 2 — Diagonal Convergence (N_k = N_d)
# ═══════════════════════════════════════════════════════════════════════════

def plot_diagonal_convergence(n_k_arr, n_d_arr, moments, save_dir):
    """
    Convergence along the diagonal where N_k == N_d.
    Includes ±2 % safety band around the finest-grid prediction.
    """
    k_unique = np.sort(np.unique(n_k_arr))
    d_unique = np.sort(np.unique(n_d_arr))
    diag_sizes = np.intersect1d(k_unique, d_unique)

    fig, axes = plt.subplots(4, 4, figsize=(22, 18))
    axes = axes.flatten()

    for idx, key in enumerate(MOMENT_KEYS):
        ax = axes[idx]
        _, _, mat = build_2d_grid(n_k_arr, n_d_arr, moments[key])

        # Diagonal values
        diag_vals = []
        for ds in diag_sizes:
            ki = np.where(k_unique == ds)[0]
            di = np.where(d_unique == ds)[0]
            if len(ki) > 0 and len(di) > 0:
                diag_vals.append(mat[ki[0], di[0]])
            else:
                diag_vals.append(np.nan)
        diag_vals = np.array(diag_vals)

        ax.plot(diag_sizes, diag_vals, 'D-', color='#2980b9', linewidth=2,
                markersize=8, label='Diagonal (N_k = N_d)', zorder=4)

        # ±2 % safety region around the finest grid prediction
        finest_val = diag_vals[-1]
        ax.axhline(finest_val, color='#2ecc71', linestyle='-', linewidth=1.5,
                   alpha=0.7, label=f'Finest = {finest_val:.4f}')
        ax.axhspan(finest_val * 0.98, finest_val * 1.02,
                   alpha=0.12, color='#2ecc71', zorder=0, label='±2% band')

        ax.set_xticks(diag_sizes)
        ax.set_xticklabels(
            [f'{ds}\n({ds * ds // 1000}k pts)' for ds in diag_sizes],
            fontsize=7,
        )
        ax.set_xlabel('Grid Size N  (N_k = N_d = N)\ntotal points = N²')
        ax.set_ylabel(MOMENT_LABELS[key])
        ax.set_title(MOMENT_LABELS[key], fontweight='bold')
        ax.legend(fontsize=6, loc='best')
        ax.grid(True, alpha=0.2)

    for i in range(len(MOMENT_KEYS), len(axes)):
        axes[i].axis('off')

    fig.suptitle(
        'Diagonal Convergence (N_k = N_d)',
        fontsize=13, fontweight='bold', y=1.02,
    )
    plt.tight_layout()
    path = os.path.join(save_dir, '02_diagonal_convergence.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════════════════
#  Convergence Diagnostics
# ═══════════════════════════════════════════════════════════════════════════

def print_convergence_diagnostics(n_k_arr, n_d_arr, moments):
    """Print the same diagnostics table as risky_convergence_analysis.py."""
    print("\n" + "=" * 70)
    print("CONVERGENCE DIAGNOSTICS")
    print("=" * 70)

    large_mask = (n_k_arr >= 250) & (n_d_arr >= 250)

    print(f"\n{'Moment':<25s} {'Grand Mean':>12s} {'High-Res Mean':>14s} "
          f"{'High-Res σ':>12s} {'CV(%)':>8s} {'Range':>12s}")
    print("-" * 90)

    for key in MOMENT_KEYS:
        vals = moments[key]
        gm = np.mean(vals)
        hr_vals = vals[large_mask]
        hr_mean = np.mean(hr_vals)
        hr_std = np.std(hr_vals)
        cv = hr_std / abs(hr_mean) * 100 if abs(hr_mean) > 1e-10 else 0
        rng = np.max(vals) - np.min(vals)

        print(f"  {MOMENT_LABELS[key]:<23s} {gm:>12.6f} {hr_mean:>14.6f} "
              f"{hr_std:>12.6f} {cv:>7.2f}% {rng:>12.6f}")

    print("\n" + "=" * 70)
    print(f"All plots saved to: {os.path.abspath(SAVE_DIR)}/")
    print("=" * 70)


# ═══════════════════════════════════════════════════════════════════════════
#  Save moments table (for external reference)
# ═══════════════════════════════════════════════════════════════════════════

def save_moments_table(n_k_arr, n_d_arr, moments, all_moments_list):
    """Write moments_table.txt in the same format as risky_golden_vfi_finder."""
    table_path = os.path.join(RESULTS_DIR, 'moments_table.txt')
    os.makedirs(RESULTS_DIR, exist_ok=True)

    with open(table_path, 'w') as f:
        header = f"{'Grid':>12s}"
        for key in MOMENT_KEYS:
            header += f"  {key:>22s}"
        f.write(header + '\n')
        f.write('-' * len(header) + '\n')

        for idx in range(len(n_k_arr)):
            row = f"({n_k_arr[idx]:3d},{n_d_arr[idx]:3d})"
            row = f"{row:>12s}"
            for key in MOMENT_KEYS:
                row += f"  {all_moments_list[idx][key]:>22.6f}"
            f.write(row + '\n')

    logger.info(f"Moments table saved to {table_path}")


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """Run the full convergence diagnostic over a grid of VFI resolutions."""
    # ══════════════════════════════════════════════════════════════════
    #  GRID CONFIGURATION
    # ══════════════════════════════════════════════════════════════════
    n_capital_list = [
        nk for nk in range(50, 550, 50) for _ in range(10)
    ]
    n_debt_list = list(range(50, 550, 50)) * 10
    n_productivity = 12
    burn_in_periods = 400
    # ══════════════════════════════════════════════════════════════════

    # -- Configure single GPU ------------------------------------------
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    import tensorflow as tf

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info(f"Using GPU: {gpus[0].name}")
    else:
        logger.warning("No GPU found — running on CPU (will be slow)")

    logger.info(f"Grid: {len(n_capital_list)} experiments  "
                f"({min(n_capital_list)}–{max(n_capital_list)} × "
                f"{min(n_debt_list)}–{max(n_debt_list)})")
    logger.info(f"n_productivity: {n_productivity}, burn_in: {burn_in_periods}")

    # -- Shared synthetic data -----------------------------------------
    logger.info("Setting up shared synthetic data ...")
    shared_data = setup_shared_data()

    initial_states_np = shared_data['initial_states_np']
    shock_sequence_np = shared_data['shock_sequence_np']
    econ_params = shared_data['econ_params']
    bounds = shared_data['bounds']

    # -- Run all jobs sequentially -------------------------------------
    results_ordered: List[Dict[str, Any]] = []
    total_t0 = time.time()

    for idx, (n_k, n_d) in enumerate(zip(n_capital_list, n_debt_list)):
        result = run_single_job(
            idx, n_k, n_d, n_productivity, burn_in_periods,
            initial_states_np, shock_sequence_np, econ_params, bounds,
        )
        results_ordered.append(result)

    total_time = time.time() - total_t0
    logger.info(f"\nTotal wall time: {total_time:.1f}s ({total_time / 60:.1f} min)")

    # -- Convert to arrays for visualisation ---------------------------
    all_moments_list = [r['moments'] for r in results_ordered]
    n_k_arr = np.array(n_capital_list)
    n_d_arr = np.array(n_debt_list)
    moments = {
        key: np.array([m[key] for m in all_moments_list])
        for key in MOMENT_KEYS
    }

    # -- Save canonical copy of final VFI ------------------------------
    tag = econ_tag(econ_list[econ_id])
    last_cache = get_vfi_cache_path(tag, n_capital_list[-1], n_debt_list[-1])
    canonical_path = f'./ground_truth_risky/golden_vfi_risky_{tag}.npz'
    if os.path.exists(last_cache):
        shutil.copy2(last_cache, canonical_path)
        logger.info(f"Copied final VFI results to {canonical_path}")

    # -- Save moments table for reference ------------------------------
    save_moments_table(n_k_arr, n_d_arr, moments, all_moments_list)

    # -- Generate convergence plots ------------------------------------
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"\nGenerating visualisations ...\n")

    plot_moment_convergence(n_k_arr, n_d_arr, moments, SAVE_DIR)
    plot_diagonal_convergence(n_k_arr, n_d_arr, moments, SAVE_DIR)

    # -- Diagnostics ---------------------------------------------------
    print_convergence_diagnostics(n_k_arr, n_d_arr, moments)


if __name__ == "__main__":
    main()
