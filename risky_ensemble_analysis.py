#!/usr/bin/env python3
"""
Ensemble Convergence Analysis — Pre-solved VFI (Risky Debt Model)
═════════════════════════════════════════════════════════════════

Loads pre-solved VFI cache files, runs simulations, computes moments,
and produces the ensemble convergence diagnostic plot.

**Requires**: All VFI solutions must already exist in ground_truth_risky/.
              Run  risky_ensemble_solver.py  first if any are missing.

For density level N (representing ≈ N² total grid points):
  • (N+30, N−30), (N+20, N−20), (N+10, N−10)  — capital-heavy
  • (N,    N   )                                — balanced
  • (N−10, N+10), (N−20, N+20), (N−30, N+30)  — debt-heavy

Density levels:  N ∈ {100, 200, 300, 400, 500}  (5 levels)
Total configs:   5 × 7 = 35

Outputs
───────
• results/golden_vfi_risky/ensemble_convergence.png
• Console convergence diagnostics table

Usage
─────
  python risky_ensemble_analysis.py
"""

import logging
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

# ── Project imports ──────────────────────────────────────────────────────
from src.econ_models.config.economic_params import EconomicParams
from src.econ_models.config.bond_config import BondsConfig
from src.econ_models.io.file_utils import load_json_file

from risky_common import (
    DEFAULT_ECON_LIST,
    BASE_DIR,
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
#  Configuration  (must stay in sync with risky_ensemble_solver.py)
# ═══════════════════════════════════════════════════════════════════════════

econ_list = DEFAULT_ECON_LIST
econ_id = 0

ECON_PARAMS_FILE_RISKY = get_econ_params_path_by_list(econ_list[econ_id])
BOUNDARY_RISKY = get_bounds_path_by_list(econ_list[econ_id])

RESULTS_DIR = './results/golden_vfi_risky'

# ── Density levels & ensemble design ─────────────────────────────────────
DENSITY_LEVELS = [100, 200, 300, 400, 500]
NUM_ENSEMBLE_MEMBERS = 7
BURN_IN_PERIODS = 400


def get_ensemble_configs(N: int) -> List[Tuple[int, int]]:
    """Return the 7 (N_k, N_d) configurations for density level N."""
    return [
        (N - 30, N + 30),
        (N - 20, N + 20),
        (N - 10, N + 10),
        (N,      N),
        (N + 10, N - 10),
        (N + 20, N - 20),
        (N + 30, N - 30),
    ]


def cache_path_for(n_k: int, n_d: int) -> str:
    """Return the ground-truth cache path for a given (N_k, N_d) config."""
    tag = econ_tag(econ_list[econ_id])
    return get_vfi_cache_path(tag, n_k, n_d)


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

ENSEMBLE_MEMBER_LABELS = [
    'K-heavy +30',
    'K-heavy +20',
    'K-heavy +10',
    'balanced',
    'D-heavy +10',
    'D-heavy +20',
    'D-heavy +30',
]

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
#  Pre-flight: verify all cache files exist
# ═══════════════════════════════════════════════════════════════════════════

def verify_all_cached() -> List[Tuple[int, int]]:
    """
    Check that every required VFI cache file exists.
    Returns list of missing (n_k, n_d) configs.
    """
    missing = []
    for N in DENSITY_LEVELS:
        for n_k, n_d in get_ensemble_configs(N):
            if not os.path.exists(cache_path_for(n_k, n_d)):
                missing.append((n_k, n_d))
    return missing


# ═══════════════════════════════════════════════════════════════════════════
#  Data Setup
# ═══════════════════════════════════════════════════════════════════════════

def setup_shared_data() -> Dict[str, Any]:
    """Create shared synthetic data for simulation (CPU-only)."""
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
#  Simulation & Moments
# ═══════════════════════════════════════════════════════════════════════════

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


def compute_all_moments(
    simulation_results: Dict[str, np.ndarray],
    econ_params: EconomicParams,
) -> Dict[str, float]:
    """Compute all 15 target moments."""
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
            'mean_K': to_python_float(compute_global_mean(derived['capital'])),
            'mean_B': to_python_float(compute_global_mean(simulation_results['B_curr'])),
            'mean_investment_rate': to_python_float(compute_global_mean(derived['investment_rate'])),
            'mean_leverage': to_python_float(compute_global_mean(leverage)),
            'std_K': to_python_float(compute_global_std(derived['capital'])),
            'std_B': to_python_float(compute_global_std(simulation_results['B_curr'])),
            'std_output': to_python_float(compute_global_std(derived['output'])),
            'std_leverage': to_python_float(compute_global_std(leverage)),
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
            'issuance_freq': to_python_float(compute_global_mean(derived['issuance_binary'])),
            'investment_freq': to_python_float(compute_global_mean(derived['investment_binary'])),
            'inaction_rate': to_python_float(
                compute_inaction_rate(derived['investment_rate'], -0.001, 0.001)
            ),
        }


def average_moments(moment_list: List[Dict[str, float]]) -> Dict[str, float]:
    """Element-wise average of a list of moment dictionaries."""
    avg = {}
    for key in MOMENT_KEYS:
        vals = [m[key] for m in moment_list]
        avg[key] = float(np.mean(vals))
    return avg


# ═══════════════════════════════════════════════════════════════════════════
#  Single-Config Runner (load → simulate → moments)
# ═══════════════════════════════════════════════════════════════════════════

def run_single_analysis(
    n_k: int,
    n_d: int,
    initial_states_np: Tuple[np.ndarray, ...],
    shock_sequence_np: np.ndarray,
    econ_params: EconomicParams,
) -> Dict[str, Any]:
    """Load a cached VFI solution, simulate, and compute moments."""
    cache_path = cache_path_for(n_k, n_d)

    logger.info(f"    Loading VFI ({n_k},{n_d}) from {cache_path}")
    vfi_results = np.load(cache_path, allow_pickle=True)

    t0 = time.time()
    sim_results = run_simulation(
        initial_states_np, shock_sequence_np, vfi_results, econ_params,
    )
    sim_time = time.time() - t0

    sim_stationary = apply_burn_in(sim_results, BURN_IN_PERIODS)
    moments = compute_all_moments(sim_stationary, econ_params)

    logger.info(
        f"    ({n_k},{n_d}): sim={sim_time:.1f}s | "
        f"inv_rate={moments['mean_investment_rate']:.4f}, "
        f"leverage={moments['mean_leverage']:.4f}"
    )

    return {
        'n_k': n_k,
        'n_d': n_d,
        'moments': moments,
        'sim_time': sim_time,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Visualisation — Ensemble Convergence Plot
# ═══════════════════════════════════════════════════════════════════════════

def plot_ensemble_convergence(
    density_levels: List[int],
    ensemble_means: List[Dict[str, float]],
    member_moments: List[List[Dict[str, float]]],
    save_dir: str,
) -> None:
    """
    Plot ensemble-averaged moment convergence with individual member scatter.

    For each of the 15 moments:
      • Blue diamond line = ensemble mean (average of 7 members)
      • Coloured scatter   = individual ensemble member values
      • Green band         = ±2% around the finest-level ensemble mean
    """
    n_members = len(member_moments[0]) if member_moments else NUM_ENSEMBLE_MEMBERS

    fig, axes = plt.subplots(4, 4, figsize=(22, 18))
    axes = axes.flatten()

    cmap = plt.cm.coolwarm
    member_colors = [cmap(i / max(n_members - 1, 1)) for i in range(n_members)]
    member_markers = ['o'] * n_members

    for idx, key in enumerate(MOMENT_KEYS):
        ax = axes[idx]

        ens_vals = np.array([m[key] for m in ensemble_means])
        ax.plot(density_levels, ens_vals, 'D-', color='#2980b9', linewidth=2.5,
                markersize=9, label='Ensemble mean', zorder=5)

        for level_i, N in enumerate(density_levels):
            for mem_j in range(n_members):
                mv = member_moments[level_i][mem_j][key]
                ax.scatter(
                    N, mv, color=member_colors[mem_j],
                    marker=member_markers[mem_j], s=30, alpha=0.6,
                    zorder=4, edgecolors='white', linewidths=0.3,
                )

        member_min = np.array([
            min(member_moments[i][j][key] for j in range(n_members))
            for i in range(len(density_levels))
        ])
        member_max = np.array([
            max(member_moments[i][j][key] for j in range(n_members))
            for i in range(len(density_levels))
        ])
        ax.fill_between(density_levels, member_min, member_max,
                        alpha=0.10, color='#2980b9', zorder=1,
                        label='Ensemble spread')

        finest_val = ens_vals[-1]
        ax.axhline(finest_val, color='#2ecc71', linestyle='-', linewidth=1.5,
                   alpha=0.7, label=f'Finest = {finest_val:.4f}')
        ax.axhspan(finest_val * 0.98, finest_val * 1.02,
                   alpha=0.12, color='#2ecc71', zorder=0, label='±2% band')

        ax.set_xticks(density_levels)
        ax.set_xticklabels(
            [f'{N}\n({N * N // 1000}k pts)' for N in density_levels],
            fontsize=7,
        )
        ax.set_xlabel('Density Level N  (N_k ≈ N_d ≈ N)\n≈ N² total grid points')
        ax.set_ylabel(MOMENT_LABELS[key])
        ax.set_title(MOMENT_LABELS[key], fontweight='bold')
        ax.legend(fontsize=5.5, loc='best')
        ax.grid(True, alpha=0.2)

    for i in range(len(MOMENT_KEYS), len(axes)):
        ax = axes[i]
        ax.axis('off')
        if i == len(MOMENT_KEYS):
            for mem_j in range(n_members):
                ax.scatter([], [], color=member_colors[mem_j],
                           marker=member_markers[mem_j], s=50,
                           label=ENSEMBLE_MEMBER_LABELS[mem_j])
            ax.legend(fontsize=8, loc='center', frameon=True,
                      title='Ensemble Members', title_fontsize=10)

    fig.suptitle(
        f'Ensemble Convergence — {n_members}-Member Average per Density Level\n'
        'Blue = ensemble mean;  coloured dots = individual members;  '
        'Green band = ±2% of finest',
        fontsize=13, fontweight='bold', y=1.03,
    )
    plt.tight_layout()
    path = os.path.join(save_dir, 'ensemble_convergence.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════════════════
#  Convergence Diagnostics
# ═══════════════════════════════════════════════════════════════════════════

def print_convergence_diagnostics(
    density_levels: List[int],
    ensemble_means: List[Dict[str, float]],
    member_moments: List[List[Dict[str, float]]],
) -> None:
    """Print convergence analysis: when does each moment enter the ±2% band?"""
    finest = ensemble_means[-1]

    print("\n" + "=" * 90)
    print("ENSEMBLE CONVERGENCE DIAGNOSTICS")
    print("=" * 90)

    print(f"\n  {'Moment':<25s} {'Finest Ens.':>12s} | ", end="")
    for N in density_levels:
        print(f" N={N:3d}", end="")
    print()
    print("  " + "-" * 25 + " " + "-" * 12 + " +" + "-" * (7 * len(density_levels)))

    for key in MOMENT_KEYS:
        fv = finest[key]
        print(f"  {MOMENT_LABELS[key]:<25s} {fv:>12.6f} | ", end="")
        for i, N in enumerate(density_levels):
            v = ensemble_means[i][key]
            pct = abs(v - fv) / abs(fv) * 100 if abs(fv) > 1e-10 else 0
            marker = "✓" if pct <= 2.0 else "✗"
            print(f" {pct:4.1f}%{marker}", end="")
        print()

    print(f"\n  {'Moment':<25s} {'Converges':>10s} {'Spread at finest':>18s}")
    print("  " + "-" * 55)

    all_converged_at = 0
    for key in MOMENT_KEYS:
        fv = finest[key]
        pcts = [abs(ensemble_means[i][key] - fv) / abs(fv) * 100
                if abs(fv) > 1e-10 else 0
                for i in range(len(density_levels))]

        conv_N = None
        for i in range(len(density_levels)):
            if all(p <= 2.0 for p in pcts[i:]):
                conv_N = density_levels[i]
                break

        n_mem = len(member_moments[-1])
        member_vals = [member_moments[-1][j][key] for j in range(n_mem)]
        spread_pct = (max(member_vals) - min(member_vals)) / abs(fv) * 100 if abs(fv) > 1e-10 else 0

        if conv_N:
            all_converged_at = max(all_converged_at, conv_N)
            print(f"  {MOMENT_LABELS[key]:<25s} N = {conv_N:>4d} {'':>4s} {spread_pct:>14.2f}%")
        else:
            print(f"  {MOMENT_LABELS[key]:<25s} {'NOT conv.':>10s} {spread_pct:>14.2f}%")

    print(f"\n  ► ALL ensemble moments converged at N = {all_converged_at} "
          f"(≈ {all_converged_at**2:,} grid points)")

    print("\n" + "=" * 90)
    print(f"VFI cache:  {os.path.abspath('./ground_truth_risky')}/")
    print(f"Plot:       {os.path.abspath(RESULTS_DIR)}/ensemble_convergence.png")
    print("=" * 90)


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """Run ensemble analysis: load VFI solutions, simulate, and compute moments."""
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    import tensorflow as tf

    # Minimal GPU setup (simulation is CPU-based, but TF moment ops need init)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # ── Pre-flight: verify all cache files ────────────────────────────
    missing = verify_all_cached()
    if missing:
        print(f"\nERROR: {len(missing)} VFI cache files are missing.")
        print("  Run  risky_ensemble_solver.py  first to generate them.\n")
        print("  Missing configs:")
        for n_k, n_d in missing:
            print(f"    ({n_k}, {n_d}) → {cache_path_for(n_k, n_d)}")
        sys.exit(1)

    logger.info(f"All {sum(len(get_ensemble_configs(N)) for N in DENSITY_LEVELS)} "
                f"VFI cache files verified.")
    logger.info(f"Density levels: {DENSITY_LEVELS}")
    logger.info(f"Ensemble: {NUM_ENSEMBLE_MEMBERS} members per level")

    # ── Shared synthetic data ─────────────────────────────────────────
    logger.info("Setting up shared synthetic data ...")
    shared_data = setup_shared_data()

    initial_states_np = shared_data['initial_states_np']
    shock_sequence_np = shared_data['shock_sequence_np']
    econ_params = shared_data['econ_params']

    # ── Run analysis per density level ────────────────────────────────
    ensemble_means: List[Dict[str, float]] = []
    member_moments: List[List[Dict[str, float]]] = []

    total_t0 = time.time()

    for level_idx, N in enumerate(DENSITY_LEVELS):
        configs = get_ensemble_configs(N)
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Density Level {N}  (≈ {N*N:,} grid points)")
        logger.info(f"  Ensemble: {configs}")
        logger.info('=' * 60)

        level_moments: List[Dict[str, float]] = []
        for mem_idx, (n_k, n_d) in enumerate(configs):
            tag = ENSEMBLE_MEMBER_LABELS[mem_idx]
            logger.info(f"  Member {mem_idx+1}/{len(configs)}: ({n_k},{n_d})  [{tag}]")
            result = run_single_analysis(
                n_k, n_d,
                initial_states_np, shock_sequence_np,
                econ_params,
            )
            level_moments.append(result['moments'])

        ens_mean = average_moments(level_moments)
        ensemble_means.append(ens_mean)
        member_moments.append(level_moments)

        logger.info(f"  Ensemble mean: inv_rate={ens_mean['mean_investment_rate']:.4f}, "
                    f"leverage={ens_mean['mean_leverage']:.4f}")

    total_time = time.time() - total_t0
    logger.info(f"\nTotal analysis time: {total_time:.1f}s ({total_time / 60:.1f} min)")

    # ── Visualisation ─────────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"\nGenerating visualisation ...\n")
    plot_ensemble_convergence(
        DENSITY_LEVELS, ensemble_means, member_moments, RESULTS_DIR,
    )

    # ── Diagnostics ───────────────────────────────────────────────────
    print_convergence_diagnostics(
        DENSITY_LEVELS, ensemble_means, member_moments,
    )


if __name__ == "__main__":
    main()
