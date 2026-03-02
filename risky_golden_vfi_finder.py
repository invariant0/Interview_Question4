# risky_golden_vfi_finder.py
"""
Convergence testing for VFI solutions — Risky Debt Model (Multi-GPU).

Tests moment convergence as grid resolution increases,
using synthetic_data_generator for consistent comparisons.

Multi-GPU parallel scheduling:
  • VFI solves are distributed across all available GPUs
  • Each GPU gets its own worker process (via multiprocessing)
  • Jobs are sorted longest-first (by n_k × n_d descending)
  • A shared work queue feeds jobs to GPU workers on demand
  • Simulation + moments computed sequentially in main process

Usage
─────
  python risky_golden_vfi_finder.py              # solve & analyse
  python risky_golden_vfi_finder.py --gpus 0,1   # use only GPUs 0 and 1
  python risky_golden_vfi_finder.py --dry-run    # show job plan only
  python risky_golden_vfi_finder.py --force      # re-solve even if cached
"""

import argparse
import dataclasses
import logging
import multiprocessing as mp
import os
import sys
import time
from typing import Dict, Any, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

# NOTE: TF-dependent imports are deferred to after GPU configuration
# in worker processes.  Main process imports only for simulation/moments.
from src.econ_models.config.economic_params import EconomicParams
from src.econ_models.config.bond_config import BondsConfig
from src.econ_models.io.file_utils import load_json_file
from src.econ_models.vfi.grids.grid_utils import compute_optimal_chunks
import tensorflow as tf
from src.econ_models.moment_calculator.compute_derived_quantities import compute_all_derived_quantities
from src.econ_models.moment_calculator.compute_mean import compute_global_mean
from src.econ_models.moment_calculator.compute_std import compute_global_std
from src.econ_models.moment_calculator.compute_autocorrelation import compute_autocorrelation_lags_1_to_5
from src.econ_models.moment_calculator.compute_inaction_rate import compute_inaction_rate

from risky_common import (
    DEFAULT_ECON_LIST,
    BASE_DIR,
    N_PRODUCTIVITY,
    econ_tag,
    get_econ_params_path_by_list,
    get_bounds_path_by_list,
    get_vfi_cache_path,
    to_python_float,
    apply_burn_in,
)


# ── Path Configuration ───────────────────────────────────────────────────
econ_list = DEFAULT_ECON_LIST
CONFIG_PARAMS_FILE = os.path.join(BASE_DIR, "hyperparam/prefixed/vfi_params.json")

RESULTS_DIR = './results/golden_vfi_risky'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(process)d] %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def cache_path_for(n_k: int, n_d: int, econ_id: int) -> str:
    """Return the ground-truth cache path for a given (N_k, N_d) config."""
    tag = econ_tag(econ_list[econ_id])
    return get_vfi_cache_path(tag, n_k, n_d)


# ═══════════════════════════════════════════════════════════════════════════
#  Data Setup
# ═══════════════════════════════════════════════════════════════════════════

def setup_shared_data(econ_id: int) -> Dict[str, Any]:
    """
    Create shared synthetic data for all convergence tests.

    Returns NumPy arrays for consistent cross-resolution comparison.
    Data generation runs on CPU to avoid sm_120 eager-op crashes
    (tfp.distributions.gamma uses non-XLA kernels).
    """
    import tensorflow as tf

    from src.econ_models.simulator.synthetic_data_gen import synthetic_data_generator

    econ_params_file = get_econ_params_path_by_list(econ_list[econ_id])
    boundary_file = get_bounds_path_by_list(econ_list[econ_id])

    BOUNDS = load_json_file(boundary_file)
    ECON_PARAMS = EconomicParams(**load_json_file(econ_params_file))

    sample_bonds_config = BondsConfig.validate_and_load(
        bounds_file=boundary_file,
        current_params=ECON_PARAMS
    )

    data_gen = synthetic_data_generator(
        econ_params_benchmark=ECON_PARAMS,
        sample_bonds_config=sample_bonds_config,
        batch_size=10000,
        T_periods=1000,
        include_debt=True,
    )

    # Force data generation on CPU to avoid sm_120 eager-op PTX crashes
    # (tfp.distributions.gamma uses non-XLA CUDA kernels)
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
#  GPU Worker Process (VFI Solve Only)
# ═══════════════════════════════════════════════════════════════════════════

def _gpu_worker(
    gpu_id: int,
    econ_id: int,
    job_queue: mp.Queue,
    result_queue: mp.Queue,
) -> None:
    """
    Worker process pinned to a single GPU.

    Pulls (n_k, n_d) jobs from the shared queue until a sentinel (None)
    is received, solves VFI, and saves to cache.
    """
    # Pin this process to exactly one GPU BEFORE importing TensorFlow
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    import tensorflow as tf
    from src.econ_models.config.economic_params import EconomicParams
    from src.econ_models.config.vfi_config import load_grid_config
    from src.econ_models.io.file_utils import load_json_file
    from src.econ_models.io.artifacts import save_vfi_results
    from src.econ_models.vfi.risky import RiskyDebtModelVFI
    from src.econ_models.vfi.grids.grid_utils import compute_optimal_chunks

    # Configure the (now sole visible) GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
        logger.info(f"Worker GPU-{gpu_id}: using {gpus[0].name}")
    else:
        logger.warning(f"Worker GPU-{gpu_id}: no GPU visible — using CPU")

    # Load economic parameters once per worker
    econ_params_file = get_econ_params_path_by_list(econ_list[econ_id])
    boundary_file = get_bounds_path_by_list(econ_list[econ_id])
    
    bounds = load_json_file(boundary_file)
    econ_params = EconomicParams(**load_json_file(econ_params_file))

    while True:
        job = job_queue.get()
        if job is None:
            logger.info(f"Worker GPU-{gpu_id}: received shutdown signal")
            break

        n_k, n_d = job
        cp = cache_path_for(n_k, n_d, econ_id)

        # Double-check: another worker may have solved it in the meantime
        if os.path.exists(cp):
            logger.info(f"Worker GPU-{gpu_id}: ({n_k},{n_d}) already cached, skipping")
            result_queue.put((gpu_id, n_k, n_d, 0.0, 'cached'))
            continue

        logger.info(f"Worker GPU-{gpu_id}: solving ({n_k},{n_d}) ...")
        t0 = time.time()

        try:
            k_chunk, b_chunk, kp_chunk, bp_chunk = compute_optimal_chunks(
                n_k, n_d, n_z=N_PRODUCTIVITY,
            )

            config = load_grid_config(CONFIG_PARAMS_FILE, 'risky')
            config = dataclasses.replace(
                config,
                n_capital=n_k,
                n_productivity=N_PRODUCTIVITY,
                n_debt=n_d,
            )

            model = RiskyDebtModelVFI(
                econ_params,
                config,
                k_bounds=(bounds['bounds']['k_min'], bounds['bounds']['k_max']),
                b_bounds=(bounds['bounds']['b_min'], bounds['bounds']['b_max']),
                k_chunk_size=k_chunk,
                b_chunk_size=b_chunk,
                kp_chunk_size=kp_chunk,
                bp_chunk_size=bp_chunk,
            )
            vfi_results = model.solve()

            os.makedirs(os.path.dirname(cp), exist_ok=True)
            save_vfi_results(vfi_results, cp)

            elapsed = time.time() - t0
            logger.info(
                f"Worker GPU-{gpu_id}: ({n_k},{n_d}) done in {elapsed:.1f}s "
                f"→ {cp}"
            )
            result_queue.put((gpu_id, n_k, n_d, elapsed, 'solved'))

            # Clear TF session state between solves to reclaim VRAM
            tf.keras.backend.clear_session()

        except Exception as e:
            elapsed = time.time() - t0
            logger.error(
                f"Worker GPU-{gpu_id}: ({n_k},{n_d}) FAILED after {elapsed:.1f}s — {e}"
            )
            result_queue.put((gpu_id, n_k, n_d, elapsed, f'error: {e}'))


# ═══════════════════════════════════════════════════════════════════════════
#  Parallel VFI Scheduler
# ═══════════════════════════════════════════════════════════════════════════

def run_parallel_solver(
    gpu_ids: List[int],
    econ_id: int,
    jobs: List[Tuple[int, int]],
) -> None:
    """
    Distribute VFI jobs across GPU workers using longest-job-first scheduling.

    Workers pull from a shared queue, so faster GPUs automatically pick up
    more work — no manual balancing needed.
    """
    n_gpus = len(gpu_ids)
    n_jobs = len(jobs)

    logger.info(f"\n{'=' * 70}")
    logger.info(f"GOLDEN VFI SOLVER — {n_gpus} GPUs × {n_jobs} jobs")
    logger.info(f"GPUs: {gpu_ids}")
    logger.info(f"{'=' * 70}")

    if n_jobs == 0:
        logger.info("All jobs already cached — nothing to solve.")
        return

    # Print job plan
    logger.info(f"\nJob queue (longest-first):")
    for i, (nk, nd) in enumerate(jobs):
        logger.info(f"  [{i+1:2d}] ({nk:4d}, {nd:4d})  ≈ {nk*nd:>8,} grid pts")

    # Shared queues
    job_queue: mp.Queue = mp.Queue()
    result_queue: mp.Queue = mp.Queue()

    # Enqueue all jobs
    for job in jobs:
        job_queue.put(job)

    # Enqueue sentinels (one per worker)
    for _ in gpu_ids:
        job_queue.put(None)

    # Spawn workers
    workers = []
    for gid in gpu_ids:
        p = mp.Process(target=_gpu_worker, args=(gid, econ_id, job_queue, result_queue))
        p.start()
        workers.append(p)
        logger.info(f"  Started worker PID={p.pid} → GPU {gid}")

    # Collect results
    t_start = time.time()
    completed = 0
    gpu_times: Dict[int, float] = {gid: 0.0 for gid in gpu_ids}
    gpu_counts: Dict[int, int] = {gid: 0 for gid in gpu_ids}

    while completed < n_jobs:
        gid, nk, nd, elapsed, status = result_queue.get()
        completed += 1
        gpu_times[gid] += elapsed
        gpu_counts[gid] += 1
        logger.info(
            f"  [{completed}/{n_jobs}] GPU-{gid}: ({nk},{nd}) → {status} "
            f"({elapsed:.1f}s)"
        )

    # Wait for workers to exit
    for p in workers:
        p.join(timeout=30)
        if p.is_alive():
            logger.warning(f"  Worker PID={p.pid} did not exit cleanly, terminating")
            p.terminate()

    total_wall = time.time() - t_start
    total_compute = sum(gpu_times.values())

    # Print summary
    print(f"\n{'=' * 70}")
    print(f"SOLVER COMPLETE — {n_jobs} jobs in {total_wall:.1f}s wall time")
    print(f"  Total compute: {total_compute:.1f}s across {n_gpus} GPUs")
    if total_wall > 0:
        print(f"  Speedup:       {total_compute / total_wall:.2f}× vs sequential")
    print(f"\n  Per-GPU breakdown:")
    for gid in gpu_ids:
        print(f"    GPU {gid}: {gpu_counts[gid]:2d} jobs, "
              f"{gpu_times[gid]:.1f}s total compute")
    print(f"{'=' * 70}")


# ═══════════════════════════════════════════════════════════════════════════
#  Simulation (runs in main process)
# ═══════════════════════════════════════════════════════════════════════════

def run_simulation(
    initial_states_np: Tuple[np.ndarray, ...],
    shock_sequence_np: np.ndarray,
    vfi_results: Dict,
    econ_params: EconomicParams
) -> Dict[str, np.ndarray]:
    """Run simulation using VFI simulator.

    Uses CPU NumPy-based simulation.
    """
    from src.econ_models.simulator.vfi.risky import VFISimulator_risky

    simulator = VFISimulator_risky(econ_params)
    simulator.load_solved_vfi_solution(vfi_results)

    return simulator.simulate(initial_states_np, shock_sequence_np)


# ═══════════════════════════════════════════════════════════════════════════
#  Post-processing helpers
# ═══════════════════════════════════════════════════════════════════════════


def compute_all_moments(simulation_results: Dict[str, np.ndarray], econ_params: EconomicParams) -> Dict[str, float]:
    """
    Compute all specified moments from simulation results using standard calculators.
    """
    with tf.device('/CPU:0'):
        # 1. Derived quantities (include debt-related: leverage, issuance, etc.)
        derived = compute_all_derived_quantities(
            simulation_results,
            econ_params.depreciation_rate,
            econ_params.capital_share,
            include_debt=True,
        )

        leverage = derived['leverage']
        
        # 2. Compute moments
        moments = {
            # Row 1 — Means
            'mean_K': to_python_float(compute_global_mean(derived['capital'])),
            'mean_B': to_python_float(compute_global_mean(simulation_results['B_curr'])),
            'mean_investment_rate': to_python_float(compute_global_mean(derived['investment_rate'])),
            'mean_leverage': to_python_float(compute_global_mean(leverage)),
            
            # Row 2 — Standard deviations
            'std_K': to_python_float(compute_global_std(derived['capital'])),
            'std_B': to_python_float(compute_global_std(simulation_results['B_curr'])),
            'std_output': to_python_float(compute_global_std(derived['output'])),
            'std_investment_rate': to_python_float(compute_global_std(derived['investment_rate'])),
            'std_leverage': to_python_float(compute_global_std(leverage)),
            
            # Row 3 — Autocorrelations
            'autocorr_investment_rate': to_python_float(compute_autocorrelation_lags_1_to_5(
                derived['investment_rate']
            )['lag_1']),
            'autocorr_leverage': to_python_float(compute_autocorrelation_lags_1_to_5(
                leverage
            )['lag_1']),
            'autocorr_output': to_python_float(compute_autocorrelation_lags_1_to_5(
                derived['output']
            )['lag_1']),
            'autocorr_issuance': to_python_float(compute_autocorrelation_lags_1_to_5(
                derived['equity_issuance_rate']
            )['lag_1']),
            
            # Row 4 — Frequencies & inaction
            'issuance_freq': to_python_float(compute_global_mean(derived['issuance_binary'])),
            'investment_freq': to_python_float(compute_global_mean(derived['investment_binary'])),
            'inaction_rate': to_python_float(compute_inaction_rate(
                derived['investment_rate'], -0.001, 0.001
            )),
        }
    return moments


def compute_percent_changes(
    previous_moments: Dict[str, float],
    current_moments: Dict[str, float]
) -> Dict[str, float]:
    """Calculate percent change between moment sets."""
    changes = {}
    for key in previous_moments.keys():
        prev = previous_moments[key]
        curr = current_moments[key]
        if np.isnan(prev) or np.isnan(curr):
            changes[key] = 0.0
            continue
        if abs(prev) > 1e-10:
            changes[key] = abs((curr - prev) / prev) * 100
        else:
            changes[key] = 0.0 if abs(curr) < 1e-10 else float('inf')
    return changes


# ═══════════════════════════════════════════════════════════════════════════
#  Plotting
# ═══════════════════════════════════════════════════════════════════════════

MOMENT_KEYS = [
    'mean_K', 'mean_B', 'mean_leverage', 'mean_investment_rate',
    'std_K', 'std_B', 'std_leverage', 'std_output',
    'autocorr_investment_rate', 'autocorr_leverage', 'autocorr_output', 'autocorr_issuance',
    'issuance_freq', 'investment_freq', 'inaction_rate',
]

MOMENT_LABELS = [
    'Mean Capital', 'Mean Debt', 'Mean Leverage', 'Mean Investment Rate',
    'Std Capital', 'Std Debt', 'Std Leverage', 'Std Output',
    'Autocorr(Inv Rate)', 'Autocorr(Leverage)', 'Autocorr(Output)', 'Autocorr(Eq. Issuance)',
    'Eq. Issuance Freq.', 'Investment Freq.', 'Inaction Rate',
]


def plot_moment_convergence(
    all_moments_list: List[Dict[str, float]],
    n_capital_list: List[int],
    n_debt_list: List[int],
    save_path: str
) -> None:
    """Plot moment values vs grid size to visualize convergence."""
    x_labels = [f'({nk}, {nd})' for nk, nd in zip(n_capital_list, n_debt_list)]
    x_positions = list(range(len(x_labels)))

    fig, axes = plt.subplots(4, 4, figsize=(20, 16))
    axes = axes.flatten()

    # Window size for moving average (adaptive to number of data points)
    ma_window = max(3, len(x_positions) // 5)

    for idx, (key, label) in enumerate(zip(MOMENT_KEYS, MOMENT_LABELS)):
        ax = axes[idx]
        values = [m[key] for m in all_moments_list]

        ax.plot(x_positions, values, 'o-', linewidth=2, markersize=8, color='#1f77b4', label='Raw')

        # Moving average line
        if len(values) >= ma_window:
            ma = np.convolve(values, np.ones(ma_window) / ma_window, mode='valid')
            ma_x = x_positions[ma_window - 1:]
            ax.plot(ma_x, ma, '-', linewidth=2.5, color='#e67e22', alpha=0.85,
                    label=f'MA({ma_window})')
            ax.legend(fontsize=7, loc='best')

        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, fontsize=8, rotation=45, ha='right')
        ax.set_xlabel('(N Capital, N Debt)', fontsize=10)
        ax.set_ylabel('Value', fontsize=10)
        ax.set_title(label, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)

        final_val = values[-1]
        if abs(final_val) > 1e-10:
            ax.axhline(final_val * 0.98, color='red', linestyle='--', alpha=0.5, linewidth=1)
            ax.axhline(final_val * 1.02, color='red', linestyle='--', alpha=0.5, linewidth=1)
            ax.fill_between(
                x_positions,
                final_val * 0.98,
                final_val * 1.02,
                alpha=0.1,
                color='red'
            )

    for i in range(len(MOMENT_KEYS), len(axes)):
        axes[i].axis('off')

    plt.suptitle(
        'Moment Convergence vs Grid Size (N Capital, N Debt)',
        fontsize=14, fontweight='bold', y=1.02
    )
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info(f"Moment convergence plot saved to {save_path}")
    plt.close()


def plot_moment_changes(
    changes_list: List[Dict[str, float]],
    n_capital_list: List[int],
    n_debt_list: List[int],
    save_path: str
) -> None:
    """Plot percent changes in moments between consecutive grid sizes."""
    x_labels = [
        f'({n_capital_list[i]},{n_debt_list[i]})\u2192({n_capital_list[i+1]},{n_debt_list[i+1]})'
        for i in range(len(n_capital_list) - 1)
    ]

    fig, axes = plt.subplots(4, 4, figsize=(20, 16))
    axes = axes.flatten()

    for idx, (key, label) in enumerate(zip(MOMENT_KEYS, MOMENT_LABELS)):
        ax = axes[idx]
        values = [c[key] for c in changes_list]
        values_capped = [min(v, 100) if not np.isinf(v) else 100 for v in values]
        colors = ['#2ecc71' if v < 2.0 else '#e74c3c' for v in values]

        bars = ax.bar(
            range(len(x_labels)), values_capped,
            color=colors, edgecolor='black', linewidth=0.5
        )

        ax.axhline(2.0, color='black', linestyle='--', linewidth=1.5, label='2% threshold')
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, fontsize=7, rotation=45, ha='right')
        ax.set_xlabel('Grid Size Transition', fontsize=10)
        ax.set_ylabel('% Change', fontsize=10)
        ax.set_title(label, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        for bar, val in zip(bars, values):
            height = bar.get_height()
            display_val = f'{val:.1f}%' if not np.isinf(val) else 'inf'
            ax.annotate(
                display_val,
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=8
            )

    for i in range(len(MOMENT_KEYS), len(axes)):
        axes[i].axis('off')

    plt.suptitle(
        'Moment % Change Between Grid Sizes',
        fontsize=14, fontweight='bold', y=1.02
    )
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info(f"Moment changes plot saved to {save_path}")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════
#  GPU Detection & CLI
# ═══════════════════════════════════════════════════════════════════════════

def detect_gpus() -> List[int]:
    """Detect available GPU IDs without initialising TensorFlow in main process."""
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--list-gpus'],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            lines = [l for l in result.stdout.strip().split('\n') if l.strip()]
            gpu_ids = list(range(len(lines)))
            for line in lines:
                logger.info(f"  Detected: {line.strip()}")
            return gpu_ids
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Fallback: import TF (will be re-imported in workers anyway)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    return list(range(len(gpus)))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the golden VFI convergence finder."""
    parser = argparse.ArgumentParser(
        description='Golden VFI convergence finder — multi-GPU parallel solver',
    )
    parser.add_argument(
        '--gpus', type=str, default=None,
        help='Comma-separated GPU IDs to use (default: auto-detect all)',
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Show job plan without solving or simulating',
    )
    parser.add_argument(
        '--force', action='store_true',
        help='Re-solve even if cached .npz exists',
    )
    parser.add_argument(
        '--econ_id', type=int, default=1,
        help='Economic configuration ID (default: 1)',
    )
    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """Find golden VFI solutions via multi-GPU parallel convergence search."""
    mp.set_start_method('spawn', force=True)
    args = parse_args()

    econ_id = args.econ_id

    # ══════════════════════════════════════════════════════════════════
    #  CONFIGURATION — edit these lists to change the convergence test
    # ══════════════════════════════════════════════════════════════════
    n_capital_list = [100, 150, 200, 250, 300, 350, 400, 450, 500]
    n_debt_list    = [100, 150, 200, 250, 300, 350, 400, 450, 500]

    # n_capital_list = [100, 560]
    # n_debt_list    = [100, 560]

    n_productivity = N_PRODUCTIVITY
    burn_in_periods = 400
    # ══════════════════════════════════════════════════════════════════

    # -- Determine GPU set ---------------------------------------------
    if args.gpus is not None:
        gpu_ids = [int(g.strip()) for g in args.gpus.split(',')]
    else:
        gpu_ids = detect_gpus()

    if not gpu_ids:
        logger.error("No GPUs detected. Cannot run parallel solver.")
        sys.exit(1)

    logger.info(f"Using GPUs: {gpu_ids}")
    logger.info(f"Grid sequence: {list(zip(n_capital_list, n_debt_list))}")
    logger.info(f"n_productivity: {n_productivity}, burn_in: {burn_in_periods}")

    # -- Collect VFI solve jobs ----------------------------------------
    all_configs = list(zip(n_capital_list, n_debt_list))
    if args.force:
        solve_jobs = list(all_configs)
    else:
        solve_jobs = [(nk, nd) for nk, nd in all_configs
                      if not os.path.exists(cache_path_for(nk, nd, econ_id))]

    # Sort longest-first for optimal GPU scheduling
    solve_jobs.sort(key=lambda x: x[0] * x[1], reverse=True)

    print(f"\n  Total grid configs: {len(all_configs)}")
    print(f"  Already cached:     {len(all_configs) - len(solve_jobs)}")
    print(f"  To solve:           {len(solve_jobs)}")
    print(f"  GPUs available:     {len(gpu_ids)}")

    if args.dry_run:
        print(f"\n  Job plan (longest-first):")
        for i, (nk, nd) in enumerate(solve_jobs):
            print(f"    [{i+1:2d}] ({nk:4d}, {nd:4d})  ≈ {nk*nd:>8,} grid pts")
        print(f"\n  [DRY RUN — no solves or simulations performed]")
        return

    # ══════════════════════════════════════════════════════════════════
    #  PHASE 1: Parallel VFI Solve across GPUs
    # ══════════════════════════════════════════════════════════════════
    if solve_jobs:
        run_parallel_solver(gpu_ids, econ_id, solve_jobs)
    else:
        logger.info("All VFI solutions already cached — skipping solve phase.")

    # Verify all expected cache files exist
    missing = [(nk, nd) for nk, nd in all_configs
               if not os.path.exists(cache_path_for(nk, nd, econ_id))]
    if missing:
        logger.error(f"{len(missing)} cache files still missing after solve phase:")
        for nk, nd in missing:
            logger.error(f"  ({nk}, {nd}) → {cache_path_for(nk, nd, econ_id)}")
        logger.error("Cannot proceed with simulation — aborting.")
        sys.exit(1)

    # ══════════════════════════════════════════════════════════════════
    #  PHASE 2: Sequential Simulation + Moments (main process)
    # ══════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 2: Sequential Simulation + Moments")
    logger.info("=" * 70)

    # Configure main-process GPU for moment computation (CPU-only)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    gpus_main = tf.config.list_physical_devices('GPU')
    if gpus_main:
        for gpu in gpus_main:
            tf.config.experimental.set_memory_growth(gpu, True)

    # Setup shared synthetic data
    logger.info("Setting up shared synthetic data ...")
    shared_data = setup_shared_data(econ_id)

    initial_states_np = shared_data['initial_states_np']
    shock_sequence_np = shared_data['shock_sequence_np']
    econ_params = shared_data['econ_params']

    # Run simulation + moments for each grid config (in order)
    all_moments_list = []
    results_ordered = []
    total_t0 = time.time()

    for idx, (n_k, n_d) in enumerate(zip(n_capital_list, n_debt_list)):
        logger.info(f"\n{'='*60}")
        logger.info(f"Sim {idx+1}/{len(n_capital_list)}: (n_k, n_d) = ({n_k}, {n_d})")
        logger.info('='*60)

        # Load cached VFI
        cp = cache_path_for(n_k, n_d, econ_id)
        vfi_results = np.load(cp, allow_pickle=True)

        # Simulate
        t1 = time.time()
        sim_results = run_simulation(
            initial_states_np,
            shock_sequence_np,
            vfi_results,
            econ_params,
        )
        sim_time = time.time() - t1

        sim_stationary = apply_burn_in(sim_results, burn_in_periods)

        # Compute moments
        moments = compute_all_moments(sim_stationary, econ_params)

        logger.info(
            f"  Done ({n_k},{n_d}): sim={sim_time:.1f}s | "
            f"inv_rate={moments['mean_investment_rate']:.4f}, "
            f"leverage={moments['mean_leverage']:.4f}"
        )

        all_moments_list.append(moments)
        results_ordered.append({
            'job_idx': idx,
            'n_k': n_k,
            'n_d': n_d,
            'moments': moments,
            'sim_time': sim_time,
        })

    total_time = time.time() - total_t0
    logger.info(f"\nTotal simulation wall time: {total_time:.1f}s ({total_time/60:.1f} min)")

    # -- Log all moments -----------------------------------------------
    for idx, (n_k, n_d) in enumerate(zip(n_capital_list, n_debt_list)):
        moments = all_moments_list[idx]
        logger.info(f"\n{'='*50}")
        logger.info(f"Results for (n_k, n_d) = ({n_k}, {n_d})")
        logger.info(f"  sim_time={results_ordered[idx]['sim_time']:.1f}s")
        logger.info('='*50)
        for key, val in moments.items():
            logger.info(f"  {key}: {val:.6f}")

    # -- Compute sequential changes ------------------------------------
    changes_list = []
    for i in range(1, len(all_moments_list)):
        changes = compute_percent_changes(all_moments_list[i-1], all_moments_list[i])
        changes_list.append(changes)
        max_change = max(v for v in changes.values() if not np.isinf(v))
        logger.info(
            f"  Max % change ({n_capital_list[i-1]},{n_debt_list[i-1]}) -> "
            f"({n_capital_list[i]},{n_debt_list[i]}): {max_change:.2f}%"
        )

    # -- Save canonical copy of final VFI ------------------------------
    last_cache = cache_path_for(n_capital_list[-1], n_debt_list[-1], econ_id)
    canonical_path = (
        f'./ground_truth_risky/golden_vfi_risky_'
        f'{econ_tag(econ_list[econ_id])}.npz'
    )
    if os.path.exists(last_cache):
        import shutil
        shutil.copy2(last_cache, canonical_path)
        logger.info(f"Copied final VFI results to {canonical_path}")

    # -- Generate plots ------------------------------------------------
    tag = econ_tag(econ_list[econ_id])
    save_dir = os.path.join(RESULTS_DIR, tag)

    plot_moment_convergence(
        all_moments_list,
        n_capital_list,
        n_debt_list,
        save_path=os.path.join(save_dir, 'moment_convergence.png')
    )

    if changes_list:
        plot_moment_changes(
            changes_list,
            n_capital_list,
            n_debt_list,
            save_path=os.path.join(save_dir, 'moment_changes_vs_grid_size.png')
        )

    # -- Convergence summary -------------------------------------------
    logger.info("\n" + "="*60)
    logger.info("CONVERGENCE SUMMARY")
    logger.info("="*60)

    if len(all_moments_list) >= 2:
        final_changes = compute_percent_changes(
            all_moments_list[-2], all_moments_list[-1]
        )
        logger.info(
            f"\nChanges between (n_k,n_d)=({n_capital_list[-2]},{n_debt_list[-2]}) "
            f"and ({n_capital_list[-1]},{n_debt_list[-1]}):"
        )
        converged_count = 0
        for key, val in final_changes.items():
            if np.isinf(val):
                status = "\u2717"
                val_str = "inf"
            else:
                converged = val < 2.0
                status = "\u2713" if converged else "\u2717"
                val_str = f"{val:.2f}"
                if converged:
                    converged_count += 1
            logger.info(f"  {status} {key}: {val_str}%")

        logger.info(f"\n  {converged_count}/{len(final_changes)} moments converged (< 2% change)")

    # -- Save moments table --------------------------------------------
    table_path = os.path.join(save_dir, 'moments_table.txt')
    os.makedirs(save_dir, exist_ok=True)
    with open(table_path, 'w') as f:
        header = f"{'Grid':>12s}"
        for key in MOMENT_KEYS:
            header += f"  {key:>22s}"
        f.write(header + '\n')
        f.write('-' * len(header) + '\n')
        for idx, (n_k, n_d) in enumerate(zip(n_capital_list, n_debt_list)):
            row = f"({n_k:3d},{n_d:3d})"
            row = f"{row:>12s}"
            for key in MOMENT_KEYS:
                row += f"  {all_moments_list[idx][key]:>22.6f}"
            f.write(row + '\n')
    logger.info(f"Moments table saved to {table_path}")


if __name__ == "__main__":
    main()
