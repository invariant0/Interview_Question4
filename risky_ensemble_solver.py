#!/usr/bin/env python3
"""
Ensemble VFI Solver — 4-GPU Parallel Scheduler (Risky Debt Model)
═════════════════════════════════════════════════════════════════

Solves all VFI grid configurations required by the ensemble convergence
analysis, distributing jobs across **4 GPUs** for maximum throughput.

Job scheduling strategy:
  • Each GPU gets its own worker process (via multiprocessing)
  • Jobs are sorted longest-first (by n_k × n_d descending) so the
    heaviest work starts immediately — avoids GPU idle tails
  • A shared work queue feeds jobs to GPU workers on demand
  • Each worker pins itself to one GPU via CUDA_VISIBLE_DEVICES
  • Already-cached solutions are skipped automatically

Grid configurations (7 per density level × 5 levels = 35 jobs):
  Density N ∈ {100, 200, 300, 400, 500}
  Per level: (N±30), (N±20), (N±10), (N,N)

Additionally solves any DOWNSTREAM_ANALYSIS configs (e.g., (500,500))
that aren't already part of the ensemble.

Usage
─────
  python risky_ensemble_solver.py              # solve all missing
  python risky_ensemble_solver.py --dry-run    # show job plan only
  python risky_ensemble_solver.py --gpus 0,1   # use only GPUs 0 and 1

Outputs
───────
  ground_truth_risky/golden_vfi_risky_*.npz   (one per grid config)
"""

import argparse
import dataclasses
import logging
import math
import multiprocessing as mp
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np

# ── Logging ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [GPU-%(process)d] %(levelname)s - %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  Shared Configuration (must match risky_ensemble_analysis.py)
# ═══════════════════════════════════════════════════════════════════════════

econ_list = [
    [0.6, 0.17, 1.0, 0.02, 0.1, 0.08],
    [0.5, 0.23, 1.5, 0.01, 0.1, 0.1],
]
econ_id = 0

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname('./')))
CONFIG_PARAMS_FILE = os.path.join(BASE_DIR, "hyperparam/prefixed/vfi_params.json")
ECON_PARAMS_FILE_RISKY = os.path.join(
    BASE_DIR,
    f"hyperparam/prefixed/econ_params_risky_"
    f"{econ_list[econ_id][0]}_{econ_list[econ_id][1]}_"
    f"{econ_list[econ_id][2]}_{econ_list[econ_id][3]}_"
    f"{econ_list[econ_id][4]}_{econ_list[econ_id][5]}.json"
)
BOUNDARY_RISKY = os.path.join(
    BASE_DIR,
    f"hyperparam/autogen/bounds_risky_"
    f"{econ_list[econ_id][0]}_{econ_list[econ_id][1]}_"
    f"{econ_list[econ_id][2]}_{econ_list[econ_id][3]}_"
    f"{econ_list[econ_id][4]}_{econ_list[econ_id][5]}.json"
)

N_PRODUCTIVITY = 12

DENSITY_LEVELS = [360, 380, 400, 420, 440, 460, 480, 500, 520, 540, 560]

DOWNSTREAM_ANALYSIS_HIGHEST_CONFIGS = [
    (500, 500),
]


def get_ensemble_configs(N: int) -> List[Tuple[int, int]]:
    """Return the 7 (N_k, N_d) configurations for density level N."""
    return [
        (N - 30, N - 30),
        (N - 20, N - 20),
        (N - 10, N - 10),
        (N,      N),
    ]



def cache_path_for(n_k: int, n_d: int) -> str:
    """Return the ground-truth cache path for a given (N_k, N_d) config."""
    return (
        f'./ground_truth_risky/golden_vfi_risky_'
        f'{econ_list[econ_id][0]}_{econ_list[econ_id][1]}_'
        f'{econ_list[econ_id][2]}_{econ_list[econ_id][3]}_'
        f'{econ_list[econ_id][4]}_{econ_list[econ_id][5]}_'
        f'{n_k}_{n_d}.npz'
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Job Collection
# ═══════════════════════════════════════════════════════════════════════════

def collect_all_jobs(skip_cached: bool = True) -> List[Tuple[int, int]]:
    """
    Gather every (n_k, n_d) that needs solving.  De-duplicated, optionally
    skipping configs that already have a cached .npz.

    Returns jobs sorted by descending workload (n_k × n_d) for
    longest-job-first scheduling.
    """
    seen = set()
    jobs: List[Tuple[int, int]] = []

    for N in DENSITY_LEVELS:
        for cfg in get_ensemble_configs(N):
            if cfg not in seen:
                seen.add(cfg)
                jobs.append(cfg)

    for cfg in DOWNSTREAM_ANALYSIS_HIGHEST_CONFIGS:
        if cfg not in seen:
            seen.add(cfg)
            jobs.append(cfg)

    if skip_cached:
        jobs = [(nk, nd) for nk, nd in jobs
                if not os.path.exists(cache_path_for(nk, nd))]

    # Longest-job-first: sort by descending grid size
    jobs.sort(key=lambda x: x[0] * x[1], reverse=True)
    return jobs


# ═══════════════════════════════════════════════════════════════════════════
#  GPU Worker Process
# ═══════════════════════════════════════════════════════════════════════════

def _gpu_worker(
    gpu_id: int,
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
    bounds = load_json_file(BOUNDARY_RISKY)
    econ_params = EconomicParams(**load_json_file(ECON_PARAMS_FILE_RISKY))

    while True:
        job = job_queue.get()
        if job is None:
            logger.info(f"Worker GPU-{gpu_id}: received shutdown signal")
            break

        n_k, n_d = job
        cache_path = cache_path_for(n_k, n_d)

        # Double-check: another worker may have solved it in the meantime
        if os.path.exists(cache_path):
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

            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            save_vfi_results(vfi_results, cache_path)

            elapsed = time.time() - t0
            logger.info(
                f"Worker GPU-{gpu_id}: ({n_k},{n_d}) done in {elapsed:.1f}s "
                f"→ {cache_path}"
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
#  Scheduler
# ═══════════════════════════════════════════════════════════════════════════

def run_parallel_solver(gpu_ids: List[int], jobs: List[Tuple[int, int]]) -> None:
    """
    Distribute VFI jobs across GPU workers using longest-job-first scheduling.

    Workers pull from a shared queue, so faster GPUs automatically pick up
    more work — no manual balancing needed.
    """
    n_gpus = len(gpu_ids)
    n_jobs = len(jobs)

    logger.info(f"\n{'=' * 70}")
    logger.info(f"ENSEMBLE VFI SOLVER — {n_gpus} GPUs × {n_jobs} jobs")
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
        p = mp.Process(target=_gpu_worker, args=(gid, job_queue, result_queue))
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
    print(f"  Speedup:       {total_compute / total_wall:.2f}× vs sequential")
    print(f"\n  Per-GPU breakdown:")
    for gid in gpu_ids:
        print(f"    GPU {gid}: {gpu_counts[gid]:2d} jobs, "
              f"{gpu_times[gid]:.1f}s total compute")
    print(f"{'=' * 70}")


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Solve ensemble VFI configurations across multiple GPUs',
    )
    parser.add_argument(
        '--gpus', type=str, default=None,
        help='Comma-separated GPU IDs to use (default: auto-detect all)',
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Show job plan without solving',
    )
    parser.add_argument(
        '--force', action='store_true',
        help='Re-solve even if cached .npz exists',
    )
    return parser.parse_args()


def detect_gpus() -> List[int]:
    """Detect available GPU IDs without initialising TensorFlow in main process."""
    # Peek at NVIDIA GPUs via nvidia-smi or fall back to TF
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


def main():
    mp.set_start_method('spawn', force=True)
    args = parse_args()

    # Determine GPU set
    if args.gpus is not None:
        gpu_ids = [int(g.strip()) for g in args.gpus.split(',')]
    else:
        gpu_ids = detect_gpus()

    if not gpu_ids:
        logger.error("No GPUs detected. Cannot run parallel solver.")
        sys.exit(1)

    logger.info(f"Using GPUs: {gpu_ids}")

    # Collect jobs
    jobs = collect_all_jobs(skip_cached=not args.force)
    all_jobs = collect_all_jobs(skip_cached=False)

    print(f"\n  Total configs:    {len(all_jobs)}")
    print(f"  Already cached:   {len(all_jobs) - len(jobs)}")
    print(f"  To solve:         {len(jobs)}")
    print(f"  GPUs available:   {len(gpu_ids)}")

    if args.dry_run:
        print(f"\n  Job plan (longest-first):")
        for i, (nk, nd) in enumerate(jobs):
            print(f"    [{i+1:2d}] ({nk:4d}, {nd:4d})  ≈ {nk*nd:>8,} grid pts")
        print(f"\n  [DRY RUN — no solves performed]")
        return

    if not jobs:
        print("  All solutions already cached — nothing to solve.")
        return

    # Run parallel solver
    run_parallel_solver(gpu_ids, jobs)

    # Verify all expected files now exist
    missing = []
    for nk, nd in all_jobs:
        if not os.path.exists(cache_path_for(nk, nd)):
            missing.append((nk, nd))

    if missing:
        print(f"\n  WARNING: {len(missing)} cache files still missing:")
        for nk, nd in missing:
            print(f"    ({nk}, {nd}) → {cache_path_for(nk, nd)}")
    else:
        print(f"\n  ✓ All {len(all_jobs)} VFI cache files verified.")


if __name__ == "__main__":
    main()
