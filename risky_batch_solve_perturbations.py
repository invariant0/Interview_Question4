#!/usr/bin/env python3
"""
Batch VFI Solver — 13 Perturbation Configs × 1 Grid Resolution (560), 4-GPU Parallel (Risky Debt)
═══════════════════════════════════════════════════════════════════════════════════════════════════

Solves VFI for all 13 configurations specified in the identification blueprint
at grid resolution 560×560:
  - 1 baseline  (C0)
  - 12 perturbations (C1–C12): ±Δ for each of 6 structural parameters
  = 13 total VFI solves

Each VFI solve is dispatched to one of 4 GPUs via multiprocessing. Workers
pull jobs from a shared queue so faster GPUs automatically pick up more work
(no manual load-balancing needed). Jobs are sorted largest-resolution-first
for optimal scheduling.

Prerequisites (econ params JSON, bounds JSON) are prepared in the main
process before dispatching GPU work.

Results are cached to ground_truth_risky/ and skipped if already solved.

Reference: docs/risky_blueprint_identification_check.md, Section 1.3

Run:
  python risky_batch_solve_perturbations.py

Optional flags:
  --dry-run       Print what would be solved without actually solving
  --gpus 0,1,2,3  Comma-separated GPU IDs (default: auto-detect all)
  --force         Re-solve even if cached .npz exists
"""

import argparse
import dataclasses
import json
import logging
import multiprocessing as mp
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np

from src.econ_models.config.economic_params import EconomicParams
from src.econ_models.config.vfi_config import GridConfig, load_grid_config
from src.econ_models.config.bond_config import BondsConfig
from src.econ_models.io.file_utils import load_json_file, save_json_file, save_boundary_to_json
from src.econ_models.io.artifacts import save_vfi_results
from src.econ_models.vfi.grids.grid_utils import compute_optimal_chunks

from risky_common import (
    FIXED_PARAMS,
    BASELINE,
    STEP_SIZES,
    PARAM_BOUNDS,
    PARAM_SYMBOLS,
    PARAM_KEYS,
    PERTURBATION_CONFIGS as CONFIGS,
    N_PRODUCTIVITY,
    BASE_DIR,
    build_full_params,
    param_tag,
    get_econ_params_path as econ_params_path,
    get_bounds_path as bounds_path,
    get_vfi_cache_path as vfi_cache_path,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════════

# VFI grid resolution — single high-resolution grid
GRID_RESOLUTIONS = [560]

# Paths
CONFIG_PARAMS_FILE = os.path.join(BASE_DIR, "hyperparam/prefixed/vfi_params.json")

# Boundary finder settings (lower resolution for speed)
BOUND_FINDER_N_K = 100
BOUND_FINDER_N_D = 100
BOUND_FINDER_N_STEPS = 100
BOUND_FINDER_N_BATCHES = 3000


# ═══════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════


def validate_perturbations():
    """Verify all perturbations remain inside feasible bounds."""
    logger.info("\n── Validating perturbation bounds ──")
    all_ok = True
    for pkey in BASELINE:
        base = BASELINE[pkey]
        h = STEP_SIZES[pkey]
        lo, hi = PARAM_BOUNDS[pkey]
        theta_minus = base - h
        theta_plus = base + h
        in_bounds = (theta_minus >= lo) and (theta_plus <= hi)
        status = "✓" if in_bounds else "✗ OUT OF BOUNDS"
        logger.info(f"  {PARAM_SYMBOLS[pkey]:>3s} ({pkey}): "
                     f"θ⁰={base}, h={h}, "
                     f"θ⁻={theta_minus:.4f}, θ⁺={theta_plus:.4f}, "
                     f"range=[{lo}, {hi}]  {status}")
        if not in_bounds:
            all_ok = False
    if not all_ok:
        logger.error("FATAL: Some perturbations violate parameter bounds!")
        sys.exit(1)
    logger.info("  All perturbations within feasible range.")


# ═══════════════════════════════════════════════════════════════════════════
#  Ensure econ params JSON
# ═══════════════════════════════════════════════════════════════════════════

def ensure_econ_params_json(full_params: Dict[str, float], tag: str) -> str:
    """Create econ params JSON if it doesn't exist. Return path."""
    path = econ_params_path(tag)
    if os.path.exists(path):
        logger.info(f"  Econ params exist: {path}")
        return path

    logger.info(f"  Creating econ params: {path}")
    save_json_file(full_params, path)
    return path


# ═══════════════════════════════════════════════════════════════════════════
#  Ensure bounds JSON
# ═══════════════════════════════════════════════════════════════════════════

def ensure_bounds(econ_params: EconomicParams, tag: str, n_k: int, n_d: int) -> Dict:
    """Ensure bounds JSON exists; run BoundaryFinder if missing."""
    bp = bounds_path(tag)
    if os.path.exists(bp):
        logger.info(f"  Bounds exist: {bp}")
        return load_json_file(bp)

    logger.info(f"  Bounds MISSING — running BoundaryFinder (risky) for {tag} ...")
    from src.econ_models.vfi.bounds import BoundaryFinder

    config = load_grid_config(CONFIG_PARAMS_FILE, 'risky')
    config = dataclasses.replace(
        config,
        n_capital=BOUND_FINDER_N_K,
        n_debt=BOUND_FINDER_N_D,
        n_productivity=N_PRODUCTIVITY,
    )

    k_chunk, b_chunk, kp_chunk, bp_chunk = compute_optimal_chunks(
        BOUND_FINDER_N_K, BOUND_FINDER_N_D, n_z=N_PRODUCTIVITY,
    )

    finder = BoundaryFinder(
        econ_params, config,
        n_steps=BOUND_FINDER_N_STEPS,
        n_batches=BOUND_FINDER_N_BATCHES,
        k_chunk_size=k_chunk,
        b_chunk_size=b_chunk,
        kp_chunk_size=kp_chunk,
        bp_chunk_size=bp_chunk,
    )
    bounds_result = finder.find_risky_bounds()

    k_bounds = bounds_result['k_bounds_add_margin']
    b_bounds = bounds_result['b_bounds_add_margin']
    z_bounds = bounds_result['z_bounds_original']

    bounds_data = {
        "k_min": float(k_bounds[0]),
        "k_max": float(k_bounds[1]),
        "b_min": float(b_bounds[0]),
        "b_max": float(b_bounds[1]),
        "z_min": float(z_bounds[0]),
        "z_max": float(z_bounds[1]),
    }
    save_boundary_to_json(bp, bounds_data, econ_params)
    logger.info(f"    Bounds saved → {bp}")
    logger.info(f"    K=[{bounds_data['k_min']:.4f}, {bounds_data['k_max']:.4f}], "
                f"B=[{bounds_data['b_min']:.4f}, {bounds_data['b_max']:.4f}], "
                f"Z=[{bounds_data['z_min']:.4f}, {bounds_data['z_max']:.4f}]")
    return load_json_file(bp)


# ═══════════════════════════════════════════════════════════════════════════
#  VFI Solver (called inside GPU worker subprocess)
# ═══════════════════════════════════════════════════════════════════════════

def solve_vfi_on_device(econ_params_dict: Dict, bounds: Dict, n_k: int, n_d: int) -> Dict:
    """
    Solve risky debt model VFI at specified grid resolution.

    Takes a plain dict (not EconomicParams) so it can be safely serialised
    across process boundaries via multiprocessing Queue.
    """
    from src.econ_models.config.economic_params import EconomicParams as EP
    from src.econ_models.config.vfi_config import load_grid_config as lgc
    from src.econ_models.vfi.risky import RiskyDebtModelVFI
    from src.econ_models.vfi.grids.grid_utils import compute_optimal_chunks as coc

    econ_params = EP(**econ_params_dict)

    config = lgc(CONFIG_PARAMS_FILE, 'risky')
    config = dataclasses.replace(
        config,
        n_capital=n_k,
        n_productivity=N_PRODUCTIVITY,
        n_debt=n_d,
    )

    k_chunk, b_chunk, kp_chunk, bp_chunk = coc(n_k, n_d, n_z=N_PRODUCTIVITY)

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
    return model.solve()


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

    Pulls jobs from the shared queue until a sentinel (None) is received,
    solves VFI, and saves to cache. Each job dict contains:
      cfg_tag, tag, n_k, n_d, cache, econ_params_dict, bounds
    """
    # Pin to GPU BEFORE importing TensorFlow
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    import tensorflow as tf
    from src.econ_models.io.artifacts import save_vfi_results as svr

    # Configure the (now sole visible) GPU for memory growth
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
        logger.info(f"Worker GPU-{gpu_id}: using {gpus[0].name}")
    else:
        logger.warning(f"Worker GPU-{gpu_id}: no GPU visible — using CPU")

    while True:
        job = job_queue.get()
        if job is None:
            logger.info(f"Worker GPU-{gpu_id}: received shutdown signal")
            break

        cfg_tag = job['cfg_tag']
        n_k = job['n_k']
        n_d = job['n_d']
        cache_path = job['cache']

        # Double-check: another worker may have solved it concurrently
        if os.path.exists(cache_path):
            logger.info(f"Worker GPU-{gpu_id}: {cfg_tag} res={n_k} already cached, skipping")
            result_queue.put((gpu_id, cfg_tag, n_k, 0.0, 'cached'))
            continue

        logger.info(f"Worker GPU-{gpu_id}: solving {cfg_tag} res={n_k} ...")
        t0 = time.time()

        try:
            vfi_results = solve_vfi_on_device(
                job['econ_params_dict'], job['bounds'], n_k, n_d,
            )

            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            svr(vfi_results, cache_path)

            elapsed = time.time() - t0
            logger.info(
                f"Worker GPU-{gpu_id}: {cfg_tag} res={n_k} done in {elapsed:.1f}s "
                f"→ {cache_path}"
            )
            result_queue.put((gpu_id, cfg_tag, n_k, elapsed, 'solved'))

            # Clear TF session state between solves to reclaim VRAM
            tf.keras.backend.clear_session()

        except Exception as e:
            elapsed = time.time() - t0
            logger.error(
                f"Worker GPU-{gpu_id}: {cfg_tag} res={n_k} FAILED "
                f"after {elapsed:.1f}s — {e}"
            )
            result_queue.put((gpu_id, cfg_tag, n_k, elapsed, f'error: {e}'))


# ═══════════════════════════════════════════════════════════════════════════
#  GPU Detection
# ═══════════════════════════════════════════════════════════════════════════

def detect_gpus() -> List[int]:
    """Detect available GPU IDs without initialising TensorFlow in the main process."""
    try:
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


# ═══════════════════════════════════════════════════════════════════════════
#  Parallel Scheduler
# ═══════════════════════════════════════════════════════════════════════════

def run_parallel_solver(gpu_ids: List[int], jobs: List[Dict]) -> None:
    """
    Distribute VFI jobs across GPU workers using longest-job-first scheduling.

    Workers pull from a shared queue, so faster GPUs automatically pick up
    more work — no manual balancing needed.
    """
    n_gpus = len(gpu_ids)
    n_jobs = len(jobs)

    logger.info(f"\n{'=' * 80}")
    logger.info(f"PARALLEL VFI SOLVER — {n_gpus} GPUs × {n_jobs} jobs")
    logger.info(f"GPUs: {gpu_ids}")
    logger.info(f"{'=' * 80}")

    if n_jobs == 0:
        logger.info("All jobs already cached — nothing to solve.")
        return

    # Print job plan
    logger.info(f"\nJob queue (largest-resolution-first):")
    for i, job in enumerate(jobs):
        desc = job['cfg_tag']
        if job['overrides']:
            key = list(job['overrides'].keys())[0]
            desc += f" ({PARAM_SYMBOLS[key]})"
        logger.info(f"  [{i+1:2d}] {desc:25s}  res={job['n_k']:4d}  "
                     f"≈ {job['n_k']*job['n_d']:>8,} grid pts")

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
    errors: List[str] = []

    while completed < n_jobs:
        gid, cfg_tag, n_k, elapsed, status = result_queue.get()
        completed += 1
        gpu_times[gid] += elapsed
        gpu_counts[gid] += 1
        logger.info(
            f"  [{completed}/{n_jobs}] GPU-{gid}: {cfg_tag} res={n_k} → {status} "
            f"({elapsed:.1f}s)"
        )
        if status.startswith('error'):
            errors.append(f"GPU-{gid}: {cfg_tag} res={n_k} → {status}")

    # Wait for workers to exit
    for p in workers:
        p.join(timeout=60)
        if p.is_alive():
            logger.warning(f"  Worker PID={p.pid} did not exit cleanly, terminating")
            p.terminate()

    total_wall = time.time() - t_start
    total_compute = sum(gpu_times.values())

    # Print summary
    print(f"\n{'=' * 80}")
    print(f"PARALLEL SOLVE COMPLETE — {n_jobs} jobs in {total_wall:.1f}s wall time")
    print(f"  Total compute: {total_compute:.1f}s across {n_gpus} GPUs")
    if total_wall > 0:
        print(f"  Speedup:       {total_compute / total_wall:.2f}× vs sequential")
    print(f"\n  Per-GPU breakdown:")
    for gid in gpu_ids:
        print(f"    GPU {gid}: {gpu_counts[gid]:2d} jobs, "
              f"{gpu_times[gid]:.1f}s total compute")
    if errors:
        print(f"\n  ERRORS ({len(errors)}):")
        for e in errors:
            print(f"    {e}")
    print(f"{'=' * 80}")


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """Batch-solve all risky perturbation configurations via multi-GPU parallelism."""
    mp.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser(
        description="Batch solve 13 risky perturbation configs at 560×560 on multi-GPU"
    )
    parser.add_argument('--dry-run', action='store_true',
                        help="Print plan without solving")
    parser.add_argument('--gpus', type=str, default=None,
                        help="Comma-separated GPU IDs (default: auto-detect all)")
    parser.add_argument('--force', action='store_true',
                        help="Re-solve even if cached .npz exists")
    args = parser.parse_args()

    # GPU setup
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # Determine GPU set
    if args.gpus is not None:
        gpu_ids = [int(g.strip()) for g in args.gpus.split(',')]
    else:
        gpu_ids = detect_gpus()

    if not gpu_ids:
        logger.error("No GPUs detected. Cannot run parallel solver.")
        sys.exit(1)

    # Validate bound adherence
    validate_perturbations()

    # ── Phase 1: Prepare all prerequisites on CPU (serial) ───────────
    # Ensure econ params JSONs and bounds JSONs exist for all 13 configs
    # BEFORE dispatching any GPU work. This is lightweight CPU-only work.
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 1: Prepare prerequisites (econ params + bounds) for all 13 configs")
    logger.info("=" * 80)

    config_prereqs = {}  # cfg_tag -> {tag, econ_params_dict, bounds}
    for cfg_tag, overrides in CONFIGS:
        full_params = build_full_params(overrides)
        tag = param_tag(full_params)
        logger.info(f"\n  [{cfg_tag}] tag={tag}")

        ep_path = ensure_econ_params_json(full_params, tag)
        econ_params = EconomicParams(**load_json_file(ep_path))

        # Bounds are resolution-independent: use max resolution for finder
        max_res = max(GRID_RESOLUTIONS)
        bounds = ensure_bounds(econ_params, tag, max_res, max_res)

        # Store as plain dict for pickling across process boundaries
        config_prereqs[cfg_tag] = {
            'tag': tag,
            'econ_params_dict': load_json_file(ep_path),
            'bounds': bounds,
            'overrides': overrides,
        }

    # ── Phase 2: Build job list ──────────────────────────────────────
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 2: Build job list (13 configs × 1 resolution = 13 jobs)")
    logger.info("=" * 80)

    all_jobs = []
    for cfg_tag, overrides in CONFIGS:
        prereq = config_prereqs[cfg_tag]
        for res in GRID_RESOLUTIONS:
            cache = vfi_cache_path(prereq['tag'], res, res)
            all_jobs.append({
                'cfg_tag': cfg_tag,
                'tag': prereq['tag'],
                'n_k': res,
                'n_d': res,
                'cache': cache,
                'cached': os.path.exists(cache),
                'overrides': overrides,
                'econ_params_dict': prereq['econ_params_dict'],
                'bounds': prereq['bounds'],
            })

    # Filter to uncached unless --force
    if args.force:
        pending_jobs = list(all_jobs)
    else:
        pending_jobs = [j for j in all_jobs if not j['cached']]

    # Sort largest-resolution-first for optimal scheduling
    # (big jobs start early, small ones fill gaps at the end)
    pending_jobs.sort(key=lambda j: j['n_k'] * j['n_d'], reverse=True)

    n_total = len(all_jobs)
    n_cached = sum(1 for j in all_jobs if j['cached'])
    n_pending = len(pending_jobs)

    print(f"\n{'=' * 80}")
    print(f"BATCH VFI SOLVER — 13 configs × 1 resolution = {n_total} jobs, {len(gpu_ids)} GPUs")
    print(f"{'=' * 80}")
    print(f"  Baseline: ρ={BASELINE['productivity_persistence']}, "
          f"σ={BASELINE['productivity_std_dev']}, "
          f"ξ={BASELINE['adjustment_cost_convex']}, "
          f"F={BASELINE['adjustment_cost_fixed']}, "
          f"η₀={BASELINE['equity_issuance_cost_fixed']}, "
          f"η₁={BASELINE['equity_issuance_cost_linear']}")
    print(f"  Grid resolutions: {GRID_RESOLUTIONS} (n_k=n_d, n_z={N_PRODUCTIVITY})")
    print(f"  GPUs:            {gpu_ids}")
    print(f"  Already cached:  {n_cached}/{n_total}")
    print(f"  To solve:        {n_pending}")
    print(f"{'=' * 80}\n")

    for idx, job in enumerate(all_jobs):
        status = "✓ cached" if job['cached'] else "✗ TO SOLVE"
        desc = job['cfg_tag']
        if job['overrides']:
            key = list(job['overrides'].keys())[0]
            desc += f" ({PARAM_SYMBOLS[key]}={job['overrides'][key]})"
        else:
            desc += " (reference)"
        print(f"  [J{idx:2d}] {desc:40s}  res={job['n_k']:4d}  {status}")

    if args.dry_run:
        print(f"\n  Pending queue (largest-resolution-first, {n_pending} jobs):")
        for i, job in enumerate(pending_jobs):
            desc = job['cfg_tag']
            if job['overrides']:
                key = list(job['overrides'].keys())[0]
                desc += f" ({PARAM_SYMBOLS[key]})"
            gpu_hint = f"(initial → GPU {gpu_ids[i % len(gpu_ids)]})" if i < len(gpu_ids) else ""
            print(f"    [{i+1:2d}] {desc:25s} res={job['n_k']:4d}  "
                  f"≈ {job['n_k']*job['n_d']:>8,} pts  {gpu_hint}")
        print(f"\n  DRY RUN — exiting without solving.")
        return

    if n_pending == 0:
        print("  All solutions already cached — nothing to solve.")
        print(f"  Ready for: python risky_identification_check.py")
        return

    # ── Phase 3: Parallel GPU solve ──────────────────────────────────
    run_parallel_solver(gpu_ids, pending_jobs)

    # ── Phase 4: Verify all expected cache files ─────────────────────
    print(f"\n{'=' * 80}")
    print(f"VERIFICATION")
    print(f"{'=' * 80}")

    all_ok = True
    for idx, job in enumerate(all_jobs):
        exists = os.path.exists(job['cache'])
        status = "✓" if exists else "✗ MISSING"
        print(f"  {status}  [J{idx:2d}] {job['cfg_tag']:16s}  res={job['n_k']}  {job['cache']}")
        if not exists:
            all_ok = False

    if all_ok:
        print(f"\n  All {n_total} VFI solutions cached (13 configs × 1 resolution).")
        print(f"  Ready for: python risky_identification_check.py")
    else:
        print(f"\n  WARNING: Some solutions still missing. Check error logs above.")


if __name__ == "__main__":
    main()
