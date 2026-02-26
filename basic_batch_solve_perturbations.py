#!/usr/bin/env python3
"""Batch VFI solver for the nine perturbation configurations.

Solves Value Function Iteration (VFI) for all nine configurations used
in the Jacobian-based identification analysis:

- 1 baseline  (C0)
- 8 perturbations (C1-C8): +/- step for each of four structural parameters

Each solve uses ``n_capital=3000``, ``n_productivity=50`` (high resolution).
Results are cached to ``ground_truth_basic/`` and skipped if already present.

Reference
---------
``docs/basic_blueprint_identification_check.md``, Section 1.3.

Usage::

    python basic_batch_solve_perturbations.py [--dry-run] [--start 0] [--n-capital 3000]
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import time
from typing import Dict

import numpy as np

from src.econ_models.config.economic_params import EconomicParams
from src.econ_models.config.vfi_config import load_grid_config
from src.econ_models.config.bond_config import BondsConfig
from src.econ_models.io.file_utils import load_json_file, save_json_file, save_boundary_to_json
from src.econ_models.io.artifacts import save_vfi_results

from basic_common import (
    BASE_DIR,
    BASELINE,
    FIXED_PARAMS,
    PARAM_SYMBOLS,
    PERTURBATION_CONFIGS,
    STEP_SIZES,
    build_full_params,
    get_bounds_path,
    get_econ_params_path,
    get_vfi_cache_path,
    param_tag,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  Grid resolution defaults
# ---------------------------------------------------------------------------

N_CAPITAL_DEFAULT: int = 3000
N_PRODUCTIVITY: int = 50

# BoundaryFinder settings (low resolution for speed)
BOUND_FINDER_N_K: int = 200
BOUND_FINDER_N_STEPS: int = 100
BOUND_FINDER_N_BATCHES: int = 5000

CONFIG_PARAMS_FILE: str = os.path.join(BASE_DIR, "hyperparam/prefixed/vfi_params.json")
"""Path to the VFI grid configuration JSON."""


# ---------------------------------------------------------------------------
#  Prerequisite helpers
# ---------------------------------------------------------------------------

def ensure_econ_params_json(full_params: Dict[str, float], tag: str) -> str:
    """Create the economic-parameters JSON file if it does not exist.

    Parameters
    ----------
    full_params:
        Complete parameter dictionary.
    tag:
        Filename tag derived from structural parameters.

    Returns
    -------
    str
        Path to the (possibly newly created) JSON file.
    """
    path = get_econ_params_path(tag)
    if os.path.exists(path):
        logger.info("  Econ params exist: %s", path)
        return path
    logger.info("  Creating econ params: %s", path)
    save_json_file(full_params, path)
    return path


def ensure_bounds(econ_params: EconomicParams, tag: str) -> Dict:
    """Ensure bounds JSON exists; run the BoundaryFinder if missing.

    Parameters
    ----------
    econ_params:
        Economic parameters for this configuration.
    tag:
        Filename tag.

    Returns
    -------
    dict
        Loaded bounds dictionary.
    """
    bp = get_bounds_path(tag)
    if os.path.exists(bp):
        logger.info("  Bounds exist: %s", bp)
        return load_json_file(bp)

    logger.info("  Bounds MISSING - running BoundaryFinder for %s ...", tag)
    from src.econ_models.vfi.bounds import BoundaryFinder

    config = load_grid_config(CONFIG_PARAMS_FILE, "basic")
    config = dataclasses.replace(config, n_capital=BOUND_FINDER_N_K)

    finder = BoundaryFinder(
        econ_params, config,
        n_steps=BOUND_FINDER_N_STEPS,
        n_batches=BOUND_FINDER_N_BATCHES,
    )
    bounds_result = finder.find_basic_bounds()

    k_bounds = bounds_result["k_bounds_add_margin"]
    z_bounds = bounds_result["z_bounds_original"]
    bounds_data = {
        "k_min": float(k_bounds[0]),
        "k_max": float(k_bounds[1]),
        "z_min": float(z_bounds[0]),
        "z_max": float(z_bounds[1]),
    }
    save_boundary_to_json(bp, bounds_data, econ_params)
    logger.info("    Bounds saved -> %s", bp)
    logger.info("    K=[%.2f, %.2f], Z=[%.4f, %.4f]",
                bounds_data["k_min"], bounds_data["k_max"],
                bounds_data["z_min"], bounds_data["z_max"])
    return load_json_file(bp)


def solve_vfi(econ_params: EconomicParams, bounds: Dict, n_capital: int) -> Dict:
    """Solve basic-model VFI at the specified grid resolution.

    Parameters
    ----------
    econ_params:
        Economic parameters.
    bounds:
        State-space bounds dictionary (with nested ``bounds`` key).
    n_capital:
        Number of capital grid points.

    Returns
    -------
    dict
        VFI solution arrays.
    """
    from src.econ_models.vfi.basic import BasicModelVFI

    config = load_grid_config(CONFIG_PARAMS_FILE, "basic")
    config = dataclasses.replace(config, n_capital=n_capital, n_productivity=N_PRODUCTIVITY)

    model = BasicModelVFI(
        econ_params, config,
        k_bounds=(bounds["bounds"]["k_min"], bounds["bounds"]["k_max"]),
    )
    return model.solve()


# ---------------------------------------------------------------------------
#  Status display
# ---------------------------------------------------------------------------

def print_plan(configs, n_capital: int, start: int) -> None:
    """Print a summary of the solve plan.

    Parameters
    ----------
    configs:
        List of prepared configuration dicts.
    n_capital:
        Capital grid resolution.
    start:
        Starting configuration index.
    """
    n_cached = sum(1 for c in configs if c["cached"])
    n_todo = sum(1 for c in configs[start:] if not c["cached"])

    print(f"\n{'=' * 70}")
    print(f"BATCH VFI SOLVER - 9 basic-model perturbations at n_k={n_capital}")
    print(f"{'=' * 70}")
    print(f"  Baseline: rho={BASELINE['productivity_persistence']}, "
          f"sigma={BASELINE['productivity_std_dev']}, "
          f"xi={BASELINE['adjustment_cost_convex']}, "
          f"F={BASELINE['adjustment_cost_fixed']}")
    print(f"  Already cached:  {n_cached}/9")
    print(f"  To solve (from --start={start}):  {n_todo}")
    print(f"{'=' * 70}\n")

    for i, cfg in enumerate(configs):
        status = "CACHED" if cfg["cached"] else "TO SOLVE"
        marker = ">>>" if (i >= start and not cfg["cached"]) else "   "
        desc = cfg["cfg_tag"]
        if cfg["overrides"]:
            key = list(cfg["overrides"].keys())[0]
            desc += f" ({PARAM_SYMBOLS[key]}={cfg['overrides'][key]})"
        else:
            desc += " (reference)"
        print(f"  {marker} [C{i}] {desc:35s}  tag={cfg['tag']:30s}  {status}")
    print()


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse arguments and batch-solve the nine VFI configurations."""
    parser = argparse.ArgumentParser(description="Batch solve 9 basic-model perturbation VFI configs")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without solving")
    parser.add_argument("--start", type=int, default=0, help="Start from config index (0-8)")
    parser.add_argument("--n-capital", type=int, default=N_CAPITAL_DEFAULT,
                        help=f"Capital grid size (default {N_CAPITAL_DEFAULT})")
    args = parser.parse_args()

    n_capital = args.n_capital
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    # Build all nine configurations
    configs = []
    for cfg_tag, overrides in PERTURBATION_CONFIGS:
        full_params = build_full_params(overrides)
        tag = param_tag(full_params)
        cache = get_vfi_cache_path(tag, n_capital)
        configs.append({
            "cfg_tag": cfg_tag,
            "full_params": full_params,
            "tag": tag,
            "cache": cache,
            "cached": os.path.exists(cache),
            "overrides": overrides,
        })

    print_plan(configs, n_capital, args.start)

    if args.dry_run:
        print("DRY RUN - exiting without solving.")
        return

    # Solve loop
    total_t0 = time.time()
    n_todo = sum(1 for c in configs[args.start:] if not c["cached"])
    solved_count = 0

    for i, cfg in enumerate(configs):
        if i < args.start:
            continue
        if cfg["cached"]:
            logger.info("[C%d/9] %s - already cached, skipping", i, cfg["cfg_tag"])
            continue

        logger.info("\n%s", "=" * 70)
        logger.info("[C%d/9] Solving: %s", i, cfg["cfg_tag"])
        logger.info("  tag    = %s", cfg["tag"])
        logger.info("  grid   = (n_k=%d, n_z=%d)", n_capital, N_PRODUCTIVITY)
        logger.info("  cache  = %s", cfg["cache"])
        logger.info("=" * 70)

        ep_path = ensure_econ_params_json(cfg["full_params"], cfg["tag"])
        econ_params = EconomicParams(**load_json_file(ep_path))
        bounds = ensure_bounds(econ_params, cfg["tag"])

        t0 = time.time()
        vfi_results = solve_vfi(econ_params, bounds, n_capital)
        solve_time = time.time() - t0

        os.makedirs(os.path.dirname(cfg["cache"]), exist_ok=True)
        save_vfi_results(vfi_results, cfg["cache"])

        solved_count += 1
        elapsed = time.time() - total_t0
        logger.info("  Solved in %.1fs (total elapsed: %.1f min)", solve_time, elapsed / 60)
        logger.info("  Saved -> %s", cfg["cache"])
        logger.info("  Progress: %d/%d solves done", solved_count, n_todo)

    total_time = time.time() - total_t0

    # Final report
    print(f"\n{'=' * 70}")
    print("BATCH COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Solved:       {solved_count} new VFI solutions")
    print(f"  Total time:   {total_time:.1f}s ({total_time / 60:.1f} min)")
    print(f"  Grid:         (n_k={n_capital}, n_z={N_PRODUCTIVITY})")
    print(f"  Cache dir:    {os.path.abspath('./ground_truth_basic/')}")
    print(f"{'=' * 70}")

    all_ok = True
    for i, cfg in enumerate(configs):
        exists = os.path.exists(cfg["cache"])
        status = "OK" if exists else "MISSING"
        print(f"  {status}  [C{i}] {cfg['cfg_tag']:16s}  {cfg['cache']}")
        if not exists:
            all_ok = False

    if all_ok:
        print(f"\n  All 9 perturbation VFI solutions cached.")
        print("  Ready for: python basic_identification_check.py")
    else:
        print(f"\n  WARNING: Some solutions still missing. Re-run with --start=<idx>.")


if __name__ == "__main__":
    main()
