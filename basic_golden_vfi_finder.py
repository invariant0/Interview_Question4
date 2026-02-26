#!/usr/bin/env python3
"""Test VFI moment convergence as capital-grid resolution increases.

For a sequence of capital-grid sizes (e.g. 500-3000) this script:

1. Solves VFI at each resolution.
2. Simulates panels using **identical** initial conditions and shocks.
3. Computes a standard set of macroeconomic moments.
4. Produces convergence plots showing how moments stabilize with finer grids.

The final (highest-resolution) VFI solution is saved as the golden
reference for downstream comparison scripts.

Usage::

    python basic_golden_vfi_finder.py [--econ-id 0]
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from src.econ_models.config.economic_params import EconomicParams
from src.econ_models.config.vfi_config import load_grid_config
from src.econ_models.vfi.basic import BasicModelVFI
from src.econ_models.io.artifacts import save_vfi_results
from src.econ_models.simulator.vfi.basic import VFISimulator_basic
from src.econ_models.moment_calculator.compute_derived_quantities import (
    compute_all_derived_quantities,
)

from basic_common import (
    BASE_DIR,
    DEFAULT_ECON_LIST,
    apply_burn_in,
    compute_standard_moments,
    get_bounds_path_by_list,
    get_econ_params_path_by_list,
    get_golden_vfi_path,
    load_bonds_config,
    load_econ_params,
    setup_simulation_data,
    to_python_float,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------

CONFIG_PARAMS_FILE: str = os.path.join(BASE_DIR, "hyperparam/prefixed/vfi_params.json")
RESULTS_DIR: str = "./results/golden_vfi_basic"

N_CAPITAL_LIST: List[int] = [500, 1000, 1500, 2000, 2500, 3000]
"""Capital grid sizes to test for convergence."""

N_PRODUCTIVITY: int = 50
"""Productivity grid size (held constant)."""

BURN_IN: int = 200
"""Burn-in periods discarded before moment calculation."""


# ---------------------------------------------------------------------------
#  VFI solving and simulation
# ---------------------------------------------------------------------------

def solve_basic_model(
    econ_params: EconomicParams,
    bounds: Dict,
    n_capital: int,
) -> Dict:
    """Solve VFI at a given capital-grid resolution.

    Parameters
    ----------
    econ_params:
        Economic parameters.
    bounds:
        Loaded bounds dictionary (with nested ``bounds`` key).
    n_capital:
        Number of capital grid points.

    Returns
    -------
    dict
        VFI solution arrays.
    """
    config = load_grid_config(CONFIG_PARAMS_FILE, "basic")
    config = dataclasses.replace(config, n_capital=n_capital, n_productivity=N_PRODUCTIVITY)
    model = BasicModelVFI(
        econ_params, config,
        k_bounds=(bounds["k_min"], bounds["k_max"]),
    )
    return model.solve()


def run_simulation(
    initial_states,
    shock_sequence,
    vfi_results: Dict,
    econ_params: EconomicParams,
) -> Dict:
    """Simulate using VFISimulator with continuous interpolation.

    Parameters
    ----------
    initial_states:
        Initial-state tensors.
    shock_sequence:
        Shock sequence array.
    vfi_results:
        Solved VFI arrays.
    econ_params:
        Economic parameters.

    Returns
    -------
    dict
        Simulation results with ``K_curr``, ``K_next``, etc.
    """
    simulator = VFISimulator_basic(econ_params)
    simulator.load_solved_vfi_solution(vfi_results)
    return simulator.simulate(
        tuple(np.asarray(s) for s in initial_states),
        np.asarray(shock_sequence),
    )


# ---------------------------------------------------------------------------
#  Convergence analysis
# ---------------------------------------------------------------------------

def compute_percent_changes(
    previous: Dict[str, float],
    current: Dict[str, float],
) -> Dict[str, float]:
    """Calculate percentage change between two moment dictionaries.

    Parameters
    ----------
    previous:
        Moments from the coarser grid.
    current:
        Moments from the finer grid.

    Returns
    -------
    dict
        Per-moment absolute percentage change.
    """
    changes: Dict[str, float] = {}
    for key in previous:
        prev = to_python_float(previous[key])
        curr = to_python_float(current[key])
        if abs(prev) > 1e-10:
            changes[key] = abs((curr - prev) / prev) * 100
        else:
            changes[key] = 0.0 if abs(curr) < 1e-10 else float("inf")
    return changes


# ---------------------------------------------------------------------------
#  Plotting
# ---------------------------------------------------------------------------

def plot_convergence(
    all_moments: List[Dict[str, float]],
    n_capital_list: List[int],
    save_path: str,
) -> None:
    """Plot moment values vs. grid size to visualize convergence.

    Parameters
    ----------
    all_moments:
        List of moment dictionaries, one per grid resolution.
    n_capital_list:
        Corresponding capital grid sizes.
    save_path:
        File path for the saved figure.
    """
    moment_keys = list(all_moments[0].keys())
    n_plots = len(moment_keys)
    n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 12))
    for idx, key in enumerate(moment_keys):
        if idx >= len(axes.flat):
            break
        ax = axes.flat[idx]
        values = [m[key] for m in all_moments]
        ax.plot(n_capital_list, values, "o-", linewidth=2, markersize=8)
        ax.set_xlabel("N Capital Grid Points")
        ax.set_ylabel(key)
        ax.set_title(key)
        ax.grid(True, alpha=0.3)
        # +/- 2 % convergence band around the finest-grid value
        final = values[-1]
        ax.axhline(final * 0.98, color="r", linestyle="--", alpha=0.5)
        ax.axhline(final * 1.02, color="r", linestyle="--", alpha=0.5)

    for idx in range(n_plots, len(axes.flat)):
        axes.flat[idx].set_visible(False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    logger.info("Convergence plot saved to %s", save_path)


def plot_percent_changes(
    changes_list: List[Dict[str, float]],
    n_capital_list: List[int],
    save_path: str,
) -> None:
    """Plot percentage moment changes between successive grids.

    Parameters
    ----------
    changes_list:
        One dict per pair of successive grids.
    n_capital_list:
        Capital grid sizes (first entry has no change).
    save_path:
        File path for the saved figure.
    """
    plt.figure(figsize=(12, 8))
    for key in changes_list[0]:
        values = [c[key] for c in changes_list]
        plt.plot(n_capital_list[1:], values, marker="o", label=key)
    plt.axhline(2.0, color="r", linestyle="--", label="2 % threshold")
    plt.title("Moment Changes vs. Grid Size")
    plt.xlabel("Number of Capital Grid Points")
    plt.ylabel("Percent Change from Previous (%)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.info("Changes plot saved to %s", save_path)


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the convergence test across capital grid resolutions."""
    parser = argparse.ArgumentParser(description="Test VFI moment convergence")
    parser.add_argument(
        "--econ-id", type=int, default=0, choices=range(len(DEFAULT_ECON_LIST)),
        help="Index into the default economy list (default: 0)",
    )
    args = parser.parse_args()

    econ_id: int = args.econ_id
    econ_spec = DEFAULT_ECON_LIST[econ_id]

    # Shared synthetic data (same for all grid sizes)
    logger.info("Setting up shared synthetic data...")
    econ_params = load_econ_params(econ_spec)
    bonds_config = load_bonds_config(econ_spec, econ_params)
    bounds = bonds_config  # used for VFI grid bounds

    initial_states, shock_sequence = setup_simulation_data(
        econ_params, bonds_config, batch_size=5000, time_periods=700,
    )

    all_moments: List[Dict[str, float]] = []
    changes_list: List[Dict[str, float]] = []
    previous_moments = None
    vfi_results = None

    for n_k in N_CAPITAL_LIST:
        logger.info("\n%s\nTesting n_capital = %d\n%s", "=" * 50, n_k, "=" * 50)

        vfi_results = solve_basic_model(econ_params, bounds, n_capital=n_k)
        sim_results = run_simulation(initial_states, shock_sequence, vfi_results, econ_params)
        sim_stationary = apply_burn_in(sim_results, BURN_IN)

        moments = compute_standard_moments(
            sim_stationary, econ_params.depreciation_rate, econ_params.capital_share,
        )
        all_moments.append(moments)

        logger.info("  inaction_rate: %.4f", moments["inaction_rate"])
        logger.info("  mean I/K: %.4f", moments["Inv. Rate_mean"])
        logger.info("  autocorr I/K: %.4f", moments["Inv. Rate_ac1"])

        if previous_moments is not None:
            changes = compute_percent_changes(previous_moments, moments)
            changes_list.append(changes)
            logger.info("  Max %% change from previous: %.2f%%", max(changes.values()))

        previous_moments = moments

    # Save the finest-grid VFI solution as the golden reference
    golden_path = get_golden_vfi_path(econ_spec)
    os.makedirs(os.path.dirname(golden_path), exist_ok=True)
    save_vfi_results(vfi_results, golden_path)
    logger.info("Golden VFI solution saved to %s", golden_path)

    # Plots
    plot_convergence(
        all_moments, N_CAPITAL_LIST,
        os.path.join(RESULTS_DIR, "moment_convergence.png"),
    )
    if changes_list:
        plot_percent_changes(
            changes_list, N_CAPITAL_LIST,
            os.path.join(RESULTS_DIR, "moment_changes_vs_grid_size.png"),
        )

    # Convergence summary
    logger.info("\n%s\nCONVERGENCE SUMMARY\n%s", "=" * 60, "=" * 60)
    if len(all_moments) >= 2:
        final_changes = compute_percent_changes(all_moments[-2], all_moments[-1])
        logger.info(
            "Changes between n_k=%d and n_k=%d:",
            N_CAPITAL_LIST[-2], N_CAPITAL_LIST[-1],
        )
        for key, val in final_changes.items():
            status = "PASS" if val < 2.0 else "FAIL"
            logger.info("  %s %s: %.2f%%", status, key, val)


if __name__ == "__main__":
    main()
