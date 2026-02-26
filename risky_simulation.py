#!/usr/bin/env python3
"""Compare VFI and Deep Learning simulation results for the risky debt model.

This script loads a ground-truth VFI solution and DL checkpoints at
several training epochs, simulates both, and produces two figures:

1. **Stationary distributions** — overlapping histograms for output,
   capital, debt, leverage, investment, investment rate, equity
   issuance, and equity issuance rate.
2. **Moments comparison** — grouped bar charts for mean, standard
   deviation, and lag-1 autocorrelation of the same variables, plus
   frequency statistics (inaction rate, equity issuance frequency)
   and a weighted-average error score across epochs.

Usage::

    python risky_simulation.py [--econ-id 0]
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Sequence, Tuple

# Ensure we load 'econ_models' from the local src directory
sys.path.insert(0, os.path.join(os.getcwd(), "src"))

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from src.econ_models.config.dl_config import load_dl_config
from src.econ_models.config.economic_params import EconomicParams
from src.econ_models.moment_calculator.compute_derived_quantities import (
    compute_all_derived_quantities,
)
from src.econ_models.simulator import DLSimulatorRiskyFinal

from risky_common import (
    DEFAULT_ECON_LIST,
    MOMENT_KEYS,
    MOMENT_LABELS,
    VARIABLE_NAMES,
    VFI_N_K,
    VFI_N_D,
    apply_burn_in,
    compute_variable_moments,
    compute_weighted_error,
    econ_tag,
    extract_freq_stats,
    extract_variable_moments,
    load_bonds_config,
    load_econ_params,
    plot_grouped_bars,
    plot_histogram_comparison,
    run_vfi_simulation,
    setup_simulation_data,
    to_python_float,
)

# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------

DL_EPOCHS: List[int] = [200 * i for i in range(1, 11)]
# DL_EPOCHS: List[int] = [600, 680, 780, 840]
"""Training epochs at which DL checkpoints are evaluated."""

CHECKPOINT_DIR: str = "./checkpoints_final/risky"
"""Directory containing DL weight files."""

BATCH_SIZE: int = 10000
T_PERIODS: int = 1000
BURN_IN: int = 400

RESULTS_DIR: str = "./results/effectiveness_risky"
"""Output directory for figures."""


# ---------------------------------------------------------------------------
#  Data loading / simulation
# ---------------------------------------------------------------------------

def load_dl_simulator(econ_params, bonds_config):
    """Create and configure a DL simulator for the risky model."""
    dl_config = load_dl_config("./hyperparam/prefixed/dl_params.json", "risky_final")
    dl_config.capital_max = bonds_config["k_max"]
    dl_config.capital_min = bonds_config["k_min"]
    dl_config.productivity_max = bonds_config["z_max"]
    dl_config.productivity_min = bonds_config["z_min"]
    dl_config.debt_max = bonds_config["b_max"]
    dl_config.debt_min = bonds_config["b_min"]
    return DLSimulatorRiskyFinal(dl_config, econ_params)


def run_dl_epoch(
    dl_simulator: DLSimulatorRiskyFinal,
    epoch: int,
    initial_states,
    shock_sequence,
    burn_in: int,
    delta: float,
    alpha: float,
) -> Dict:
    """Load a DL checkpoint, simulate, and compute derived quantities.

    Returns dict with keys: stationary, derived, and flattened arrays
    for each tracked variable.
    """
    dl_simulator.load_solved_dl_solution(
        capital_policy_filepath=os.path.join(
            CHECKPOINT_DIR, f"risky_capital_policy_net_{epoch}.weights.h5"
        ),
        debt_filepath=os.path.join(
            CHECKPOINT_DIR, f"risky_debt_policy_net_{epoch}.weights.h5"
        ),
        investment_policy_filepath=os.path.join(
            CHECKPOINT_DIR, f"risky_investment_decision_net_{epoch}.weights.h5"
        ),
        default_policy_filepath=os.path.join(
            CHECKPOINT_DIR, f"risky_default_policy_net_{epoch}.weights.h5"
        ),
        value_function_filepath=os.path.join(
            CHECKPOINT_DIR, f"risky_value_net_{epoch}.weights.h5"
        ),
        equity_issuance_invest_filepath=os.path.join(
            CHECKPOINT_DIR, f"risky_equity_issuance_net_{epoch}.weights.h5"
        ),
        equity_issuance_noinvest_filepath=os.path.join(
            CHECKPOINT_DIR, f"risky_equity_issuance_net_noinvest_{epoch}.weights.h5"
        ),
    )
    raw_results = dl_simulator.simulate(initial_states, shock_sequence)
    stationary = apply_burn_in(raw_results, burn_in)
    derived = compute_all_derived_quantities(
        stationary, delta, alpha, include_debt=True
    )
    return {
        "stationary": stationary,
        "derived": derived,
        "output": derived["output"].flatten(),
        "capital": stationary["K_curr"].flatten(),
        "debt": stationary["B_curr"].flatten(),
        "leverage": derived["leverage"].flatten(),
        "investment": derived["investment"].flatten(),
        "investment_rate": derived["investment_rate"].flatten(),
        "equity_issuance": derived["equity_issuance"].flatten(),
        "equity_issuance_rate": derived["equity_issuance_rate"].flatten(),
    }


# ---------------------------------------------------------------------------
#  Plotting
# ---------------------------------------------------------------------------

def plot_stationary_distributions(
    vfi_flat: Dict[str, np.ndarray],
    dl_epoch_results: Dict[int, Dict],
    epoch_colors: np.ndarray,
    econ_id: int,
) -> plt.Figure:
    """Create the 5x2 stationary-distribution comparison figure."""
    plot_specs = [
        ("Output (Y)", "Y", "output"),
        ("Capital Stock (K)", "K", "capital"),
        ("Debt (B)", "B", "debt"),
        ("Leverage (B/K)", "B/K", "leverage"),
        ("Investment (I)", "I", "investment"),
        ("Investment Rate (I/K)", "I/K", "investment_rate"),
        ("Equity Issuance (Eq)", "Eq", "equity_issuance"),
        ("Eq. Issuance Rate (Eq/K)", "Eq/K", "equity_issuance_rate"),
    ]

    fig, axes = plt.subplots(4, 2, figsize=(18, 24))
    fig.suptitle(
        f"Risky Model: Stationary Distribution Comparison  (Econ ID: {econ_id})",
        fontsize=18, fontweight="bold", y=0.99,
    )

    for ax, (title, xlabel, key) in zip(axes.flat[:len(plot_specs)], plot_specs):
        epoch_data = {ep: dl_epoch_results[ep][key] for ep in dl_epoch_results}
        plot_histogram_comparison(ax, vfi_flat[key], epoch_data, title, xlabel, epoch_colors)

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    return fig


def plot_moments_figure(
    vfi_moments: Dict[str, Dict[str, float]],
    dl_moments_by_epoch: Dict[int, Dict[str, Dict[str, float]]],
    vfi_freqs: Dict[str, float],
    dl_freqs_by_epoch: Dict[int, Dict[str, float]],
    epoch_colors: np.ndarray,
    econ_id: int,
) -> plt.Figure:
    """Create the moments-comparison figure.

    Layout: variable moments in a grid, plus frequency stats.
    """
    epochs = sorted(dl_moments_by_epoch.keys())
    n_vars = len(VARIABLE_NAMES)
    # variables + 1 row for frequency stats
    n_rows_vars = (n_vars + 1) // 2  # ceil division
    n_rows = n_rows_vars + 1  # +1 row for freq

    fig, axes = plt.subplots(n_rows, 2, figsize=(24, 7 * n_rows))
    fig.suptitle(
        f"Risky Model: Moments & Frequencies — VFI vs. Deep Learning  (Econ ID: {econ_id})",
        fontsize=18, fontweight="bold", y=0.99,
    )

    # Variable moment panels
    for idx, var in enumerate(VARIABLE_NAMES):
        ax = axes[idx // 2, idx % 2]
        vfi_vals = [vfi_moments[var][k] for k in MOMENT_KEYS]
        epoch_vals = {
            ep: [dl_moments_by_epoch[ep][var][k] for k in MOMENT_KEYS]
            for ep in epochs
        }
        plot_grouped_bars(
            ax, vfi_vals, epoch_vals, MOMENT_LABELS, epoch_colors,
            title=f"{var} Moments",
        )

    # Hide unused variable subplot if odd number of variables
    if n_vars % 2 == 1:
        axes[n_rows_vars - 1, 1].set_visible(False)

    # Frequency stats
    ax_freq = axes[n_rows - 1, 0]
    freq_labels = list(vfi_freqs.keys())
    vfi_freq_vals = [vfi_freqs[k] for k in freq_labels]
    epoch_freq_vals = {
        ep: [dl_freqs_by_epoch[ep][k] for k in freq_labels] for ep in epochs
    }
    plot_grouped_bars(
        ax_freq, vfi_freq_vals, epoch_freq_vals, freq_labels, epoch_colors,
        title="Frequency Stats",
    )

    # Hide unused panel
    axes[n_rows - 1, 1].set_visible(False)

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    return fig


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def plot_sample_trajectories(
    vfi_stationary: Dict[str, np.ndarray],
    vfi_derived: Dict,
    dl_epoch_results: Dict[int, Dict],
    econ_id: int,
    *,
    firm_idx: int = 0,
    t_max: int = 200,
) -> plt.Figure:
    """Plot time-series trajectories of K, Z, and I/K for a single sampled firm.

    Parameters
    ----------
    vfi_stationary:
        Post burn-in VFI simulation output.
    vfi_derived:
        Derived quantities from VFI simulation.
    dl_epoch_results:
        ``{epoch: {stationary: …, derived: …}}``.
    econ_id:
        Economy index for the figure title.
    firm_idx:
        Index of the firm path to display.
    t_max:
        Number of time periods to show (capped to data length).

    Returns
    -------
    matplotlib.figure.Figure
    """
    vfi_K = vfi_stationary["K_curr"][firm_idx]

    # If the selected firm defaulted during the burn-in window, all its
    # post burn-in K values are NaN, producing blank panels for K, B,
    # I/K, and Eq (while Z would still show data because VFI pre-computes
    # Z_curr independently).  Auto-select the first surviving firm.
    if np.all(np.isnan(vfi_K)):
        surviving = np.where(~np.all(np.isnan(vfi_stationary["K_curr"]), axis=1))[0]
        if len(surviving) > 0:
            firm_idx = int(surviving[0])
            vfi_K = vfi_stationary["K_curr"][firm_idx]
            print(f"[trajectory] firm_idx=0 defaulted during burn-in; "
                  f"auto-selected firm_idx={firm_idx}")
        else:
            print("[trajectory] WARNING: all firms defaulted; trajectory will be blank")

    t_max = min(t_max, len(vfi_K))
    time = np.arange(t_max)

    specs = [
        ("Capital  (K)", "K", "K_curr", None),
        ("Productivity  (Z)", "Z", "Z_curr", None),
        ("Debt  (B)", "B", "B_curr", None),
        ("Investment Rate  (I/K)", "I/K", None, "investment_rate"),
        ("Equity Issuance  (Eq)", "Eq", None, "equity_issuance"),
    ]

    fig, axes = plt.subplots(5, 1, figsize=(14, 16), sharex=True)
    fig.suptitle(
        f"Sample Firm Trajectory  (Firm {firm_idx}, Econ ID: {econ_id})",
        fontsize=15, fontweight="bold", y=0.98,
    )

    dl_colors = ["#DD8452", "#55A868", "#C44E52", "#8172B3", "#CCB974",
                 "#64B5CD", "#4C72B0", "#917B56", "#D65F5F", "#8EBA42"]

    for ax, (title, ylabel, state_key, derived_key) in zip(axes, specs):
        # VFI line
        if state_key is not None:
            vfi_path = vfi_stationary[state_key][firm_idx][:t_max]
        else:
            vfi_path = vfi_derived[derived_key][firm_idx][:t_max]
        ax.plot(time, vfi_path, color="#4C72B0", linewidth=1.6, label="VFI", alpha=0.9)

        # DL lines
        for i, (epoch, res) in enumerate(sorted(dl_epoch_results.items())):
            if state_key is not None:
                dl_path = res["stationary"][state_key][firm_idx][:t_max]
            else:
                dl_path = res["derived"][derived_key][firm_idx][:t_max]
            c = dl_colors[i % len(dl_colors)]
            ax.plot(time, dl_path, color=c, linewidth=1.2, label=f"DL Ep {epoch}",
                    alpha=0.8, linestyle="--")

        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="semibold")
        ax.legend(fontsize=8, loc="upper right", framealpha=0.9,
                  edgecolor="#cccccc", ncol=2)
        ax.grid(True, alpha=0.2, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].set_xlabel("Time Period", fontsize=11)
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    return fig


def main() -> None:
    """Run VFI and multi-epoch DL simulations, then generate comparison figures."""
    # --- Global plot style ---
    plt.rcParams.update({
        "font.size": 14,
        "font.family": "sans-serif",
        "axes.titlesize": 15,
        "axes.labelsize": 13,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.dpi": 150,
        "savefig.dpi": 250,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.2,
    })

    parser = argparse.ArgumentParser(
        description="Risky debt model: DL vs VFI simulation comparison"
    )
    parser.add_argument(
        "--econ-id", type=int, default=0, choices=range(len(DEFAULT_ECON_LIST)),
        help="Index into the default economy list (default: 0)",
    )
    args = parser.parse_args()

    econ_id: int = args.econ_id
    econ_spec = DEFAULT_ECON_LIST[econ_id]

    # --- Load configuration ---
    econ_params = load_econ_params(econ_spec)
    bonds_config = load_bonds_config(econ_spec, econ_params)
    delta = econ_params.depreciation_rate
    alpha = econ_params.capital_share

    # --- Generate synthetic data ---
    initial_states, shock_sequence = setup_simulation_data(
        econ_params, bonds_config, batch_size=BATCH_SIZE, time_periods=T_PERIODS,
    )

    # --- VFI baseline ---
    vfi_raw = run_vfi_simulation(econ_params, econ_spec, initial_states, shock_sequence)
    vfi_stationary = apply_burn_in(vfi_raw, BURN_IN)
    vfi_derived = compute_all_derived_quantities(
        vfi_stationary, delta, alpha, include_debt=True
    )

    vfi_flat = {
        "output": vfi_derived["output"].flatten(),
        "capital": vfi_stationary["K_curr"].flatten(),
        "debt": vfi_stationary["B_curr"].flatten(),
        "leverage": vfi_derived["leverage"].flatten(),
        "investment": vfi_derived["investment"].flatten(),
        "investment_rate": vfi_derived["investment_rate"].flatten(),
        "equity_issuance": vfi_derived["equity_issuance"].flatten(),
        "equity_issuance_rate": vfi_derived["equity_issuance_rate"].flatten(),
    }

    vfi_moments = extract_variable_moments(vfi_stationary, vfi_derived)
    vfi_freqs = extract_freq_stats(vfi_derived)

    print("\nVFI Moments:")
    for var, stats in vfi_moments.items():
        print(f"  {var:18s}  mean={stats['mean']:.4f}  std={stats['std']:.4f}  ac1={stats['ac1']:.4f}")
    print("VFI Frequencies:")
    for k, v in vfi_freqs.items():
        print(f"  {k:18s}  {v:.4f}")

    # --- DL simulations across epochs ---
    dl_simulator = load_dl_simulator(econ_params, bonds_config)
    dl_epoch_flat: Dict[int, Dict] = {}
    dl_moments_by_epoch: Dict[int, Dict] = {}
    dl_freqs_by_epoch: Dict[int, Dict[str, float]] = {}

    for epoch in DL_EPOCHS:
        print(f"\nRunning DL epoch {epoch} ...")
        result = run_dl_epoch(
            dl_simulator, epoch, initial_states, shock_sequence,
            BURN_IN, delta, alpha,
        )
        dl_epoch_flat[epoch] = result
        dl_moments_by_epoch[epoch] = extract_variable_moments(
            result["stationary"], result["derived"]
        )
        dl_freqs_by_epoch[epoch] = extract_freq_stats(result["derived"])

        print(f"  DL Epoch {epoch} moments:")
        for var, stats in dl_moments_by_epoch[epoch].items():
            print(f"    {var:18s}  mean={stats['mean']:.4f}  std={stats['std']:.4f}  ac1={stats['ac1']:.4f}")

    # --- Generate figures ---
    os.makedirs(RESULTS_DIR, exist_ok=True)
    epoch_colors = plt.cm.Set2(np.linspace(0, 0.8, len(DL_EPOCHS) + 1))

    fig1 = plot_stationary_distributions(
        vfi_flat, dl_epoch_flat, epoch_colors, econ_id
    )
    outfile1 = os.path.join(RESULTS_DIR, "risky_simulation_distributions.png")
    fig1.savefig(outfile1)
    print(f"\nFigure 1 saved to {outfile1}")

    fig2 = plot_moments_figure(
        vfi_moments, dl_moments_by_epoch,
        vfi_freqs, dl_freqs_by_epoch,
        epoch_colors, econ_id,
    )
    outfile2 = os.path.join(RESULTS_DIR, "risky_moments_combined.png")
    fig2.savefig(outfile2)
    print(f"Figure 2 saved to {outfile2}")

    fig3 = plot_sample_trajectories(
        vfi_stationary, vfi_derived, dl_epoch_flat, econ_id,
    )
    outfile3 = os.path.join(RESULTS_DIR, "risky_simulation_trajectories.png")
    fig3.savefig(outfile3)
    print(f"Figure 3 saved to {outfile3}")

    print("\nSimulation and plotting completed successfully.")


if __name__ == "__main__":
    main()
