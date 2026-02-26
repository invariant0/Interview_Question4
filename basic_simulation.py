#!/usr/bin/env python3
"""Compare VFI and Deep Learning simulation results for the basic model.

This script loads a ground-truth VFI solution and DL checkpoints at
several training epochs, simulates both, and produces two figures:

1. **Stationary distributions** — overlapping histograms for capital,
   output, investment, and investment rate.
2. **Moments comparison** — grouped bar charts for mean, standard
   deviation, and lag-1 autocorrelation of the same variables, plus
   inaction rate and a weighted-average error score across epochs.

Usage::

    python basic_simulation.py [--econ-id 0]
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from src.econ_models.config.dl_config import load_dl_config
from src.econ_models.config.economic_params import EconomicParams
from src.econ_models.moment_calculator.compute_derived_quantities import (
    compute_all_derived_quantities,
)
from src.econ_models.moment_calculator.compute_inaction_rate import compute_inaction_rate
from src.econ_models.simulator import DLSimulatorBasicFinal

from basic_common import (
    DEFAULT_ECON_LIST,
    apply_burn_in,
    compute_variable_moments,
    get_golden_vfi_path,
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

BURN_IN_PERIODS: int = 200
"""Number of leading periods discarded to reach stationarity."""

DL_EPOCHS: List[int] = [120]
"""Training epochs at which DL checkpoints are evaluated."""

RESULTS_DIR: str = "./results/effectiveness_basic"
"""Output directory for figures."""

CHECKPOINT_DIR: str = "./checkpoints_final/basic"
"""Directory containing DL weight files."""

VARIABLE_NAMES: List[str] = ["Capital", "Investment", "Inv. Rate", "Output"]
"""Economic variables tracked across moments and plots."""


# ---------------------------------------------------------------------------
#  Data loading / simulation
# ---------------------------------------------------------------------------

def load_dl_simulator(
    econ_params: EconomicParams,
    bonds_config: Dict,
) -> DLSimulatorBasicFinal:
    """Create and configure a DL simulator for the basic model.

    Parameters
    ----------
    econ_params:
        Benchmark economic parameters.
    bonds_config:
        Validated bonds/bounds configuration.

    Returns
    -------
    DLSimulatorBasicFinal
    """
    dl_config = load_dl_config("./hyperparam/prefixed/dl_params.json", "basic")
    dl_config.capital_max = bonds_config["k_max"]
    dl_config.capital_min = bonds_config["k_min"]
    dl_config.productivity_max = bonds_config["z_max"]
    dl_config.productivity_min = bonds_config["z_min"]
    return DLSimulatorBasicFinal(dl_config, econ_params)


def run_dl_epoch(
    dl_simulator: DLSimulatorBasicFinal,
    epoch: int,
    initial_states,
    shock_sequence,
    burn_in: int,
    delta: float,
    alpha: float,
) -> Dict:
    """Load a DL checkpoint, simulate, and compute derived quantities.

    Parameters
    ----------
    dl_simulator:
        Configured DL simulator instance.
    epoch:
        Training epoch to load.
    initial_states:
        Initial-state tensors.
    shock_sequence:
        Innovation/shock sequence tensor.
    burn_in:
        Number of burn-in periods.
    delta:
        Depreciation rate.
    alpha:
        Capital share.

    Returns
    -------
    dict
        Keys: ``stationary``, ``derived``, ``capital``, ``investment``,
        ``investment_rate``, ``output`` (flattened arrays).
    """
    dl_simulator.load_solved_dl_solution(
        os.path.join(CHECKPOINT_DIR, f"basic_capital_policy_net_{epoch}.weights.h5"),
        os.path.join(CHECKPOINT_DIR, f"basic_investment_policy_net_{epoch}.weights.h5"),
        os.path.join(CHECKPOINT_DIR, f"basic_value_net_{epoch}.weights.h5"),
    )
    raw_results = dl_simulator.simulate(initial_states, shock_sequence)
    stationary = apply_burn_in(raw_results, burn_in)
    derived = compute_all_derived_quantities(stationary, delta, alpha)
    return {
        "stationary": stationary,
        "derived": derived,
        "capital": stationary["K_curr"].flatten(),
        "investment": derived["investment"].flatten(),
        "investment_rate": derived["investment_rate"].flatten(),
        "output": derived["output"].flatten(),
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
    """Create the 2x2 stationary-distribution comparison figure.

    Parameters
    ----------
    vfi_flat:
        ``{variable_key: flattened_array}`` for VFI baseline.
    dl_epoch_results:
        ``{epoch: {variable_key: flattened_array}}``.
    epoch_colors:
        Color palette array.
    econ_id:
        Economy index used in the figure title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    plot_specs = [
        ("Capital Stock (K)", "Capital", "capital"),
        ("Output (Y)", "Output", "output"),
        ("Investment (I)", "Investment", "investment"),
        ("Investment Rate (I/K)", "I/K", "investment_rate"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        f"Stationary Distribution Comparison  (Econ ID: {econ_id})",
        fontsize=16, fontweight="bold", y=0.98,
    )

    for ax, (title, xlabel, key) in zip(axes.flat, plot_specs):
        epoch_data = {ep: dl_epoch_results[ep][key] for ep in dl_epoch_results}
        plot_histogram_comparison(ax, vfi_flat[key], epoch_data, title, xlabel, epoch_colors)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


def plot_moments_figure(
    vfi_moments: Dict[str, Dict[str, float]],
    dl_moments_by_epoch: Dict[int, Dict[str, Dict[str, float]]],
    vfi_inaction: float,
    dl_inaction_by_epoch: Dict[int, float],
    epoch_colors: np.ndarray,
    econ_id: int,
) -> plt.Figure:
    """Create the moments-comparison figure (2x3 grid).

    Parameters
    ----------
    vfi_moments:
        ``{variable: {mean, std, ac1}}`` for VFI.
    dl_moments_by_epoch:
        ``{epoch: {variable: {mean, std, ac1}}}``.
    vfi_inaction:
        VFI inaction rate.
    dl_inaction_by_epoch:
        ``{epoch: inaction_rate}``.
    epoch_colors:
        Color palette.
    econ_id:
        Economy index.

    Returns
    -------
    matplotlib.figure.Figure
    """
    moment_keys = ["mean", "std", "ac1"]
    moment_labels = ["Mean", "Std Dev", "AutoCorr (L1)"]
    epochs = sorted(dl_moments_by_epoch.keys())

    # 5 panels: 4 variable moments + inaction rate → 2 rows x 3 cols (last cell empty)
    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    fig.suptitle(
        f"Moments Comparison — VFI vs. Deep Learning  (Econ ID: {econ_id})",
        fontsize=16, fontweight="bold", y=0.98,
    )

    # Variable moments panels
    panel_axes = [axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0]]
    for ax, var in zip(panel_axes, VARIABLE_NAMES):
        vfi_vals = [vfi_moments[var][k] for k in moment_keys]
        epoch_vals = {ep: [dl_moments_by_epoch[ep][var][k] for k in moment_keys] for ep in epochs}
        plot_grouped_bars(ax, vfi_vals, epoch_vals, moment_labels, epoch_colors, title=f"{var} Moments")

    # Inaction rate panel
    ax_inaction = axes[1, 1]
    epoch_ir = {ep: [dl_inaction_by_epoch[ep]] for ep in epochs}
    plot_grouped_bars(ax_inaction, [vfi_inaction], epoch_ir, ["Inaction Rate"], epoch_colors, title="Inaction Rate")

    # Hide unused panel
    axes[1, 2].set_visible(False)

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    return fig


def plot_sample_trajectories(
    vfi_stationary: Dict[str, np.ndarray],
    vfi_derived: Dict[str, np.ndarray],
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
    # Extract VFI paths
    vfi_K = vfi_stationary["K_curr"][firm_idx]
    vfi_Z = vfi_stationary["Z_curr"][firm_idx]
    vfi_IR = vfi_derived["investment_rate"][firm_idx]
    t_max = min(t_max, len(vfi_K))
    time = np.arange(t_max)

    specs = [
        ("Capital  (K)", "K", "capital_level", "K_curr"),
        ("Productivity  (Z)", "Z", "productivity", "Z_curr"),
        ("Investment Rate  (I/K)", "I/K", "inv_rate", None),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(
        f"Sample Firm Trajectory  (Firm {firm_idx}, Econ ID: {econ_id})",
        fontsize=15, fontweight="bold", y=0.98,
    )

    for ax, (title, ylabel, spec_key, state_key) in zip(axes, specs):
        # VFI line
        if state_key is not None:
            vfi_path = vfi_stationary[state_key][firm_idx][:t_max]
        else:
            vfi_path = vfi_IR[:t_max]
        ax.plot(time, vfi_path, color="#4C72B0", linewidth=1.6, label="VFI", alpha=0.9)

        # DL lines
        dl_colors = ["#DD8452", "#55A868", "#C44E52", "#8172B3", "#CCB974"]
        for i, (epoch, res) in enumerate(sorted(dl_epoch_results.items())):
            if state_key is not None:
                dl_path = res["stationary"][state_key][firm_idx][:t_max]
            else:
                dl_path = res["derived"]["investment_rate"][firm_idx][:t_max]
            c = dl_colors[i % len(dl_colors)]
            ax.plot(time, dl_path, color=c, linewidth=1.2, label=f"DL Ep {epoch}",
                    alpha=0.8, linestyle="--")

        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="semibold")
        ax.legend(fontsize=9, loc="upper right", framealpha=0.9, edgecolor="#cccccc")
        ax.grid(True, alpha=0.2, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].set_xlabel("Time Period", fontsize=11)
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    return fig


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run VFI and DL simulations, then generate comparison figures."""
    # --- Global plot style ---
    plt.rcParams.update({
        "font.size": 12,
        "font.family": "sans-serif",
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
    })

    parser = argparse.ArgumentParser(description="Basic model simulation comparison")
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
    initial_states, shock_sequence = setup_simulation_data(econ_params, bonds_config)

    # --- VFI baseline ---
    vfi_solution = np.load(get_golden_vfi_path(econ_spec))
    print("Loaded VFI solution. Starting simulation...")
    vfi_raw = run_vfi_simulation(econ_params, vfi_solution, initial_states, shock_sequence)
    vfi_stationary = apply_burn_in(vfi_raw, BURN_IN_PERIODS)
    vfi_derived = compute_all_derived_quantities(vfi_stationary, delta, alpha)

    vfi_flat = {
        "capital": vfi_stationary["K_curr"].flatten(),
        "investment": vfi_derived["investment"].flatten(),
        "investment_rate": vfi_derived["investment_rate"].flatten(),
        "output": vfi_derived["output"].flatten(),
    }

    vfi_moments: Dict[str, Dict[str, float]] = {}
    for var_name, data in [
        ("Capital", vfi_stationary["K_curr"]),
        ("Investment", vfi_derived["investment"]),
        ("Inv. Rate", vfi_derived["investment_rate"]),
        ("Output", vfi_derived["output"]),
    ]:
        vfi_moments[var_name] = compute_variable_moments(data)
    vfi_inaction = to_python_float(compute_inaction_rate(vfi_derived["investment_rate"]))

    # --- DL simulations across epochs ---
    dl_simulator = load_dl_simulator(econ_params, bonds_config)
    dl_epoch_flat: Dict[int, Dict] = {}
    dl_moments_by_epoch: Dict[int, Dict] = {}
    dl_inaction_by_epoch: Dict[int, float] = {}

    for epoch in DL_EPOCHS:
        print(f"Loaded DL solution from epoch {epoch}. Starting simulation...")
        result = run_dl_epoch(dl_simulator, epoch, initial_states, shock_sequence, BURN_IN_PERIODS, delta, alpha)
        dl_epoch_flat[epoch] = result

        dl_moments_by_epoch[epoch] = {}
        for var_name, data in [
            ("Capital", result["stationary"]["K_curr"]),
            ("Investment", result["derived"]["investment"]),
            ("Inv. Rate", result["derived"]["investment_rate"]),
            ("Output", result["derived"]["output"]),
        ]:
            dl_moments_by_epoch[epoch][var_name] = compute_variable_moments(data)
        dl_inaction_by_epoch[epoch] = to_python_float(
            compute_inaction_rate(result["derived"]["investment_rate"])
        )

    # --- Generate figures ---
    os.makedirs(RESULTS_DIR, exist_ok=True)
    epoch_colors = plt.cm.tab10(np.linspace(0, 1, max(len(DL_EPOCHS) + 1, 10)))

    fig1 = plot_stationary_distributions(vfi_flat, dl_epoch_flat, epoch_colors, econ_id)
    fig1.savefig(os.path.join(RESULTS_DIR, "basic_simulation_distributions.png"))

    fig2 = plot_moments_figure(
        vfi_moments, dl_moments_by_epoch, vfi_inaction, dl_inaction_by_epoch, epoch_colors, econ_id,
    )
    fig2.savefig(os.path.join(RESULTS_DIR, "basic_simulation_moments.png"))

    fig3 = plot_sample_trajectories(
        vfi_stationary, vfi_derived, dl_epoch_flat, econ_id,
    )
    fig3.savefig(os.path.join(RESULTS_DIR, "basic_simulation_trajectories.png"))

    print("Simulation and plotting completed successfully.")


if __name__ == "__main__":
    main()
