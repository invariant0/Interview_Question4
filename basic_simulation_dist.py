#!/usr/bin/env python3
"""Compare VFI and distributional DL simulations across multiple economies.

This script evaluates a *distributional* deep-learning model that operates
across different economic parameter sets against VFI ground truths.  For
**each** economy in the default list it:

1. Loads the VFI golden solution and a distributional DL checkpoint.
2. Simulates panels for both approaches.
3. Computes stationary distributions and summary moments.

Two multi-panel figures are produced:

* **Figure 1** — Stationary distributions (4 variables x N economies).
* **Figure 2** — Moments comparison with grouped bar charts, inaction
  rate, and weighted average error (6 rows x N economies).

A detailed moments table is printed to stdout for each economy / epoch.

Usage::

    python basic_simulation_dist.py
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from src.econ_models.config.dl_config import load_dl_config
from src.econ_models.config.economic_params import EconomicParams
from src.econ_models.config.bond_config import BondsConfig
from src.econ_models.io.file_utils import load_json_file
from src.econ_models.moment_calculator.compute_derived_quantities import (
    compute_all_derived_quantities,
)
from src.econ_models.moment_calculator.compute_inaction_rate import compute_inaction_rate
from src.econ_models.simulator import DLSimulatorBasicFinal_dist

from basic_common import (
    BASE_DIR,
    DEFAULT_ECON_LIST,
    apply_burn_in,
    compute_variable_moments,
    get_golden_vfi_path,
    load_bonds_config,
    load_econ_params,
    plot_grouped_bars,
    run_vfi_simulation,
    setup_simulation_data,
    tensor_to_numpy,
    to_python_float,
)

# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------

BURN_IN_PERIODS: int = 200
"""Number of leading periods discarded to reach stationarity."""

DL_EPOCHS: List[int] = [1200]
"""Training epochs at which DL distributional checkpoints are evaluated."""

RESULTS_DIR: str = "./results_dist"
"""Output directory for figures."""

CHECKPOINT_DIR: str = "./checkpoints_final_dist/basic"
"""Directory containing distributional DL weight files."""

VARIABLE_NAMES: List[str] = ["Capital", "Investment", "Inv. Rate", "Output"]
"""Economic variables tracked across moments and plots."""

VARIABLE_XLABELS: List[str] = ["Capital (K)", "Investment (I)", "Investment Rate (I/K)", "Output (Y)"]
"""Axis labels corresponding to each variable."""

# Inaction rate thresholds (symmetric about zero)
INACTION_LOWER: float = 0.0
INACTION_UPPER: float = 0.0

# Distributional DL configuration paths
DL_PARAMS_PATH: str = "./hyperparam_dist/prefixed/dl_params_dist.json"
DL_BONDS_PATH: str = os.path.join(BASE_DIR, "hyperparam_dist/autogen/bounds_basic_dist.json")
DL_ECON_PARAMS_PATH: str = os.path.join(BASE_DIR, "hyperparam_dist/prefixed/econ_params_basic_dist.json")


# ---------------------------------------------------------------------------
#  Distributional DL simulator setup
# ---------------------------------------------------------------------------

def create_distributional_dl_simulator() -> DLSimulatorBasicFinal_dist:
    """Build and configure the distributional DL simulator.

    The distributional model is trained once and can simulate under
    *different* economic parameters without re-training.

    Returns
    -------
    DLSimulatorBasicFinal_dist
    """
    dl_config = load_dl_config(DL_PARAMS_PATH, "basic_final")
    econ_params_dist = EconomicParams(**load_json_file(DL_ECON_PARAMS_PATH))
    bonds_dist = BondsConfig.validate_and_load(
        bounds_file=DL_BONDS_PATH, current_params=econ_params_dist,
    )
    dl_config.capital_max = bonds_dist["k_max"]
    dl_config.capital_min = bonds_dist["k_min"]
    dl_config.productivity_max = bonds_dist["z_max"]
    dl_config.productivity_min = bonds_dist["z_min"]
    return DLSimulatorBasicFinal_dist(dl_config, bonds_dist)


# ---------------------------------------------------------------------------
#  Per-economy simulation pipeline
# ---------------------------------------------------------------------------

def simulate_single_economy(
    econ_spec: List[float],
    dl_simulator: DLSimulatorBasicFinal_dist,
) -> Dict[str, Any]:
    """Run VFI and multi-epoch DL simulations for one economy.

    Parameters
    ----------
    econ_spec:
        Economy specification ``[rho, sigma, gamma, F]``.
    dl_simulator:
        Pre-built distributional DL simulator.

    Returns
    -------
    dict
        Contains VFI moments, DL epoch data, flat arrays for plotting,
        and a human-readable label.
    """
    label = f"\u03c1={econ_spec[0]}, \u03c3={econ_spec[1]}, \u03b3={econ_spec[2]}, F={econ_spec[3]}"
    print(f"\n{'=' * 60}\nRunning: {label}\n{'=' * 60}")

    econ_params = load_econ_params(econ_spec)
    bonds_config = load_bonds_config(econ_spec, econ_params)
    delta = econ_params.depreciation_rate
    alpha = econ_params.capital_share

    initial_states, shock_sequence = setup_simulation_data(econ_params, bonds_config)

    # --- VFI baseline ---
    vfi_solution = np.load(get_golden_vfi_path(econ_spec))
    vfi_raw = run_vfi_simulation(econ_params, vfi_solution, initial_states, shock_sequence)
    vfi_stationary = apply_burn_in(vfi_raw, BURN_IN_PERIODS)
    vfi_derived = compute_all_derived_quantities(vfi_stationary, delta, alpha)

    print(f"  VFI done. K mean={np.mean(vfi_raw['K_curr']):.4f}, std={np.std(vfi_raw['K_curr']):.4f}")

    # Compute VFI moments and flat data
    vfi_moments, vfi_flat = _extract_moments_and_flat(vfi_stationary, vfi_derived)
    vfi_inaction = to_python_float(compute_inaction_rate(
        vfi_derived["investment_rate"],
        lower_threshold=INACTION_LOWER,
        upper_threshold=INACTION_UPPER,
    ))

    # --- DL simulations across epochs ---
    dl_epoch_data: Dict[int, Dict[str, Any]] = {}
    for epoch in DL_EPOCHS:
        dl_simulator.load_solved_dl_solution(
            os.path.join(CHECKPOINT_DIR, f"basic_capital_policy_net_{epoch}.weights.h5"),
            os.path.join(CHECKPOINT_DIR, f"basic_investment_policy_net_{epoch}.weights.h5"),
        )
        dl_raw = dl_simulator.simulate(initial_states, shock_sequence, econ_params)
        print(f"  DL Epoch {epoch} done.")

        dl_stationary = apply_burn_in(dl_raw, BURN_IN_PERIODS)
        dl_derived = compute_all_derived_quantities(dl_stationary, delta, alpha)
        dl_moments, dl_flat = _extract_moments_and_flat(dl_stationary, dl_derived)
        dl_inaction = to_python_float(compute_inaction_rate(
            dl_derived["investment_rate"],
            lower_threshold=INACTION_LOWER,
            upper_threshold=INACTION_UPPER,
        ))
        dl_epoch_data[epoch] = {
            "moments": dl_moments,
            "inaction_rate": dl_inaction,
            "flat": dl_flat,
            "stationary": dl_stationary,
            "derived": dl_derived,
        }

    return {
        "label": label,
        "econ_spec": econ_spec,
        "vfi_moments": vfi_moments,
        "vfi_inaction": vfi_inaction,
        "vfi_flat": vfi_flat,
        "vfi_stationary": vfi_stationary,
        "vfi_derived": vfi_derived,
        "dl_epoch_data": dl_epoch_data,
    }


def _extract_moments_and_flat(
    stationary: Dict[str, np.ndarray],
    derived: Dict[str, np.ndarray],
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, np.ndarray]]:
    """Compute per-variable moments and flatten data for one simulation.

    Parameters
    ----------
    stationary:
        Burn-in-removed simulation output.
    derived:
        Derived economic quantities.

    Returns
    -------
    tuple
        ``(moments_dict, flat_dict)`` — moments keyed by variable name
        and flattened numpy arrays for histogram plotting.
    """
    variable_map = {
        "Capital": stationary["K_curr"],
        "Investment": derived["investment"],
        "Inv. Rate": derived["investment_rate"],
        "Output": derived["output"],
    }
    moments: Dict[str, Dict[str, float]] = {}
    flat: Dict[str, np.ndarray] = {}
    for name, data in variable_map.items():
        moments[name] = compute_variable_moments(data)
        flat[name] = tensor_to_numpy(data).flatten()
    return moments, flat


# ---------------------------------------------------------------------------
#  Plotting — Figure 1: stationary distributions
# ---------------------------------------------------------------------------

def plot_distributions_figure(
    all_results: List[Dict[str, Any]],
    epoch_colors: np.ndarray,
) -> plt.Figure:
    """Create multi-economy stationary-distribution comparison.

    Parameters
    ----------
    all_results:
        One entry per economy from :func:`simulate_single_economy`.
    epoch_colors:
        Colour array.

    Returns
    -------
    matplotlib.figure.Figure
    """
    n_vars = len(VARIABLE_NAMES)
    n_econ = len(all_results)
    hist_kwargs_vfi = {"bins": 100, "density": True, "alpha": 0.55, "edgecolor": "none"}
    vfi_color = "#1B4965"

    fig, axes = plt.subplots(n_vars, n_econ, figsize=(7 * n_econ, 4 * n_vars))
    fig.suptitle(
        "Multi-Economy Stationary Distributions \u2014 VFI vs Deep Learning",
        fontsize=15, fontweight="bold", y=0.98,
    )
    if n_econ == 1:
        axes = axes[:, np.newaxis]

    for col, result in enumerate(all_results):
        for row, (var, xlabel) in enumerate(zip(VARIABLE_NAMES, VARIABLE_XLABELS)):
            ax = axes[row, col]
            vfi_data = result["vfi_flat"][var]
            vfi_data = vfi_data[~np.isnan(vfi_data)]
            ax.hist(vfi_data, **hist_kwargs_vfi, color=vfi_color, label="VFI")

            annotation = f"VFI: \u03bc={np.mean(vfi_data):.3f}, \u03c3={np.std(vfi_data):.3f}"
            for ep_idx, epoch in enumerate(DL_EPOCHS):
                dl_data = result["dl_epoch_data"][epoch]["flat"][var]
                dl_data = dl_data[~np.isnan(dl_data)]
                ax.hist(
                    dl_data, bins=100, density=True, alpha=0.35,
                    color=epoch_colors[ep_idx + 1],
                    label=f"DL Ep {epoch}", histtype="step", linewidth=2,
                )
                annotation += f"\nDL Ep{epoch}: \u03bc={np.mean(dl_data):.3f}, \u03c3={np.std(dl_data):.3f}"

            ax.set_xlabel(xlabel)
            ax.set_ylabel("Density")
            if row == 0:
                ax.set_title(f"Econ {col}: {result['label']}", fontsize=10, fontweight="bold")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.text(
                0.97, 0.95, annotation, transform=ax.transAxes, fontsize=7,
                verticalalignment="top", horizontalalignment="right", fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
            )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# ---------------------------------------------------------------------------
#  Plotting — Figure 2: moments comparison
# ---------------------------------------------------------------------------

def plot_moments_figure(
    all_results: List[Dict[str, Any]],
    epoch_colors: np.ndarray,
) -> plt.Figure:
    """Create multi-economy moments comparison figure.

    Layout: 5 rows x N columns.
      - Rows 0-3: Capital / Output / Investment / Inv. Rate moments.
      - Row 4: Inaction rate.

    Parameters
    ----------
    all_results:
        One entry per economy.
    epoch_colors:
        Colour array.

    Returns
    -------
    matplotlib.figure.Figure
    """
    moment_keys = ["mean", "std", "ac1"]
    moment_labels = ["Mean", "Std", "AC(1)"]
    moment_vars = ["Capital", "Output", "Investment", "Inv. Rate"]
    n_econ = len(all_results)
    vfi_color = "#1B4965"  # deep navy for VFI baseline

    fig, axes = plt.subplots(5, n_econ, figsize=(7 * n_econ, 19))
    fig.suptitle(
        "Multi-Economy Moments Comparison \u2014 VFI vs Deep Learning",
        fontsize=15, fontweight="bold", y=0.98,
    )
    if n_econ == 1:
        axes = axes[:, np.newaxis]

    for col, result in enumerate(all_results):
        vm = result["vfi_moments"]

        # Rows 0-3: per-variable moments
        for row, var in enumerate(moment_vars):
            ax = axes[row, col]
            vfi_vals = [vm[var][k] for k in moment_keys]
            epoch_vals = {
                ep: [result["dl_epoch_data"][ep]["moments"][var][k] for k in moment_keys]
                for ep in DL_EPOCHS
            }
            title = f"Econ {col}: {result['label']}" if row == 0 else None
            plot_grouped_bars(ax, vfi_vals, epoch_vals, moment_labels, epoch_colors, title=title, vfi_color=vfi_color)
            ax.set_ylabel(var)

        # Row 4: inaction rate
        ax_ir = axes[4, col]
        vfi_ir = [result["vfi_inaction"]]
        epoch_ir = {ep: [result["dl_epoch_data"][ep]["inaction_rate"]] for ep in DL_EPOCHS}
        plot_grouped_bars(
            ax_ir, vfi_ir, epoch_ir, ["Inaction Rate"], epoch_colors,
            title="Inaction Rate (|I/K| \u2264 threshold)" if col == 0 else None,
            vfi_color=vfi_color,
        )
        ax_ir.set_ylabel("Rate")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# ---------------------------------------------------------------------------
#  Plotting — Figure 3: sample trajectories
# ---------------------------------------------------------------------------

def plot_trajectories_figure(
    all_results: List[Dict[str, Any]],
    *,
    firm_idx: int = 0,
    t_max: int = 200,
) -> plt.Figure:
    """Create multi-economy sample-firm trajectory comparison.

    For each economy a column of 3 subplots is drawn showing the time
    paths of Capital (K), Productivity (Z), and Investment Rate (I/K)
    for a single firm.

    Parameters
    ----------
    all_results:
        One entry per economy from :func:`simulate_single_economy`.
    firm_idx:
        Index of the firm path to display.
    t_max:
        Number of time periods to show (capped to data length).

    Returns
    -------
    matplotlib.figure.Figure
    """
    n_econ = len(all_results)
    specs = [
        ("Capital  ($K_t$)", "$K$", "K_curr"),
        ("Productivity  ($Z_t$)", "$Z$", "Z_curr"),
        ("Investment Rate  ($I_t / K_t$)", "$I/K$", None),
    ]

    # --- Colour palette ---------------------------------------------------
    vfi_color = "#A0C4E1"  # lighter blue so DL lines remain visible
    dl_palette = ["#E8553A", "#3BB273", "#7E57C2", "#FF9800", "#00ACC1"]

    # --- Figure setup ------------------------------------------------------
    fig, axes = plt.subplots(
        3, n_econ,
        figsize=(7.5 * n_econ, 9),
        sharex="col",
        gridspec_kw={"hspace": 0.28, "wspace": 0.22},
    )
    fig.suptitle(
        f"Multi-Economy Sample Firm Trajectory \u2014 VFI vs Deep Learning  (Firm {firm_idx})",
        fontsize=14, fontweight="bold", y=0.99,
    )
    if n_econ == 1:
        axes = axes[:, np.newaxis]

    for col, result in enumerate(all_results):
        vfi_stat = result["vfi_stationary"]
        vfi_der = result["vfi_derived"]
        T = min(
            t_max,
            vfi_stat["K_curr"].shape[1]
            if vfi_stat["K_curr"].ndim > 1
            else len(vfi_stat["K_curr"]),
        )
        time = np.arange(T)

        for row, (title, ylabel, state_key) in enumerate(specs):
            ax = axes[row, col]

            # --- VFI path --------------------------------------------------
            if state_key is not None:
                vfi_path = vfi_stat[state_key][firm_idx][:T]
            else:
                vfi_path = vfi_der["investment_rate"][firm_idx][:T]

            ax.plot(
                time, vfi_path,
                color=vfi_color, linewidth=1.2,
                label="VFI (ground truth)", alpha=0.55, zorder=2,
            )

            # --- DL paths per epoch ----------------------------------------
            for ep_idx, epoch in enumerate(DL_EPOCHS):
                ep_data = result["dl_epoch_data"][epoch]
                if state_key is not None:
                    dl_path = ep_data["stationary"][state_key][firm_idx][:T]
                else:
                    dl_path = ep_data["derived"]["investment_rate"][firm_idx][:T]

                c = dl_palette[ep_idx % len(dl_palette)]
                ax.plot(
                    time, dl_path,
                    color=c, linewidth=1.4,
                    label=f"DL Epoch {epoch}",
                    alpha=0.90, linestyle="--", zorder=3,
                )

            # --- Annotation: summary stats in a corner box ----------------
            if state_key is not None:
                vfi_mu = np.mean(vfi_path)
                vfi_sig = np.std(vfi_path)
                ann = f"VFI: \u03bc={vfi_mu:.3f}, \u03c3={vfi_sig:.3f}"
                for ep_idx, epoch in enumerate(DL_EPOCHS):
                    ep_data = result["dl_epoch_data"][epoch]
                    dl_arr = ep_data["stationary"][state_key][firm_idx][:T]
                    ann += f"\nEp{epoch}: \u03bc={np.mean(dl_arr):.3f}, \u03c3={np.std(dl_arr):.3f}"
                ax.text(
                    0.98, 0.96, ann,
                    transform=ax.transAxes, fontsize=7,
                    verticalalignment="top", horizontalalignment="right",
                    fontfamily="monospace",
                    bbox=dict(
                        boxstyle="round,pad=0.3", facecolor="white",
                        edgecolor="#cccccc", alpha=0.85,
                    ),
                )

            # --- Axes styling ---------------------------------------------
            ax.set_ylabel(ylabel, fontsize=11, labelpad=6)
            ax.tick_params(axis="both", labelsize=9, direction="in",
                           top=False, right=False)
            ax.grid(True, alpha=0.20, linestyle=":", linewidth=0.6)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_linewidth(0.6)
            ax.spines["bottom"].set_linewidth(0.6)

            if row == 0:
                ax.set_title(
                    f"Econ {col}: {result['label']}",
                    fontsize=10, fontweight="bold", pad=8,
                )

        # x-label only on the bottom row
        axes[-1, col].set_xlabel("Time Period ($t$)", fontsize=11, labelpad=6)

    # --- Shared legend at the bottom of the figure -------------------------
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=len(labels),
        fontsize=10,
        frameon=True,
        fancybox=True,
        shadow=False,
        edgecolor="#cccccc",
        borderpad=0.6,
        columnspacing=1.8,
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    return fig


# ---------------------------------------------------------------------------
#  Console output
# ---------------------------------------------------------------------------

def print_moments_tables(all_results: List[Dict[str, Any]]) -> None:
    """Print detailed per-economy moments tables to stdout.

    Parameters
    ----------
    all_results:
        One entry per economy.
    """
    moment_vars = ["Capital", "Output", "Investment", "Inv. Rate"]

    for idx, result in enumerate(all_results):
        vm = result["vfi_moments"]
        for epoch in DL_EPOCHS:
            dm = result["dl_epoch_data"][epoch]["moments"]
            print(f"\n{'=' * 90}")
            print(f"MOMENTS TABLE \u2014 Econ {idx}: {result['label']}  |  DL Epoch {epoch}")
            print(f"{'=' * 90}")
            print(f"{'Variable':<16} {'Moment':<10} {'VFI':<14} {'DL':<14} {'Rel Err (%)':<14}")
            print("-" * 68)

            for var in moment_vars:
                for label, key in [("Mean", "mean"), ("Std", "std"), ("AC(1)", "ac1")]:
                    vfi_v = vm[var][key]
                    dl_v = dm[var][key]
                    denom = max(abs(vfi_v), 1e-12)
                    rel_err = abs(dl_v - vfi_v) / denom * 100
                    row_label = var if label == "Mean" else ""
                    print(f"{row_label:<16} {label:<10} {vfi_v:<14.6f} {dl_v:<14.6f} {rel_err:<14.2f}")
                print("-" * 68)

            vfi_ir = result["vfi_inaction"]
            dl_ir = result["dl_epoch_data"][epoch]["inaction_rate"]
            denom_ir = max(abs(vfi_ir), 1e-12)
            ir_err = abs(dl_ir - vfi_ir) / denom_ir * 100
            print(f"{'Inaction':<16} {'Rate':<10} {vfi_ir:<14.6f} {dl_ir:<14.6f} {ir_err:<14.2f}")
            print("=" * 90)


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run distributional DL vs VFI comparison for all economies."""
    dl_simulator = create_distributional_dl_simulator()

    all_results: List[Dict[str, Any]] = []
    for econ_spec in DEFAULT_ECON_LIST:
        result = simulate_single_economy(econ_spec, dl_simulator)
        all_results.append(result)

    # Colours — distinct, accessible palette
    _base_palette = ["#1B4965", "#E8553A", "#3BB273", "#7E57C2", "#FF9800", "#00ACC1"]
    epoch_colors = np.array(
        [plt.cm.colors.to_rgba(c) for c in _base_palette[: len(DL_EPOCHS) + 1]]
    )

    # Figure 1: stationary distributions
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fig1 = plot_distributions_figure(all_results, epoch_colors)
    fig1.savefig(
        os.path.join(RESULTS_DIR, "figure1_stationary_distributions.png"),
        dpi=300, bbox_inches="tight",
    )
    print(f"\nFigure 1 saved \u2192 {RESULTS_DIR}/figure1_stationary_distributions.png")

    # Figure 2: moments comparison
    fig2 = plot_moments_figure(all_results, epoch_colors)
    fig2.savefig(
        os.path.join(RESULTS_DIR, "figure2_moments_comparison.png"),
        dpi=300, bbox_inches="tight",
    )
    print(f"Figure 2 saved \u2192 {RESULTS_DIR}/figure2_moments_comparison.png")

    # Figure 3: sample trajectories
    fig3 = plot_trajectories_figure(all_results)
    fig3.savefig(
        os.path.join(RESULTS_DIR, "figure3_sample_trajectories.png"),
        dpi=300, bbox_inches="tight",
    )
    print(f"Figure 3 saved \u2192 {RESULTS_DIR}/figure3_sample_trajectories.png")

    # Detailed console tables
    print_moments_tables(all_results)

    plt.show()


if __name__ == "__main__":
    main()
