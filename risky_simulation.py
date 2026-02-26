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

from src.econ_models.config.bond_config import BondsConfig
from src.econ_models.config.dl_config import load_dl_config
from src.econ_models.config.economic_params import EconomicParams
from src.econ_models.io.file_utils import load_json_file
from src.econ_models.moment_calculator.compute_autocorrelation import (
    compute_autocorrelation_lags_1_to_5,
)
from src.econ_models.moment_calculator.compute_derived_quantities import (
    compute_all_derived_quantities,
)
from src.econ_models.moment_calculator.compute_inaction_rate import compute_inaction_rate
from src.econ_models.moment_calculator.compute_mean import compute_global_mean
from src.econ_models.moment_calculator.compute_std import compute_global_std
from src.econ_models.simulator import DLSimulatorRiskyFinal
from src.econ_models.simulator import VFISimulator_risky
from src.econ_models.simulator import synthetic_data_generator

# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------

ECON_LIST: List[List[float]] = [
    [0.6, 0.17, 1.0, 0.02, 0.1, 0.08],
    [0.5, 0.23, 1.5, 0.01, 0.1, 0.1],
]
"""Default economy parameter sets: [theta, rho, psi0, psi1, eta0, eta1]."""

BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.dirname("./")))

VFI_N_K: int = 560
VFI_N_D: int = 560
"""Golden VFI grid resolution."""

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

VARIABLE_NAMES: List[str] = [
    "Output", "Capital", "Debt", "Leverage",
    "Investment", "Inv. Rate", "Eq. Issuance", "Eq. Iss. Rate",
]
"""Economic variables tracked across moments and plots."""

MOMENT_KEYS: List[str] = ["mean", "std", "ac1"]
MOMENT_LABELS: List[str] = ["Mean", "Std Dev", "AutoCorr (L1)"]


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def to_python_float(value: Any) -> float:
    """Coerce a tensor, numpy scalar, or number to a plain Python float."""
    if value is None:
        return 0.0
    if hasattr(value, "numpy"):
        return float(value.numpy())
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def econ_tag(econ_spec: Sequence[float]) -> str:
    """Build a filename-safe tag from the economy parameter list."""
    return "_".join(str(x) for x in econ_spec)


def apply_burn_in(
    results: Dict[str, np.ndarray], burn_in: int
) -> Dict[str, np.ndarray]:
    """Discard the first *burn_in* time-periods from simulation output."""
    return {
        key: val[:, burn_in:]
        for key, val in results.items()
        if isinstance(val, np.ndarray) and val.ndim == 2
    }


def compute_variable_moments(data: np.ndarray) -> Dict[str, float]:
    """Compute mean, std, and lag-1 autocorrelation for a 2-D array."""
    ac = compute_autocorrelation_lags_1_to_5(data)
    return {
        "mean": to_python_float(compute_global_mean(data)),
        "std": to_python_float(compute_global_std(data)),
        "ac1": to_python_float(ac["lag_1"]),
    }


def compute_weighted_error(
    vfi_moments: Dict[str, Dict[str, float]],
    dl_moments: Dict[str, Dict[str, float]],
    variable_names: Sequence[str],
    moment_keys: Sequence[str] = ("mean", "std", "ac1"),
    *,
    vfi_freqs: Dict[str, float] | None = None,
    dl_freqs: Dict[str, float] | None = None,
) -> float:
    """Weighted average percentage error across all moments and frequencies."""
    categories: List[float] = []
    for var in variable_names:
        sub_errors: List[float] = []
        for mk in moment_keys:
            vfi_v = float(vfi_moments[var][mk])
            dl_v = float(dl_moments[var][mk])
            denom = abs(vfi_v) if abs(vfi_v) > 1e-12 else 1.0
            sub_errors.append(abs(dl_v - vfi_v) / denom * 100)
        categories.append(float(np.mean(sub_errors)))

    if vfi_freqs and dl_freqs:
        for key in vfi_freqs:
            vfi_v = vfi_freqs[key]
            dl_v = dl_freqs.get(key, 0.0)
            denom = abs(vfi_v) if abs(vfi_v) > 1e-12 else 1.0
            categories.append(abs(dl_v - vfi_v) / denom * 100)

    return float(np.mean(categories))


# ---------------------------------------------------------------------------
#  Data loading / simulation
# ---------------------------------------------------------------------------

def load_config(econ_spec: Sequence[float]):
    """Load EconomicParams and BondsConfig for a risky economy."""
    tag = econ_tag(econ_spec)
    econ_params_file = os.path.join(
        BASE_DIR, f"hyperparam/prefixed/econ_params_risky_{tag}.json"
    )
    boundary_file = os.path.join(
        BASE_DIR, f"hyperparam/autogen/bounds_risky_{tag}.json"
    )
    econ_params = EconomicParams(**load_json_file(econ_params_file))
    bonds_config = BondsConfig.validate_and_load(
        bounds_file=boundary_file, current_params=econ_params
    )
    return econ_params, bonds_config


def setup_simulation_data(econ_params, bonds_config):
    """Generate synthetic initial states and shock sequences."""
    data_gen = synthetic_data_generator(
        econ_params_benchmark=econ_params,
        sample_bonds_config=bonds_config,
        batch_size=BATCH_SIZE,
        T_periods=T_PERIODS,
        include_debt=True,
    )
    return data_gen.gen()


def run_vfi_simulation(econ_params, econ_spec, initial_states, shock_sequence):
    """Load golden VFI solution and simulate."""
    tag = econ_tag(econ_spec)
    vfi_file = f"./ground_truth_risky/golden_vfi_risky_{tag}_{VFI_N_K}_{VFI_N_D}.npz"
    print(f"Loading VFI golden benchmark ({VFI_N_K},{VFI_N_D}) from {vfi_file} ...")
    vfi_sim = VFISimulator_risky(econ_params)
    solved = np.load(vfi_file, allow_pickle=True)
    vfi_sim.load_solved_vfi_solution(solved)
    results = vfi_sim.simulate(
        tuple(s.numpy() if hasattr(s, "numpy") else s for s in initial_states),
        shock_sequence.numpy() if hasattr(shock_sequence, "numpy") else shock_sequence,
    )
    print("  VFI simulation complete.")
    return results


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
#  Moments extraction helpers
# ---------------------------------------------------------------------------

def extract_variable_moments(
    stationary: Dict[str, np.ndarray],
    derived: Dict,
) -> Dict[str, Dict[str, float]]:
    """Compute moments for all tracked variables."""
    var_data_map = {
        "Output": derived["output"],
        "Capital": stationary["K_curr"],
        "Debt": stationary["B_curr"],
        "Leverage": derived["leverage"],
        "Investment": derived["investment"],
        "Inv. Rate": derived["investment_rate"],
        "Eq. Issuance": derived["equity_issuance"],
        "Eq. Iss. Rate": derived["equity_issuance_rate"],
    }
    return {name: compute_variable_moments(data) for name, data in var_data_map.items()}


def extract_freq_stats(derived: Dict) -> Dict[str, float]:
    """Compute inaction rate and equity issuance frequency."""
    return {
        "Inaction Rate": to_python_float(
            compute_inaction_rate(derived["investment_rate"], -0.001, 0.001)
        ),
        "Issuance Freq": to_python_float(
            compute_global_mean(derived["issuance_binary"])
        ),
    }


# ---------------------------------------------------------------------------
#  Plotting
# ---------------------------------------------------------------------------

def plot_histogram_comparison(
    ax,
    vfi_data: np.ndarray,
    dl_epoch_data: Dict[int, np.ndarray],
    title: str,
    xlabel: str,
    epoch_colors: np.ndarray,
    *,
    bins: int = 50,
) -> None:
    """Draw overlapping VFI vs. multi-epoch DL histograms on *ax*."""
    clean_vfi = vfi_data[np.isfinite(vfi_data)]
    ax.hist(
        clean_vfi, bins=bins, density=True, alpha=0.5,
        color="#2C7BB6", label="VFI (Ground Truth)",
        edgecolor="white", linewidth=0.6,
    )
    for idx, (epoch, dl_data) in enumerate(dl_epoch_data.items()):
        clean_dl = dl_data[np.isfinite(dl_data)]
        ax.hist(
            clean_dl, bins=bins, density=True, alpha=0.7,
            color=epoch_colors[idx + 1],
            label=f"DL Epoch {epoch}",
            histtype="step", linewidth=2.5,
        )
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.legend(fontsize=11, framealpha=0.9, edgecolor="#bbbbbb", fancybox=True)
    ax.grid(True, alpha=0.2, linestyle="--")
    ax.tick_params(labelsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_grouped_bars(
    ax,
    vfi_stats: Sequence[float],
    epoch_stats: Dict[int, Sequence[float]],
    labels: Sequence[str],
    epoch_colors: np.ndarray,
    *,
    title: str | None = None,
) -> None:
    """Draw grouped bar charts comparing VFI vs. multiple DL epochs."""
    vfi_vals = [float(v) for v in vfi_stats]
    n_groups = len(labels)
    n_bars = 1 + len(epoch_stats)
    width = 0.8 / n_bars
    x = np.arange(n_groups)

    rects_vfi = ax.bar(
        x - 0.4 + width / 2, vfi_vals, width,
        label="VFI", color="#2C7BB6", alpha=0.85,
        edgecolor="white", linewidth=0.6,
    )
    ax.bar_label(rects_vfi, padding=3, fmt="%.4f", fontsize=9)

    for idx, (epoch, dl_vals) in enumerate(epoch_stats.items()):
        dl_vals = [float(v) for v in dl_vals]
        offset = -0.4 + (idx + 1.5) * width
        rects_dl = ax.bar(
            x + offset, dl_vals, width,
            label=f"DL Ep {epoch}", color=epoch_colors[idx + 1], alpha=0.8,
            edgecolor="white", linewidth=0.6,
        )
        ax.bar_label(rects_dl, padding=3, fmt="%.4f", fontsize=9)

    ax.set_ylabel("Value", fontsize=12)
    if title:
        ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(fontsize=10, framealpha=0.9, edgecolor="#bbbbbb", fancybox=True)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.tick_params(labelsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


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
        "--econ-id", type=int, default=0, choices=range(len(ECON_LIST)),
        help="Index into the default economy list (default: 0)",
    )
    args = parser.parse_args()

    econ_id: int = args.econ_id
    econ_spec = ECON_LIST[econ_id]

    # --- Load configuration ---
    econ_params, bonds_config = load_config(econ_spec)
    delta = econ_params.depreciation_rate
    alpha = econ_params.capital_share

    # --- Generate synthetic data ---
    initial_states, shock_sequence = setup_simulation_data(econ_params, bonds_config)

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
