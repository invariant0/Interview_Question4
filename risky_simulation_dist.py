#!/usr/bin/env python3
"""Compare VFI and distributional DL simulations for the risky debt model.

This script evaluates a *distributional* deep-learning model that operates
across different economic parameter sets against VFI ground truths.  For
**each** economy in the default list it:

1. Loads the VFI golden solution and a distributional DL checkpoint.
2. Simulates panels for both approaches.
3. Computes stationary distributions and summary moments.

Three multi-panel figures are produced:

* **Figure 1** — Stationary distributions (8 variables x N economies).
* **Figure 2** — Moments comparison with grouped bar charts, frequency
  stats, and weighted average error (rows x N economies).
* **Figure 3** — Sample firm trajectories.

A detailed moments table is printed to stdout for each economy / epoch.

Usage::

    python risky_simulation_dist.py [--gpu GPU_ID]
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare VFI and distributional DL simulations (risky model).",
    )
    parser.add_argument(
        "--gpu", type=int, default=None,
        help="GPU device ID to use for TensorFlow (e.g. --gpu 0). "
             "If not specified, TF default behaviour is used.",
    )
    return parser.parse_args()


def configure_gpu(gpu_id: int | None) -> None:
    """Restrict TensorFlow to a single GPU identified by *gpu_id*.

    Sets ``CUDA_VISIBLE_DEVICES`` *before* TensorFlow is imported so
    that TF only ever sees the selected device.  After importing TF,
    memory growth is enabled to avoid reserving all VRAM.
    """
    if gpu_id is None:
        return
    # Set env var BEFORE TF is loaded so it only sees the target GPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"CUDA_VISIBLE_DEVICES set to {gpu_id}")

    import tensorflow as tf  # first TF import happens here

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("WARNING: No GPUs detected by TensorFlow after setting "
              f"CUDA_VISIBLE_DEVICES={gpu_id}.")
        return
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"TensorFlow sees {len(gpus)} GPU(s) after isolation: "
          f"{[g.name for g in gpus]}")


# Parse early so GPU env var is set before any TF import from submodules
_ARGS = parse_args()
configure_gpu(_ARGS.gpu)

import matplotlib.pyplot as plt
import numpy as np

# Ensure we load 'econ_models' from the local src directory
sys.path.insert(0, os.path.join(os.getcwd(), "src"))

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
from src.econ_models.simulator import DLSimulatorRiskyFinal_dist
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

DL_EPOCHS: List[int] = [1400]
"""Training epochs at which DL distributional checkpoints are evaluated."""

CHECKPOINT_DIR: str = "./checkpoints_final_dist/risky"
"""Directory containing distributional DL weight files."""

BATCH_SIZE: int = 10000
T_PERIODS: int = 1000
BURN_IN: int = 400

RESULTS_DIR: str = "./results_dist/effectiveness_risky"
"""Output directory for figures."""

VARIABLE_NAMES: List[str] = [
    "Output", "Capital", "Debt", "Leverage",
    "Investment", "Inv. Rate", "Eq. Issuance", "Eq. Iss. Rate",
]
"""Economic variables tracked across moments and plots."""

VARIABLE_XLABELS: List[str] = [
    "Output (Y)", "Capital (K)", "Debt (B)", "Leverage (B/K)",
    "Investment (I)", "Investment Rate (I/K)",
    "Equity Issuance (Eq)", "Eq. Issuance Rate (Eq/K)",
]
"""Axis labels corresponding to each variable."""

MOMENT_KEYS: List[str] = ["mean", "std", "ac1"]
MOMENT_LABELS: List[str] = ["Mean", "Std Dev", "AutoCorr (L1)"]

# Inaction rate thresholds
INACTION_LOWER: float = -0.001
INACTION_UPPER: float = 0.001

# Distributional DL configuration paths
DL_PARAMS_PATH: str = "./hyperparam_dist/prefixed/dl_params_dist.json"
DL_BONDS_PATH: str = os.path.join(BASE_DIR, "hyperparam_dist/autogen/bounds_risky_dist.json")
DL_ECON_PARAMS_PATH: str = os.path.join(BASE_DIR, "hyperparam_dist/prefixed/econ_params_risky_dist.json")


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


def tensor_to_numpy(tensor: Any) -> np.ndarray:
    """Convert a TensorFlow tensor to a NumPy array (no-op if already NumPy)."""
    import tensorflow as tf
    if isinstance(tensor, tf.Tensor):
        return tensor.numpy()
    return np.asarray(tensor)


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
#  Distributional DL simulator setup
# ---------------------------------------------------------------------------

def create_distributional_dl_simulator() -> DLSimulatorRiskyFinal_dist:
    """Build and configure the distributional DL simulator.

    The distributional model is trained once and can simulate under
    *different* economic parameters without re-training.

    Returns
    -------
    DLSimulatorRiskyFinal_dist
    """
    dl_config = load_dl_config(DL_PARAMS_PATH, "risky_final")
    econ_params_dist = EconomicParams(**load_json_file(DL_ECON_PARAMS_PATH))
    bonds_dist = BondsConfig.validate_and_load(
        bounds_file=DL_BONDS_PATH, current_params=econ_params_dist,
    )
    dl_config.capital_max = bonds_dist["k_max"]
    dl_config.capital_min = bonds_dist["k_min"]
    dl_config.productivity_max = bonds_dist["z_max"]
    dl_config.productivity_min = bonds_dist["z_min"]
    dl_config.debt_max = bonds_dist["b_max"]
    dl_config.debt_min = bonds_dist["b_min"]
    return DLSimulatorRiskyFinal_dist(dl_config, bonds_dist)


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


# ---------------------------------------------------------------------------
#  Moments extraction helpers
# ---------------------------------------------------------------------------

def extract_moments_and_flat(
    stationary: Dict[str, np.ndarray],
    derived: Dict,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, np.ndarray]]:
    """Compute per-variable moments and flatten data for one simulation.

    Returns
    -------
    tuple
        ``(moments_dict, flat_dict)``
    """
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
    moments: Dict[str, Dict[str, float]] = {}
    flat: Dict[str, np.ndarray] = {}
    for name, data in var_data_map.items():
        moments[name] = compute_variable_moments(data)
        flat[name] = tensor_to_numpy(data).flatten()
    return moments, flat


def extract_freq_stats(derived: Dict) -> Dict[str, float]:
    """Compute inaction rate and equity issuance frequency."""
    return {
        "Inaction Rate": to_python_float(
            compute_inaction_rate(
                derived["investment_rate"],
                INACTION_LOWER,
                INACTION_UPPER,
            )
        ),
        "Issuance Freq": to_python_float(
            compute_global_mean(derived["issuance_binary"])
        ),
    }


# ---------------------------------------------------------------------------
#  Per-economy simulation pipeline
# ---------------------------------------------------------------------------

def simulate_single_economy(
    econ_spec: List[float],
    dl_simulator: DLSimulatorRiskyFinal_dist,
) -> Dict[str, Any]:
    """Run VFI and multi-epoch DL simulations for one economy.

    Parameters
    ----------
    econ_spec:
        Economy specification ``[theta, rho, psi0, psi1, eta0, eta1]``.
    dl_simulator:
        Pre-built distributional DL simulator.

    Returns
    -------
    dict
        Contains VFI moments, DL epoch data, flat arrays for plotting,
        and a human-readable label.
    """
    label = (
        f"\u03b8={econ_spec[0]}, \u03c1={econ_spec[1]}, "
        f"\u03c8\u2080={econ_spec[2]}, \u03c8\u2081={econ_spec[3]}, "
        f"\u03b7\u2080={econ_spec[4]}, \u03b7\u2081={econ_spec[5]}"
    )
    print(f"\n{'=' * 70}\nRunning: {label}\n{'=' * 70}")

    econ_params, bonds_config = load_config(econ_spec)
    delta = econ_params.depreciation_rate
    alpha = econ_params.capital_share

    initial_states, shock_sequence = setup_simulation_data(econ_params, bonds_config)

    # --- VFI baseline ---
    tag = econ_tag(econ_spec)
    vfi_file = f"./ground_truth_risky/golden_vfi_risky_{tag}_{VFI_N_K}_{VFI_N_D}.npz"
    has_vfi = os.path.exists(vfi_file)

    vfi_moments = None
    vfi_freqs = None
    vfi_flat = None
    vfi_stationary = None
    vfi_derived = None

    if has_vfi:
        vfi_raw = run_vfi_simulation(econ_params, econ_spec, initial_states, shock_sequence)
        vfi_stationary = apply_burn_in(vfi_raw, BURN_IN)
        vfi_derived = compute_all_derived_quantities(
            vfi_stationary, delta, alpha, include_debt=True
        )
        vfi_moments, vfi_flat = extract_moments_and_flat(vfi_stationary, vfi_derived)
        vfi_freqs = extract_freq_stats(vfi_derived)

        print(f"  VFI done. K mean={np.nanmean(vfi_stationary['K_curr']):.4f}")
        print("  VFI Moments:")
        for var, stats in vfi_moments.items():
            print(f"    {var:18s}  mean={stats['mean']:.4f}  std={stats['std']:.4f}  ac1={stats['ac1']:.4f}")
        print("  VFI Frequencies:")
        for k, v in vfi_freqs.items():
            print(f"    {k:18s}  {v:.4f}")
    else:
        print(f"  WARNING: VFI golden file not found at {vfi_file} — skipping VFI baseline.")

    # --- DL simulations across epochs ---
    dl_epoch_data: Dict[int, Dict[str, Any]] = {}
    for epoch in DL_EPOCHS:
        print(f"\n  Running DL epoch {epoch} ...")
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
        dl_raw = dl_simulator.simulate(initial_states, shock_sequence, econ_params)
        print(f"  DL Epoch {epoch} done.")

        dl_stationary = apply_burn_in(dl_raw, BURN_IN)
        dl_derived = compute_all_derived_quantities(
            dl_stationary, delta, alpha, include_debt=True
        )
        dl_moments, dl_flat = extract_moments_and_flat(dl_stationary, dl_derived)
        dl_freqs = extract_freq_stats(dl_derived)

        dl_epoch_data[epoch] = {
            "moments": dl_moments,
            "freqs": dl_freqs,
            "flat": dl_flat,
            "stationary": dl_stationary,
            "derived": dl_derived,
        }

        print(f"  DL Epoch {epoch} moments:")
        for var, stats in dl_moments.items():
            print(f"    {var:18s}  mean={stats['mean']:.4f}  std={stats['std']:.4f}  ac1={stats['ac1']:.4f}")

    return {
        "label": label,
        "econ_spec": econ_spec,
        "has_vfi": has_vfi,
        "vfi_moments": vfi_moments,
        "vfi_freqs": vfi_freqs,
        "vfi_flat": vfi_flat,
        "vfi_stationary": vfi_stationary,
        "vfi_derived": vfi_derived,
        "dl_epoch_data": dl_epoch_data,
    }


# ---------------------------------------------------------------------------
#  Plotting helpers
# ---------------------------------------------------------------------------

def plot_grouped_bars(
    ax,
    vfi_stats: Sequence[float],
    epoch_stats: Dict[int, Sequence[float]],
    labels: Sequence[str],
    epoch_colors: np.ndarray,
    *,
    title: str | None = None,
    vfi_color: str = "#2E86AB",
) -> None:
    """Draw grouped bar charts comparing VFI vs. multiple DL epochs."""
    vfi_vals = [float(v) for v in vfi_stats]
    n_groups = len(labels)
    n_bars = 1 + len(epoch_stats)
    width = 0.8 / n_bars
    x = np.arange(n_groups)

    rects_vfi = ax.bar(
        x - 0.4 + width / 2, vfi_vals, width,
        label="VFI", color=vfi_color, alpha=0.85,
        edgecolor="white", linewidth=0.8,
    )
    ax.bar_label(rects_vfi, padding=3, fmt="%.4f", fontsize=11, fontweight="semibold")

    for idx, (epoch, dl_vals) in enumerate(epoch_stats.items()):
        dl_vals = [float(v) for v in dl_vals]
        offset = -0.4 + (idx + 1.5) * width
        rects_dl = ax.bar(
            x + offset, dl_vals, width,
            label="DL", color=epoch_colors[idx + 1], alpha=0.85,
            edgecolor="white", linewidth=0.8,
        )
        ax.bar_label(rects_dl, padding=3, fmt="%.4f", fontsize=11, fontweight="semibold")

    ax.set_ylabel("Value", fontsize=15)
    if title:
        ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=14)
    ax.legend(fontsize=13, framealpha=0.92, edgecolor="#999999", fancybox=True,
              shadow=True, borderpad=0.8)
    ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.7)
    ax.tick_params(labelsize=13, width=1.2, length=5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)


# ---------------------------------------------------------------------------
#  Plotting — Figure 1: stationary distributions
# ---------------------------------------------------------------------------

def _plot_distributions_panel(
    all_results: List[Dict[str, Any]],
    var_names: List[str],
    var_xlabels: List[str],
    part_label: str,
) -> plt.Figure:
    """Create a stationary-distribution comparison for a subset of variables.

    Parameters
    ----------
    all_results:
        One entry per economy from :func:`simulate_single_economy`.
    var_names:
        Variable names to include in this panel.
    var_xlabels:
        Axis labels corresponding to *var_names*.
    part_label:
        Human-readable part label (e.g. ``"Part A"``)

    Returns
    -------
    matplotlib.figure.Figure
    """
    n_vars = len(var_names)
    n_econ = len(all_results)
    hist_kwargs_vfi = {"bins": 100, "density": True, "alpha": 0.45, "edgecolor": "none"}
    vfi_color = "#2E86AB"
    dl_dist_color = "#E04040"  # bold red for DL — high contrast on white

    fig, axes = plt.subplots(n_vars, n_econ, figsize=(7 * n_econ, 4 * n_vars))
    fig.suptitle(
        f"Multi-Econ Risky Model: Stationary Distributions ({part_label})",
        fontsize=20, fontweight="bold", y=0.995,
    )
    if n_econ == 1:
        axes = axes[:, np.newaxis]
    if n_vars == 1:
        axes = axes[np.newaxis, :]

    for col, result in enumerate(all_results):
        for row, (var, xlabel) in enumerate(zip(var_names, var_xlabels)):
            ax = axes[row, col]
            ax.set_facecolor("#F7F7F7")  # very light grey background

            # VFI histogram
            annotation = ""
            if result["has_vfi"] and result["vfi_flat"] is not None:
                vfi_data = result["vfi_flat"][var]
                vfi_data = vfi_data[~np.isnan(vfi_data)]
                ax.hist(vfi_data, **hist_kwargs_vfi, color=vfi_color, label="VFI")
                annotation = f"VFI: \u03bc={np.mean(vfi_data):.3f}, \u03c3={np.std(vfi_data):.3f}"

            # DL epochs
            for ep_idx, epoch in enumerate(DL_EPOCHS):
                if epoch not in result["dl_epoch_data"]:
                    continue
                dl_data = result["dl_epoch_data"][epoch]["flat"][var]
                dl_data = dl_data[~np.isnan(dl_data)]
                # Filled histogram with transparency + bold step outline
                ax.hist(
                    dl_data, bins=100, density=True, alpha=0.25,
                    color=dl_dist_color,
                    label="_nolegend_", edgecolor="none",
                )
                ax.hist(
                    dl_data, bins=100, density=True,
                    color=dl_dist_color,
                    label="DL", histtype="step", linewidth=2.5, alpha=0.9,
                )
                annotation += f"\nDL: \u03bc={np.mean(dl_data):.3f}, \u03c3={np.std(dl_data):.3f}"

            ax.set_xlabel(xlabel, fontsize=14)
            ax.set_ylabel("Density", fontsize=14)
            if row == 0:
                ax.set_title(f"Econ {col}: {result['label']}", fontsize=13, fontweight="bold")
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.7)
            ax.tick_params(labelsize=12)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if annotation:
                ax.text(
                    0.97, 0.95, annotation, transform=ax.transAxes, fontsize=9,
                    verticalalignment="top", horizontalalignment="right",
                    fontfamily="monospace",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.85,
                              edgecolor="#999999"),
                )

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def plot_distributions_figures(
    all_results: List[Dict[str, Any]],
    epoch_colors: np.ndarray,
) -> Tuple[plt.Figure, plt.Figure]:
    """Create two stationary-distribution figures (Part A & Part B).

    Part A covers Output, Capital, Debt, Leverage.
    Part B covers Investment, Inv. Rate, Eq. Issuance, Eq. Iss. Rate.

    Returns
    -------
    tuple of (Figure, Figure)
    """
    mid = len(VARIABLE_NAMES) // 2  # 4
    fig_a = _plot_distributions_panel(
        all_results,
        VARIABLE_NAMES[:mid], VARIABLE_XLABELS[:mid],
        "Part A",
    )
    fig_b = _plot_distributions_panel(
        all_results,
        VARIABLE_NAMES[mid:], VARIABLE_XLABELS[mid:],
        "Part B",
    )
    return fig_a, fig_b


# ---------------------------------------------------------------------------
#  Plotting — Figure 2: moments comparison
# ---------------------------------------------------------------------------

def _plot_moments_panel(
    all_results: List[Dict[str, Any]],
    epoch_colors: np.ndarray,
    var_names: List[str],
    part_label: str,
    *,
    include_freq: bool = False,
) -> plt.Figure:
    """Create a moments comparison figure for a subset of variables.

    Parameters
    ----------
    all_results:
        One entry per economy.
    epoch_colors:
        Colour array.
    var_names:
        Variable names to include.
    part_label:
        Human-readable part label (e.g. ``"Part A"``).
    include_freq:
        If True, add a frequency-stats row at the bottom.

    Returns
    -------
    matplotlib.figure.Figure
    """
    n_econ = len(all_results)
    vfi_color = "#2E86AB"

    n_rows = len(var_names) + (1 if include_freq else 0)
    fig, axes = plt.subplots(n_rows, n_econ, figsize=(8 * n_econ, 4 * n_rows))
    fig.suptitle(
        f"Multi-Econ Risky Model: Moments & Frequencies ({part_label})",
        fontsize=20, fontweight="bold", y=0.995,
    )
    if n_econ == 1:
        axes = axes[:, np.newaxis]
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for col, result in enumerate(all_results):
        has_vfi = result["has_vfi"]
        vm = result["vfi_moments"]

        for row, var in enumerate(var_names):
            ax = axes[row, col]
            if has_vfi and vm is not None:
                vfi_vals = [vm[var][k] for k in MOMENT_KEYS]
            else:
                vfi_vals = [0.0] * len(MOMENT_KEYS)
            epoch_vals = {
                ep: [result["dl_epoch_data"][ep]["moments"][var][k] for k in MOMENT_KEYS]
                for ep in DL_EPOCHS
                if ep in result["dl_epoch_data"]
            }
            title = f"Econ {col}: {result['label']}" if row == 0 else None
            plot_grouped_bars(
                ax, vfi_vals, epoch_vals, MOMENT_LABELS, epoch_colors,
                title=title, vfi_color=vfi_color,
            )
            ax.set_ylabel(var)

        if include_freq:
            ax_freq = axes[len(var_names), col]
            freq_labels = ["Inaction Rate", "Issuance Freq"]
            if has_vfi and result["vfi_freqs"] is not None:
                vfi_freq_vals = [result["vfi_freqs"][k] for k in freq_labels]
            else:
                vfi_freq_vals = [0.0] * len(freq_labels)
            epoch_freq_vals = {
                ep: [result["dl_epoch_data"][ep]["freqs"][k] for k in freq_labels]
                for ep in DL_EPOCHS
                if ep in result["dl_epoch_data"]
            }
            plot_grouped_bars(
                ax_freq, vfi_freq_vals, epoch_freq_vals, freq_labels, epoch_colors,
                title="Frequency Stats" if col == 0 else None,
                vfi_color=vfi_color,
            )
            ax_freq.set_ylabel("Rate")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def plot_moments_figures(
    all_results: List[Dict[str, Any]],
    epoch_colors: np.ndarray,
) -> Tuple[plt.Figure, plt.Figure]:
    """Create two moments comparison figures (Part A & Part B).

    Part A covers Output, Capital, Debt, Leverage.
    Part B covers Investment, Inv. Rate, Eq. Issuance, Eq. Iss. Rate,
    plus the frequency-stats row.

    Returns
    -------
    tuple of (Figure, Figure)
    """
    mid = len(VARIABLE_NAMES) // 2  # 4
    fig_a = _plot_moments_panel(
        all_results, epoch_colors,
        VARIABLE_NAMES[:mid], "Part A",
        include_freq=False,
    )
    fig_b = _plot_moments_panel(
        all_results, epoch_colors,
        VARIABLE_NAMES[mid:], "Part B",
        include_freq=True,
    )
    return fig_a, fig_b


# ---------------------------------------------------------------------------
#  Plotting — Figure 3: sample trajectories
# ---------------------------------------------------------------------------

def plot_sample_trajectories(
    all_results: List[Dict[str, Any]],
    *,
    firm_idx: int = 0,
    t_max: int = 200,
) -> plt.Figure | None:
    """Plot time-series trajectories of K, Z, B, I/K, and Eq for a single firm.

    All economies are shown side-by-side (one column per economy).

    Parameters
    ----------
    all_results:
        One entry per economy from :func:`simulate_single_economy`.
    firm_idx:
        Index of the firm path to display.
    t_max:
        Number of time periods to show.

    Returns
    -------
    matplotlib.figure.Figure or None
        None if no economy has DL data.
    """
    # Filter to economies that have DL data
    valid_indices = [
        i for i, r in enumerate(all_results)
        if r["dl_epoch_data"]
    ]
    if not valid_indices:
        print("  No DL data available; skipping trajectory plot.")
        return None

    n_econ = len(valid_indices)

    specs = [
        ("Capital  (K)", "K", "K_curr", None),
        ("Productivity  (Z)", "Z", "Z_curr", None),
        ("Debt  (B)", "B", "B_curr", None),
        ("Investment Rate  (I/K)", "I/K", None, "investment_rate"),
        ("Equity Issuance  (Eq)", "Eq", None, "equity_issuance"),
    ]
    n_rows = len(specs)

    dl_colors = ["#DD8452", "#55A868", "#C44E52", "#8172B3", "#CCB974",
                 "#64B5CD", "#4C72B0", "#917B56", "#D65F5F", "#8EBA42"]

    fig, axes = plt.subplots(
        n_rows, n_econ,
        figsize=(10 * n_econ, 4.5 * n_rows),
        sharex="col",
        squeeze=False,
    )
    fig.suptitle(
        "Multi-Econ Risky Model: Sample Firm Trajectories",
        fontsize=20, fontweight="bold", y=0.995,
    )

    vfi_color = "#2E86AB"

    for col_idx, econ_idx in enumerate(valid_indices):
        result = all_results[econ_idx]
        first_epoch = DL_EPOCHS[0]
        dl_stat = result["dl_epoch_data"][first_epoch]["stationary"]

        # Auto-select a surviving firm for this economy
        local_firm = firm_idx
        K_path = dl_stat["K_curr"][local_firm]
        if np.all(np.isnan(K_path)):
            surviving = np.where(~np.all(np.isnan(dl_stat["K_curr"]), axis=1))[0]
            if len(surviving) > 0:
                local_firm = int(surviving[0])
                print(f"  [trajectory] Econ {econ_idx}: firm_idx=0 defaulted; "
                      f"auto-selected firm_idx={local_firm}")
            else:
                print(f"  [trajectory] Econ {econ_idx} WARNING: all firms defaulted")

        t_end = min(t_max, dl_stat["K_curr"].shape[1])
        time = np.arange(t_end)

        # Check if VFI stationary data is available for this economy
        has_vfi_traj = (
            result["has_vfi"]
            and result.get("vfi_stationary") is not None
            and result.get("vfi_derived") is not None
        )

        for row_idx, (title, ylabel, state_key, derived_key) in enumerate(specs):
            ax = axes[row_idx, col_idx]

            # --- VFI trajectory ---
            if has_vfi_traj:
                if state_key is not None:
                    vfi_path = result["vfi_stationary"][state_key][local_firm][:t_end]
                else:
                    vfi_path = result["vfi_derived"][derived_key][local_firm][:t_end]
                ax.plot(
                    time, vfi_path, color=vfi_color, linewidth=2.0,
                    label="VFI", alpha=0.9, linestyle="-",
                )

            # --- DL trajectories ---
            for i, (epoch, ep_data) in enumerate(sorted(result["dl_epoch_data"].items())):
                if state_key is not None:
                    dl_path = ep_data["stationary"][state_key][local_firm][:t_end]
                else:
                    dl_path = ep_data["derived"][derived_key][local_firm][:t_end]
                c = dl_colors[i % len(dl_colors)]
                ax.plot(
                    time, dl_path, color=c, linewidth=1.8,
                    label="DL", alpha=0.85,
                    linestyle="--",
                )

            # Styling
            ax.set_ylabel(ylabel, fontsize=15, fontweight="semibold")
            if row_idx == 0:
                ax.set_title(
                    f"Econ {econ_idx}: {result['label']}\n(Firm {local_firm})",
                    fontsize=16, fontweight="bold", pad=12,
                )
            else:
                ax.set_title(title, fontsize=14, fontweight="semibold")
            ax.legend(
                fontsize=12, loc="upper right", framealpha=0.92,
                edgecolor="#999999", ncol=min(len(DL_EPOCHS) + 1, 4),
                shadow=True, borderpad=0.8,
            )
            ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.7)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(labelsize=13, width=1.2, length=5)
            ax.spines["left"].set_linewidth(1.2)
            ax.spines["bottom"].set_linewidth(1.2)

        # X-axis label on the bottom row only
        axes[-1, col_idx].set_xlabel("Time Period", fontsize=15, fontweight="semibold")

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    return fig


# ---------------------------------------------------------------------------
#  Console output
# ---------------------------------------------------------------------------

def print_moments_tables(all_results: List[Dict[str, Any]]) -> None:
    """Print detailed per-economy moments tables to stdout."""
    for idx, result in enumerate(all_results):
        vm = result["vfi_moments"]
        vf = result["vfi_freqs"]
        has_vfi = result["has_vfi"]

        for epoch in DL_EPOCHS:
            if epoch not in result["dl_epoch_data"]:
                continue
            dm = result["dl_epoch_data"][epoch]["moments"]
            df = result["dl_epoch_data"][epoch]["freqs"]

            print(f"\n{'=' * 100}")
            print(f"MOMENTS TABLE \u2014 Econ {idx}: {result['label']}  |  DL Epoch {epoch}")
            print(f"{'=' * 100}")
            print(f"{'Variable':<20} {'Moment':<10} {'VFI':<14} {'DL':<14} {'Rel Err (%)':<14}")
            print("-" * 72)

            for var in VARIABLE_NAMES:
                for label, key in [("Mean", "mean"), ("Std", "std"), ("AC(1)", "ac1")]:
                    if has_vfi and vm is not None:
                        vfi_v = vm[var][key]
                    else:
                        vfi_v = float("nan")
                    dl_v = dm[var][key]
                    if has_vfi and vm is not None:
                        denom = max(abs(vfi_v), 1e-12)
                        rel_err = abs(dl_v - vfi_v) / denom * 100
                    else:
                        rel_err = float("nan")
                    row_label = var if label == "Mean" else ""
                    print(f"{row_label:<20} {label:<10} {vfi_v:<14.6f} {dl_v:<14.6f} {rel_err:<14.2f}")
                print("-" * 72)

            # Frequency stats
            for freq_key in ["Inaction Rate", "Issuance Freq"]:
                if has_vfi and vf is not None:
                    vfi_v = vf[freq_key]
                    denom = max(abs(vfi_v), 1e-12)
                    dl_v = df[freq_key]
                    rel_err = abs(dl_v - vfi_v) / denom * 100
                else:
                    vfi_v = float("nan")
                    dl_v = df[freq_key]
                    rel_err = float("nan")
                print(f"{freq_key:<20} {'Rate':<10} {vfi_v:<14.6f} {dl_v:<14.6f} {rel_err:<14.2f}")

            print("=" * 100)


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run distributional DL vs VFI comparison for all risky economies."""
    # --- Global plot style ---
    plt.rcParams.update({
        "font.size": 16,
        "font.family": "sans-serif",
        "axes.titlesize": 18,
        "axes.labelsize": 15,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 13,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.3,
        "axes.linewidth": 1.2,
        "lines.linewidth": 1.8,
    })

    dl_simulator = create_distributional_dl_simulator()

    all_results: List[Dict[str, Any]] = []
    for econ_spec in ECON_LIST:
        result = simulate_single_economy(econ_spec, dl_simulator)
        all_results.append(result)

    # Colours
    epoch_colors = plt.cm.viridis(np.linspace(0, 1, len(DL_EPOCHS) + 1))

    # Figure 1: stationary distributions (split into Part A & Part B)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fig1a, fig1b = plot_distributions_figures(all_results, epoch_colors)
    outfile1a = os.path.join(RESULTS_DIR, "risky_dist_stationary_distributions_part_a.png")
    fig1a.savefig(outfile1a, dpi=300, bbox_inches="tight")
    print(f"\nFigure 1a saved \u2192 {outfile1a}")
    outfile1b = os.path.join(RESULTS_DIR, "risky_dist_stationary_distributions_part_b.png")
    fig1b.savefig(outfile1b, dpi=300, bbox_inches="tight")
    print(f"Figure 1b saved \u2192 {outfile1b}")

    # Figure 2: moments comparison (split into Part A & Part B)
    fig2a, fig2b = plot_moments_figures(all_results, epoch_colors)
    outfile2a = os.path.join(RESULTS_DIR, "risky_dist_moments_comparison_part_a.png")
    fig2a.savefig(outfile2a, dpi=300, bbox_inches="tight")
    print(f"Figure 2a saved \u2192 {outfile2a}")
    outfile2b = os.path.join(RESULTS_DIR, "risky_dist_moments_comparison_part_b.png")
    fig2b.savefig(outfile2b, dpi=300, bbox_inches="tight")
    print(f"Figure 2b saved \u2192 {outfile2b}")

    # Figure 3: sample trajectories (all economies side by side)
    fig3 = plot_sample_trajectories(all_results)
    if fig3 is not None:
        outfile3 = os.path.join(RESULTS_DIR, "risky_dist_trajectories.png")
        fig3.savefig(outfile3, dpi=300, bbox_inches="tight")
        print(f"Figure 3 saved \u2192 {outfile3}")

    # Detailed console tables
    print_moments_tables(all_results)

    print("\nSimulation and plotting completed successfully.")


if __name__ == "__main__":
    main()
