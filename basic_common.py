"""Shared utilities for basic model scripts.

This module consolidates duplicated functions and configuration used across
the ``basic_*.py`` family of scripts.  By importing from here each script
avoids re-implementing burn-in removal, type coercion, path construction,
moment computation and plotting helpers.

Typical usage::

    from basic_common import (
        apply_burn_in,
        to_python_float,
        tensor_to_numpy,
        compute_standard_moments,
        build_full_params,
        param_tag,
        get_econ_params_path,
        get_bounds_path,
        get_vfi_cache_path,
        setup_simulation_data,
        run_vfi_simulation,
        FIXED_PARAMS,
        BASELINE,
        STEP_SIZES,
        PARAM_KEYS,
        PARAM_SYMBOLS,
        PERTURBATION_CONFIGS,
        BASE_DIR,
    )
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf

from src.econ_models.config.bond_config import BondsConfig
from src.econ_models.config.economic_params import EconomicParams
from src.econ_models.io.file_utils import load_json_file
from src.econ_models.moment_calculator.compute_autocorrelation import (
    compute_autocorrelation_lags_1_to_5,
)
from src.econ_models.moment_calculator.compute_derived_quantities import (
    compute_all_derived_quantities,
)
from src.econ_models.moment_calculator.compute_inaction_rate import (
    compute_inaction_rate,
)
from src.econ_models.moment_calculator.compute_mean import compute_global_mean
from src.econ_models.moment_calculator.compute_std import compute_global_std
from src.econ_models.simulator import synthetic_data_generator
from src.econ_models.simulator.vfi.basic import VFISimulator_basic

# ---------------------------------------------------------------------------
#  Project-level path / directory constants
# ---------------------------------------------------------------------------

BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.dirname("./")))
"""Root directory resolved relative to the workspace layout."""

GROUND_TRUTH_DIR: str = "./ground_truth_basic"
"""Directory containing cached golden VFI results."""

# ---------------------------------------------------------------------------
#  Structural-parameter configuration (basic model without frictions)
# ---------------------------------------------------------------------------

FIXED_PARAMS: Dict[str, float] = {
    "discount_factor": 0.96,
    "capital_share": 0.70,
    "depreciation_rate": 0.15,
    "risk_free_rate": 0.04,
    "equity_issuance_cost_fixed": 0.0,
    "equity_issuance_cost_linear": 0.0,
    "default_cost_proportional": 0.0,
    "corporate_tax_rate": 0.0,
    "collateral_recovery_fraction": 0.0,
}
"""Parameters held constant across all perturbation experiments."""

BASELINE: Dict[str, float] = {
    "productivity_persistence": 0.600,
    "productivity_std_dev": 0.175,
    "adjustment_cost_convex": 1.005,
    "adjustment_cost_fixed": 0.030,
}
"""Baseline structural parameters (midpoints of feasible ranges)."""

STEP_SIZES: Dict[str, float] = {
    "productivity_persistence": 0.030,
    "productivity_std_dev": 0.010,
    "adjustment_cost_convex": 0.050,
    "adjustment_cost_fixed": 0.0015,
}
"""Step sizes for central-finite-difference perturbations (~5 % of baseline)."""

PARAM_KEYS: List[str] = [
    "productivity_persistence",
    "productivity_std_dev",
    "adjustment_cost_convex",
    "adjustment_cost_fixed",
]
"""Ordered list of structural parameter keys (column order for Jacobian)."""

PARAM_SYMBOLS: Dict[str, str] = {
    "productivity_persistence": "\u03c1",   # ρ
    "productivity_std_dev": "\u03c3",        # σ
    "adjustment_cost_convex": "\u03be",      # ξ
    "adjustment_cost_fixed": "F",
}
"""Human-readable Unicode symbols for each structural parameter."""

PERTURBATION_CONFIGS: List[Tuple[str, Dict[str, float]]] = [
    ("baseline", {}),
    ("rho_plus", {"productivity_persistence": BASELINE["productivity_persistence"] + STEP_SIZES["productivity_persistence"]}),
    ("rho_minus", {"productivity_persistence": BASELINE["productivity_persistence"] - STEP_SIZES["productivity_persistence"]}),
    ("sigma_plus", {"productivity_std_dev": BASELINE["productivity_std_dev"] + STEP_SIZES["productivity_std_dev"]}),
    ("sigma_minus", {"productivity_std_dev": BASELINE["productivity_std_dev"] - STEP_SIZES["productivity_std_dev"]}),
    ("xi_plus", {"adjustment_cost_convex": BASELINE["adjustment_cost_convex"] + STEP_SIZES["adjustment_cost_convex"]}),
    ("xi_minus", {"adjustment_cost_convex": BASELINE["adjustment_cost_convex"] - STEP_SIZES["adjustment_cost_convex"]}),
    ("F_plus", {"adjustment_cost_fixed": BASELINE["adjustment_cost_fixed"] + STEP_SIZES["adjustment_cost_fixed"]}),
    ("F_minus", {"adjustment_cost_fixed": BASELINE["adjustment_cost_fixed"] - STEP_SIZES["adjustment_cost_fixed"]}),
]
"""The nine configurations used for Jacobian-based identification analysis.

``PERTURBATION_CONFIGS[0]`` is the baseline; indices 1-8 are ± perturbations
of each of the four structural parameters.
"""

PERTURBATION_MAP: Dict[str, Tuple[int, int]] = {
    "productivity_persistence": (1, 2),
    "productivity_std_dev": (3, 4),
    "adjustment_cost_convex": (5, 6),
    "adjustment_cost_fixed": (7, 8),
}
"""Maps each parameter key to ``(forward_index, backward_index)`` into
:data:`PERTURBATION_CONFIGS`.
"""

# Default economic-parameter lists used by simulation / sensitivity scripts
DEFAULT_ECON_LIST: List[List[float]] = [
    [0.6, 0.17, 1.0, 0.02],
    [0.57, 0.2, 0.8, 0.022],
    # [0.65, 0.1, 1.5, 0.01],
    # [0.6, 0.175, 1.005, 0.03]
]
"""Default two-economy benchmark list ``[ρ, σ, γ, F]``."""


# ---------------------------------------------------------------------------
#  Numeric-conversion helpers
# ---------------------------------------------------------------------------

def to_python_float(value: Any) -> float:
    """Coerce a tensor, numpy scalar, or number to a plain Python *float*.

    Parameters
    ----------
    value:
        The value to convert.  Accepts TensorFlow tensors, NumPy arrays,
        Python scalars, or ``None``.

    Returns
    -------
    float
        The value as a native Python float.  Returns ``0.0`` for ``None``.
    """
    if value is None:
        return 0.0
    if hasattr(value, "numpy"):
        return float(value.numpy())
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def tensor_to_numpy(tensor: Any) -> np.ndarray:
    """Convert a TensorFlow tensor to a NumPy array (no-op if already NumPy).

    Parameters
    ----------
    tensor:
        A ``tf.Tensor`` or ``np.ndarray``.

    Returns
    -------
    np.ndarray
    """
    if isinstance(tensor, tf.Tensor):
        return tensor.numpy()
    return np.asarray(tensor)


# ---------------------------------------------------------------------------
#  Simulation data helpers
# ---------------------------------------------------------------------------

def apply_burn_in(results: Dict[str, np.ndarray], burn_in: int) -> Dict[str, np.ndarray]:
    """Discard the first *burn_in* time-periods from simulation output.

    Parameters
    ----------
    results:
        Dictionary with keys ``K_curr``, ``K_next``, ``Z_curr``, ``Z_next``,
        each shaped ``(batch, T)``.
    burn_in:
        Number of leading periods to discard.

    Returns
    -------
    dict
        A new dictionary with the same keys, sliced to ``[:, burn_in:]``.

    Raises
    ------
    ValueError
        If *burn_in* exceeds the time dimension.
    """
    sample_key = next(iter(results))
    time_dim = results[sample_key].shape[1] if results[sample_key].ndim > 1 else len(results[sample_key])
    if burn_in >= time_dim:
        raise ValueError(
            f"burn_in ({burn_in}) must be less than the time dimension ({time_dim})."
        )
    return {key: array[:, burn_in:] for key, array in results.items()}


# ---------------------------------------------------------------------------
#  Path construction helpers
# ---------------------------------------------------------------------------

def build_full_params(overrides: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """Merge fixed params, baseline, and optional overrides into one dict.

    Parameters
    ----------
    overrides:
        Parameter values that differ from :data:`BASELINE`.

    Returns
    -------
    dict
        Complete parameter dictionary ready for ``EconomicParams(**…)``.
    """
    params: Dict[str, float] = {}
    params.update(FIXED_PARAMS)
    params.update(BASELINE)
    if overrides:
        params.update(overrides)
    return params


def param_tag(params_dict: Dict[str, float]) -> str:
    """Build a filename-safe tag from the four structural parameters.

    Parameters
    ----------
    params_dict : dict
        Must contain ``productivity_persistence``, ``productivity_std_dev``,
        ``adjustment_cost_convex``, and ``adjustment_cost_fixed``.

    Returns
    -------
    str
        Underscore-separated parameter values, e.g. ``"0.6_0.175_1.005_0.03"``.
    """
    return (
        f"{params_dict['productivity_persistence']}"
        f"_{params_dict['productivity_std_dev']}"
        f"_{params_dict['adjustment_cost_convex']}"
        f"_{params_dict['adjustment_cost_fixed']}"
    )


def get_econ_params_path(tag: str) -> str:
    """Return the JSON path for economic parameters identified by *tag*.

    Parameters
    ----------
    tag : str
        Filename tag produced by :func:`param_tag`.

    Returns
    -------
    str
    """
    return os.path.join(BASE_DIR, f"hyperparam/prefixed/econ_params_basic_{tag}.json")


def get_bounds_path(tag: str) -> str:
    """Return the JSON path for state-space bounds identified by *tag*.

    Parameters
    ----------
    tag : str
        Filename tag produced by :func:`param_tag`.

    Returns
    -------
    str
    """
    return os.path.join(BASE_DIR, f"hyperparam/autogen/bounds_basic_{tag}.json")


def get_vfi_cache_path(tag: str, n_capital: int) -> str:
    """Return the ``.npz`` cache path for a VFI solution.

    Parameters
    ----------
    tag : str
        Filename tag produced by :func:`param_tag`.
    n_capital : int
        Number of capital grid points used in the solve.

    Returns
    -------
    str
    """
    return os.path.join(GROUND_TRUTH_DIR, f"golden_vfi_basic_{tag}_{n_capital}.npz")


def get_golden_vfi_path(econ_params_list: Sequence[float]) -> str:
    """Return the golden VFI result path for an economy specified as ``[ρ, σ, γ, F]``.

    Parameters
    ----------
    econ_params_list : sequence of float
        Four structural parameters ``[ρ, σ, γ, F]``.

    Returns
    -------
    str
    """
    rho, sigma, gamma, fixed = econ_params_list
    return os.path.join(GROUND_TRUTH_DIR, f"golden_vfi_results_{rho}_{sigma}_{gamma}_{fixed}.npz")


def get_econ_params_path_by_list(econ_params_list: Sequence[float]) -> str:
    """Return the economic parameters JSON path for ``[ρ, σ, γ, F]``.

    Parameters
    ----------
    econ_params_list : sequence of float
        Four structural parameters.

    Returns
    -------
    str
    """
    rho, sigma, gamma, fixed = econ_params_list
    return os.path.join(
        BASE_DIR,
        f"hyperparam/prefixed/econ_params_basic_{rho}_{sigma}_{gamma}_{fixed}.json",
    )


def get_bounds_path_by_list(econ_params_list: Sequence[float]) -> str:
    """Return the bounds JSON path for ``[ρ, σ, γ, F]``.

    Parameters
    ----------
    econ_params_list : sequence of float
        Four structural parameters.

    Returns
    -------
    str
    """
    rho, sigma, gamma, fixed = econ_params_list
    return os.path.join(
        BASE_DIR,
        f"hyperparam/autogen/bounds_basic_{rho}_{sigma}_{gamma}_{fixed}.json",
    )


# ---------------------------------------------------------------------------
#  Setup / simulation convenience wrappers
# ---------------------------------------------------------------------------

def load_econ_params(econ_params_list: Sequence[float]) -> EconomicParams:
    """Load and return an :class:`EconomicParams` for the given economy.

    Parameters
    ----------
    econ_params_list : sequence of float
        Four structural parameters ``[ρ, σ, γ, F]``.

    Returns
    -------
    EconomicParams
    """
    path = get_econ_params_path_by_list(econ_params_list)
    return EconomicParams(**load_json_file(path))


def load_bonds_config(
    econ_params_list: Sequence[float],
    econ_params: Optional[EconomicParams] = None,
) -> Dict[str, Any]:
    """Load and validate the bonds/bounds configuration for the given economy.

    Parameters
    ----------
    econ_params_list:
        Economy specification ``[ρ, σ, γ, F]``.
    econ_params:
        Pre-loaded :class:`EconomicParams`; loaded from disk if ``None``.

    Returns
    -------
    dict
        Validated bonds configuration.
    """
    if econ_params is None:
        econ_params = load_econ_params(econ_params_list)
    bounds_file = get_bounds_path_by_list(econ_params_list)
    return BondsConfig.validate_and_load(bounds_file=bounds_file, current_params=econ_params)


def setup_simulation_data(
    econ_params: EconomicParams,
    bonds_config: Dict[str, Any],
    batch_size: int = 8000,
    time_periods: int = 800,
) -> Tuple[Any, Any]:
    """Generate synthetic initial states and shock sequences.

    Parameters
    ----------
    econ_params:
        The economic parameters for the simulation.
    bonds_config:
        Validated bonds/bounds configuration.
    batch_size:
        Number of firms to simulate.
    time_periods:
        Number of time periods.

    Returns
    -------
    tuple
        ``(initial_states, shock_sequence)`` tensors.
    """
    data_gen = synthetic_data_generator(
        econ_params_benchmark=econ_params,
        sample_bonds_config=bonds_config,
        batch_size=batch_size,
        T_periods=time_periods,
    )
    return data_gen.gen()


def run_vfi_simulation(
    econ_params: EconomicParams,
    vfi_solution: Any,
    initial_states: Any,
    shock_sequence: Any,
) -> Dict[str, np.ndarray]:
    """Run a VFI simulation with pre-loaded solution matrices.

    Parameters
    ----------
    econ_params:
        Economic parameters for this simulation.
    vfi_solution:
        Loaded ``.npz`` archive (or dict) with VFI solution matrices.
    initial_states:
        Tuple of initial-state tensors.
    shock_sequence:
        Tensor of innovation/shock sequences.

    Returns
    -------
    dict
        Simulation output with keys ``K_curr``, ``K_next``, ``Z_curr``, ``Z_next``.
    """
    simulator = VFISimulator_basic(econ_params)
    simulator.load_solved_vfi_solution(vfi_solution)
    return simulator.simulate(
        tuple(s.numpy() if hasattr(s, "numpy") else s for s in initial_states),
        shock_sequence.numpy() if hasattr(shock_sequence, "numpy") else shock_sequence,
    )


# ---------------------------------------------------------------------------
#  Moment computation
# ---------------------------------------------------------------------------

def compute_standard_moments(
    simulation_results: Dict[str, np.ndarray],
    depreciation_rate: float,
    capital_share: float,
    *,
    inaction_lower: float = -0.001,
    inaction_upper: float = 0.001,
) -> Dict[str, float]:
    """Compute a standard set of summary moments from simulation output.

    The moments include means, standard deviations, first-order
    autocorrelations, and the inaction rate.

    Parameters
    ----------
    simulation_results:
        Post burn-in simulation output (``K_curr``, ``K_next``, etc.).
    depreciation_rate:
        Economic depreciation rate δ.
    capital_share:
        Capital share parameter α.
    inaction_lower:
        Lower threshold for the inaction-rate calculation.
    inaction_upper:
        Upper threshold for the inaction-rate calculation.

    Returns
    -------
    dict
        Mapping from human-readable moment names to float values.
    """
    derived = compute_all_derived_quantities(simulation_results, depreciation_rate, capital_share)

    variables = {
        "Capital": derived["capital"],
        "Investment": derived["investment"],
        "Inv. Rate": derived["investment_rate"],
        "Output": derived["output"],
    }

    moments: Dict[str, float] = {}
    for name, data in variables.items():
        ac_results = compute_autocorrelation_lags_1_to_5(data)
        moments[f"{name}_mean"] = to_python_float(compute_global_mean(data))
        moments[f"{name}_std"] = to_python_float(compute_global_std(data))
        moments[f"{name}_ac1"] = to_python_float(ac_results["lag_1"])

    moments["inaction_rate"] = to_python_float(
        compute_inaction_rate(
            derived["investment_rate"],
            lower_threshold=inaction_lower,
            upper_threshold=inaction_upper,
        )
    )
    return moments


def compute_variable_moments(
    data: np.ndarray,
) -> Dict[str, float]:
    """Compute mean, std, and lag-1 autocorrelation for a single variable.

    Parameters
    ----------
    data:
        2-D array shaped ``(batch, T)``.

    Returns
    -------
    dict
        Keys ``mean``, ``std``, ``ac1``.
    """
    ac_results = compute_autocorrelation_lags_1_to_5(data)
    return {
        "mean": to_python_float(compute_global_mean(data)),
        "std": to_python_float(compute_global_std(data)),
        "ac1": to_python_float(ac_results["lag_1"]),
    }


# ---------------------------------------------------------------------------
#  Plotting helpers
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
    """Draw overlapping VFI vs. multi-epoch DL histograms on *ax*.

    Parameters
    ----------
    ax:
        Matplotlib axes to draw on.
    vfi_data:
        Flattened VFI ground-truth data.
    dl_epoch_data:
        Mapping from epoch number to flattened DL data.
    title:
        Subplot title.
    xlabel:
        X-axis label.
    epoch_colors:
        Color array (one per epoch + 1 for VFI).
    bins:
        Number of histogram bins.
    """
    clean_vfi = vfi_data[np.isfinite(vfi_data)]
    ax.hist(clean_vfi, bins=bins, density=True, alpha=0.45, color="#4C72B0", label="VFI (Ground Truth)", edgecolor="white", linewidth=0.5)
    for idx, (epoch, dl_data) in enumerate(dl_epoch_data.items()):
        clean_dl = dl_data[np.isfinite(dl_data)]
        ax.hist(
            clean_dl,
            bins=bins,
            density=True,
            alpha=0.6,
            color=epoch_colors[idx + 1],
            label=f"DL Epoch {epoch}",
            histtype="step",
            linewidth=2.2,
        )
    ax.set_title(title, fontsize=12, fontweight="semibold")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.legend(fontsize=9, framealpha=0.9, edgecolor="#cccccc")
    ax.grid(True, alpha=0.2, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_grouped_bars(
    ax,
    vfi_stats: Sequence[float],
    epoch_stats: Dict[int, Sequence[float]],
    labels: Sequence[str],
    epoch_colors: np.ndarray,
    *,
    title: Optional[str] = None,
    vfi_color: str = "tab:blue",
) -> None:
    """Draw grouped bar charts comparing VFI vs. multiple DL epochs.

    Parameters
    ----------
    ax:
        Matplotlib axes.
    vfi_stats:
        Values for the VFI baseline (one per *label*).
    epoch_stats:
        ``{epoch: [values…]}`` for each DL epoch.
    labels:
        X-tick labels for the groups.
    epoch_colors:
        Color array for epoch bars.
    title:
        Optional subplot title.
    vfi_color:
        Bar colour for VFI.
    """
    vfi_vals = [float(v) for v in vfi_stats]
    n_groups = len(labels)
    n_bars = 1 + len(epoch_stats)
    width = 0.8 / n_bars
    x = np.arange(n_groups)

    rects_vfi = ax.bar(x - 0.4 + width / 2, vfi_vals, width, label="VFI", color=vfi_color, alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.bar_label(rects_vfi, padding=3, fmt="%.4f", fontsize=7)

    for idx, (epoch, dl_vals) in enumerate(epoch_stats.items()):
        dl_vals = [float(v) for v in dl_vals]
        offset = -0.4 + (idx + 1.5) * width
        rects_dl = ax.bar(x + offset, dl_vals, width, label=f"DL Ep {epoch}", color=epoch_colors[idx + 1], alpha=0.8, edgecolor="white", linewidth=0.5)
        ax.bar_label(rects_dl, padding=3, fmt="%.4f", fontsize=7)

    ax.set_ylabel("Value", fontsize=10)
    if title:
        ax.set_title(title, fontsize=12, fontweight="semibold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend(fontsize=8, framealpha=0.9, edgecolor="#cccccc")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def compute_weighted_error(
    vfi_moments: Dict[str, Any],
    dl_moments: Dict[str, Any],
    variable_names: Sequence[str],
    moment_keys: Sequence[str] = ("mean", "std", "ac1"),
    *,
    include_inaction: bool = True,
    vfi_inaction: float = 0.0,
    dl_inaction: float = 0.0,
) -> float:
    """Compute the weighted average percentage error across moments.

    Each variable-moment pair is weighted equally within categories,
    and each category (including inaction) receives equal weight.

    Parameters
    ----------
    vfi_moments:
        Nested ``{variable: {moment_key: value}}``.
    dl_moments:
        Same structure as *vfi_moments*.
    variable_names:
        Which variables to include.
    moment_keys:
        Sub-moment keys.
    include_inaction:
        Whether to add an inaction-rate category.
    vfi_inaction:
        VFI inaction rate.
    dl_inaction:
        DL inaction rate.

    Returns
    -------
    float
        Weighted average absolute relative error in percent.
    """
    categories: List[float] = []
    for var in variable_names:
        sub_errors: List[float] = []
        for mk in moment_keys:
            vfi_v = float(vfi_moments[var][mk])
            dl_v = float(dl_moments[var][mk])
            denom = abs(vfi_v) if abs(vfi_v) > 1e-12 else 1.0
            sub_errors.append(abs(dl_v - vfi_v) / denom * 100)
        categories.append(float(np.mean(sub_errors)))

    if include_inaction:
        denom = abs(vfi_inaction) if abs(vfi_inaction) > 1e-12 else 1.0
        categories.append(abs(dl_inaction - vfi_inaction) / denom * 100)

    return float(np.mean(categories))
