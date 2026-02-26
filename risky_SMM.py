#!/usr/bin/env python3
"""Simulated Method of Moments (SMM) estimation for the risky debt model.

Implements the full SMM workflow adapted from the basic model (basic_SMM.py)
for the risky debt case with 6 structural parameters:

  - CMA-ES global search + Nelder-Mead local refinement (10 Sobol starts)
  - Bootstrap-based inverse-covariance weighting matrix (B = 500)
  - Asymptotic standard errors via central-difference Jacobian
  - J-test for overidentifying restrictions (9 moments, 6 parameters)
  - Monte Carlo experiment framework for bias/RMSE/coverage analysis
  - Structured result serialization (CSV, JSON, LaTeX tables)

**Moment selection** follows the Jacobian-based D-optimal identification
analysis (see ``results/moments_identification_risky/``).  The
recommended 9-moment set (base-6 + k=9 augmentation) is::

    m03: AC[I/K]             — identifies ρ (investment persistence)
    m04: Skew[I/K]           — identifies ξ (adjustment cost curvature)
    m09: SD[Y/K]             — identifies σ (shock volatility)
    m12: Corr(I/K, lag Y/K)  — identifies ρ (lagged profitability response)
    m27: P(e_raw<0)          — identifies η₀ (equity issuance frequency)
    m28: E[|e|/K | e<0]      — identifies η₁ (equity issuance intensity)
    m29: E[D/K | D>0]        — identifies F (dividend payout level)
    m22: Corr(ΔB/K, Y/K)     — identifies F/ξ (debt-output comovement)
    m20: Median[B/K]          — identifies ξ/F (capital structure level)

  Condition number κ = 82.26 (vs 2825 for the previous 8-moment set).
  Overidentified with 3 degrees of freedom for J-test.

Usage::

    # Single estimation run
    python risky_SMM.py

    # Monte Carlo with 50 replications
    python risky_SMM.py --mc-replications 50

    # Use diagonal weighting (legacy fallback)
    python risky_SMM.py --diagonal-W
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import cma
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
from scipy import stats as sp_stats
from scipy.optimize import minimize
from scipy.stats import chi2, skew as sp_skew
from scipy.stats.qmc import Sobol

from src.econ_models.config.dl_config import load_dl_config
from src.econ_models.config.economic_params import EconomicParams
from src.econ_models.config.bond_config import BondsConfig
from src.econ_models.io.file_utils import load_json_file
from src.econ_models.simulator import (
    DLSimulatorRiskyFinal_dist,
    VFISimulator_risky,
)
from src.econ_models.simulator import synthetic_data_generator
from src.econ_models.moment_calculator.compute_derived_quantities import (
    compute_all_derived_quantities,
)
from src.econ_models.moment_calculator.compute_autocorrelation import (
    compute_autocorrelation_lags_1_to_5,
)

from risky_common import (
    FIXED_PARAMS,
    BASE_DIR,
    VFI_N_K,
    VFI_N_D,
    PARAM_KEYS as PARAM_ORDER,
    PARAM_SYMBOLS as PARAM_LABELS,
    to_python_float,
    apply_burn_in,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# =========================================================================
#  §0  Parameter Alignment
# =========================================================================

TRUE_PARAMS: Dict[str, float] = {
    "productivity_persistence": 0.600,
    "productivity_std_dev": 0.170,
    "adjustment_cost_convex": 1.000,
    "adjustment_cost_fixed": 0.020,
    "equity_issuance_cost_fixed": 0.080,
    "equity_issuance_cost_linear": 0.100,
}
"""True parameter values for the risky debt baseline economy."""

SEARCH_BOUNDS: Dict[str, Tuple[float, float]] = {
    "productivity_persistence": (0.40, 0.80),
    "productivity_std_dev": (0.05, 0.30),
    "adjustment_cost_convex": (0.01, 2.00),
    "adjustment_cost_fixed": (0.01, 0.05),
    "equity_issuance_cost_fixed": (0.01, 0.20),
    "equity_issuance_cost_linear": (0.01, 0.20),
}
"""Search bounds from distributional training config (econ_params_risky_dist.json)."""

# PARAM_ORDER and PARAM_LABELS are imported from risky_common
# as PARAM_KEYS and PARAM_SYMBOLS respectively.


# =========================================================================
#  §1  Panel Dimensions
# =========================================================================

N_DATA: int = 3000
T_DATA_EFF: int = 200
"""Effective data-panel periods *after* burn-in."""

N_SIM: int = 5000
T_SIM_EFF: int = 500
"""Effective simulation periods *after* burn-in."""

T_BURN: int = 200
"""Burn-in periods discarded before moment computation (risky model
requires a longer burn-in than the basic model due to debt dynamics)."""

T_DATA_RAW: int = T_DATA_EFF + T_BURN + 1
T_SIM_RAW: int = T_SIM_EFF + T_BURN + 1

J_RATIO: float = (N_SIM * T_SIM_EFF) / (N_DATA * T_DATA_EFF)
"""Simulation-to-data ratio ≈ 8.33, used in SE formula."""


# =========================================================================
#  Other defaults
# =========================================================================

DEFAULT_DL_EPOCH: int = 1400
DEFAULT_N_BOOTSTRAP: int = 500
DEFAULT_N_SOBOL_STARTS: int = 10
DEFAULT_CMA_MAX_EVALS: int = 200
DEFAULT_MC_REPLICATIONS: int = 1

# VFI_N_K, VFI_N_D imported from risky_common (560, 560)

DL_CONFIG_PATH: str = "./hyperparam_dist/prefixed/dl_params_dist.json"
DL_CHECKPOINT_DIR: str = "./checkpoints_final_dist/risky"
BONDS_FILE_DIST: str = os.path.join(
    BASE_DIR, "hyperparam_dist/autogen/bounds_risky_dist.json"
)
ECON_PARAMS_FILE_DIST: str = os.path.join(
    BASE_DIR, "hyperparam_dist/prefixed/econ_params_risky_dist.json"
)

# ── 9-moment set from D-optimal identification analysis ─────────────
# Base 6 (just-identified) + k=9 augmentation for robustness + J-test.
# Condition number κ = 82.26 (vs 2825 for the old 8-moment set).
#
#   [0]  AC[I/K]             — investment persistence (ρ)
#   [1]  Skew[I/K]           — adjustment cost curvature (ξ)
#   [2]  SD[Y/K]             — shock volatility (σ)
#   [3]  Corr(I/K, lag Y/K)  — lagged profitability response (ρ)
#   [4]  P(equity iss.)      — issuance frequency (η₀)
#   [5]  E[|e|/K | e<0]      — issuance intensity (η₁)
#   [6]  E[D/K | D>0]        — dividend payout level (F)
#   [7]  Corr(ΔB/K, Y/K)     — debt-output comovement (F/ξ)
#   [8]  Median[B/K]          — capital structure level (ξ/F)

MOMENT_NAMES: List[str] = [
    "AC[I/K]",
    "Skew[I/K]",
    "SD[Y/K]",
    "Corr(I/K,lagY/K)",
    "P(equity iss.)",
    "E[|e|/K|e<0]",
    "E[D/K|D>0]",
    "Corr(ΔB/K,Y/K)",
    "Median[B/K]",
]
"""9-moment set from D-optimal identification analysis (base-6 + k=9)."""

K_MOMENTS: int = 9
P_PARAMS: int = 6
DF_OVERID: int = K_MOMENTS - P_PARAMS  # = 3


# =========================================================================
#  Moment-computation helpers
# =========================================================================

# to_python_float and apply_burn_in imported from risky_common


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation (NaN-safe)."""
    x_flat = np.asarray(x).flatten()
    y_flat = np.asarray(y).flatten()
    mask = np.isfinite(x_flat) & np.isfinite(y_flat)
    xv, yv = x_flat[mask], y_flat[mask]
    if len(xv) < 3:
        return 0.0
    r = np.corrcoef(xv, yv)[0, 1]
    return float(np.clip(r, -1.0, 1.0))


def _safe_fraction(arr: np.ndarray, condition_fn) -> float:
    """Fraction of finite elements satisfying *condition_fn*."""
    flat = np.asarray(arr).flatten()
    valid = flat[np.isfinite(flat)]
    if len(valid) == 0:
        return 0.0
    return float(np.sum(condition_fn(valid))) / len(valid)


def _safe_nanmean(arr: np.ndarray) -> float:
    flat = np.asarray(arr).flatten()
    valid = flat[np.isfinite(flat)]
    return float(np.mean(valid)) if len(valid) > 0 else 0.0


def _safe_nanstd(arr: np.ndarray) -> float:
    flat = np.asarray(arr).flatten()
    valid = flat[np.isfinite(flat)]
    return float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0


def _safe_nanmedian(arr: np.ndarray) -> float:
    flat = np.asarray(arr).flatten()
    valid = flat[np.isfinite(flat)]
    return float(np.median(valid)) if len(valid) > 0 else 0.0


def _safe_skew(arr: np.ndarray) -> float:
    """Pool-skewness (NaN-safe) via scipy.stats.skew."""
    flat = np.asarray(arr).flatten()
    valid = flat[np.isfinite(flat)]
    if len(valid) < 3:
        return 0.0
    return float(sp_skew(valid, nan_policy='omit'))


def compute_identification_moments(
    sim_results: Dict[str, np.ndarray],
    delta: float,
    alpha: float,
    r_f: float = 0.04,
) -> np.ndarray:
    """Compute the 9-moment vector selected by D-optimal identification.

    These are the base-6 moments from the identification analysis plus
    the k=9 augmentation set, giving condition number κ = 82.26.

    Index  Moment              Role
    -----  ------------------  --------------------------------
    0      AC[I/K]             Investment persistence (ρ)
    1      Skew[I/K]           Adjustment cost curvature (ξ)
    2      SD[Y/K]             Shock volatility (σ)
    3      Corr(I/K, lag Y/K)  Lagged profitability response (ρ)
    4      P(equity iss.)      Equity issuance frequency (η₀)
    5      E[|e|/K | e<0]      Equity issuance intensity (η₁)
    6      E[D/K | D>0]        Dividend payout level (F)
    7      Corr(ΔB/K, Y/K)     Debt-output comovement (F/ξ)
    8      Median[B/K]          Capital structure level (ξ/F)
    """
    derived = compute_all_derived_quantities(
        sim_results, delta, alpha, include_debt=True
    )
    capital = derived["capital"]
    output = derived["output"]
    inv_rate = derived["investment_rate"]
    leverage = derived["leverage"]
    eq_iss = derived["equity_issuance"]
    eq_iss_rate = derived["equity_issuance_rate"]

    # Compute Y/K (revenue-to-capital ratio)
    capital_arr = np.asarray(capital)
    output_arr = np.asarray(output)
    inv_rate_arr = np.asarray(inv_rate)
    leverage_arr = np.asarray(leverage)
    eq_iss_arr = np.asarray(eq_iss)
    eq_iss_rate_arr = np.asarray(eq_iss_rate)

    K_safe = np.maximum(capital_arr, 1e-5)
    rev_cap = output_arr / K_safe  # Y/K

    moments = np.zeros(K_MOMENTS)

    # m0: AC[I/K] — autocorrelation of investment rate (lag 1)
    moments[0] = to_python_float(
        compute_autocorrelation_lags_1_to_5(inv_rate)["lag_1"]
    )

    # m1: Skew[I/K] — skewness of investment rate
    moments[1] = _safe_skew(inv_rate_arr)

    # m2: SD[Y/K] — standard deviation of revenue-to-capital ratio
    moments[2] = _safe_nanstd(rev_cap)

    # m3: Corr(I/K, lag Y/K) — correlation between current I/K and
    #     lagged Y/K (captures delayed response to profitability shocks)
    ir_2d = inv_rate_arr if inv_rate_arr.ndim == 2 else inv_rate_arr.reshape(1, -1)
    rc_2d = rev_cap if rev_cap.ndim == 2 else rev_cap.reshape(1, -1)
    if ir_2d.shape[1] > 1:
        moments[3] = _safe_corr(ir_2d[:, 1:], rc_2d[:, :-1])
    else:
        moments[3] = 0.0

    # m4: P(equity issuance) — fraction of firm-periods with equity issuance
    #     equity_issuance = max(0, -payout), so P(e_raw<0) ≡ P(equity_iss > 0)
    eq_flat = eq_iss_arr.flatten()
    eq_valid = eq_flat[np.isfinite(eq_flat)]
    if len(eq_valid) > 0:
        moments[4] = float(np.sum(eq_valid > 1e-6)) / len(eq_valid)
    else:
        moments[4] = 0.0

    # m5: E[|e|/K | e<0] — mean equity issuance rate conditional on issuance
    eq_rate_flat = eq_iss_rate_arr.flatten()
    eq_rate_valid = eq_rate_flat[np.isfinite(eq_rate_flat)]
    issuance_mask = eq_rate_valid > 1e-6
    if np.sum(issuance_mask) > 0:
        moments[5] = float(np.mean(eq_rate_valid[issuance_mask]))
    else:
        moments[5] = 0.0

    # m6: E[D/K | D>0] — mean dividend rate conditional on positive dividends
    #     D/K ≈ Y/K - I/K - (B/K)·r_f - equity_issuance_rate  (cash-flow identity)
    rc_flat = rev_cap.flatten()
    ir_flat = inv_rate_arr.flatten()
    lev_flat = leverage_arr.flatten()
    eq_rate_all = eq_iss_rate_arr.flatten()
    fin_mask = (
        np.isfinite(rc_flat) & np.isfinite(ir_flat)
        & np.isfinite(lev_flat) & np.isfinite(eq_rate_all)
    )
    if np.sum(fin_mask) > 0:
        div_proxy = (
            rc_flat[fin_mask] - ir_flat[fin_mask]
            - lev_flat[fin_mask] * r_f - eq_rate_all[fin_mask]
        )
        pos_div = div_proxy[div_proxy > 0]
        moments[6] = float(np.mean(pos_div)) if len(pos_div) > 0 else 0.0
    else:
        moments[6] = 0.0

    # m7: Corr(ΔB/K, Y/K) — correlation between change in leverage and
    #     contemporaneous output-to-capital ratio
    lev_2d = leverage_arr if leverage_arr.ndim == 2 else leverage_arr.reshape(1, -1)
    if lev_2d.shape[1] > 1:
        delta_lev = lev_2d[:, 1:] - lev_2d[:, :-1]
        moments[7] = _safe_corr(delta_lev, rc_2d[:, 1:])
    else:
        moments[7] = 0.0

    # m8: Median[B/K] — median leverage
    moments[8] = _safe_nanmedian(leverage_arr)

    return moments


def compute_moments_from_panel(
    sim_results: Dict[str, np.ndarray],
    delta: float,
    alpha: float,
    r_f: float = 0.04,
    firm_indices: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute moments, optionally on a subset of firms (for bootstrap).

    Parameters
    ----------
    sim_results : dict
        Post burn-in simulation output with 2-D arrays (N, T).
    delta, alpha : float
        Depreciation rate and capital share.
    r_f : float
        Risk-free rate (used for dividend proxy in E[D/K|D>0]).
    firm_indices : np.ndarray, optional
        If provided, select these row indices before computing moments.

    Returns
    -------
    np.ndarray  shape (9,)
    """
    if firm_indices is not None:
        sub = {k: v[firm_indices] for k, v in sim_results.items()}
    else:
        sub = sim_results
    return compute_identification_moments(sub, delta, alpha, r_f=r_f)


# =========================================================================
#  Golden Moments (VFI truth)
# =========================================================================

def _econ_tag() -> str:
    """Build filename-safe tag from TRUE_PARAMS (6 structural params)."""
    p = TRUE_PARAMS
    return (
        f"{p['productivity_persistence']}_"
        f"{p['productivity_std_dev']}_"
        f"{p['adjustment_cost_convex']}_"
        f"{p['adjustment_cost_fixed']}_"
        f"{p['equity_issuance_cost_linear']}_"
        f"{p['equity_issuance_cost_fixed']}"
    )


def _golden_vfi_path() -> str:
    """Canonical path for the golden VFI solution at TRUE_PARAMS."""
    tag = _econ_tag()
    return (
        f"./ground_truth_risky/golden_vfi_risky_{tag}_{VFI_N_K}_{VFI_N_D}.npz"
    )


def _econ_params_path() -> str:
    """Path to the econ params JSON for the true parameter set."""
    tag = _econ_tag()
    return os.path.join(
        BASE_DIR,
        f"hyperparam/prefixed/econ_params_risky_{tag}.json",
    )


def _bounds_path() -> str:
    """Path to the bounds JSON for the true parameter set."""
    tag = _econ_tag()
    return os.path.join(
        BASE_DIR,
        f"hyperparam/autogen/bounds_risky_{tag}.json",
    )


def compute_golden_moments(
    econ_params: EconomicParams,
    initial_states: Any,
    shock_sequence: Any,
    bounds: Optional[Dict] = None,
    vfi_solution: Optional[Any] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Compute golden moments via VFI simulation on the data panel.

    Returns both the 8-element moment vector AND the stationary simulation
    results (needed for bootstrap weighting matrix).

    If *vfi_solution* is provided (pre-loaded), disk I/O is skipped.
    """
    if vfi_solution is not None:
        solved = vfi_solution
        logger.info("Using pre-loaded golden VFI solution")
    else:
        golden_path = _golden_vfi_path()
        if not os.path.exists(golden_path):
            raise FileNotFoundError(
                f"Golden VFI file not found: {golden_path}. "
                "Run risky_golden_vfi_finder.py first."
            )
        logger.info("Loading golden VFI: %s", golden_path)
        solved = np.load(golden_path, allow_pickle=True)

    simulator = VFISimulator_risky(econ_params)
    simulator.load_solved_vfi_solution(solved)
    results = simulator.simulate(
        tuple(s.numpy() if hasattr(s, "numpy") else s for s in initial_states),
        shock_sequence.numpy() if hasattr(shock_sequence, "numpy") else shock_sequence,
    )
    stationary = apply_burn_in(results, T_BURN)

    delta = econ_params.depreciation_rate
    alpha = econ_params.capital_share
    r_f = econ_params.risk_free_rate
    moments = compute_identification_moments(stationary, delta, alpha, r_f=r_f)

    logger.info("Golden moments (9-moment identification set):")
    for name, val in zip(MOMENT_NAMES, moments):
        logger.info("  %s: %.6f", name, val)

    return moments, stationary


# =========================================================================
#  §2  Bootstrap Weighting Matrix
# =========================================================================

def bootstrap_weighting_matrix(
    stationary_data: Dict[str, np.ndarray],
    delta: float,
    alpha: float,
    r_f: float = 0.04,
    B: int = DEFAULT_N_BOOTSTRAP,
    ridge_threshold: float = 1000.0,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Compute bootstrap inverse-covariance weighting matrix.

    Block-bootstrap at the firm level: resample N_DATA firms with
    replacement B times, recompute moments each time, form the
    sample covariance :math:`\hat\Omega`, invert to get
    :math:`W = \hat\Omega^{-1}`.

    Parameters
    ----------
    stationary_data : dict
        Post burn-in VFI simulation (N_DATA × T_effective).
    delta, alpha : float
        Depreciation and capital share.
    r_f : float
        Risk-free rate (passed through to moment computation).
    B : int
        Number of bootstrap replications.
    ridge_threshold : float
        If condition number of Ω̂ exceeds this, add ridge ε·I.
    rng : np.random.Generator, optional

    Returns
    -------
    W : np.ndarray, shape (K, K)
        Inverse-covariance weighting matrix.
    Omega_hat : np.ndarray, shape (K, K)
        Bootstrap covariance of moments.
    """
    if rng is None:
        rng = np.random.default_rng()

    sample_key = next(iter(stationary_data))
    N = stationary_data[sample_key].shape[0]

    moment_draws = np.zeros((B, K_MOMENTS))
    for b in range(B):
        idx = rng.integers(0, N, size=N)
        moment_draws[b] = compute_moments_from_panel(
            stationary_data, delta, alpha, r_f=r_f, firm_indices=idx
        )
        if (b + 1) % 100 == 0:
            logger.info("  Bootstrap %d / %d", b + 1, B)

    Omega_hat = np.cov(moment_draws, rowvar=False)

    # Regularize if ill-conditioned
    cond = np.linalg.cond(Omega_hat)
    if cond > ridge_threshold:
        eps = 1e-6 * np.trace(Omega_hat) / K_MOMENTS
        Omega_hat += eps * np.eye(K_MOMENTS)
        logger.info(
            "Bootstrap Ω̂ regularized (cond was %.1f, ridge ε=%.2e)", cond, eps
        )

    W = np.linalg.inv(Omega_hat)

    logger.info(
        "Bootstrap weighting matrix computed (B=%d, cond(Ω̂)=%.1f)",
        B, np.linalg.cond(Omega_hat),
    )
    return W, Omega_hat


def build_diagonal_weighting_matrix(golden_moments: np.ndarray) -> np.ndarray:
    """Legacy diagonal weighting W_ii = 1/m_i² (fallback for debugging)."""
    MAX_WEIGHT_RATIO = 100.0
    abs_m = np.abs(golden_moments)
    floor = 0.10 * np.median(abs_m[abs_m > 0])
    denominators = np.maximum(abs_m, floor) ** 2
    W_diag = 1.0 / denominators
    w_min = np.min(W_diag)
    W_diag = np.minimum(W_diag, MAX_WEIGHT_RATIO * w_min)
    return np.diag(W_diag)


# =========================================================================
#  DL Simulator Setup
# =========================================================================

def create_dl_simulator(epoch: int) -> DLSimulatorRiskyFinal_dist:
    """Create and load the distributional risky DL simulator."""
    dl_config = load_dl_config(DL_CONFIG_PATH, "risky_final")
    econ_dist = EconomicParams(**load_json_file(ECON_PARAMS_FILE_DIST))
    bonds_dist = BondsConfig.validate_and_load(
        bounds_file=BONDS_FILE_DIST, current_params=econ_dist,
    )
    dl_config.capital_max = bonds_dist["k_max"]
    dl_config.capital_min = bonds_dist["k_min"]
    dl_config.productivity_max = bonds_dist["z_max"]
    dl_config.productivity_min = bonds_dist["z_min"]
    dl_config.debt_max = bonds_dist["b_max"]
    dl_config.debt_min = bonds_dist["b_min"]

    sim = DLSimulatorRiskyFinal_dist(dl_config, bonds_dist)

    # Load all 7 policy network weights for the risky model
    sim.load_solved_dl_solution(
        capital_policy_filepath=os.path.join(
            DL_CHECKPOINT_DIR,
            f"risky_capital_policy_net_{epoch}.weights.h5",
        ),
        debt_filepath=os.path.join(
            DL_CHECKPOINT_DIR,
            f"risky_debt_policy_net_{epoch}.weights.h5",
        ),
        investment_policy_filepath=os.path.join(
            DL_CHECKPOINT_DIR,
            f"risky_investment_decision_net_{epoch}.weights.h5",
        ),
        default_policy_filepath=os.path.join(
            DL_CHECKPOINT_DIR,
            f"risky_default_policy_net_{epoch}.weights.h5",
        ),
        value_function_filepath=os.path.join(
            DL_CHECKPOINT_DIR,
            f"risky_value_net_{epoch}.weights.h5",
        ),
        equity_issuance_invest_filepath=os.path.join(
            DL_CHECKPOINT_DIR,
            f"risky_equity_issuance_net_{epoch}.weights.h5",
        ),
        equity_issuance_noinvest_filepath=os.path.join(
            DL_CHECKPOINT_DIR,
            f"risky_equity_issuance_net_noinvest_{epoch}.weights.h5",
        ),
    )
    logger.info("DL simulator loaded (epoch %d)", epoch)
    return sim


# =========================================================================
#  §3  SMM Objective (standalone callable)
# =========================================================================

def compute_smm_loss(
    moments_estimated: np.ndarray,
    moments_golden: np.ndarray,
    weighting_matrix: np.ndarray,
) -> float:
    r"""Compute the weighted quadratic SMM loss.

    .. math::

        Q(\theta) = (\hat m - m^*)^\top W (\hat m - m^*)
    """
    diff = moments_estimated - moments_golden
    return float(diff @ weighting_matrix @ diff)


def make_smm_objective(
    dl_simulator: DLSimulatorRiskyFinal_dist,
    econ_params_template: EconomicParams,
    sim_initial_states: Any,
    sim_shock_sequence: Any,
    golden_moments: np.ndarray,
    weighting_matrix: np.ndarray,
) -> Callable[[np.ndarray], float]:
    """Build a standalone objective θ → Q(θ) for derivative-free optimizers.

    Uses common random numbers: the simulation shock sequence is fixed
    across all evaluations.

    Parameters
    ----------
    dl_simulator : DLSimulatorRiskyFinal_dist
    econ_params_template : EconomicParams
        Template with fixed structural params; free params overridden.
    sim_initial_states, sim_shock_sequence
        Fixed simulation panel tensors (N_SIM × T_SIM_RAW).
    golden_moments : np.ndarray, shape (9,)
    weighting_matrix : np.ndarray, shape (9, 9)

    Returns
    -------
    callable
        objective(theta) -> float, where theta = [ρ, σ, ξ, F, η₀, η₁].
    """
    delta = econ_params_template.depreciation_rate
    alpha = econ_params_template.capital_share
    r_f = econ_params_template.risk_free_rate
    bounds_lo = np.array([SEARCH_BOUNDS[k][0] for k in PARAM_ORDER])
    bounds_hi = np.array([SEARCH_BOUNDS[k][1] for k in PARAM_ORDER])

    eval_count = [0]
    best_Q_seen = [np.inf]
    t_obj_start = [time.perf_counter()]
    best_eval = {"theta": None, "moments": None, "Q": np.inf}

    def objective(theta: np.ndarray) -> float:
        # Clip to bounds (soft enforcement for Nelder-Mead)
        theta_clipped = np.clip(theta, bounds_lo, bounds_hi)

        # Penalty for out-of-bounds (large but smooth)
        penalty = 1e4 * np.sum((theta - theta_clipped) ** 2)

        current_params = dataclasses.replace(
            econ_params_template,
            productivity_persistence=float(theta_clipped[0]),
            productivity_std_dev=float(theta_clipped[1]),
            adjustment_cost_convex=float(theta_clipped[2]),
            adjustment_cost_fixed=float(theta_clipped[3]),
            equity_issuance_cost_fixed=float(theta_clipped[4]),
            equity_issuance_cost_linear=float(theta_clipped[5]),
        )

        dl_results = dl_simulator.simulate(
            sim_initial_states, sim_shock_sequence, current_params,
            jit_compile=True,
        )
        dl_stationary = apply_burn_in(dl_results, T_BURN)
        moments_est = compute_identification_moments(dl_stationary, delta, alpha, r_f=r_f)
        Q = compute_smm_loss(moments_est, golden_moments, weighting_matrix)

        val = Q + penalty
        eval_count[0] += 1
        if val < best_Q_seen[0]:
            best_Q_seen[0] = val
        if val < best_eval["Q"]:
            best_eval["Q"] = val
            best_eval["theta"] = theta_clipped.copy()
            best_eval["moments"] = moments_est.copy()

        if eval_count[0] % 10 == 0:
            elapsed = time.perf_counter() - t_obj_start[0]
            logger.info(
                "    [eval %4d | %.0fs] Q=%.6e  best=%.6e  "
                "θ=[%.4f, %.4f, %.4f, %.4f, %.4f, %.4f]",
                eval_count[0], elapsed, val, best_Q_seen[0],
                *theta_clipped,
            )

        return val

    objective.eval_count = eval_count  # type: ignore[attr-defined]
    objective.best_Q_seen = best_Q_seen  # type: ignore[attr-defined]
    objective.t_obj_start = t_obj_start  # type: ignore[attr-defined]
    objective.best_eval = best_eval  # type: ignore[attr-defined]

    def reset_counters():
        """Reset eval counter and timer (call before each CMA-ES start)."""
        eval_count[0] = 0
        best_Q_seen[0] = np.inf
        t_obj_start[0] = time.perf_counter()

    objective.reset_counters = reset_counters  # type: ignore[attr-defined]
    return objective


# =========================================================================
#  §3  CMA-ES + Nelder-Mead Optimizer
# =========================================================================

def run_cmaes_nelder_mead(
    objective: Callable[[np.ndarray], float],
    n_starts: int = DEFAULT_N_SOBOL_STARTS,
    cma_max_evals: int = DEFAULT_CMA_MAX_EVALS,
    nm_maxiter: int = 500,
) -> Dict[str, Any]:
    """Two-stage optimizer: CMA-ES global + Nelder-Mead local.

    Stage 1: Launch CMA-ES from each of ``n_starts`` Sobol-sequence
             points.  Terminate when relative Q change < 1e-3 over the
             last 50 evaluations, or after ``cma_max_evals`` evaluations.
    Stage 2: Take the best CMA-ES solution across all starts and refine
             with Nelder-Mead (xatol=1e-6, fatol=1e-4).

    Returns
    -------
    dict
        theta_hat, Q_min, n_evals, wall_time, stage1_results
    """
    bounds_lo = np.array([SEARCH_BOUNDS[k][0] for k in PARAM_ORDER])
    bounds_hi = np.array([SEARCH_BOUNDS[k][1] for k in PARAM_ORDER])
    ranges = bounds_hi - bounds_lo

    # Generate Sobol starting points
    sobol = Sobol(d=P_PARAMS, scramble=True)
    sobol_points = sobol.random(n_starts)
    starting_points = bounds_lo + sobol_points * ranges

    t_start = time.perf_counter()
    total_evals = 0
    best_Q = np.inf
    best_theta = None
    stage1_results: List[Dict[str, Any]] = []

    logger.info(
        "Stage 1: CMA-ES from %d Sobol starting points (max %d evals each)",
        n_starts, cma_max_evals,
    )

    for i, x0 in enumerate(starting_points):
        sigma0 = 0.2 * np.mean(ranges)
        opts = {
            "bounds": [bounds_lo.tolist(), bounds_hi.tolist()],
            "maxfevals": cma_max_evals,
            "tolx": 1e-6,
            "tolfun": 1e-8,
            "verbose": -9,
            "seed": i + 42,
        }

        logger.info(
            "  ── CMA-ES start %d/%d  "
            "x0=[%.3f, %.3f, %.3f, %.3f, %.3f, %.3f] ──",
            i + 1, n_starts, *x0,
        )
        if hasattr(objective, "reset_counters"):
            objective.reset_counters()

        try:
            es = cma.CMAEvolutionStrategy(x0.tolist(), sigma0, opts)
            es.optimize(objective)
            result_x = np.array(es.result.xbest)
            result_f = float(es.result.fbest)
            n_ev = int(es.result.evaluations)
        except Exception as e:
            logger.warning("CMA-ES start %d failed: %s", i, e)
            continue

        total_evals += n_ev
        stage1_results.append({
            "start": i,
            "theta": result_x.copy(),
            "Q": result_f,
            "evals": n_ev,
        })

        marker = ""
        if result_f < best_Q:
            best_Q = result_f
            best_theta = result_x.copy()
            marker = " ★ NEW BEST"

        logger.info(
            "  Start %2d/%d DONE: Q=%.6e  evals=%d  "
            "θ=[%.4f, %.4f, %.4f, %.4f, %.4f, %.4f]%s",
            i + 1, n_starts, result_f, n_ev, *result_x, marker,
        )

    if best_theta is None:
        raise RuntimeError("All CMA-ES starts failed.")

    logger.info(
        "Stage 1 best: Q=%.6e  θ=%s",
        best_Q, np.array2string(best_theta, precision=5),
    )

    # Stage 2: Nelder-Mead local refinement
    logger.info("Stage 2: Nelder-Mead refinement from best CMA-ES solution")
    logger.info(
        "  Starting θ=[%.5f, %.5f, %.5f, %.5f, %.5f, %.5f], Q=%.6e",
        *best_theta, best_Q,
    )

    if hasattr(objective, "reset_counters"):
        objective.reset_counters()

    nm_result = minimize(
        objective,
        best_theta,
        method="Nelder-Mead",
        options={
            "xatol": 1e-6,
            "fatol": 1e-4,
            "maxiter": nm_maxiter,
            "maxfev": nm_maxiter * 2,
            "adaptive": True,
        },
    )

    total_evals += nm_result.nfev
    wall_time = time.perf_counter() - t_start

    theta_hat = np.clip(nm_result.x, bounds_lo, bounds_hi)
    Q_min = float(nm_result.fun)

    logger.info(
        "Stage 2 done: Q=%.6e  NM_evals=%d  converged=%s",
        Q_min, nm_result.nfev, nm_result.success,
    )
    logger.info(
        "Total wall time: %.1f s, total evals: %d", wall_time, total_evals
    )

    # Retrieve cached moments at the best point found across all stages
    cached_moments = None
    if hasattr(objective, "best_eval") and objective.best_eval["moments"] is not None:
        cached_moments = objective.best_eval["moments"]

    return {
        "theta_hat": theta_hat,
        "Q_min": Q_min,
        "n_evals": total_evals,
        "wall_time": wall_time,
        "nm_result": nm_result,
        "stage1_results": stage1_results,
        "cached_model_moments": cached_moments,
    }


# =========================================================================
#  §4  Standard Error Computation
# =========================================================================

def compute_smm_jacobian(
    theta_hat: np.ndarray,
    dl_simulator: DLSimulatorRiskyFinal_dist,
    econ_params_template: EconomicParams,
    sim_initial_states: Any,
    sim_shock_sequence: Any,
    h_rel: float = 0.01,
) -> np.ndarray:
    r"""Compute the moment Jacobian G = ∂m/∂θ at θ̂ via central differences.

    Uses 1 % relative perturbation: :math:`h_j = h_{\text{rel}} \times |\hat\theta_j|`
    (minimum 1e-5).  Requires :math:`2 \times 6 = 12` DL simulator evaluations.

    Returns
    -------
    G : np.ndarray, shape (9, 6)
    """
    delta = econ_params_template.depreciation_rate
    alpha = econ_params_template.capital_share
    r_f = econ_params_template.risk_free_rate

    def _moments_at(theta: np.ndarray) -> np.ndarray:
        params = dataclasses.replace(
            econ_params_template,
            productivity_persistence=float(theta[0]),
            productivity_std_dev=float(theta[1]),
            adjustment_cost_convex=float(theta[2]),
            adjustment_cost_fixed=float(theta[3]),
            equity_issuance_cost_fixed=float(theta[4]),
            equity_issuance_cost_linear=float(theta[5]),
        )
        dl_results = dl_simulator.simulate(
            sim_initial_states, sim_shock_sequence, params,
            jit_compile=True,
        )
        stationary = apply_burn_in(dl_results, T_BURN)
        return compute_identification_moments(stationary, delta, alpha, r_f=r_f)

    G = np.zeros((K_MOMENTS, P_PARAMS))
    for j in range(P_PARAMS):
        h_j = max(h_rel * abs(theta_hat[j]), 1e-5)
        theta_plus = theta_hat.copy()
        theta_minus = theta_hat.copy()
        theta_plus[j] += h_j
        theta_minus[j] -= h_j

        logger.info(
            "  Jacobian: computing ∂m/∂%s  (%d/%d) ...",
            PARAM_LABELS[j], j + 1, P_PARAMS,
        )
        m_plus = _moments_at(theta_plus)
        m_minus = _moments_at(theta_minus)
        G[:, j] = (m_plus - m_minus) / (2.0 * h_j)

    logger.info("Jacobian G computed (shape %s)", G.shape)
    return G


def compute_smm_standard_errors(
    theta_hat: np.ndarray,
    G: np.ndarray,
    W: np.ndarray,
    N: int,
    J: float,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Compute asymptotic SMM standard errors.

    .. math::

        \Sigma_{\hat\theta} = \frac{1}{N}\left(1 + \frac{1}{J}\right)
                              \left(G^\top W G \right)^{-1}

    Parameters
    ----------
    theta_hat : np.ndarray, shape (6,)
    G : np.ndarray, shape (9, 6)
    W : np.ndarray, shape (9, 9)
    N : int — N_DATA × T_DATA_EFF
    J : float — simulation-to-data ratio

    Returns
    -------
    se : np.ndarray, shape (6,) — standard errors
    Sigma : np.ndarray, shape (6, 6) — full covariance matrix
    """
    GtWG = G.T @ W @ G
    try:
        GtWG_inv = np.linalg.inv(GtWG)
    except np.linalg.LinAlgError:
        logger.warning("G'WG singular — using pseudo-inverse")
        GtWG_inv = np.linalg.pinv(GtWG)

    Sigma = (1.0 / N) * (1.0 + 1.0 / J) * GtWG_inv
    se = np.sqrt(np.abs(np.diag(Sigma)))

    logger.info("Standard errors: %s", np.array2string(se, precision=6))
    return se, Sigma


# =========================================================================
#  §5  J-Test for Overidentifying Restrictions
# =========================================================================

def compute_j_test(
    Q_min: float,
    N: int,
    J: float,
    k: int = K_MOMENTS,
    p: int = P_PARAMS,
) -> Dict[str, float]:
    r"""Compute the J-test statistic for overidentification.

    .. math::

        T_J = \frac{N \cdot J}{1 + J} \cdot Q(\hat\theta) \;\sim\; \chi^2(k - p)

    Returns
    -------
    dict with keys: T_J, p_value, df
    """
    T_J = (N * J / (1.0 + J)) * Q_min
    df = k - p
    p_value = 1.0 - chi2.cdf(T_J, df)

    logger.info("J-test: T_J=%.4f, df=%d, p-value=%.4f", T_J, df, p_value)
    return {"T_J": float(T_J), "p_value": float(p_value), "df": int(df)}


# =========================================================================
#  §6  Moment Fit Table
# =========================================================================

def compute_moment_fit_table(
    theta_hat: np.ndarray,
    golden_moments: np.ndarray,
    dl_simulator: DLSimulatorRiskyFinal_dist,
    econ_params_template: EconomicParams,
    sim_initial_states: Any,
    sim_shock_sequence: Any,
    Omega_hat: Optional[np.ndarray] = None,
    model_moments: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Compute the moment fit table at θ̂.

    Returns data moment, model moment, absolute error, and t-statistic
    for each of the 9 moments.
    """
    if model_moments is None:
        params_hat = dataclasses.replace(
            econ_params_template,
            productivity_persistence=float(theta_hat[0]),
            productivity_std_dev=float(theta_hat[1]),
            adjustment_cost_convex=float(theta_hat[2]),
            adjustment_cost_fixed=float(theta_hat[3]),
            equity_issuance_cost_fixed=float(theta_hat[4]),
            equity_issuance_cost_linear=float(theta_hat[5]),
        )
        dl_results = dl_simulator.simulate(
            sim_initial_states, sim_shock_sequence, params_hat,
            jit_compile=True,
        )
        stationary = apply_burn_in(dl_results, T_BURN)
        delta = econ_params_template.depreciation_rate
        alpha = econ_params_template.capital_share
        r_f = econ_params_template.risk_free_rate
        model_moments = compute_identification_moments(stationary, delta, alpha, r_f=r_f)
    else:
        logger.info(
            "Using cached model moments for moment fit table "
            "(skipping DL simulation)"
        )

    table: List[Dict[str, Any]] = []
    for i, name in enumerate(MOMENT_NAMES):
        data_m = golden_moments[i]
        model_m = model_moments[i]
        abs_err = abs(model_m - data_m)

        if Omega_hat is not None and Omega_hat[i, i] > 0:
            se_data = np.sqrt(Omega_hat[i, i])
            t_stat = (model_m - data_m) / se_data
        else:
            t_stat = float("nan")

        table.append({
            "moment": name,
            "data": float(data_m),
            "model": float(model_m),
            "abs_error": float(abs_err),
            "t_stat": float(t_stat),
        })

    # Log table
    logger.info("\n" + "=" * 72)
    logger.info("MOMENT FIT TABLE")
    logger.info(
        "%-22s  %10s  %10s  %10s  %10s",
        "Moment", "Data", "Model", "|Error|", "t-stat",
    )
    logger.info("-" * 72)
    for row in table:
        logger.info(
            "%-22s  %10.6f  %10.6f  %10.6f  %10.3f",
            row["moment"], row["data"], row["model"],
            row["abs_error"], row["t_stat"],
        )
    logger.info("=" * 72)

    return {"table": table, "model_moments": model_moments}


# =========================================================================
#  §8  Confidence Ellipse (η₀, η₁)
# =========================================================================

def plot_confidence_ellipse(
    theta_hat: np.ndarray,
    Sigma: np.ndarray,
    output_path: str,
    mc_thetas: Optional[np.ndarray] = None,
) -> None:
    """Plot the 95 % confidence ellipse for the (η₀, η₁) pair.

    Uses the equity issuance cost parameters (indices 4, 5) since these
    are the novel parameters in the risky debt model.

    Parameters
    ----------
    theta_hat : np.ndarray, shape (6,)
    Sigma : np.ndarray, shape (6, 6)
    output_path : str
    mc_thetas : np.ndarray, shape (R, 6), optional
    """
    eta0_hat, eta1_hat = theta_hat[4], theta_hat[5]
    sub_cov = Sigma[np.ix_([4, 5], [4, 5])]

    eigenvalues, eigenvectors = np.linalg.eigh(sub_cov)
    chi2_val = chi2.ppf(0.95, 2)

    angle = np.degrees(np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1]))
    width = 2.0 * np.sqrt(chi2_val * max(eigenvalues[1], 0))
    height = 2.0 * np.sqrt(chi2_val * max(eigenvalues[0], 0))

    fig, ax = plt.subplots(figsize=(8, 6))

    ellipse = Ellipse(
        xy=(eta0_hat, eta1_hat),
        width=width, height=height, angle=angle,
        edgecolor="blue", facecolor="lightblue", alpha=0.3, linewidth=2,
        label="95% confidence ellipse",
    )
    ax.add_patch(ellipse)

    # True values
    ax.plot(
        TRUE_PARAMS["equity_issuance_cost_fixed"],
        TRUE_PARAMS["equity_issuance_cost_linear"],
        "r*", markersize=15,
        label=r"True ($\eta_0^*$, $\eta_1^*$)", zorder=5,
    )
    ax.plot(
        eta0_hat, eta1_hat,
        "b^", markersize=10,
        label=r"Estimate ($\hat\eta_0$, $\hat\eta_1$)", zorder=5,
    )

    if mc_thetas is not None and len(mc_thetas) > 1:
        ax.scatter(
            mc_thetas[:, 4], mc_thetas[:, 5],
            s=15, alpha=0.4, c="gray",
            label="MC replications", zorder=3,
        )

    ax.set_xlabel(r"$\eta_0$ (fixed equity issuance cost)", fontsize=12)
    ax.set_ylabel(r"$\eta_1$ (linear equity issuance cost)", fontsize=12)
    ax.set_title(
        r"Pairwise ($\eta_0$, $\eta_1$) 95% Confidence Ellipse",
        fontsize=13,
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(
        os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
        exist_ok=True,
    )
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Confidence ellipse saved: %s", output_path)


# =========================================================================
#  §9  Condition Number at θ̂
# =========================================================================

def compute_condition_number(
    G: np.ndarray,
    theta: np.ndarray,
    golden_moments: np.ndarray,
) -> float:
    r"""Compute condition number of the elasticity-normalized Jacobian.

    .. math::

        E_{ij} = G_{ij} \times \frac{\theta_j}{m_i}, \qquad
        \kappa = \frac{s_1}{s_p}
    """
    m_abs = np.maximum(np.abs(golden_moments), 1e-10)
    theta_abs = np.maximum(np.abs(theta), 1e-10)

    E = G * (theta_abs[np.newaxis, :] / m_abs[:, np.newaxis])
    s = np.linalg.svd(E, compute_uv=False)
    kappa = float(s[0] / s[-1]) if s[-1] > 1e-15 else float("inf")

    logger.info(
        "Condition number κ(E) = %.2f  (singular values: %s)",
        kappa, np.array2string(s, precision=4),
    )
    return kappa


# =========================================================================
#  §7  Monte Carlo — Single Estimation
# =========================================================================

@dataclasses.dataclass
class MCReplicationResult:
    """Stores results from a single MC replication."""
    replication_id: int
    theta_hat: np.ndarray
    se: np.ndarray
    Q_min: float
    j_stat: float
    j_pvalue: float
    n_evals: int
    wall_time: float
    golden_moments: np.ndarray
    model_moments: np.ndarray


def run_single_estimation(
    econ_params: EconomicParams,
    bonds_config: Dict,
    dl_simulator: DLSimulatorRiskyFinal_dist,
    replication_id: int = 0,
    use_diagonal_W: bool = False,
    n_sobol_starts: int = DEFAULT_N_SOBOL_STARTS,
    cma_max_evals: int = DEFAULT_CMA_MAX_EVALS,
    rng: Optional[np.random.Generator] = None,
    vfi_solution: Optional[Any] = None,
) -> Tuple[MCReplicationResult, Dict[str, Any]]:
    """Run one complete SMM estimation.

    Pipeline:
      1. Generate data-panel shocks → VFI golden moments
      2. Bootstrap (or diagonal) weighting matrix
      3. Generate sim-panel shocks (fixed CRN for all evals)
      4. CMA-ES + Nelder-Mead optimization
      5. Post-estimation: SE, J-test, moment fit, condition number

    Parameters
    ----------
    vfi_solution : optional
        Pre-loaded VFI solution arrays (avoids disk I/O per replication).

    Returns
    -------
    result : MCReplicationResult
    diagnostics : dict
        G, Sigma, Omega_hat, W, kappa, moment_fit, opt_result, j_test,
        sim_init, sim_shocks, dl_moments_at_true
    """
    if rng is None:
        rng = np.random.default_rng()

    delta = econ_params.depreciation_rate
    alpha = econ_params.capital_share
    r_f = econ_params.risk_free_rate

    # ── Data panel ──────────────────────────────────────────────────
    logger.info(
        "[Rep %d] Generating data panel (N=%d, T=%d) ...",
        replication_id, N_DATA, T_DATA_EFF,
    )
    data_gen = synthetic_data_generator(
        econ_params_benchmark=econ_params,
        sample_bonds_config=bonds_config,
        batch_size=N_DATA,
        T_periods=T_DATA_RAW,
        include_debt=True,
    )
    data_init, data_shocks = data_gen.gen()

    golden_moments, stationary_data = compute_golden_moments(
        econ_params, data_init, data_shocks, bounds=bonds_config,
        vfi_solution=vfi_solution,
    )

    # ── Weighting matrix ────────────────────────────────────────────
    if use_diagonal_W:
        W = build_diagonal_weighting_matrix(golden_moments)
        Omega_hat = None
        logger.info("[Rep %d] Using diagonal weighting matrix", replication_id)
    else:
        W, Omega_hat = bootstrap_weighting_matrix(
            stationary_data, delta, alpha, r_f=r_f,
            B=DEFAULT_N_BOOTSTRAP, rng=rng,
        )

    # ── Simulation panel (CRN) ──────────────────────────────────────
    logger.info(
        "[Rep %d] Generating simulation panel (N=%d, T=%d) ...",
        replication_id, N_SIM, T_SIM_EFF,
    )
    sim_gen = synthetic_data_generator(
        econ_params_benchmark=econ_params,
        sample_bonds_config=bonds_config,
        batch_size=N_SIM,
        T_periods=T_SIM_RAW,
        include_debt=True,
    )
    sim_init, sim_shocks = sim_gen.gen()

    # ── Warm-up: trace @tf.function graph once before timed loop ───
    logger.info(
        "[Rep %d] Warming up DL simulator (@tf.function tracing) ...",
        replication_id,
    )
    _warmup_t0 = time.perf_counter()
    warmup_results = dl_simulator.simulate(sim_init, sim_shocks, econ_params, jit_compile=True)
    logger.info(
        "[Rep %d] Warm-up done (%.1f s, subsequent calls will be faster)",
        replication_id, time.perf_counter() - _warmup_t0,
    )
    warmup_stationary = apply_burn_in(warmup_results, T_BURN)
    dl_moments_at_true = compute_identification_moments(
        warmup_stationary, delta, alpha, r_f=r_f,
    )

    # ── Build objective ─────────────────────────────────────────────
    objective = make_smm_objective(
        dl_simulator, econ_params, sim_init, sim_shocks,
        golden_moments, W,
    )

    # ── Optimize ────────────────────────────────────────────────────
    logger.info(
        "[Rep %d] Starting CMA-ES + Nelder-Mead optimization ...",
        replication_id,
    )
    opt_result = run_cmaes_nelder_mead(
        objective,
        n_starts=n_sobol_starts,
        cma_max_evals=cma_max_evals,
    )
    theta_hat = opt_result["theta_hat"]
    Q_min = opt_result["Q_min"]

    # ── Post-estimation diagnostics ─────────────────────────────────
    N_total = N_DATA * T_DATA_EFF

    # Jacobian
    G = compute_smm_jacobian(
        theta_hat, dl_simulator, econ_params,
        sim_init, sim_shocks,
    )

    # Standard errors
    se, Sigma = compute_smm_standard_errors(theta_hat, G, W, N_total, J_RATIO)

    # J-test
    j_result = compute_j_test(Q_min, N_total, J_RATIO)

    # Moment fit (use cached moments from optimizer to avoid re-simulation)
    cached_moments = opt_result.get("cached_model_moments")
    fit_result = compute_moment_fit_table(
        theta_hat, golden_moments, dl_simulator, econ_params,
        sim_init, sim_shocks, Omega_hat,
        model_moments=cached_moments,
    )

    # Condition number
    kappa = compute_condition_number(G, theta_hat, golden_moments)

    # ── Log parameter recovery ──────────────────────────────────────
    logger.info("\n" + "=" * 72)
    logger.info("PARAMETER RECOVERY (Rep %d)", replication_id)
    logger.info(
        "%-10s  %10s  %10s  %10s  %10s  %10s",
        "Param", "True", "Estimate", "SE", "Error", "Error%",
    )
    logger.info("-" * 72)
    for j, key in enumerate(PARAM_ORDER):
        true_val = TRUE_PARAMS[key]
        est_val = theta_hat[j]
        err = est_val - true_val
        pct_err = 100.0 * err / true_val if abs(true_val) > 1e-10 else 0.0
        logger.info(
            "%-10s  %10.5f  %10.5f  %10.5f  %+10.5f  %+9.2f%%",
            PARAM_LABELS[j], true_val, est_val, se[j], err, pct_err,
        )
    logger.info("Q(θ̂) = %.6e", Q_min)
    logger.info(
        "J-stat = %.4f (p = %.4f)", j_result["T_J"], j_result["p_value"]
    )
    logger.info("κ(E) = %.2f", kappa)
    logger.info(
        "Wall time = %.1f s, evals = %d",
        opt_result["wall_time"], opt_result["n_evals"],
    )
    logger.info("=" * 72)

    mc_result = MCReplicationResult(
        replication_id=replication_id,
        theta_hat=theta_hat,
        se=se,
        Q_min=Q_min,
        j_stat=j_result["T_J"],
        j_pvalue=j_result["p_value"],
        n_evals=opt_result["n_evals"],
        wall_time=opt_result["wall_time"],
        golden_moments=golden_moments,
        model_moments=fit_result["model_moments"],
    )

    diagnostics = {
        "G": G,
        "Sigma": Sigma,
        "Omega_hat": Omega_hat,
        "W": W,
        "kappa": kappa,
        "moment_fit": fit_result,
        "opt_result": opt_result,
        "j_test": j_result,
        "sim_init": sim_init,
        "sim_shocks": sim_shocks,
        "dl_moments_at_true": dl_moments_at_true,
    }

    return mc_result, diagnostics


# =========================================================================
#  §7 + §10  MC Aggregation
# =========================================================================

def aggregate_mc_results(
    results: List[MCReplicationResult],
) -> Dict[str, Any]:
    """Aggregate Monte Carlo replications into summary statistics.

    Computes: median θ̂, mean SE, SD(θ̂), bias%, RMSE%, 95% CI coverage,
    J-test rejection rate.
    """
    R = len(results)
    thetas = np.array([r.theta_hat for r in results])     # (R, 6)
    ses = np.array([r.se for r in results])                # (R, 6)
    true_vals = np.array([TRUE_PARAMS[k] for k in PARAM_ORDER])

    median_theta = np.median(thetas, axis=0)
    mean_se = np.mean(ses, axis=0)
    sd_theta = (
        np.std(thetas, axis=0, ddof=1) if R > 1 else np.zeros(P_PARAMS)
    )

    bias = np.mean(thetas, axis=0) - true_vals
    bias_pct = 100.0 * bias / true_vals

    rmse = np.sqrt(np.mean((thetas - true_vals) ** 2, axis=0))
    rmse_pct = 100.0 * rmse / true_vals

    # 95 % CI coverage
    coverage = np.zeros(P_PARAMS)
    for j in range(P_PARAMS):
        lo = thetas[:, j] - 1.96 * ses[:, j]
        hi = thetas[:, j] + 1.96 * ses[:, j]
        coverage[j] = np.mean(
            (true_vals[j] >= lo) & (true_vals[j] <= hi)
        )

    # J-test rejection rate
    j_pvalues = np.array([r.j_pvalue for r in results])
    rejection_rate = float(np.mean(j_pvalues < 0.05))

    mean_Q = float(np.mean([r.Q_min for r in results]))
    mean_evals = float(np.mean([r.n_evals for r in results]))
    mean_time = float(np.mean([r.wall_time for r in results]))

    summary: Dict[str, Any] = {
        "R": R,
        "param_labels": PARAM_LABELS,
        "param_keys": PARAM_ORDER,
        "true_values": true_vals.tolist(),
        "median_theta": median_theta.tolist(),
        "mean_se": mean_se.tolist(),
        "sd_theta": sd_theta.tolist(),
        "bias": bias.tolist(),
        "bias_pct": bias_pct.tolist(),
        "rmse": rmse.tolist(),
        "rmse_pct": rmse_pct.tolist(),
        "coverage_95": coverage.tolist(),
        "j_rejection_rate": rejection_rate,
        "mean_Q": mean_Q,
        "mean_evals": mean_evals,
        "mean_wall_time": mean_time,
    }

    # Log summary
    logger.info("\n" + "=" * 80)
    logger.info("MONTE CARLO SUMMARY  (%d replications)", R)
    logger.info("=" * 80)
    logger.info(
        "%-10s  %8s  %8s  %8s  %8s  %8s  %8s  %8s",
        "Param", "True", "Median", "SE_avg", "SD(θ̂)",
        "Bias%", "RMSE%", "Cover95",
    )
    logger.info("-" * 80)
    for j in range(P_PARAMS):
        logger.info(
            "%-10s  %8.4f  %8.4f  %8.4f  %8.4f  %+7.2f%%  %7.2f%%  %7.1f%%",
            PARAM_LABELS[j], true_vals[j], median_theta[j],
            mean_se[j], sd_theta[j], bias_pct[j], rmse_pct[j],
            100.0 * coverage[j],
        )
    logger.info("-" * 80)
    logger.info(
        "J-test rejection rate (5%%): %.1f%%", 100.0 * rejection_rate
    )
    logger.info("Mean Q(θ̂): %.6e", mean_Q)
    logger.info(
        "Mean evals: %.0f, Mean time: %.1f s", mean_evals, mean_time
    )
    logger.info("=" * 80)

    return summary


# =========================================================================
#  §10  Serialization
# =========================================================================

def save_replication_csv(
    results: List[MCReplicationResult],
    output_path: str,
) -> None:
    """Save per-replication results to CSV."""
    os.makedirs(
        os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
        exist_ok=True,
    )
    fieldnames = [
        "replication_id",
        "rho_hat", "sigma_hat", "xi_hat", "F_hat", "eta0_hat", "eta1_hat",
        "se_rho", "se_sigma", "se_xi", "se_F", "se_eta0", "se_eta1",
        "Q_min", "j_stat", "j_pvalue", "n_evals", "wall_time",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "replication_id": r.replication_id,
                "rho_hat": r.theta_hat[0],
                "sigma_hat": r.theta_hat[1],
                "xi_hat": r.theta_hat[2],
                "F_hat": r.theta_hat[3],
                "eta0_hat": r.theta_hat[4],
                "eta1_hat": r.theta_hat[5],
                "se_rho": r.se[0],
                "se_sigma": r.se[1],
                "se_xi": r.se[2],
                "se_F": r.se[3],
                "se_eta0": r.se[4],
                "se_eta1": r.se[5],
                "Q_min": r.Q_min,
                "j_stat": r.j_stat,
                "j_pvalue": r.j_pvalue,
                "n_evals": r.n_evals,
                "wall_time": r.wall_time,
            })
    logger.info("Replication CSV saved: %s", output_path)


def save_summary_json(summary: Dict[str, Any], output_path: str) -> None:
    """Save MC summary statistics to JSON."""
    os.makedirs(
        os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
        exist_ok=True,
    )
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary JSON saved: %s", output_path)


def generate_latex_table_recovery(summary: Dict[str, Any]) -> str:
    r"""Generate LaTeX string for Parameter Recovery table."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{SMM Parameter Recovery --- Risky Debt Model}",
        r"\label{tab:smm_risky_recovery}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Parameter & True & Median $\hat{\theta}$ & SE "
        r"& Bias (\%) & RMSE (\%) & Coverage (95\%) \\",
        r"\midrule",
    ]
    for j in range(P_PARAMS):
        lines.append(
            f"${PARAM_LABELS[j]}$ & "
            f"{summary['true_values'][j]:.4f} & "
            f"{summary['median_theta'][j]:.4f} & "
            f"{summary['mean_se'][j]:.4f} & "
            f"{summary['bias_pct'][j]:+.2f} & "
            f"{summary['rmse_pct'][j]:.2f} & "
            f"{100 * summary['coverage_95'][j]:.1f}\\% \\\\"
        )
    lines += [
        r"\midrule",
        (
            r"\multicolumn{7}{l}{J-test rejection rate (5\%): "
            f"{100 * summary['j_rejection_rate']:.1f}\\%}} \\\\"
        ),
        f"\\multicolumn{{7}}{{l}}{{MC replications: {summary['R']}}} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def generate_latex_table_moment_fit(
    moment_fit: Dict[str, Any],
    j_test: Dict[str, float],
) -> str:
    r"""Generate LaTeX string for Moment Fit table."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Moment Fit at $\hat{\theta}$ --- Risky Debt Model}",
        r"\label{tab:smm_risky_moment_fit}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Moment & Data & Model & $|$Error$|$ & $t$-stat \\",
        r"\midrule",
    ]
    for row in moment_fit["table"]:
        t_str = (
            f"{row['t_stat']:.3f}" if np.isfinite(row["t_stat"]) else "---"
        )
        lines.append(
            f"{row['moment']} & "
            f"{row['data']:.6f} & "
            f"{row['model']:.6f} & "
            f"{row['abs_error']:.6f} & "
            f"{t_str} \\\\"
        )
    lines += [
        r"\midrule",
        (
            r"\multicolumn{5}{l}{J-statistic: "
            f"{j_test['T_J']:.4f} (p = {j_test['p_value']:.4f})}} \\\\"
        ),
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def generate_latex_table_design(summary: Dict[str, Any]) -> str:
    r"""Generate LaTeX string for MC Design table."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Monte Carlo Experimental Design --- Risky Debt Model}",
        r"\label{tab:smm_risky_mc_design}",
        r"\begin{tabular}{lc}",
        r"\toprule",
        r"Setting & Value \\",
        r"\midrule",
        f"$N_{{\\text{{data}}}}$ (firms) & {N_DATA} \\\\",
        f"$T_{{\\text{{data}}}}$ (periods) & {T_DATA_EFF} \\\\",
        f"$N_{{\\text{{sim}}}}$ (firms) & {N_SIM} \\\\",
        f"$T_{{\\text{{sim}}}}$ (periods) & {T_SIM_EFF} \\\\",
        f"Burn-in & {T_BURN} \\\\",
        f"$J$ (sim/data ratio) & {J_RATIO:.2f} \\\\",
        f"Bootstrap resamples ($B$) & {DEFAULT_N_BOOTSTRAP} \\\\",
        f"CMA-ES starts & {DEFAULT_N_SOBOL_STARTS} \\\\",
        f"MC replications ($R$) & {summary['R']} \\\\",
        f"Mean wall time per run (s) & {summary['mean_wall_time']:.1f} \\\\",
        f"Mean function evaluations & {summary['mean_evals']:.0f} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# =========================================================================
#  Main
# =========================================================================

def main() -> None:
    """Run the complete risky debt SMM estimation pipeline."""
    parser = argparse.ArgumentParser(
        description=(
            "SMM estimation with DL simulator — risky debt model"
        )
    )
    parser.add_argument(
        "--mc-replications", type=int, default=DEFAULT_MC_REPLICATIONS,
        help="Number of Monte Carlo replications (default: 1)",
    )
    parser.add_argument(
        "--dl-epoch", type=int, default=DEFAULT_DL_EPOCH,
        help=f"DL checkpoint epoch (default: {DEFAULT_DL_EPOCH})",
    )
    parser.add_argument(
        "--n-sobol-starts", type=int, default=DEFAULT_N_SOBOL_STARTS,
        help=f"CMA-ES starting points (default: {DEFAULT_N_SOBOL_STARTS})",
    )
    parser.add_argument(
        "--cma-max-evals", type=int, default=DEFAULT_CMA_MAX_EVALS,
        help=(
            "Max CMA-ES evaluations per start "
            f"(default: {DEFAULT_CMA_MAX_EVALS})"
        ),
    )
    parser.add_argument(
        "--diagonal-W", action="store_true",
        help="Use diagonal 1/m² weighting instead of bootstrap",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./results/smm_risky",
        help="Output directory for results",
    )
    parser.add_argument(
        "--seed", type=int, default=2026,
        help="Master random seed",
    )
    args = parser.parse_args()

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    master_rng = np.random.default_rng(args.seed)

    logger.info("=" * 72)
    logger.info("SMM ESTIMATION — RISKY DEBT MODEL")
    logger.info("=" * 72)
    logger.info("  MC replications  : %d", args.mc_replications)
    logger.info("  DL epoch         : %d", args.dl_epoch)
    logger.info("  Sobol starts     : %d", args.n_sobol_starts)
    logger.info("  CMA-ES max evals : %d", args.cma_max_evals)
    logger.info(
        "  Weighting        : %s",
        "diagonal" if args.diagonal_W else "bootstrap",
    )
    logger.info(
        "  Data panel       : N=%d, T_eff=%d (T_raw=%d)",
        N_DATA, T_DATA_EFF, T_DATA_RAW,
    )
    logger.info(
        "  Sim panel        : N=%d, T_eff=%d (T_raw=%d), burn=%d",
        N_SIM, T_SIM_EFF, T_SIM_RAW, T_BURN,
    )
    logger.info("  J ratio          : %.2f", J_RATIO)
    logger.info("  Output dir       : %s", args.output_dir)
    logger.info("  TRUE_PARAMS      : %s", TRUE_PARAMS)
    logger.info("  SEARCH_BOUNDS    : %s", SEARCH_BOUNDS)

    # ─── Load economic parameters ───────────────────────────────────
    econ_path = _econ_params_path()
    logger.info("Loading econ params: %s", econ_path)
    econ_params = EconomicParams(**load_json_file(econ_path))

    bounds_path_ = _bounds_path()
    logger.info("Loading bounds: %s", bounds_path_)
    bonds_config = BondsConfig.validate_and_load(
        bounds_file=bounds_path_, current_params=econ_params,
    )

    # Load golden VFI solution once for all replications
    golden_path = _golden_vfi_path()
    if not os.path.exists(golden_path):
        raise FileNotFoundError(
            f"Golden VFI file not found: {golden_path}. "
            "Run risky_golden_vfi_finder.py first."
        )
    vfi_solution = np.load(golden_path, allow_pickle=True)
    logger.info("Golden VFI solution loaded once: %s", golden_path)

    # Load DL simulator (shared across all replications)
    dl_simulator = create_dl_simulator(args.dl_epoch)

    # ─── Monte Carlo loop ───────────────────────────────────────────
    all_results: List[MCReplicationResult] = []
    all_diagnostics: List[Dict[str, Any]] = []

    for rep in range(args.mc_replications):
        logger.info(
            "\n%s\n  MC REPLICATION %d / %d\n%s",
            "=" * 72, rep + 1, args.mc_replications, "=" * 72,
        )
        rep_rng = np.random.default_rng(master_rng.integers(0, 2**32))

        result, diagnostics = run_single_estimation(
            econ_params=econ_params,
            bonds_config=bonds_config,
            dl_simulator=dl_simulator,
            replication_id=rep,
            use_diagonal_W=args.diagonal_W,
            n_sobol_starts=args.n_sobol_starts,
            cma_max_evals=args.cma_max_evals,
            rng=rep_rng,
            vfi_solution=vfi_solution,
        )
        all_results.append(result)
        all_diagnostics.append(diagnostics)

    # ─── Aggregate and serialize ────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)

    summary = aggregate_mc_results(all_results)

    # Per-replication CSV
    save_replication_csv(
        all_results, os.path.join(args.output_dir, "replications.csv")
    )

    # Summary JSON
    save_summary_json(
        summary, os.path.join(args.output_dir, "summary.json")
    )

    # §8: Confidence ellipse (using last replication diagnostics)
    if all_diagnostics and all_diagnostics[-1]["Sigma"] is not None:
        mc_thetas = (
            np.array([r.theta_hat for r in all_results])
            if len(all_results) > 1 else None
        )
        plot_confidence_ellipse(
            theta_hat=all_results[-1].theta_hat,
            Sigma=all_diagnostics[-1]["Sigma"],
            output_path=os.path.join(
                args.output_dir, "eta0_eta1_confidence_ellipse.png"
            ),
            mc_thetas=mc_thetas,
        )

    # §10: LaTeX tables
    latex_recovery = generate_latex_table_recovery(summary)
    latex_design = generate_latex_table_design(summary)

    last_diag = all_diagnostics[-1]
    latex_fit = generate_latex_table_moment_fit(
        last_diag["moment_fit"], last_diag["j_test"]
    )

    latex_path = os.path.join(args.output_dir, "latex_tables.tex")
    with open(latex_path, "w") as f:
        f.write("% Auto-generated LaTeX tables — risky debt model SMM\n")
        f.write("% Generated by risky_SMM.py\n\n")
        f.write("% MC Design\n")
        f.write(latex_design + "\n\n")
        f.write("% Parameter Recovery\n")
        f.write(latex_recovery + "\n\n")
        f.write("% Moment Fit\n")
        f.write(latex_fit + "\n")
    logger.info("LaTeX tables saved: %s", latex_path)

    # ─── §9: DL approximation bias at θ* ───────────────────────────
    logger.info("\n§9: DL approximation bias at θ* ...")
    true_theta = np.array([TRUE_PARAMS[k] for k in PARAM_ORDER])

    sim_init_last = last_diag["sim_init"]
    sim_shocks_last = last_diag["sim_shocks"]
    dl_moments_star = last_diag["dl_moments_at_true"]

    # VFI moments at θ* on the same simulation panel
    vfi_sim = VFISimulator_risky(econ_params)
    vfi_sim.load_solved_vfi_solution(vfi_solution)
    vfi_results_star = vfi_sim.simulate(
        tuple(
            s.numpy() if hasattr(s, "numpy") else s for s in sim_init_last
        ),
        (
            sim_shocks_last.numpy()
            if hasattr(sim_shocks_last, "numpy") else sim_shocks_last
        ),
    )
    vfi_stationary_star = apply_burn_in(vfi_results_star, T_BURN)
    vfi_moments_star = compute_identification_moments(
        vfi_stationary_star,
        econ_params.depreciation_rate,
        econ_params.capital_share,
        r_f=econ_params.risk_free_rate,
    )

    dl_bias = dl_moments_star - vfi_moments_star
    logger.info("\nDL APPROXIMATION BIAS AT θ*:")
    logger.info(
        "%-22s  %10s  %10s  %10s  %10s",
        "Moment", "VFI", "DL", "Δm", "|Δm/m|%",
    )
    for i, name in enumerate(MOMENT_NAMES):
        pct = (
            100.0 * abs(dl_bias[i]) / max(abs(vfi_moments_star[i]), 1e-10)
        )
        logger.info(
            "%-22s  %10.6f  %10.6f  %+10.6f  %9.2f%%",
            name, vfi_moments_star[i], dl_moments_star[i], dl_bias[i], pct,
        )

    # Save DL bias info
    dl_bias_info = {
        "moment_names": MOMENT_NAMES,
        "vfi_moments_at_true": vfi_moments_star.tolist(),
        "dl_moments_at_true": dl_moments_star.tolist(),
        "dl_bias": dl_bias.tolist(),
        "dl_bias_pct": [
            100.0 * abs(dl_bias[i]) / max(abs(vfi_moments_star[i]), 1e-10)
            for i in range(K_MOMENTS)
        ],
    }
    with open(os.path.join(args.output_dir, "dl_bias.json"), "w") as f:
        json.dump(dl_bias_info, f, indent=2)

    # §9: Condition number comparison at θ̂ vs θ*
    if all_diagnostics:
        kappa_hat = last_diag["kappa"]
        G_star = compute_smm_jacobian(
            true_theta, dl_simulator, econ_params,
            sim_init_last, sim_shocks_last,
        )
        kappa_star = compute_condition_number(
            G_star, true_theta, vfi_moments_star
        )
        logger.info("\nCONDITION NUMBER COMPARISON:")
        logger.info("  κ(E) at θ*: %.2f", kappa_star)
        logger.info("  κ(E) at θ̂:  %.2f", kappa_hat)

        condition_info = {
            "kappa_at_true": float(kappa_star),
            "kappa_at_estimate": float(kappa_hat),
        }
        with open(
            os.path.join(args.output_dir, "condition_numbers.json"), "w"
        ) as f:
            json.dump(condition_info, f, indent=2)

    logger.info("\n" + "=" * 72)
    logger.info("ALL DONE — results saved to %s", args.output_dir)
    logger.info("=" * 72)


if __name__ == "__main__":
    main()
