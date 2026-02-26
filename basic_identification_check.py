#!/usr/bin/env python3
"""Jacobian-based moment selection for the basic model.

Full implementation of the identification blueprint:

1. Load 9 VFI solutions (baseline + 8 perturbations).
2. Simulate panels with identical random seeds.
3. Compute 17 candidate moments per configuration.
4. Build raw Jacobian via central finite differences.
5. Normalize to elasticities.
6. SVD analysis and collinearity diagnostics.
7. D-optimal moment selection (base 4 + augmentation to 5, 6, 7).
8. Generate comprehensive plots and summary tables.

Reference
---------
``docs/basic_blueprint_identification_check.md``

Usage::

    python basic_identification_check.py [--solve-missing] [--n-capital 3000]
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns
from scipy import stats as sp_stats

from src.econ_models.config.economic_params import EconomicParams
from src.econ_models.config.vfi_config import load_grid_config
from src.econ_models.config.bond_config import BondsConfig
from src.econ_models.io.file_utils import load_json_file, save_json_file, save_boundary_to_json
from src.econ_models.io.artifacts import load_vfi_results, save_vfi_results
from src.econ_models.simulator.vfi.basic import VFISimulator_basic
from src.econ_models.simulator.synthetic_data_gen import synthetic_data_generator
from src.econ_models.moment_calculator.compute_derived_quantities import (
    compute_all_derived_quantities,
)
from src.econ_models.moment_calculator.compute_mean import compute_global_mean
from src.econ_models.moment_calculator.compute_std import compute_global_std
from src.econ_models.moment_calculator.compute_autocorrelation import (
    compute_autocorrelation_lags_1_to_5,
)
from src.econ_models.moment_calculator.compute_inaction_rate import compute_inaction_rate

from basic_common import (
    BASE_DIR,
    BASELINE,
    PARAM_KEYS,
    PARAM_SYMBOLS,
    PERTURBATION_CONFIGS,
    PERTURBATION_MAP,
    STEP_SIZES,
    apply_burn_in,
    build_full_params,
    get_bounds_path,
    get_econ_params_path,
    get_vfi_cache_path,
    param_tag,
    to_python_float,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# =========================================================================
#  17 candidate moment labels (Section 0.3)
# =========================================================================

MOMENT_LABELS: List[str] = [
    "m01: E[I/K]",
    "m02: SD[I/K]",
    "m03: AC[I/K]",
    "m04: Skew[I/K]",
    "m05: P(I/K>0.20)",
    "m06: P(I/K<0)",
    "m07: Inaction rate",
    "m08: E[Y/K]",
    "m09: SD[Y/K]",
    "m10: AC[Y/K]",
    "m11: Corr(I/K,Y/K)",
    "m12: Corr(I/K,lag Y/K)",
    "m13: SD[log K]",
    "m14: AC[log K]",
    "m15: IQR[I/K]",
    "m16: Kurt[I/K]",
    "m17: P(|delta(I/K)|>0.10)",
]
"""Full descriptive labels for the 17 candidate moments."""

MOMENT_SHORT: List[str] = [
    "E[I/K]", "SD[I/K]", "AC[I/K]", "Skew[I/K]",
    "Spike+", "Neg I/K", "Inaction",
    "E[Y/K]", "SD[Y/K]", "AC[Y/K]",
    "Corr(I,Y)", "Corr(I,lagY)",
    "SD[logK]", "AC[logK]",
    "IQR[I/K]", "Kurt[I/K]", "LrgAdj",
]
"""Short-form labels used in tables and plot legends."""


# =========================================================================
#  Simulation and grid settings
# =========================================================================

BATCH_SIZE: int = 5000
T_PERIODS: int = 700
BURN_IN: int = 200
N_CAPITAL: int = 3000
N_PRODUCTIVITY: int = 50

CONFIG_PARAMS_FILE: str = os.path.join(BASE_DIR, "hyperparam/prefixed/vfi_params.json")
RESULTS_DIR: str = "./results/moments_identification_basic"


# =========================================================================
#  Prerequisite helpers (econ params, bounds, VFI solutions)
# =========================================================================

def ensure_econ_params(full_params: Dict[str, float], tag: str) -> EconomicParams:
    """Ensure the economic-parameters JSON exists and return the object.

    Parameters
    ----------
    full_params:
        Complete parameter dictionary.
    tag:
        Filename tag.

    Returns
    -------
    EconomicParams
    """
    path = get_econ_params_path(tag)
    if not os.path.exists(path):
        logger.info("  Creating econ params: %s", path)
        save_json_file(full_params, path)
    return EconomicParams(**load_json_file(path))


def ensure_bounds(econ_params: EconomicParams, tag: str) -> Dict:
    """Ensure bounds JSON exists; run the BoundaryFinder if missing.

    Parameters
    ----------
    econ_params:
        Economic parameters.
    tag:
        Filename tag.

    Returns
    -------
    dict
        Loaded bounds dictionary.
    """
    bp = get_bounds_path(tag)
    if os.path.exists(bp):
        return load_json_file(bp)

    logger.info("  Bounds MISSING for %s - running BoundaryFinder ...", tag)
    from src.econ_models.vfi.bounds import BoundaryFinder

    config = load_grid_config(CONFIG_PARAMS_FILE, "basic")
    config = dataclasses.replace(config, n_capital=200)

    finder = BoundaryFinder(econ_params, config, n_steps=100, n_batches=5000)
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
    return load_json_file(bp)


def ensure_vfi_solution(
    econ_params: EconomicParams,
    bounds: Dict,
    tag: str,
    n_k: int,
    solve_missing: bool,
) -> Dict:
    """Load a cached VFI solution or solve it on the fly.

    Parameters
    ----------
    econ_params:
        Economic parameters.
    bounds:
        Bounds dictionary.
    tag:
        Filename tag.
    n_k:
        Capital grid size.
    solve_missing:
        If ``True``, solve missing solutions; otherwise exit with error.

    Returns
    -------
    dict
        VFI solution arrays.
    """
    cache = get_vfi_cache_path(tag, n_k)
    if os.path.exists(cache):
        logger.info("  Loading VFI: %s", cache)
        return load_vfi_results(cache)

    if not solve_missing:
        logger.error("  VFI solution MISSING: %s", cache)
        logger.error("  Run: python basic_batch_solve_perturbations.py")
        logger.error("  Or use --solve-missing to solve on the fly.")
        sys.exit(1)

    logger.info("  Solving VFI for %s at n_k=%d ...", tag, n_k)
    from src.econ_models.vfi.basic import BasicModelVFI

    config = load_grid_config(CONFIG_PARAMS_FILE, "basic")
    config = dataclasses.replace(config, n_capital=n_k, n_productivity=N_PRODUCTIVITY)
    model = BasicModelVFI(
        econ_params, config,
        k_bounds=(bounds["bounds"]["k_min"], bounds["bounds"]["k_max"]),
    )
    t0 = time.time()
    vfi_results = model.solve()
    logger.info("    Solved in %.1fs", time.time() - t0)

    os.makedirs(os.path.dirname(cache), exist_ok=True)
    save_vfi_results(vfi_results, cache)
    logger.info("    Cached -> %s", cache)
    return vfi_results


# =========================================================================
#  17-moment computation (Section 0.3)
# =========================================================================

def _safe_nanmean(arr: np.ndarray) -> float:
    """NaN-safe mean over a flattened array."""
    flat = np.asarray(arr).flatten()
    valid = flat[np.isfinite(flat)]
    return float(np.mean(valid)) if len(valid) > 0 else 0.0


def _safe_nanstd(arr: np.ndarray) -> float:
    """NaN-safe standard deviation (ddof=1) over a flattened array."""
    flat = np.asarray(arr).flatten()
    valid = flat[np.isfinite(flat)]
    return float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0


def _safe_fraction(arr: np.ndarray, condition_fn) -> float:
    """Fraction of finite elements satisfying *condition_fn*."""
    flat = np.asarray(arr).flatten()
    valid = flat[np.isfinite(flat)]
    if len(valid) == 0:
        return 0.0
    return float(np.sum(condition_fn(valid))) / len(valid)


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation between two arrays (NaN-safe)."""
    x_flat = np.asarray(x).flatten()
    y_flat = np.asarray(y).flatten()
    mask = np.isfinite(x_flat) & np.isfinite(y_flat)
    xv, yv = x_flat[mask], y_flat[mask]
    if len(xv) < 3:
        return 0.0
    r = np.corrcoef(xv, yv)[0, 1]
    return float(np.clip(r, -1.0, 1.0))


def _safe_autocorr(data_2d: np.ndarray, lag: int = 1) -> float:
    """Autocorrelation at *lag*, pooled across the batch dimension."""
    data = np.asarray(data_2d)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] <= lag:
        return 0.0
    return _safe_corr(data[:, lag:], data[:, :-lag])


def compute_17_moments(
    sim_results: Dict[str, np.ndarray],
    delta: float,
    alpha: float,
) -> np.ndarray:
    """Compute all 17 candidate moments from simulation results.

    Parameters
    ----------
    sim_results:
        Post burn-in simulation output.
    delta:
        Depreciation rate.
    alpha:
        Capital share.

    Returns
    -------
    np.ndarray
        17-element array in the order defined by :data:`MOMENT_LABELS`.
    """
    derived = compute_all_derived_quantities(sim_results, delta, alpha)

    capital = derived["capital"]
    inv_rate = derived["investment_rate"]
    output = derived["output"]

    capital_safe = np.maximum(capital, 1e-5)
    revenue_capital = output / capital_safe
    log_capital = np.log(np.maximum(capital, 1e-10))

    flat_ir = np.asarray(inv_rate).flatten()
    valid_ir = flat_ir[np.isfinite(flat_ir)]
    ir_2d = np.asarray(inv_rate)
    rc_2d = np.asarray(revenue_capital)

    moments = np.zeros(17)

    # m01-m03: basic investment-rate moments
    moments[0] = to_python_float(compute_global_mean(inv_rate))
    moments[1] = to_python_float(compute_global_std(inv_rate))
    moments[2] = to_python_float(compute_autocorrelation_lags_1_to_5(inv_rate)["lag_1"])

    # m04: skewness of I/K
    moments[3] = float(sp_stats.skew(valid_ir, nan_policy="omit")) if len(valid_ir) > 2 else 0.0

    # m05-m07: threshold-based rates
    moments[4] = _safe_fraction(inv_rate, lambda x: x > 0.20)
    moments[5] = _safe_fraction(inv_rate, lambda x: x < 0.0)
    moments[6] = to_python_float(compute_inaction_rate(inv_rate, -0.001, 0.001))

    # m08-m10: revenue-to-capital moments
    moments[7] = _safe_nanmean(revenue_capital)
    moments[8] = _safe_nanstd(revenue_capital)
    moments[9] = _safe_autocorr(revenue_capital, lag=1)

    # m11-m12: cross-correlations
    moments[10] = _safe_corr(inv_rate, revenue_capital)
    if ir_2d.ndim == 2 and ir_2d.shape[1] > 1:
        moments[11] = _safe_corr(ir_2d[:, 1:], rc_2d[:, :-1])

    # m13-m14: log-capital moments
    moments[12] = _safe_nanstd(log_capital)
    moments[13] = _safe_autocorr(log_capital, lag=1)

    # m15-m16: distributional shape of I/K
    if len(valid_ir) > 0:
        q75, q25 = np.nanpercentile(valid_ir, [75, 25])
        moments[14] = float(q75 - q25)
    moments[15] = float(sp_stats.kurtosis(valid_ir, nan_policy="omit")) if len(valid_ir) > 3 else 0.0

    # m17: large adjustment fraction
    if ir_2d.ndim == 2 and ir_2d.shape[1] > 1:
        delta_ir = ir_2d[:, 1:] - ir_2d[:, :-1]
        moments[16] = _safe_fraction(delta_ir, lambda x: np.abs(x) > 0.10)

    return moments


# =========================================================================
#  Jacobian construction (Section 2)
# =========================================================================

def build_raw_jacobian(moment_vectors: Dict[str, np.ndarray]) -> np.ndarray:
    """Build the raw Jacobian via central finite differences.

    .. math::

        J_{\\text{raw}}[i, j] = \\frac{m_i(\\theta_j^+) - m_i(\\theta_j^-)}{2 h_j}

    Parameters
    ----------
    moment_vectors:
        ``{config_tag: 17-element moment array}``.

    Returns
    -------
    np.ndarray
        Shape ``(17, 4)`` raw Jacobian.
    """
    n_moments = 17
    n_params = len(PARAM_KEYS)
    j_raw = np.zeros((n_moments, n_params))

    for col, pkey in enumerate(PARAM_KEYS):
        fwd_idx, bwd_idx = PERTURBATION_MAP[pkey]
        fwd_tag = PERTURBATION_CONFIGS[fwd_idx][0]
        bwd_tag = PERTURBATION_CONFIGS[bwd_idx][0]
        h = STEP_SIZES[pkey]
        j_raw[:, col] = (moment_vectors[fwd_tag] - moment_vectors[bwd_tag]) / (2.0 * h)

    return j_raw


def normalize_jacobian(
    j_raw: np.ndarray,
    baseline_moments: np.ndarray,
) -> Tuple[np.ndarray, List[int]]:
    """Normalize the raw Jacobian to elasticities.

    Full elasticity:
    ``J_norm[i,j] = J_raw[i,j] * theta_j^0 / m_i^0``

    Semi-elasticity (when ``|m_i^0| < threshold``):
    ``J_norm[i,j] = J_raw[i,j] * theta_j^0``

    Parameters
    ----------
    j_raw:
        Raw Jacobian, shape ``(17, 4)``.
    baseline_moments:
        Baseline (C0) moment vector.

    Returns
    -------
    tuple
        ``(j_norm, semi_elasticity_rows)`` where *semi_elasticity_rows*
        lists moment indices that used semi-elasticity.
    """
    n_moments, n_params = j_raw.shape
    j_norm = np.zeros_like(j_raw)
    semi_rows: List[int] = []
    baseline_param_values = np.array([BASELINE[k] for k in PARAM_KEYS])

    for i in range(n_moments):
        m0 = baseline_moments[i]
        if abs(m0) > 1e-10:
            for j in range(n_params):
                j_norm[i, j] = j_raw[i, j] * baseline_param_values[j] / m0
        else:
            semi_rows.append(i)
            for j in range(n_params):
                j_norm[i, j] = j_raw[i, j] * baseline_param_values[j]

    return j_norm, semi_rows


# =========================================================================
#  Sanity checks (Section 2.5)
# =========================================================================

def sanity_check_jacobian(j_norm: np.ndarray) -> Dict[str, bool]:
    """Run sanity checks on the normalized Jacobian.

    Checks for NaN/Inf entries, zero rows (insensitive moments), and
    zero columns (unidentified parameters).

    Parameters
    ----------
    j_norm:
        Normalized Jacobian, shape ``(17, 4)``.

    Returns
    -------
    dict
        Boolean pass/fail results keyed by check name.
    """
    checks: Dict[str, bool] = {}

    checks["no_nan_inf"] = bool(np.all(np.isfinite(j_norm)))

    row_maxabs = np.max(np.abs(j_norm), axis=1)
    checks["no_zero_rows"] = bool(np.all(row_maxabs > 1e-8))
    zero_rows = np.where(row_maxabs <= 1e-8)[0]

    col_maxabs = np.max(np.abs(j_norm), axis=0)
    checks["no_zero_cols"] = bool(np.all(col_maxabs > 1e-8))
    zero_cols = np.where(col_maxabs <= 1e-8)[0]

    logger.info("\n-- Sanity Checks on Normalized Jacobian --")
    for name, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        logger.info("  %s  %s", status, name)
    if len(zero_rows) > 0:
        logger.warning("  Zero rows (insensitive moments): %s",
                        [MOMENT_LABELS[i] for i in zero_rows])
    if len(zero_cols) > 0:
        logger.warning("  Zero cols (unidentified params): %s",
                        [PARAM_SYMBOLS[PARAM_KEYS[j]] for j in zero_cols])
    return checks


# =========================================================================
#  SVD analysis (Section 3.1-3.2)
# =========================================================================

def svd_analysis(j_norm: np.ndarray) -> Dict[str, Any]:
    """Perform SVD of the normalized Jacobian and return diagnostics.

    Parameters
    ----------
    j_norm:
        Normalized Jacobian, shape ``(17, 4)``.

    Returns
    -------
    dict
        Keys: ``U``, ``sigma``, ``V``, ``Vt``, ``condition_number``, ``det_JtJ``.
    """
    u_mat, sigma, vt_mat = np.linalg.svd(j_norm, full_matrices=False)
    v_mat = vt_mat.T

    kappa = sigma[0] / sigma[-1] if sigma[-1] > 1e-15 else float("inf")
    det_jtj = float(np.prod(sigma ** 2))

    result = {
        "U": u_mat,
        "sigma": sigma,
        "V": v_mat,
        "Vt": vt_mat,
        "condition_number": kappa,
        "det_JtJ": det_jtj,
    }

    logger.info("\n-- SVD Analysis --")
    logger.info("  Singular values: %s", sigma)
    logger.info("  Condition number kappa = sigma_1/sigma_4 = %.2f", kappa)
    logger.info("  det(J'J) = prod(sigma_k^2) = %.6e", det_jtj)

    if kappa < 10:
        logger.info("  -> Excellent identification (kappa < 10)")
    elif kappa < 50:
        logger.info("  -> Acceptable identification (kappa < 50)")
    else:
        logger.warning("  -> Problematic identification (kappa > 50)")

    logger.info("\n  Right Singular Vectors V:")
    header = f"  {'Parameter':<30s}" + "".join(f"{'V' + str(k + 1):>10s}" for k in range(len(sigma)))
    logger.info(header)
    for j, pkey in enumerate(PARAM_KEYS):
        row = f"  {PARAM_SYMBOLS[pkey] + ' (' + pkey + ')':<30s}"
        row += "".join(f"{v_mat[j, k]:>10.4f}" for k in range(len(sigma)))
        logger.info(row)

    weakest = v_mat[:, -1]
    max_load_idx = int(np.argmax(np.abs(weakest)))
    logger.info("\n  Weakest direction (V4): heaviest loading on %s", PARAM_SYMBOLS[PARAM_KEYS[max_load_idx]])

    return result


# =========================================================================
#  Collinearity diagnostics (Section 3.3)
# =========================================================================

def collinearity_diagnostics(
    j_norm: np.ndarray,
    svd_result: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute condition indices, variance decomposition, and column correlations.

    Parameters
    ----------
    j_norm:
        Normalized Jacobian.
    svd_result:
        Output of :func:`svd_analysis`.

    Returns
    -------
    dict
        Keys: ``condition_indices``, ``variance_decomposition``,
        ``column_correlations``.
    """
    sigma = svd_result["sigma"]
    v_mat = svd_result["V"]
    n_params = len(sigma)

    # Belsley condition indices
    ci = sigma[0] / sigma
    logger.info("\n-- Collinearity Diagnostics --")
    logger.info("  Condition Indices:")
    for k in range(n_params):
        flag = " <- COLLINEARITY" if ci[k] > 30 else ""
        logger.info("    CI_%d = %.2f%s", k + 1, ci[k], flag)

    # Variance decomposition proportions
    phi = np.zeros((n_params, n_params))
    for j in range(n_params):
        denom = np.sum((v_mat[j, :] / sigma) ** 2)
        if denom > 1e-15:
            for k in range(n_params):
                phi[k, j] = (v_mat[j, k] / sigma[k]) ** 2 / denom

    logger.info("\n  Variance Decomposition Proportions:")
    header = f"  {'CI value':<15s}" + "".join(f"{PARAM_SYMBOLS[k]:>10s}" for k in PARAM_KEYS)
    logger.info(header)
    for k in range(n_params):
        row = f"  CI_{k + 1} = {ci[k]:<8.2f}  "
        row += "".join(f"{phi[k, j]:>10.4f}" for j in range(n_params))
        logger.info(row)

    for k in range(n_params):
        if ci[k] > 30:
            high_load = [PARAM_SYMBOLS[PARAM_KEYS[j]] for j in range(n_params) if phi[k, j] > 0.50]
            if len(high_load) >= 2:
                logger.warning("  Collinearity detected at CI_%d: %s", k + 1, high_load)

    # Pairwise column correlations
    corr_matrix = np.corrcoef(j_norm.T)
    logger.info("\n  Pairwise Jacobian Column Correlations:")
    header = f"  {'':>10s}" + "".join(f"{PARAM_SYMBOLS[k]:>10s}" for k in PARAM_KEYS)
    logger.info(header)
    for i in range(n_params):
        row = f"  {PARAM_SYMBOLS[PARAM_KEYS[i]]:>10s}"
        row += "".join(f"{corr_matrix[i, j]:>10.4f}" for j in range(n_params))
        logger.info(row)

    for i in range(n_params):
        for j in range(i + 1, n_params):
            if abs(corr_matrix[i, j]) > 0.90:
                logger.warning(
                    "  High correlation: |Corr(%s, %s)| = %.4f",
                    PARAM_SYMBOLS[PARAM_KEYS[i]], PARAM_SYMBOLS[PARAM_KEYS[j]],
                    abs(corr_matrix[i, j]),
                )

    return {
        "condition_indices": ci,
        "variance_decomposition": phi,
        "column_correlations": corr_matrix,
    }


# =========================================================================
#  D-optimal moment selection (Section 4)
# =========================================================================

def d_optimal_selection(j_norm: np.ndarray) -> Dict[str, Any]:
    """Two-stage D-optimal moment selection.

    Stage 1 finds the best 4-moment subset maximising ``det(J_sub' J_sub)``.
    Stage 2 augments to 5, 6, and 7 moments.

    Parameters
    ----------
    j_norm:
        Normalized Jacobian, shape ``(17, 4)``.

    Returns
    -------
    dict
        Keys: ``base4``, ``base4_det``, ``base4_kappa``, ``top5_subsets``,
        ``augmentation``.
    """
    n_moments, n_params = j_norm.shape
    all_indices = list(range(n_moments))

    # Stage 1: base 4
    logger.info("\n-- D-Optimal: Stage 1 (Base 4) --")
    best_subsets: List[Tuple[float, Tuple[int, ...], float, float]] = []
    for combo in itertools.combinations(all_indices, n_params):
        j_sub = j_norm[list(combo), :]
        det_val = np.linalg.det(j_sub.T @ j_sub)
        if np.isfinite(det_val):
            sv = np.linalg.svd(j_sub, compute_uv=False)
            kappa_sub = sv[0] / sv[-1] if sv[-1] > 1e-15 else float("inf")
            best_subsets.append((det_val, combo, kappa_sub, sv[-1]))

    best_subsets.sort(key=lambda x: -x[0])

    logger.info("  Evaluated %d subsets of size 4", len(best_subsets))
    logger.info("\n  Top 5 subsets:")
    logger.info("  %-6s %-40s %14s %10s %10s", "Rank", "Moments", "det(J'J)", "kappa", "sigma_min")
    for rank, (det_val, combo, kappa_sub, sig_min) in enumerate(best_subsets[:5]):
        moment_str = "{" + ", ".join(MOMENT_SHORT[i] for i in combo) + "}"
        logger.info("  %-6d %-40s %14.6e %10.2f %10.4f", rank + 1, moment_str, det_val, kappa_sub, sig_min)

    _, base4, base4_kappa, base4_sig_min = best_subsets[0]
    base4 = list(base4)
    base4_det = best_subsets[0][0]

    logger.info("\n  Selected base 4: %s", [MOMENT_LABELS[i] for i in base4])
    logger.info("  det(J'J) = %.6e, kappa = %.2f, sigma_min = %.4f", base4_det, base4_kappa, base4_sig_min)

    # Moment-parameter mapping
    logger.info("\n  Moment-Parameter Mapping:")
    j_base = j_norm[base4, :]
    header = f"  {'Moment':<25s} {'Primary':>10s} {'|Elast|':>10s}" + "".join(
        f"{PARAM_SYMBOLS[k]:>10s}" for k in PARAM_KEYS
    )
    logger.info(header)
    for idx, mi in enumerate(base4):
        row_abs = np.abs(j_base[idx, :])
        primary_j = int(np.argmax(row_abs))
        row_str = f"  {MOMENT_LABELS[mi]:<25s} {PARAM_SYMBOLS[PARAM_KEYS[primary_j]]:>10s} {row_abs[primary_j]:>10.4f}"
        row_str += "".join(f"{j_base[idx, j]:>10.4f}" for j in range(len(PARAM_KEYS)))
        logger.info(row_str)

    # Stage 2: augmentation to 5, 6, 7
    logger.info("\n-- D-Optimal: Stage 2 (Augmentation) --")
    remaining = [i for i in all_indices if i not in base4]
    augmentation_results: Dict[int, Dict[str, Any]] = {}

    for k_target in [5, 6, 7]:
        n_add = k_target - 4
        candidates: List[Tuple[float, Tuple[int, ...], float, float]] = []
        for combo in itertools.combinations(remaining, n_add):
            subset = base4 + list(combo)
            j_sub = j_norm[subset, :]
            det_val = np.linalg.det(j_sub.T @ j_sub)
            if np.isfinite(det_val):
                sv = np.linalg.svd(j_sub, compute_uv=False)
                kappa_sub = sv[0] / sv[-1] if sv[-1] > 1e-15 else float("inf")
                candidates.append((det_val, combo, kappa_sub, sv[-1]))
        candidates.sort(key=lambda x: -x[0])

        best = candidates[0] if candidates else None
        augmentation_results[k_target] = {
            "n_evaluated": len(candidates),
            "best": best,
            "top3": candidates[:3],
        }

        logger.info("\n  k = %d (add %d moment%s):", k_target, n_add, "s" if n_add > 1 else "")
        logger.info("    Evaluated: %d candidates", len(candidates))
        if best:
            det_val, added, kappa_sub, sig_min = best
            logger.info("    Best addition: %s", [MOMENT_LABELS[i] for i in added])
            logger.info("    det(J'J) = %.6e, kappa = %.2f, sigma_min = %.4f", det_val, kappa_sub, sig_min)

    # Marginal value analysis
    logger.info("\n-- Marginal Value of Additional Moments --")
    logger.info("  %-5s %14s %10s %10s %10s %10s", "k", "det(J'J)", "kappa", "sigma_min", "d_sigma_min", "Overid df")

    prev_sig_min = base4_sig_min
    sv_base4 = np.linalg.svd(j_norm[base4, :], compute_uv=False)
    logger.info("  %-5d %14.6e %10.2f %10.4f %10s %10d", 4, base4_det, base4_kappa, sv_base4[-1], "-", 0)

    for k_target in [5, 6, 7]:
        res = augmentation_results[k_target]
        if res["best"]:
            det_val, added, kappa_sub, sig_min = res["best"]
            delta_sig = sig_min - prev_sig_min
            logger.info("  %-5d %14.6e %10.2f %10.4f %+10.4f %10d", k_target, det_val, kappa_sub, sig_min, delta_sig, k_target - 4)
            prev_sig_min = sig_min

    # Final summary
    logger.info("\n%s\nFINAL MOMENT SELECTION SUMMARY\n%s", "=" * 70, "=" * 70)
    logger.info("\n  Base Set (4 moments - exact identification):")
    j_base = j_norm[base4, :]
    for idx, mi in enumerate(base4):
        row_abs = np.abs(j_base[idx, :])
        primary_j = int(np.argmax(row_abs))
        logger.info("    %s -> identifies %s", MOMENT_LABELS[mi], PARAM_SYMBOLS[PARAM_KEYS[primary_j]])

    logger.info("\n  Candidate Additions:")
    for priority, k_target in enumerate([5, 6, 7], 1):
        res = augmentation_results[k_target]
        if res["best"]:
            _, added, _, _ = res["best"]
            for mi in added:
                logger.info("    +%d: %s -> recommended k = %d", priority, MOMENT_LABELS[mi], k_target)

    return {
        "base4": base4,
        "base4_det": base4_det,
        "base4_kappa": base4_kappa,
        "top5_subsets": best_subsets[:5],
        "augmentation": augmentation_results,
    }


# =========================================================================
#  Plotting
# =========================================================================

def plot_jacobian_heatmap(j_norm: np.ndarray, save_dir: str) -> None:
    """Plot the normalized Jacobian as a heatmap.

    Parameters
    ----------
    j_norm:
        Normalized Jacobian.
    save_dir:
        Output directory for the figure.
    """
    fig, ax = plt.subplots(figsize=(8, 12))
    sns.heatmap(
        j_norm, annot=True, fmt=".3f",
        xticklabels=[PARAM_SYMBOLS[k] for k in PARAM_KEYS],
        yticklabels=MOMENT_LABELS,
        cmap="RdBu_r", center=0, ax=ax, linewidths=0.5,
    )
    ax.set_title("Normalized Jacobian (Elasticities)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Parameters", fontsize=12)
    ax.set_ylabel("Moments", fontsize=12)
    plt.tight_layout()
    path = os.path.join(save_dir, "jacobian_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info("  Saved: %s", path)


def plot_svd_diagnostics(svd_result: Dict[str, Any], save_dir: str) -> None:
    """Plot singular values and right singular vectors.

    Parameters
    ----------
    svd_result:
        Output of :func:`svd_analysis`.
    save_dir:
        Output directory.
    """
    sigma = svd_result["sigma"]
    v_mat = svd_result["V"]
    n_params = len(sigma)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(range(1, n_params + 1), sigma, color="steelblue", edgecolor="black")
    axes[0].set_xlabel("Singular Value Index")
    axes[0].set_ylabel("sigma")
    axes[0].set_title("Singular Values of J_norm")
    axes[0].set_xticks(range(1, n_params + 1))
    axes[0].grid(axis="y", alpha=0.3)

    sns.heatmap(
        v_mat, annot=True, fmt=".3f",
        xticklabels=[f"V{k + 1}" for k in range(n_params)],
        yticklabels=[PARAM_SYMBOLS[k] for k in PARAM_KEYS],
        cmap="RdBu_r", center=0, ax=axes[1], linewidths=0.5,
    )
    axes[1].set_title("Right Singular Vectors V")

    plt.tight_layout()
    path = os.path.join(save_dir, "svd_diagnostics.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info("  Saved: %s", path)


def plot_column_correlations(corr_matrix: np.ndarray, save_dir: str) -> None:
    """Plot the pairwise column-correlation heatmap.

    Parameters
    ----------
    corr_matrix:
        ``(4, 4)`` correlation matrix.
    save_dir:
        Output directory.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    labels = [PARAM_SYMBOLS[k] for k in PARAM_KEYS]
    sns.heatmap(
        corr_matrix, annot=True, fmt=".3f",
        xticklabels=labels, yticklabels=labels,
        cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax, linewidths=0.5,
    )
    ax.set_title("Jacobian Column Correlations", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(save_dir, "column_correlations.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info("  Saved: %s", path)


def plot_moment_sensitivity_bars(
    moment_vectors: Dict[str, np.ndarray],
    save_dir: str,
) -> None:
    """Bar chart of moment values across the nine configurations.

    Parameters
    ----------
    moment_vectors:
        ``{config_tag: 17-element array}``.
    save_dir:
        Output directory.
    """
    tags = [cfg[0] for cfg in PERTURBATION_CONFIGS]
    n_moments = 17
    n_cols = 3
    n_rows = (n_moments + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    colors = sns.color_palette("Set2", len(tags))

    for idx in range(n_moments):
        ax = axes.flat[idx]
        values = [moment_vectors[tag][idx] for tag in tags]
        ax.bar(range(len(tags)), values, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_title(MOMENT_LABELS[idx], fontsize=10, fontweight="bold")
        ax.set_xticks(range(len(tags)))
        ax.set_xticklabels([t[:6] for t in tags], fontsize=7, rotation=45, ha="right")
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for idx in range(n_moments, len(axes.flat)):
        axes.flat[idx].set_visible(False)

    fig.suptitle("Moment Values Across 9 Perturbation Configurations",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(save_dir, "moment_sensitivity_bars.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info("  Saved: %s", path)


def plot_d_optimal_comparison(
    j_norm: np.ndarray,
    selection_result: Dict[str, Any],
    save_dir: str,
) -> None:
    """Plot D-optimal diagnostics: det(J'J), sigma_min, kappa vs. k.

    Parameters
    ----------
    j_norm:
        Normalized Jacobian.
    selection_result:
        Output of :func:`d_optimal_selection`.
    save_dir:
        Output directory.
    """
    base4 = selection_result["base4"]
    aug = selection_result["augmentation"]

    sizes = [4]
    dets = [selection_result["base4_det"]]
    kappas = [selection_result["base4_kappa"]]
    sig_mins = [np.linalg.svd(j_norm[base4, :], compute_uv=False)[-1]]

    for k in [5, 6, 7]:
        if aug[k]["best"]:
            d, added, kap, sm = aug[k]["best"]
            sizes.append(k)
            dets.append(d)
            kappas.append(kap)
            sig_mins.append(sm)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].plot(sizes, dets, "o-", color="steelblue", linewidth=2, markersize=8)
    axes[0].set_xlabel("Number of Moments")
    axes[0].set_ylabel("det(J'J)")
    axes[0].set_title("Information Volume")
    axes[0].set_yscale("log")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(sizes, sig_mins, "s-", color="darkorange", linewidth=2, markersize=8)
    axes[1].set_xlabel("Number of Moments")
    axes[1].set_ylabel("sigma_min")
    axes[1].set_title("Weakest Identification Direction")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(sizes, kappas, "^-", color="forestgreen", linewidth=2, markersize=8)
    axes[2].set_xlabel("Number of Moments")
    axes[2].set_ylabel("Condition Number kappa")
    axes[2].set_title("Identification Balance")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "d_optimal_progression.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info("  Saved: %s", path)


# =========================================================================
#  Save results
# =========================================================================

def save_results(
    j_raw: np.ndarray,
    j_norm: np.ndarray,
    baseline_moments: np.ndarray,
    moment_vectors: Dict[str, np.ndarray],
    svd_result: Dict[str, Any],
    collin_result: Dict[str, Any],
    selection_result: Dict[str, Any],
) -> None:
    """Persist numerical results and a human-readable summary.

    Parameters
    ----------
    j_raw:
        Raw Jacobian.
    j_norm:
        Normalized Jacobian.
    baseline_moments:
        Baseline moment vector.
    moment_vectors:
        All config moment vectors.
    svd_result:
        SVD analysis output.
    collin_result:
        Collinearity diagnostics output.
    selection_result:
        D-optimal selection output.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    np.savez(
        os.path.join(RESULTS_DIR, "jacobian_results.npz"),
        J_raw=j_raw,
        J_norm=j_norm,
        baseline_moments=baseline_moments,
        singular_values=svd_result["sigma"],
        V_matrix=svd_result["V"],
        column_correlations=collin_result["column_correlations"],
        condition_indices=collin_result["condition_indices"],
        variance_decomposition=collin_result["variance_decomposition"],
        base4_indices=np.array(selection_result["base4"]),
    )

    np.savez(
        os.path.join(RESULTS_DIR, "all_moment_vectors.npz"),
        **{tag: moment_vectors[tag] for tag in moment_vectors},
    )

    summary_path = os.path.join(RESULTS_DIR, "identification_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Jacobian-Based Moment Selection - Basic Case\n")
        f.write("=" * 70 + "\n\n")

        f.write("Baseline Parameters:\n")
        for k in PARAM_KEYS:
            f.write(f"  {PARAM_SYMBOLS[k]} ({k}): {BASELINE[k]}\n")
        f.write(f"\nStep sizes: {STEP_SIZES}\n")

        f.write(f"\n\nSVD Results:\n")
        f.write(f"  Singular values: {svd_result['sigma']}\n")
        f.write(f"  Condition number: {svd_result['condition_number']:.4f}\n")
        f.write(f"  det(J'J): {svd_result['det_JtJ']:.6e}\n")

        f.write(f"\n\nSelected Base 4 Moments:\n")
        j_base = j_norm[selection_result["base4"], :]
        for idx, mi in enumerate(selection_result["base4"]):
            row_abs = np.abs(j_base[idx, :])
            primary_j = int(np.argmax(row_abs))
            f.write(f"  {MOMENT_LABELS[mi]:30s} -> {PARAM_SYMBOLS[PARAM_KEYS[primary_j]]}\n")
        f.write(f"\n  det(J'J) = {selection_result['base4_det']:.6e}\n")
        f.write(f"  Condition number = {selection_result['base4_kappa']:.4f}\n")

        f.write(f"\n\nAugmentation Results:\n")
        for k_target in [5, 6, 7]:
            res = selection_result["augmentation"][k_target]
            if res["best"]:
                _, added, kap, sm = res["best"]
                f.write(f"  k={k_target}: add {[MOMENT_LABELS[i] for i in added]}, "
                        f"kappa={kap:.2f}, sigma_min={sm:.4f}\n")

        f.write(f"\n\nColumn Correlations:\n")
        labels = [PARAM_SYMBOLS[k] for k in PARAM_KEYS]
        f.write(f"  {'':>8s}" + "".join(f"{l:>10s}" for l in labels) + "\n")
        for i in range(4):
            f.write(f"  {labels[i]:>8s}")
            for j in range(4):
                f.write(f"{collin_result['column_correlations'][i, j]:>10.4f}")
            f.write("\n")

        f.write(f"\n\nBaseline Moments (17 candidates):\n")
        for j, label in enumerate(MOMENT_LABELS):
            f.write(f"  {label:30s} = {baseline_moments[j]:.6f}\n")

    logger.info("  Saved: %s", summary_path)


# =========================================================================
#  Main
# =========================================================================

def main() -> None:
    """Execute the full Jacobian-based identification analysis pipeline."""
    parser = argparse.ArgumentParser(description="Jacobian-based identification check (basic model)")
    parser.add_argument("--solve-missing", action="store_true", help="Solve missing VFI solutions on the fly")
    parser.add_argument("--n-capital", type=int, default=N_CAPITAL,
                        help=f"Capital grid size for VFI (default {N_CAPITAL})")
    args = parser.parse_args()
    n_k = args.n_capital

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    logger.info("=" * 70)
    logger.info("Jacobian-Based Moment Selection - Basic Case")
    logger.info("=" * 70)
    logger.info("  Baseline: rho=%.3f, sigma=%.3f, xi=%.3f, F=%.4f",
                BASELINE["productivity_persistence"], BASELINE["productivity_std_dev"],
                BASELINE["adjustment_cost_convex"], BASELINE["adjustment_cost_fixed"])
    logger.info("  Grid: n_k=%d, n_z=%d", n_k, N_PRODUCTIVITY)
    logger.info("  Simulation: batch=%d, T=%d, burn-in=%d", BATCH_SIZE, T_PERIODS, BURN_IN)

    # Step 1: Ensure all 9 VFI solutions
    logger.info("\n%s\nSTEP 1: Ensure VFI solutions for all 9 configurations\n%s", "=" * 70, "=" * 70)
    config_data: Dict[str, Dict[str, Any]] = {}
    for i, (cfg_tag, overrides) in enumerate(PERTURBATION_CONFIGS):
        full_params = build_full_params(overrides)
        tag = param_tag(full_params)
        logger.info("\n[C%d] %s (tag=%s)", i, cfg_tag, tag)
        econ_params = ensure_econ_params(full_params, tag)
        bounds = ensure_bounds(econ_params, tag)
        vfi_results = ensure_vfi_solution(econ_params, bounds, tag, n_k, args.solve_missing)
        config_data[cfg_tag] = {
            "econ_params": econ_params,
            "bounds": bounds,
            "vfi_results": vfi_results,
            "tag": tag,
            "full_params": full_params,
        }

    # Step 2: Generate shared synthetic data
    logger.info("\n%s\nSTEP 2: Generate shared synthetic data\n%s", "=" * 70, "=" * 70)
    baseline_data = config_data["baseline"]
    baseline_econ = baseline_data["econ_params"]
    baseline_tag = baseline_data["tag"]

    sample_bonds_config = BondsConfig.validate_and_load(
        bounds_file=get_bounds_path(baseline_tag),
        current_params=baseline_econ,
    )
    data_gen = synthetic_data_generator(
        econ_params_benchmark=baseline_econ,
        sample_bonds_config=sample_bonds_config,
        batch_size=BATCH_SIZE,
        T_periods=T_PERIODS,
    )
    initial_states, shock_sequence = data_gen.gen()
    logger.info("  Generated: %d firms x %d periods", BATCH_SIZE, T_PERIODS)

    # Step 3: Simulate and compute 17 moments per config
    logger.info("\n%s\nSTEP 3: Simulate and compute 17 candidate moments\n%s", "=" * 70, "=" * 70)
    moment_vectors: Dict[str, np.ndarray] = {}
    delta = baseline_econ.depreciation_rate
    alpha = baseline_econ.capital_share

    for i, (cfg_tag, _) in enumerate(PERTURBATION_CONFIGS):
        cd = config_data[cfg_tag]
        logger.info("\n[C%d] %s", i, cfg_tag)

        simulator = VFISimulator_basic(cd["econ_params"])
        simulator.load_solved_vfi_solution(cd["vfi_results"])
        sim_results = simulator.simulate(
            tuple(s.numpy() for s in initial_states),
            shock_sequence.numpy(),
        )
        sim_stationary = apply_burn_in(sim_results, BURN_IN)
        moments = compute_17_moments(sim_stationary, delta, alpha)
        moment_vectors[cfg_tag] = moments

        logger.info("  Moments computed (%d values)", len(moments))
        for j, label in enumerate(MOMENT_LABELS):
            logger.info("    %s = %.6f", label, moments[j])

    # Step 4: Build raw Jacobian
    logger.info("\n%s\nSTEP 4: Build raw Jacobian\n%s", "=" * 70, "=" * 70)
    j_raw = build_raw_jacobian(moment_vectors)
    _log_matrix("Raw Jacobian J_raw (17 x 4)", j_raw, "%.6f")

    # Step 5: Normalize to elasticities
    logger.info("\n%s\nSTEP 5: Normalize Jacobian to elasticities\n%s", "=" * 70, "=" * 70)
    baseline_moments = moment_vectors["baseline"]
    j_norm, semi_rows = normalize_jacobian(j_raw, baseline_moments)
    if semi_rows:
        logger.info("  Semi-elasticity used for moments: %s", [MOMENT_LABELS[i] for i in semi_rows])
    _log_matrix("Normalized Jacobian J_norm", j_norm, "%.4f")

    # Step 6: Sanity checks
    logger.info("\n%s\nSTEP 6: Sanity checks\n%s", "=" * 70, "=" * 70)
    sanity_check_jacobian(j_norm)

    # Step 7: SVD analysis
    logger.info("\n%s\nSTEP 7: SVD analysis\n%s", "=" * 70, "=" * 70)
    svd_result = svd_analysis(j_norm)

    # Step 8: Collinearity diagnostics
    logger.info("\n%s\nSTEP 8: Collinearity diagnostics\n%s", "=" * 70, "=" * 70)
    collin_result = collinearity_diagnostics(j_norm, svd_result)

    # Step 9: D-optimal moment selection
    logger.info("\n%s\nSTEP 9: D-optimal moment selection\n%s", "=" * 70, "=" * 70)
    selection_result = d_optimal_selection(j_norm)

    # Step 10: Generate plots
    logger.info("\n%s\nSTEP 10: Generate plots\n%s", "=" * 70, "=" * 70)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plot_jacobian_heatmap(j_norm, RESULTS_DIR)
    plot_svd_diagnostics(svd_result, RESULTS_DIR)
    plot_column_correlations(collin_result["column_correlations"], RESULTS_DIR)
    plot_moment_sensitivity_bars(moment_vectors, RESULTS_DIR)
    plot_d_optimal_comparison(j_norm, selection_result, RESULTS_DIR)

    # Step 11: Save results
    logger.info("\n%s\nSTEP 11: Save numerical results\n%s", "=" * 70, "=" * 70)
    save_results(j_raw, j_norm, baseline_moments, moment_vectors, svd_result, collin_result, selection_result)

    # Final console summary
    sigma = svd_result["sigma"]
    logger.info("\n%s\nIDENTIFICATION ANALYSIS COMPLETE\n%s", "=" * 70, "=" * 70)
    logger.info("  Singular values: [%s]", ", ".join(f"{s:.4f}" for s in sigma))
    logger.info("  Condition number: %.2f", svd_result["condition_number"])
    logger.info("  sigma_min: %.4f", sigma[-1])
    logger.info("  Results saved to: %s", os.path.abspath(RESULTS_DIR))

    return {
        "J_raw": j_raw,
        "J_norm": j_norm,
        "baseline_moments": baseline_moments,
        "moment_vectors": moment_vectors,
        "svd": svd_result,
        "collinearity": collin_result,
        "selection": selection_result,
    }


def _log_matrix(title: str, matrix: np.ndarray, fmt: str) -> None:
    """Log a Jacobian-style matrix with row and column labels.

    Parameters
    ----------
    title:
        Section title.
    matrix:
        ``(17, 4)`` matrix.
    fmt:
        Format string for each cell.
    """
    logger.info("\n  %s:", title)
    header = f"  {'Moment':<30s}" + "".join(f"{PARAM_SYMBOLS[k]:>12s}" for k in PARAM_KEYS)
    logger.info(header)
    logger.info("  %s", "-" * 78)
    for i in range(matrix.shape[0]):
        row = f"  {MOMENT_LABELS[i]:<30s}"
        row += "".join(f"{matrix[i, j]:>12{fmt[-1]}}" for j in range(matrix.shape[1]))
        logger.info(row)


if __name__ == "__main__":
    main()
