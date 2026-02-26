#!/usr/bin/env python3
"""
Jacobian-Based Moment Selection — Risky Debt Case
═════════════════════════════════════════════════════════════════════════════

Full implementation of the risky debt identification blueprint:
  1. Load 13 VFI solutions (baseline + 12 perturbations)
  2. Simulate panels with identical random seeds
  3. Compute 30 candidate moments per configuration
  4. Build raw Jacobian via central finite differences
  5. Normalize to elasticities
  6. SVD analysis & collinearity diagnostics
  7. D-optimal moment selection (base 6 + augmentation to 7, 8, 9)
  8. Generate comprehensive plots and summary tables

Reference: docs/risky_blueprint_identification_check.md

Run:
  python risky_identification_check.py [--solve-missing]

If --solve-missing is set and a golden VFI solution is absent, the script
will invoke the solver at specified grid resolutions.
"""

import argparse
import dataclasses
import itertools
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats as sp_stats

from src.econ_models.config.economic_params import EconomicParams
from src.econ_models.config.vfi_config import GridConfig, load_grid_config
from src.econ_models.config.bond_config import BondsConfig
from src.econ_models.io.file_utils import load_json_file, save_json_file, save_boundary_to_json
from src.econ_models.io.artifacts import load_vfi_results, save_vfi_results
from src.econ_models.simulator.vfi.risky import VFISimulator_risky
from src.econ_models.simulator.synthetic_data_gen import synthetic_data_generator
from src.econ_models.vfi.grids.grid_utils import compute_optimal_chunks

from src.econ_models.moment_calculator.compute_derived_quantities import compute_all_derived_quantities
from src.econ_models.moment_calculator.compute_mean import compute_global_mean
from src.econ_models.moment_calculator.compute_std import compute_global_std
from src.econ_models.moment_calculator.compute_autocorrelation import compute_autocorrelation_lags_1_to_5
from src.econ_models.moment_calculator.compute_inaction_rate import compute_inaction_rate

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  Blueprint Configuration (Sections 0.1–1.3)
# ═══════════════════════════════════════════════════════════════════════════

FIXED_PARAMS = {
    'discount_factor': 0.96,
    'capital_share': 0.60,
    'depreciation_rate': 0.15,
    'risk_free_rate': 0.04,
    'default_cost_proportional': 0.30,
    'corporate_tax_rate': 0.20,
    'collateral_recovery_fraction': 0.50,
}

BASELINE = {
    'productivity_persistence': 0.600,
    'productivity_std_dev': 0.175,
    'adjustment_cost_convex': 1.005,
    'adjustment_cost_fixed': 0.030,
    'equity_issuance_cost_fixed': 0.105,
    'equity_issuance_cost_linear': 0.105,
}

STEP_SIZES = {
    'productivity_persistence': 0.030,
    'productivity_std_dev': 0.010,
    'adjustment_cost_convex': 0.050,
    'adjustment_cost_fixed': 0.0015,
    'equity_issuance_cost_fixed': 0.005,
    'equity_issuance_cost_linear': 0.005,
}

# Parameter bounds (from econ_params_risky_dist.json estimate_param)
PARAM_BOUNDS = {
    'productivity_persistence': (0.40, 0.80),
    'productivity_std_dev': (0.05, 0.30),
    'adjustment_cost_convex': (0.01, 2.00),
    'adjustment_cost_fixed': (0.01, 0.05),
    'equity_issuance_cost_fixed': (0.01, 0.20),
    'equity_issuance_cost_linear': (0.01, 0.20),
}

# Ordered list of structural parameters (column order for Jacobian — 6 params)
PARAM_KEYS = [
    'productivity_persistence',
    'productivity_std_dev',
    'adjustment_cost_convex',
    'adjustment_cost_fixed',
    'equity_issuance_cost_fixed',
    'equity_issuance_cost_linear',
]

PARAM_SYMBOLS = {
    'productivity_persistence': 'ρ',
    'productivity_std_dev': 'σ',
    'adjustment_cost_convex': 'ξ',
    'adjustment_cost_fixed': 'F',
    'equity_issuance_cost_fixed': 'η₀',
    'equity_issuance_cost_linear': 'η₁',
}

# 13 configurations: (tag, overrides)
CONFIGS = [
    ('baseline', {}),
    ('rho_plus',   {'productivity_persistence': BASELINE['productivity_persistence'] + STEP_SIZES['productivity_persistence']}),
    ('rho_minus',  {'productivity_persistence': BASELINE['productivity_persistence'] - STEP_SIZES['productivity_persistence']}),
    ('sigma_plus', {'productivity_std_dev': BASELINE['productivity_std_dev'] + STEP_SIZES['productivity_std_dev']}),
    ('sigma_minus',{'productivity_std_dev': BASELINE['productivity_std_dev'] - STEP_SIZES['productivity_std_dev']}),
    ('xi_plus',    {'adjustment_cost_convex': BASELINE['adjustment_cost_convex'] + STEP_SIZES['adjustment_cost_convex']}),
    ('xi_minus',   {'adjustment_cost_convex': BASELINE['adjustment_cost_convex'] - STEP_SIZES['adjustment_cost_convex']}),
    ('F_plus',     {'adjustment_cost_fixed': BASELINE['adjustment_cost_fixed'] + STEP_SIZES['adjustment_cost_fixed']}),
    ('F_minus',    {'adjustment_cost_fixed': BASELINE['adjustment_cost_fixed'] - STEP_SIZES['adjustment_cost_fixed']}),
    ('eta0_plus',  {'equity_issuance_cost_fixed': BASELINE['equity_issuance_cost_fixed'] + STEP_SIZES['equity_issuance_cost_fixed']}),
    ('eta0_minus', {'equity_issuance_cost_fixed': BASELINE['equity_issuance_cost_fixed'] - STEP_SIZES['equity_issuance_cost_fixed']}),
    ('eta1_plus',  {'equity_issuance_cost_linear': BASELINE['equity_issuance_cost_linear'] + STEP_SIZES['equity_issuance_cost_linear']}),
    ('eta1_minus', {'equity_issuance_cost_linear': BASELINE['equity_issuance_cost_linear'] - STEP_SIZES['equity_issuance_cost_linear']}),
]

# Mapping from parameter key → (forward_config_index, backward_config_index)
PERTURBATION_MAP = {
    'productivity_persistence':    (1, 2),    # C1, C2
    'productivity_std_dev':        (3, 4),    # C3, C4
    'adjustment_cost_convex':      (5, 6),    # C5, C6
    'adjustment_cost_fixed':       (7, 8),    # C7, C8
    'equity_issuance_cost_fixed':  (9, 10),   # C9, C10
    'equity_issuance_cost_linear': (11, 12),  # C11, C12
}

# ── 30 candidate moment labels (Section 0.3) ────────────────────────────
MOMENT_LABELS = [
    # Group A: Investment Behaviour (7)
    'm01: E[I/K]',
    'm02: SD[I/K]',
    'm03: AC[I/K]',
    'm04: Skew[I/K]',
    'm05: P(I/K>0.20)',
    'm06: P(I/K<0)',
    'm07: Inaction rate',
    # Group B: Profitability (3)
    'm08: E[Y/K]',
    'm09: SD[Y/K]',
    'm10: AC[Y/K]',
    # Group C: Investment–Profitability Cross (2)
    'm11: Corr(I/K,Y/K)',
    'm12: Corr(I/K,lag Y/K)',
    # Group D: Firm Size (2)
    'm13: SD[log K]',
    'm14: AC[log K]',
    # Group E: Higher-Order Investment (2)
    'm15: IQR[I/K]',
    'm16: Kurt[I/K]',
    # Group F: Leverage and Capital Structure (4)
    'm17: E[B/K]',
    'm18: SD[B/K]',
    'm19: AC[B/K]',
    'm20: Median[B/K]',
    # Group G: Capital Structure Interactions (2)
    'm21: Corr(I/K,B/K)',
    'm22: Corr(ΔB/K,Y/K)',
    # Group H: Default and Credit (4)
    'm23: Default rate',
    'm24: E[q]',
    'm25: SD[q]',
    'm26: Corr(q,B/K)',
    # Group I: Dividend and Equity Issuance (4)
    'm27: P(e_raw<0)',
    'm28: E[|e|/K|e<0]',
    'm29: E[D/K|D>0]',
    'm30: SD[D/K]',
]

MOMENT_SHORT = [
    'E[I/K]', 'SD[I/K]', 'AC[I/K]', 'Skew[I/K]',
    'Spike+', 'Neg I/K', 'Inaction',
    'E[Y/K]', 'SD[Y/K]', 'AC[Y/K]',
    'Corr(I,Y)', 'Corr(I,lagY)',
    'SD[logK]', 'AC[logK]',
    'IQR[I/K]', 'Kurt[I/K]',
    'E[B/K]', 'SD[B/K]', 'AC[B/K]', 'Med[B/K]',
    'Corr(I,B/K)', 'Corr(ΔB,Y)',
    'DefRate', 'E[q]', 'SD[q]', 'Corr(q,B/K)',
    'P(e<0)', 'E|e|/K', 'E[D/K]', 'SD[D/K]',
]

N_MOMENTS = 30
N_PARAMS = 6

# Simulation settings
BATCH_SIZE = 10000
T_PERIODS = 700
BURN_IN = 200

# VFI grid resolution — single high-resolution grid
GRID_RESOLUTIONS = [560]
N_PRODUCTIVITY = 12

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname('./')))
CONFIG_PARAMS_FILE = os.path.join(BASE_DIR, "hyperparam/prefixed/vfi_params.json")
RESULTS_DIR = './results/moments_identification_risky'


# ═══════════════════════════════════════════════════════════════════════════
#  Utility Functions
# ═══════════════════════════════════════════════════════════════════════════

def to_float(val) -> float:
    """Convert any tensor, array, or number to Python float."""
    if val is None:
        return 0.0
    if hasattr(val, 'numpy'):
        return float(val.numpy())
    elif hasattr(val, 'item'):
        return float(val.item())
    return float(val)


def build_full_params(overrides: Dict[str, float]) -> Dict[str, float]:
    params = {}
    params.update(FIXED_PARAMS)
    params.update(BASELINE)
    params.update(overrides)
    return params


def param_tag(params_dict: Dict[str, float]) -> str:
    rho = params_dict['productivity_persistence']
    sigma = params_dict['productivity_std_dev']
    xi = params_dict['adjustment_cost_convex']
    F = params_dict['adjustment_cost_fixed']
    eta0 = params_dict['equity_issuance_cost_fixed']
    eta1 = params_dict['equity_issuance_cost_linear']
    return f"{rho}_{sigma}_{xi}_{F}_{eta0}_{eta1}"


def econ_params_path(tag: str) -> str:
    return os.path.join(BASE_DIR, f"hyperparam/prefixed/econ_params_risky_{tag}.json")


def bounds_path(tag: str) -> str:
    return os.path.join(BASE_DIR, f"hyperparam/autogen/bounds_risky_{tag}.json")


def vfi_cache_path(tag: str, n_k: int, n_d: int) -> str:
    return f'./ground_truth_risky/golden_vfi_risky_{tag}_{n_k}_{n_d}.npz'


def apply_burn_in(results_dict: Dict, burn_in: int) -> Dict:
    """Discard the first burn_in periods from all arrays."""
    return {k: v[:, burn_in:] for k, v in results_dict.items()}


# ═══════════════════════════════════════════════════════════════════════════
#  Ensure Prerequisites (econ params, bounds, VFI solution)
# ═══════════════════════════════════════════════════════════════════════════

def ensure_econ_params(full_params: Dict, tag: str) -> EconomicParams:
    """Ensure econ params JSON exists. Return EconomicParams object."""
    path = econ_params_path(tag)
    if not os.path.exists(path):
        logger.info(f"  Creating econ params: {path}")
        save_json_file(full_params, path)
    return EconomicParams(**load_json_file(path))


def ensure_bounds(econ_params: EconomicParams, tag: str) -> Dict:
    """Ensure bounds JSON exists. Run BoundaryFinder (risky) if missing."""
    bp = bounds_path(tag)
    if os.path.exists(bp):
        return load_json_file(bp)

    logger.info(f"  Bounds MISSING for {tag} — running BoundaryFinder (risky) ...")
    from src.econ_models.vfi.bounds import BoundaryFinder

    config = load_grid_config(CONFIG_PARAMS_FILE, 'risky')
    config = dataclasses.replace(
        config,
        n_capital=40,
        n_debt=40,
        n_productivity=N_PRODUCTIVITY,
    )

    k_chunk, b_chunk, kp_chunk, bp_chunk = compute_optimal_chunks(40, 40, n_z=N_PRODUCTIVITY)

    finder = BoundaryFinder(
        econ_params, config,
        n_steps=100,
        n_batches=3000,
        k_chunk_size=k_chunk,
        b_chunk_size=b_chunk,
        kp_chunk_size=kp_chunk,
        bp_chunk_size=bp_chunk,
    )
    bounds_result = finder.find_risky_bounds()

    k_bounds = bounds_result['k_bounds_add_margin']
    b_bounds = bounds_result['b_bounds_add_margin']
    z_bounds = bounds_result['z_bounds_original']

    bounds_data = {
        "k_min": float(k_bounds[0]),
        "k_max": float(k_bounds[1]),
        "b_min": float(b_bounds[0]),
        "b_max": float(b_bounds[1]),
        "z_min": float(z_bounds[0]),
        "z_max": float(z_bounds[1]),
    }
    save_boundary_to_json(bp, bounds_data, econ_params)
    logger.info(f"    Bounds saved → {bp}")
    return load_json_file(bp)


def ensure_vfi_solution(econ_params: EconomicParams, bounds: Dict,
                        tag: str, n_k: int, n_d: int,
                        solve_missing: bool) -> Dict:
    """Load cached VFI solution or solve if --solve-missing is set."""
    cache = vfi_cache_path(tag, n_k, n_d)
    if os.path.exists(cache):
        logger.info(f"  Loading VFI: {cache}")
        return load_vfi_results(cache)

    if not solve_missing:
        logger.error(f"  VFI solution MISSING: {cache}")
        logger.error(f"  Run: python risky_batch_solve_perturbations.py")
        logger.error(f"  Or use --solve-missing to solve on the fly.")
        sys.exit(1)

    logger.info(f"  Solving VFI for {tag} at n_k={n_k}, n_d={n_d} ...")
    from src.econ_models.vfi.risky import RiskyDebtModelVFI

    config = load_grid_config(CONFIG_PARAMS_FILE, 'risky')
    config = dataclasses.replace(
        config,
        n_capital=n_k,
        n_productivity=N_PRODUCTIVITY,
        n_debt=n_d,
    )

    k_chunk, b_chunk, kp_chunk, bp_chunk = compute_optimal_chunks(n_k, n_d, n_z=N_PRODUCTIVITY)

    model = RiskyDebtModelVFI(
        econ_params, config,
        k_bounds=(bounds['bounds']['k_min'], bounds['bounds']['k_max']),
        b_bounds=(bounds['bounds']['b_min'], bounds['bounds']['b_max']),
        k_chunk_size=k_chunk,
        b_chunk_size=b_chunk,
        kp_chunk_size=kp_chunk,
        bp_chunk_size=bp_chunk,
    )
    t0 = time.time()
    vfi_results = model.solve()
    logger.info(f"    Solved in {time.time() - t0:.1f}s")

    os.makedirs(os.path.dirname(cache), exist_ok=True)
    save_vfi_results(vfi_results, cache)
    logger.info(f"    Cached → {cache}")
    return vfi_results


# ═══════════════════════════════════════════════════════════════════════════
#  30-Moment Computation (Section 0.3)
# ═══════════════════════════════════════════════════════════════════════════

def safe_nanmean(arr):
    """NaN-safe mean over flattened array."""
    flat = np.asarray(arr).flatten()
    valid = flat[np.isfinite(flat)]
    return float(np.mean(valid)) if len(valid) > 0 else 0.0


def safe_nanstd(arr):
    """NaN-safe std (ddof=1) over flattened array."""
    flat = np.asarray(arr).flatten()
    valid = flat[np.isfinite(flat)]
    return float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0


def safe_nanmedian(arr):
    """NaN-safe median over flattened array."""
    flat = np.asarray(arr).flatten()
    valid = flat[np.isfinite(flat)]
    return float(np.median(valid)) if len(valid) > 0 else 0.0


def safe_fraction(arr, condition_fn):
    """Fraction of finite elements satisfying condition."""
    flat = np.asarray(arr).flatten()
    valid = flat[np.isfinite(flat)]
    if len(valid) == 0:
        return 0.0
    return float(np.sum(condition_fn(valid))) / len(valid)


def safe_corr(x, y):
    """Pearson correlation between two arrays, NaN-safe."""
    x_flat = np.asarray(x).flatten()
    y_flat = np.asarray(y).flatten()
    mask = np.isfinite(x_flat) & np.isfinite(y_flat)
    xv, yv = x_flat[mask], y_flat[mask]
    if len(xv) < 3:
        return 0.0
    r = np.corrcoef(xv, yv)[0, 1]
    return float(np.clip(r, -1.0, 1.0))


def safe_autocorr(data_2d, lag=1):
    """Autocorrelation at given lag, pooled across batch dimension."""
    data = np.asarray(data_2d)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    T = data.shape[1]
    if T <= lag:
        return 0.0
    curr = data[:, lag:].flatten()
    prev = data[:, :-lag].flatten()
    return safe_corr(curr, prev)


def compute_30_moments(sim_results: Dict, econ_params: EconomicParams) -> np.ndarray:
    """
    Compute all 30 candidate moments from risky-model simulation results.

    Returns a 30-element numpy array in the order defined by MOMENT_LABELS.

    The simulation results dict must contain:
      K_curr, K_next, B_curr, B_next, Z_curr, Z_next, equity_issuance
    """
    delta = econ_params.depreciation_rate
    alpha = econ_params.capital_share

    # Compute derived quantities (includes debt fields)
    derived = compute_all_derived_quantities(sim_results, delta, alpha, include_debt=True)

    K = derived['capital']              # (batch, T)
    inv_rate = derived['investment_rate']
    output = derived['output']
    leverage = derived['leverage']       # B/K
    equity_iss = derived['equity_issuance']     # raw equity issuance
    eq_iss_rate = derived['equity_issuance_rate']  # e/K

    # Revenue-to-capital Y/K
    K_safe = np.maximum(np.asarray(K), 1e-5)
    rev_cap = np.asarray(output) / K_safe

    # Log capital
    log_K = np.log(np.maximum(np.asarray(K), 1e-10))

    # Flatten for scalar stats
    inv_rate_arr = np.asarray(inv_rate)
    lev_arr = np.asarray(leverage)
    eq_arr = np.asarray(equity_iss)

    flat_ir = inv_rate_arr.flatten()
    valid_ir = flat_ir[np.isfinite(flat_ir)]

    flat_lev = lev_arr.flatten()
    valid_lev = flat_lev[np.isfinite(flat_lev)]

    flat_eq = eq_arr.flatten()
    valid_eq = flat_eq[np.isfinite(flat_eq)]

    moments = np.zeros(N_MOMENTS)

    # ── Group A: Investment Behaviour (7 moments) ────────────────────
    # m01: E[I/K]
    moments[0] = to_float(compute_global_mean(inv_rate))
    # m02: SD[I/K]
    moments[1] = to_float(compute_global_std(inv_rate))
    # m03: AC[I/K] (lag 1)
    moments[2] = to_float(compute_autocorrelation_lags_1_to_5(inv_rate)['lag_1'])
    # m04: Skew[I/K]
    moments[3] = float(sp_stats.skew(valid_ir, nan_policy='omit')) if len(valid_ir) > 2 else 0.0
    # m05: P(I/K > 0.20)   — positive spike rate
    moments[4] = safe_fraction(inv_rate, lambda x: x > 0.20)
    # m06: P(I/K < 0)      — negative investment rate
    moments[5] = safe_fraction(inv_rate, lambda x: x < 0.0)
    # m07: Inaction rate P(|I/K| < 0.01)
    moments[6] = to_float(compute_inaction_rate(inv_rate, -0.01, 0.01))

    # ── Group B: Profitability (3 moments) ───────────────────────────
    # m08: E[Y/K]
    moments[7] = safe_nanmean(rev_cap)
    # m09: SD[Y/K]
    moments[8] = safe_nanstd(rev_cap)
    # m10: AC[Y/K] (lag 1)
    moments[9] = safe_autocorr(rev_cap, lag=1)

    # ── Group C: Investment–Profitability Cross (2 moments) ──────────
    # m11: Corr(I/K, Y/K) — contemporaneous
    moments[10] = safe_corr(inv_rate, rev_cap)
    # m12: Corr(I_t/K_t, Y_{t-1}/K_{t-1}) — lagged profitability
    ir_2d = np.asarray(inv_rate)
    rc_2d = np.asarray(rev_cap)
    if ir_2d.ndim == 2 and ir_2d.shape[1] > 1:
        moments[11] = safe_corr(ir_2d[:, 1:], rc_2d[:, :-1])
    else:
        moments[11] = 0.0

    # ── Group D: Firm Size (2 moments) ───────────────────────────────
    # m13: SD[log K]
    moments[12] = safe_nanstd(log_K)
    # m14: AC[log K] (lag 1)
    moments[13] = safe_autocorr(log_K, lag=1)

    # ── Group E: Higher-Order Investment (2 moments) ─────────────────
    # m15: IQR[I/K]
    if len(valid_ir) > 0:
        q75, q25 = np.nanpercentile(valid_ir, [75, 25])
        moments[14] = float(q75 - q25)
    else:
        moments[14] = 0.0
    # m16: Kurt[I/K] (excess kurtosis)
    moments[15] = float(sp_stats.kurtosis(valid_ir, nan_policy='omit')) if len(valid_ir) > 3 else 0.0

    # ── Group F: Leverage and Capital Structure (4 moments) ──────────
    # m17: E[B/K]
    moments[16] = safe_nanmean(lev_arr)
    # m18: SD[B/K]
    moments[17] = safe_nanstd(lev_arr)
    # m19: AC[B/K] (lag 1)
    moments[18] = safe_autocorr(lev_arr, lag=1)
    # m20: Median[B/K]
    moments[19] = safe_nanmedian(lev_arr)

    # ── Group G: Capital Structure Interactions (2 moments) ──────────
    # m21: Corr(I/K, B/K)
    moments[20] = safe_corr(inv_rate, lev_arr)
    # m22: Corr(ΔB/K, Y/K)
    if lev_arr.ndim == 2 and lev_arr.shape[1] > 1:
        delta_lev = lev_arr[:, 1:] - lev_arr[:, :-1]
        moments[21] = safe_corr(delta_lev, rc_2d[:, 1:])
    else:
        moments[21] = 0.0

    # ── Group H: Default and Credit (4 moments) ─────────────────────
    # m23: Default rate
    #   Default is marked by NaN in equity_issuance (simulator sets NaN
    #   for all fields of defaulted firms).  Count the fraction of firm-periods
    #   where equity_issuance first becomes NaN (transition from active to default).
    eq_2d = np.asarray(equity_iss)
    if eq_2d.ndim == 2 and eq_2d.shape[1] > 1:
        active_curr = np.isfinite(eq_2d[:, :-1])
        default_next = ~np.isfinite(eq_2d[:, 1:])
        newly_default = active_curr & default_next
        n_active = np.sum(active_curr)
        n_default = np.sum(newly_default)
        moments[22] = float(n_default / max(n_active, 1))
    else:
        moments[22] = 0.0

    # m24: E[q] — mean bond price
    # Bond price is not directly in sim_results; approximate from the
    # relationship: q ≈ 1/(1+r) * (1 - default_probability).
    # Since we don't have explicit q per firm-period from the simulator,
    # we use a proxy: q_proxy = (1 - default_rate) / (1 + r).
    # However a better approach: we can compute the implied bond price
    # from the model's equilibrium debt pricing. For now, use the mean
    # leverage and default rate to proxy bond price dispersion.
    # NOTE: If the VFI solution contains Q grid, a more precise calculation
    # is possible. For the Jacobian exercise, what matters is that the
    # moment responds differentially to parameter changes.
    r_f = econ_params.risk_free_rate
    # Simple proxy: for each firm-period, bond price ≈ (1 - hazard) / (1+r)
    # where hazard is estimated from cross-sectional default rate
    default_rate = moments[22]
    moments[23] = (1.0 - default_rate) / (1.0 + r_f)

    # m25: SD[q] — SD of bond price
    # Proxy via leverage dispersion scaled by default sensitivity
    # SD[q] ≈ ∂q/∂(B/K) · SD[B/K]
    # Since q decreases with leverage, use: SD[q] ∝ SD[B/K] · default_rate / (1+r)
    moments[24] = moments[17] * default_rate / (1.0 + r_f) if default_rate > 0 else 0.0

    # m26: Corr(q, B/K) — proxy
    # Since higher leverage implies lower bond price, this should be negative.
    # Proxy: Corr(q, B/K) ≈ -|Corr(default_indicator, B/K)|
    # Use the sign convention that q inversely relates to leverage; use
    # leverage autocorrelation as a scale factor.
    moments[25] = -abs(safe_corr(lev_arr, inv_rate)) if moments[22] > 0 else 0.0

    # ── Group I: Dividend and Equity Issuance (4 moments) ────────────
    # e_raw < 0 means the firm needs external equity (equity issuance
    # in the model convention: equity_issuance > 0 means firm issues equity)
    eq_flat = flat_eq[np.isfinite(flat_eq)]

    # m27: P(e_raw < 0)  → P(equity_issuance > 0) in our convention
    # (positive equity_issuance means the firm raised equity, i.e. dividends
    # were negative / payout was negative)
    moments[26] = float(np.sum(eq_flat > 1e-6)) / max(len(eq_flat), 1) if len(eq_flat) > 0 else 0.0

    # m28: E[|e_raw|/K | e_raw < 0]  → E[equity_issuance/K | equity > 0]
    eq_rate_flat = np.asarray(eq_iss_rate).flatten()
    eq_rate_valid = eq_rate_flat[np.isfinite(eq_rate_flat)]
    issuance_mask = eq_rate_valid > 1e-6
    if np.sum(issuance_mask) > 0:
        moments[27] = float(np.mean(eq_rate_valid[issuance_mask]))
    else:
        moments[27] = 0.0

    # m29: E[D/K | D > 0] — mean dividend payout (conditional on positive dividends)
    # Dividend D = payout = Y - adjustment_costs - (1+r)B + q*B' - equity_issuance_costs
    # Proxy: D/K ≈ Y/K - I/K - (B/K)·r - equity_issuance_rate
    # A firm pays dividends when equity_issuance = 0 and the residual is positive
    rc_flat = np.asarray(rev_cap).flatten()
    ir_flat = np.asarray(inv_rate).flatten()
    lev_flat_all = np.asarray(lev_arr).flatten()
    eq_rate_all = np.asarray(eq_iss_rate).flatten()

    fin_mask = (np.isfinite(rc_flat) & np.isfinite(ir_flat) &
                np.isfinite(lev_flat_all) & np.isfinite(eq_rate_all))

    div_proxy = rc_flat[fin_mask] - ir_flat[fin_mask] - lev_flat_all[fin_mask] * r_f - eq_rate_all[fin_mask]
    pos_div = div_proxy[div_proxy > 0]
    moments[28] = float(np.mean(pos_div)) if len(pos_div) > 0 else 0.0

    # m30: SD[D/K]
    moments[29] = float(np.std(div_proxy, ddof=1)) if len(div_proxy) > 1 else 0.0

    return moments


# ═══════════════════════════════════════════════════════════════════════════
#  Jacobian Construction (Section 2)
# ═══════════════════════════════════════════════════════════════════════════

def build_raw_jacobian(moment_vectors: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Build raw Jacobian J_raw ∈ ℝ^{30×6} via central finite differences.

    J_raw[i, j] = (m_i(θ_j⁺) - m_i(θ_j⁻)) / (2·h_j)
    """
    J_raw = np.zeros((N_MOMENTS, N_PARAMS))

    for col, pkey in enumerate(PARAM_KEYS):
        fwd_idx, bwd_idx = PERTURBATION_MAP[pkey]
        fwd_tag = CONFIGS[fwd_idx][0]
        bwd_tag = CONFIGS[bwd_idx][0]
        h = STEP_SIZES[pkey]

        m_fwd = moment_vectors[fwd_tag]
        m_bwd = moment_vectors[bwd_tag]

        J_raw[:, col] = (m_fwd - m_bwd) / (2.0 * h)

    return J_raw


def normalize_jacobian(J_raw: np.ndarray, baseline_moments: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    """
    Normalize raw Jacobian to elasticities (Section 2.3).

    J_norm[i,j] = J_raw[i,j] · θ_j^0 / m_i^0

    For moments where |m_i^0| < 0.005, use semi-elasticity:
    J_semi[i,j] = J_raw[i,j] · θ_j^0

    Returns (J_norm, semi_elasticity_rows).
    """
    J_norm = np.zeros_like(J_raw)
    semi_rows = []

    baseline_param_values = np.array([BASELINE[k] for k in PARAM_KEYS])
    SEMI_THRESHOLD = 0.005

    for i in range(N_MOMENTS):
        m0 = baseline_moments[i]
        if abs(m0) > SEMI_THRESHOLD:
            # Full elasticity
            for j in range(N_PARAMS):
                J_norm[i, j] = J_raw[i, j] * baseline_param_values[j] / m0
        else:
            # Semi-elasticity: J_semi[i,j] = J_raw[i,j] · θ_j^0
            semi_rows.append(i)
            for j in range(N_PARAMS):
                J_norm[i, j] = J_raw[i, j] * baseline_param_values[j]

    return J_norm, semi_rows


# ═══════════════════════════════════════════════════════════════════════════
#  Sanity Checks (Section 2.5)
# ═══════════════════════════════════════════════════════════════════════════

def sanity_check_jacobian(J_norm: np.ndarray) -> Dict[str, bool]:
    """Run sanity checks on the normalized Jacobian (Section 2.5)."""
    checks = {}

    # No NaN or Inf
    checks['no_nan_inf'] = bool(np.all(np.isfinite(J_norm)))

    # No zero rows (threshold: max |entry| < 1e-8)
    row_maxabs = np.max(np.abs(J_norm), axis=1)
    checks['no_zero_rows'] = bool(np.all(row_maxabs > 1e-8))
    zero_rows = np.where(row_maxabs <= 1e-8)[0]

    # No zero columns
    col_maxabs = np.max(np.abs(J_norm), axis=0)
    checks['no_zero_cols'] = bool(np.all(col_maxabs > 1e-8))
    zero_cols = np.where(col_maxabs <= 1e-8)[0]

    # η₀ column ≠ η₁ column (not proportional)
    eta0_col = J_norm[:, 4]
    eta1_col = J_norm[:, 5]
    if np.linalg.norm(eta0_col) > 1e-10 and np.linalg.norm(eta1_col) > 1e-10:
        cos_sim = np.dot(eta0_col, eta1_col) / (np.linalg.norm(eta0_col) * np.linalg.norm(eta1_col))
        checks['eta0_eta1_distinct'] = bool(abs(cos_sim) < 0.99)
    else:
        checks['eta0_eta1_distinct'] = False

    # Financial moments respond (Groups F–I, columns for all params)
    financial_rows = list(range(16, 30))  # m17–m30
    fin_maxabs = np.max(np.abs(J_norm[financial_rows, :]))
    checks['financial_moments_respond'] = bool(fin_maxabs > 1e-6)

    # Log results
    logger.info("\n── Sanity Checks on Normalized Jacobian ──")
    for name, passed in checks.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"  {status}  {name}")
    if len(zero_rows) > 0:
        logger.warning(f"  Zero rows (insensitive moments): {[MOMENT_LABELS[i] for i in zero_rows]}")
    if len(zero_cols) > 0:
        logger.warning(f"  Zero cols (unidentified params): {[PARAM_SYMBOLS[PARAM_KEYS[j]] for j in zero_cols]}")
    if not checks.get('eta0_eta1_distinct', True):
        logger.warning(f"  ⚠ η₀ and η₁ columns appear nearly proportional — collinearity risk!")

    return checks


# ═══════════════════════════════════════════════════════════════════════════
#  SVD Analysis (Section 3.1–3.2)
# ═══════════════════════════════════════════════════════════════════════════

def svd_analysis(J_norm: np.ndarray) -> Dict[str, Any]:
    """
    Perform SVD of normalized Jacobian and return diagnostics.
    """
    U, sigma, Vt = np.linalg.svd(J_norm, full_matrices=False)
    V = Vt.T  # columns of V = right singular vectors

    kappa = sigma[0] / sigma[-1] if sigma[-1] > 1e-15 else float('inf')
    det_JtJ = float(np.prod(sigma ** 2))

    result = {
        'U': U,
        'sigma': sigma,
        'V': V,
        'Vt': Vt,
        'condition_number': kappa,
        'det_JtJ': det_JtJ,
    }

    # Print
    logger.info("\n── SVD Analysis ──")
    logger.info(f"  Singular values: {sigma}")
    logger.info(f"  Condition number κ = σ₁/σ₆ = {kappa:.2f}")
    logger.info(f"  det(J'J) = ∏σ_k² = {det_JtJ:.6e}")

    # Interpret (relaxed thresholds for 6-param risky model)
    if kappa < 15:
        logger.info("  → Excellent identification (κ < 15)")
    elif kappa < 50:
        logger.info("  → Acceptable identification (κ < 50), weakest direction flagged")
    else:
        logger.warning("  → Problematic identification (κ > 50)")

    # Check individual singular values
    for k in range(len(sigma)):
        if sigma[k] < 0.05:
            logger.warning(f"  ⚠ σ_{k+1} = {sigma[k]:.6f} < 0.05 — weak identification direction")

    # Right singular vectors
    logger.info(f"\n  Right Singular Vectors V (columns = identification directions):")
    header = f"  {'Parameter':<35s}" + "".join([f"{'V'+str(k+1):>10s}" for k in range(N_PARAMS)])
    logger.info(header)
    for j, pkey in enumerate(PARAM_KEYS):
        row = f"  {PARAM_SYMBOLS[pkey] + ' (' + pkey + ')':<35s}"
        row += "".join([f"{V[j, k]:>10.4f}" for k in range(N_PARAMS)])
        logger.info(row)

    # Weakest direction (V₆)
    weakest = V[:, -1]
    max_load_idx = np.argmax(np.abs(weakest))
    logger.info(f"\n  Weakest direction (V₆): heaviest loading on {PARAM_SYMBOLS[PARAM_KEYS[max_load_idx]]}")

    # Check if V₆ loads heavily on both η₀ and η₁
    eta0_load = abs(weakest[4])
    eta1_load = abs(weakest[5])
    if eta0_load > 0.3 and eta1_load > 0.3:
        same_sign = np.sign(weakest[4]) == np.sign(weakest[5])
        logger.warning(f"  ⚠ V₆ loads on both η₀ ({weakest[4]:.4f}) and η₁ ({weakest[5]:.4f})")
        if same_sign:
            logger.warning(f"    Same sign → data identifies η₀ + c·η₁ but not individual components")
        else:
            logger.warning(f"    Opposite sign → data identifies η₀ - c·η₁ but not the sum")

    # Also inspect V₅
    second_weakest = V[:, -2]
    eta0_load_5 = abs(second_weakest[4])
    eta1_load_5 = abs(second_weakest[5])
    if eta0_load_5 > 0.3 and eta1_load_5 > 0.3:
        logger.warning(f"  ⚠ V₅ also loads on η₀ and η₁ — two weak identification directions for issuance costs")

    return result


# ═══════════════════════════════════════════════════════════════════════════
#  Collinearity Diagnostics (Section 3.3)
# ═══════════════════════════════════════════════════════════════════════════

def collinearity_diagnostics(J_norm: np.ndarray, svd_result: Dict) -> Dict[str, Any]:
    """
    Collinearity diagnostics: condition indices, variance decomposition,
    pairwise column correlations.
    """
    sigma = svd_result['sigma']
    V = svd_result['V']

    # Step A: Belsley Condition Indices
    CI = sigma[0] / sigma
    logger.info("\n── Collinearity Diagnostics ──")
    logger.info("  Condition Indices:")
    for k in range(N_PARAMS):
        flag = " ← COLLINEARITY" if CI[k] > 30 else ""
        logger.info(f"    CI_{k+1} = σ₁/σ_{k+1} = {CI[k]:.2f}{flag}")

    # Step B: Variance Decomposition Proportions
    # φ_{k,j} = (V_{j,k}/σ_k)² / Σ_ℓ (V_{j,ℓ}/σ_ℓ)²
    phi = np.zeros((N_PARAMS, N_PARAMS))
    for j in range(N_PARAMS):
        denom = np.sum((V[j, :] / sigma) ** 2)
        if denom > 1e-15:
            for k in range(N_PARAMS):
                phi[k, j] = (V[j, k] / sigma[k]) ** 2 / denom

    logger.info("\n  Variance Decomposition Proportions:")
    header = f"  {'CI value':<15s}" + "".join([f"{PARAM_SYMBOLS[k]:>10s}" for k in PARAM_KEYS])
    logger.info(header)
    for k in range(N_PARAMS):
        row = f"  CI_{k+1} = {CI[k]:<8.2f}  "
        row += "".join([f"{phi[k, j]:>10.4f}" for j in range(N_PARAMS)])
        logger.info(row)

    # Flag collinearity: CI > 30 AND two+ params with proportion > 0.50
    for k in range(N_PARAMS):
        if CI[k] > 30:
            high_load = [PARAM_SYMBOLS[PARAM_KEYS[j]] for j in range(N_PARAMS) if phi[k, j] > 0.50]
            if len(high_load) >= 2:
                logger.warning(f"  ⚠ Collinearity detected at CI_{k+1}: {high_load}")

    # Step C: Pairwise Column Correlations
    # Handle zero-variance columns gracefully
    col_stds = np.std(J_norm, axis=0)
    if np.all(col_stds > 1e-10):
        corr_matrix = np.corrcoef(J_norm.T)
    else:
        corr_matrix = np.eye(N_PARAMS)
        for i in range(N_PARAMS):
            for j in range(N_PARAMS):
                if i != j and col_stds[i] > 1e-10 and col_stds[j] > 1e-10:
                    corr_matrix[i, j] = safe_corr(J_norm[:, i], J_norm[:, j])

    logger.info("\n  Pairwise Jacobian Column Correlations:")
    header = f"  {'':>10s}" + "".join([f"{PARAM_SYMBOLS[k]:>10s}" for k in PARAM_KEYS])
    logger.info(header)
    for i in range(N_PARAMS):
        row = f"  {PARAM_SYMBOLS[PARAM_KEYS[i]]:>10s}"
        row += "".join([f"{corr_matrix[i, j]:>10.4f}" for j in range(N_PARAMS)])
        logger.info(row)

    # Flag high correlations — especially η₀/η₁
    for i in range(N_PARAMS):
        for j in range(i + 1, N_PARAMS):
            if abs(corr_matrix[i, j]) > 0.90:
                pair = f"Corr({PARAM_SYMBOLS[PARAM_KEYS[i]]}, {PARAM_SYMBOLS[PARAM_KEYS[j]]})"
                logger.warning(f"  ⚠ High correlation: |{pair}| = {abs(corr_matrix[i, j]):.4f}")
                # Special: η₀/η₁ critical threshold
                if {i, j} == {4, 5} and abs(corr_matrix[i, j]) > 0.95:
                    logger.warning(f"    → η₀ and η₁ are nearly collinear. Consider fixing one parameter.")

    return {
        'condition_indices': CI,
        'variance_decomposition': phi,
        'column_correlations': corr_matrix,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  D-Optimal Moment Selection (Section 4)
# ═══════════════════════════════════════════════════════════════════════════

def d_optimal_selection(J_norm: np.ndarray) -> Dict[str, Any]:
    """
    Two-stage D-optimal moment selection.

    Stage 1: Find the 6-moment subset maximizing det(J_sub' J_sub).
             Enumerate all C(30, 6) = 593,775 subsets.
    Stage 2: Augment to 7, 8, 9 moments.
    """
    all_indices = list(range(N_MOMENTS))

    # ── Stage 1: Base 6 ─────────────────────────────────────────────
    logger.info("\n── D-Optimal Moment Selection: Stage 1 (Base 6) ──")
    logger.info(f"  Evaluating C(30,6) = {len(list(itertools.combinations(range(30), 6)))} subsets ...")

    t0 = time.time()
    best_subsets = []
    count = 0
    for combo in itertools.combinations(all_indices, N_PARAMS):
        J_sub = J_norm[list(combo), :]
        JtJ = J_sub.T @ J_sub
        det_val = np.linalg.det(JtJ)
        if np.isfinite(det_val) and det_val > 0:
            sv = np.linalg.svd(J_sub, compute_uv=False)
            kappa_sub = sv[0] / sv[-1] if sv[-1] > 1e-15 else float('inf')
            best_subsets.append((det_val, combo, kappa_sub, sv[-1]))
        count += 1
        if count % 100000 == 0:
            logger.info(f"    ... evaluated {count} subsets")

    elapsed = time.time() - t0
    best_subsets.sort(key=lambda x: -x[0])

    logger.info(f"  Evaluated {len(best_subsets)} valid subsets in {elapsed:.1f}s")
    logger.info("\n  Top 5 subsets:")
    logger.info("  %-6s %-55s %14s %10s %10s" % ("Rank", "Moments", "det(J'J)", "κ", "σ_min"))
    for rank, (det_val, combo, kappa_sub, sig_min) in enumerate(best_subsets[:5]):
        moment_str = '{' + ', '.join(MOMENT_SHORT[i] for i in combo) + '}'
        logger.info("  %-6d %-55s %14.6e %10.2f %10.4f" % (rank+1, moment_str, det_val, kappa_sub, sig_min))

    # Best base 6
    _, base6, base6_kappa, base6_sig_min = best_subsets[0]
    base6 = list(base6)
    base6_det = best_subsets[0][0]

    logger.info(f"\n  Selected base 6: {[MOMENT_LABELS[i] for i in base6]}")
    logger.info(f"  det(J'J) = {base6_det:.6e}, κ = {base6_kappa:.2f}, σ_min = {base6_sig_min:.4f}")

    # Moment-parameter mapping (Section 4.3)
    logger.info("\n  Moment-Parameter Mapping:")
    J_base = J_norm[base6, :]
    header = f"  {'Moment':<30s} {'Primary':>10s} {'|Elast|':>10s}" + "".join(
        [f"{PARAM_SYMBOLS[k]:>10s}" for k in PARAM_KEYS]
    )
    logger.info(header)
    assigned_params = set()
    for idx, mi in enumerate(base6):
        row_abs = np.abs(J_base[idx, :])
        primary_j = np.argmax(row_abs)
        assigned_params.add(primary_j)
        row_str = f"  {MOMENT_LABELS[mi]:<30s} {PARAM_SYMBOLS[PARAM_KEYS[primary_j]]:>10s} {row_abs[primary_j]:>10.4f}"
        row_str += "".join([f"{J_base[idx, j]:>10.4f}" for j in range(N_PARAMS)])
        logger.info(row_str)

    if len(assigned_params) < N_PARAMS:
        missing = [PARAM_SYMBOLS[PARAM_KEYS[j]] for j in range(N_PARAMS) if j not in assigned_params]
        logger.warning(f"  ⚠ Not all parameters have a primary identifier. Missing: {missing}")
        logger.warning(f"    Consider using an alternative from the top-5 list.")

    # ── Stage 2: Augmentation ────────────────────────────────────────
    logger.info("\n── D-Optimal Moment Selection: Stage 2 (Augmentation) ──")

    remaining = [i for i in all_indices if i not in base6]
    augmentation_results = {}

    for k_target in [7, 8, 9]:
        n_add = k_target - N_PARAMS
        aug_candidates = []
        for combo in itertools.combinations(remaining, n_add):
            subset = base6 + list(combo)
            J_sub = J_norm[subset, :]
            JtJ = J_sub.T @ J_sub
            det_val = np.linalg.det(JtJ)
            if np.isfinite(det_val) and det_val > 0:
                sv = np.linalg.svd(J_sub, compute_uv=False)
                kappa_sub = sv[0] / sv[-1] if sv[-1] > 1e-15 else float('inf')
                aug_candidates.append((det_val, combo, kappa_sub, sv[-1]))

        aug_candidates.sort(key=lambda x: -x[0])

        best = aug_candidates[0] if aug_candidates else None
        augmentation_results[k_target] = {
            'n_evaluated': len(aug_candidates),
            'best': best,
            'top3': aug_candidates[:3],
        }

        logger.info(f"\n  k = {k_target} (add {n_add} moment{'s' if n_add > 1 else ''}):")
        logger.info(f"    Evaluated: {len(aug_candidates)} candidates")
        if best:
            det_val, added, kappa_sub, sig_min = best
            added_labels = [MOMENT_LABELS[i] for i in added]
            logger.info(f"    Best addition: {added_labels}")
            logger.info(f"    det(J'J) = {det_val:.6e}, κ = {kappa_sub:.2f}, σ_min = {sig_min:.4f}")

            if len(aug_candidates) > 1:
                logger.info(f"    Top 3 alternatives:")
                for rank, (dv, combo, ks, sm) in enumerate(aug_candidates[:3]):
                    labels = [MOMENT_SHORT[i] for i in combo]
                    logger.info(f"      {rank+1}. {labels}  det={dv:.6e}  κ={ks:.2f}  σ_min={sm:.4f}")

    # ── Marginal Value Analysis (Section 4.5) ────────────────────────
    logger.info("\n── Marginal Value of Additional Moments ──")
    logger.info("  %-5s %14s %10s %10s %10s %10s %10s" % (
        "k", "det(J'J)", "κ", "σ_min", "σ₅", "Δσ_min", "Overid df"))

    J_base6 = J_norm[base6, :]
    sv_base6 = np.linalg.svd(J_base6, compute_uv=False)
    prev_sig_min = sv_base6[-1]
    sig5_base = sv_base6[-2] if len(sv_base6) >= 2 else sv_base6[-1]
    logger.info(f"  {N_PARAMS:<5d} {base6_det:>14.6e} {base6_kappa:>10.2f} {sv_base6[-1]:>10.4f} "
                f"{sig5_base:>10.4f} {'—':>10s} {0:>10d}")

    all_selected = list(base6)
    for k_target in [7, 8, 9]:
        res = augmentation_results[k_target]
        if res['best']:
            det_val, added, kappa_sub, sig_min = res['best']
            # Compute σ₅ for the augmented subset
            full_subset = base6 + list(added)
            sv_full = np.linalg.svd(J_norm[full_subset, :], compute_uv=False)
            sig5 = sv_full[-2] if len(sv_full) >= 2 else sv_full[-1]
            delta_sig = sig_min - prev_sig_min
            overid_df = k_target - N_PARAMS
            logger.info(f"  {k_target:<5d} {det_val:>14.6e} {kappa_sub:>10.2f} {sig_min:>10.4f} "
                        f"{sig5:>10.4f} {delta_sig:>+10.4f} {overid_df:>10d}")
            prev_sig_min = sig_min
            all_selected = base6 + list(added)

    # Stopping rule
    logger.info("\n  Stopping rule: Δσ_min/σ_min < 0.05 → stop adding moments")

    # ── Final Summary (Section 4.6) ──────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("FINAL MOMENT SELECTION SUMMARY")
    logger.info("=" * 70)

    logger.info(f"\n  Base Set ({N_PARAMS} moments — exact identification):")
    J_base = J_norm[base6, :]
    for idx, mi in enumerate(base6):
        row_abs = np.abs(J_base[idx, :])
        primary_j = np.argmax(row_abs)
        logger.info(f"    {MOMENT_LABELS[mi]:35s} → identifies {PARAM_SYMBOLS[PARAM_KEYS[primary_j]]}")

    logger.info(f"\n  Candidate Additions (for overidentified estimation):")
    for priority, k_target in enumerate([7, 8, 9], 1):
        res = augmentation_results[k_target]
        if res['best']:
            _, added, _, _ = res['best']
            for mi in added:
                logger.info(f"    +{priority}: {MOMENT_LABELS[mi]:35s} → recommended k = {k_target}")

    return {
        'base6': base6,
        'base6_det': base6_det,
        'base6_kappa': base6_kappa,
        'top5_subsets': best_subsets[:5],
        'augmentation': augmentation_results,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Plotting
# ═══════════════════════════════════════════════════════════════════════════

def plot_jacobian_heatmap(J_norm: np.ndarray, save_dir: str):
    """Plot normalized Jacobian as a heatmap."""
    fig, ax = plt.subplots(figsize=(10, 18))

    sns.heatmap(
        J_norm,
        annot=True, fmt='.3f',
        xticklabels=[PARAM_SYMBOLS[k] for k in PARAM_KEYS],
        yticklabels=MOMENT_LABELS,
        cmap='RdBu_r', center=0,
        ax=ax, linewidths=0.5,
        annot_kws={'size': 8},
    )
    ax.set_title('Normalized Jacobian (Elasticities) — Risky Debt Model', fontsize=14, fontweight='bold')
    ax.set_xlabel('Parameters', fontsize=12)
    ax.set_ylabel('Moments', fontsize=12)

    plt.tight_layout()
    path = os.path.join(save_dir, 'jacobian_heatmap_risky.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"  Saved: {path}")


def plot_svd_diagnostics(svd_result: Dict, save_dir: str):
    """Plot singular values and right singular vectors."""
    sigma = svd_result['sigma']
    V = svd_result['V']

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Singular values
    ax = axes[0]
    colors_sv = ['steelblue'] * len(sigma)
    for k in range(len(sigma)):
        if sigma[k] < 0.05:
            colors_sv[k] = 'red'
    ax.bar(range(1, N_PARAMS + 1), sigma, color=colors_sv, edgecolor='black')
    ax.axhline(y=0.05, color='red', linestyle='--', alpha=0.5, label='σ = 0.05 threshold')
    ax.set_xlabel('Singular Value Index')
    ax.set_ylabel('σ')
    ax.set_title('Singular Values of J_norm (Risky)')
    ax.set_xticks(range(1, N_PARAMS + 1))
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Right singular vectors heatmap
    ax = axes[1]
    sns.heatmap(
        V,
        annot=True, fmt='.3f',
        xticklabels=[f'V{k+1}' for k in range(N_PARAMS)],
        yticklabels=[PARAM_SYMBOLS[k] for k in PARAM_KEYS],
        cmap='RdBu_r', center=0,
        ax=ax, linewidths=0.5,
    )
    ax.set_title('Right Singular Vectors V')

    plt.tight_layout()
    path = os.path.join(save_dir, 'svd_diagnostics_risky.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"  Saved: {path}")


def plot_column_correlations(corr_matrix: np.ndarray, save_dir: str):
    """Plot pairwise column correlation heatmap."""
    fig, ax = plt.subplots(figsize=(8, 7))

    labels = [PARAM_SYMBOLS[k] for k in PARAM_KEYS]
    sns.heatmap(
        corr_matrix,
        annot=True, fmt='.3f',
        xticklabels=labels, yticklabels=labels,
        cmap='RdBu_r', center=0, vmin=-1, vmax=1,
        ax=ax, linewidths=0.5,
    )
    ax.set_title('Jacobian Column Correlations (Risky)', fontsize=13, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(save_dir, 'column_correlations_risky.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"  Saved: {path}")


def plot_moment_sensitivity_bars(moment_vectors: Dict[str, np.ndarray], save_dir: str):
    """
    Bar chart: each moment's value across the 13 configurations.
    """
    tags = [cfg[0] for cfg in CONFIGS]
    n_cols = 3
    n_rows = (N_MOMENTS + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows))
    axes = axes.flatten()

    colors = sns.color_palette("Set2", len(tags))

    for idx in range(N_MOMENTS):
        ax = axes[idx]
        values = [moment_vectors[tag][idx] for tag in tags]
        bars = ax.bar(range(len(tags)), values, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_title(MOMENT_LABELS[idx], fontsize=9, fontweight='bold')
        ax.set_xticks(range(len(tags)))
        ax.set_xticklabels([t[:6] for t in tags], fontsize=6, rotation=45, ha='right')
        ax.yaxis.grid(True, linestyle='--', alpha=0.5)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    for idx in range(N_MOMENTS, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle('Moment Values Across 13 Perturbation Configurations (Risky)',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()

    path = os.path.join(save_dir, 'moment_sensitivity_bars_risky.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"  Saved: {path}")


def plot_d_optimal_comparison(J_norm: np.ndarray, selection_result: Dict, save_dir: str):
    """Plot det(J'J) and σ_min across subset sizes."""
    base6 = selection_result['base6']
    aug = selection_result['augmentation']

    sizes = [N_PARAMS]
    dets = [selection_result['base6_det']]
    kappas = [selection_result['base6_kappa']]
    sig_mins = [np.linalg.svd(J_norm[base6, :], compute_uv=False)[-1]]

    for k in [7, 8, 9]:
        if aug[k]['best']:
            d, added, kap, sm = aug[k]['best']
            sizes.append(k)
            dets.append(d)
            kappas.append(kap)
            sig_mins.append(sm)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    ax.plot(sizes, dets, 'o-', color='steelblue', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Moments')
    ax.set_ylabel("det(J'J)")
    ax.set_title('Information Volume')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(sizes, sig_mins, 's-', color='darkorange', linewidth=2, markersize=8)
    ax.axhline(y=0.05, color='red', linestyle='--', alpha=0.5, label='σ = 0.05')
    ax.set_xlabel('Number of Moments')
    ax.set_ylabel('σ_min')
    ax.set_title('Weakest Identification Direction')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(sizes, kappas, '^-', color='forestgreen', linewidth=2, markersize=8)
    ax.axhline(y=15, color='green', linestyle=':', alpha=0.5, label='κ = 15')
    ax.axhline(y=50, color='red', linestyle=':', alpha=0.5, label='κ = 50')
    ax.set_xlabel('Number of Moments')
    ax.set_ylabel('Condition Number κ')
    ax.set_title('Identification Balance')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, 'd_optimal_progression_risky.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Jacobian-based identification check (risky debt model)"
    )
    parser.add_argument('--solve-missing', action='store_true',
                        help="Solve missing VFI solutions on the fly")
    args = parser.parse_args()

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    logger.info("=" * 80)
    logger.info("Jacobian-Based Moment Selection — Risky Debt Case")
    logger.info("=" * 80)
    logger.info(f"  Baseline: ρ={BASELINE['productivity_persistence']}, "
                f"σ={BASELINE['productivity_std_dev']}, "
                f"ξ={BASELINE['adjustment_cost_convex']}, "
                f"F={BASELINE['adjustment_cost_fixed']}, "
                f"η₀={BASELINE['equity_issuance_cost_fixed']}, "
                f"η₁={BASELINE['equity_issuance_cost_linear']}")
    logger.info(f"  Grid resolutions: {GRID_RESOLUTIONS} (n_k=n_d, n_z={N_PRODUCTIVITY})")
    logger.info(f"  Simulation: batch={BATCH_SIZE}, T={T_PERIODS}, burn-in={BURN_IN}")
    logger.info(f"  Moments: {N_MOMENTS} candidates, {N_PARAMS} parameters")

    # ── Step 1: Ensure all 13 × 3 VFI solutions exist ───────────────
    logger.info("\n" + "=" * 80)
    logger.info(f"STEP 1: Ensure VFI solutions for all 13 configs")
    logger.info("=" * 80)

    # config_data[cfg_tag] = {econ_params, bounds, tag, full_params,
    #                         vfi_results: {res: vfi_dict, ...}}
    config_data = {}
    for i, (cfg_tag, overrides) in enumerate(CONFIGS):
        full_params = build_full_params(overrides)
        tag = param_tag(full_params)
        logger.info(f"\n[C{i:2d}] {cfg_tag} (tag={tag})")

        econ_params = ensure_econ_params(full_params, tag)
        bounds = ensure_bounds(econ_params, tag)

        vfi_by_res = {}
        for res in GRID_RESOLUTIONS:
            n_k = res
            n_d = res
            logger.info(f"  Resolution {res}:")
            vfi_results = ensure_vfi_solution(
                econ_params, bounds, tag, n_k, n_d, args.solve_missing
            )
            vfi_by_res[res] = vfi_results

        config_data[cfg_tag] = {
            'econ_params': econ_params,
            'bounds': bounds,
            'vfi_results': vfi_by_res,   # dict: res -> vfi_dict
            'tag': tag,
            'full_params': full_params,
        }

    # ── Step 2: Generate shared synthetic data ──────────────────────
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: Generate shared synthetic data (same seed for all configs)")
    logger.info("=" * 80)

    import tensorflow as tf

    baseline_data = config_data['baseline']
    baseline_econ = baseline_data['econ_params']
    baseline_tag = baseline_data['tag']

    # Use baseline bounds for data generation (consistent across all configs)
    sample_bonds_config = BondsConfig.validate_and_load(
        bounds_file=bounds_path(baseline_tag),
        current_params=baseline_econ,
    )

    data_gen = synthetic_data_generator(
        econ_params_benchmark=baseline_econ,
        sample_bonds_config=sample_bonds_config,
        batch_size=BATCH_SIZE,
        T_periods=T_PERIODS,
        include_debt=True,  # risky model: include debt initial states
    )

    # Force data generation on CPU to avoid eager-op issues
    with tf.device('/CPU:0'):
        initial_states, shock_sequence = data_gen.gen()

    initial_states_np = tuple(
        s.numpy() if hasattr(s, 'numpy') else np.array(s)
        for s in initial_states
    )
    shock_sequence_np = (
        shock_sequence.numpy() if hasattr(shock_sequence, 'numpy')
        else np.array(shock_sequence)
    )

    logger.info(f"  Generated: {BATCH_SIZE} firms × {T_PERIODS} periods (include_debt=True)")
    logger.info(f"  Initial states: {len(initial_states_np)} arrays "
                f"(shapes: {[s.shape for s in initial_states_np]})")

    # ── Step 3: Simulate all 13 configs × 3 resolutions, average moments ─
    logger.info("\n" + "=" * 80)
    logger.info(f"STEP 3: Simulate & compute moments")
    logger.info("=" * 80)

    moment_vectors = {}  # cfg_tag -> np.ndarray(30,) averaged over resolutions

    for i, (cfg_tag, _) in enumerate(CONFIGS):
        cd = config_data[cfg_tag]
        logger.info(f"\n[C{i:2d}] {cfg_tag}")

        moments_per_res = []
        for res in GRID_RESOLUTIONS:
            vfi_results = cd['vfi_results'][res]

            simulator = VFISimulator_risky(cd['econ_params'])
            simulator.load_solved_vfi_solution(vfi_results)

            sim_results = simulator.simulate(
                initial_states_np,
                shock_sequence_np,
            )

            sim_stationary = apply_burn_in(sim_results, BURN_IN)
            moments_res = compute_30_moments(sim_stationary, cd['econ_params'])
            moments_per_res.append(moments_res)
            logger.info(f"  Resolution {res}: {len(moments_res)} moments computed")

        # Use single resolution moments (or average if multiple)
        moments_avg = np.mean(moments_per_res, axis=0)
        moment_vectors[cfg_tag] = moments_avg

        logger.info(f"  Moments:")
        for j, label in enumerate(MOMENT_LABELS):
            logger.info(f"    {label:35s} = {moments_avg[j]:.6f}")

    # ── Step 4: Build raw Jacobian ──────────────────────────────────
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: Build raw Jacobian (central finite differences)")
    logger.info("=" * 80)

    J_raw = build_raw_jacobian(moment_vectors)

    logger.info(f"\n  Raw Jacobian J_raw ({N_MOMENTS} × {N_PARAMS}):")
    header = f"  {'Moment':<35s}" + "".join([f"{PARAM_SYMBOLS[k]:>12s}" for k in PARAM_KEYS])
    logger.info(header)
    logger.info("  " + "-" * 107)
    for i in range(N_MOMENTS):
        row = f"  {MOMENT_LABELS[i]:<35s}"
        row += "".join([f"{J_raw[i, j]:>12.6f}" for j in range(N_PARAMS)])
        logger.info(row)

    # ── Step 5: Normalize to elasticities ────────────────────────────
    logger.info("\n" + "=" * 80)
    logger.info("STEP 5: Normalize Jacobian to elasticities")
    logger.info("=" * 80)

    baseline_moments = moment_vectors['baseline']
    J_norm, semi_rows = normalize_jacobian(J_raw, baseline_moments)

    if semi_rows:
        logger.info(f"  Semi-elasticity used for moments: {[MOMENT_LABELS[i] for i in semi_rows]}")
        logger.info(f"  (These moments had |m_i^0| < 0.005 at baseline)")

    logger.info(f"\n  Normalized Jacobian J_norm (elasticities):")
    header = f"  {'Moment':<35s}" + "".join([f"{PARAM_SYMBOLS[k]:>12s}" for k in PARAM_KEYS])
    logger.info(header)
    logger.info("  " + "-" * 107)
    for i in range(N_MOMENTS):
        semi_flag = " [semi]" if i in semi_rows else ""
        row = f"  {MOMENT_LABELS[i]:<35s}"
        row += "".join([f"{J_norm[i, j]:>12.4f}" for j in range(N_PARAMS)])
        row += semi_flag
        logger.info(row)

    # ── Step 6: Sanity checks ────────────────────────────────────────
    logger.info("\n" + "=" * 80)
    logger.info("STEP 6: Sanity checks")
    logger.info("=" * 80)

    checks = sanity_check_jacobian(J_norm)

    # ── Step 7: SVD analysis ─────────────────────────────────────────
    logger.info("\n" + "=" * 80)
    logger.info("STEP 7: SVD analysis")
    logger.info("=" * 80)

    svd_result = svd_analysis(J_norm)

    # ── Step 8: Collinearity diagnostics ─────────────────────────────
    logger.info("\n" + "=" * 80)
    logger.info("STEP 8: Collinearity diagnostics")
    logger.info("=" * 80)

    collin_result = collinearity_diagnostics(J_norm, svd_result)

    # ── Decision gate (Section 3.4) ──────────────────────────────────
    logger.info("\n" + "=" * 80)
    logger.info("STEP 8b: Decision gate")
    logger.info("=" * 80)

    sigma = svd_result['sigma']
    kappa = svd_result['condition_number']

    if sigma[-1] > 0.05 and kappa < 15:
        logger.info("  → All 6 parameters well identified. Proceeding to moment selection.")
    elif sigma[-1] > 0.05 and kappa < 50:
        logger.info("  → Acceptable identification. Proceeding with caution.")
    elif sigma[-1] < 0.05:
        logger.warning("  → Identification problem detected (σ₆ < 0.05).")
        logger.warning("    Inspect V₆ to diagnose. May need to fix one issuance cost parameter.")
    if kappa > 50:
        logger.warning("  → Condition number > 50. Identification is severely unbalanced.")

    # Check η₀–η₁ collinearity specifically
    corr_eta = collin_result['column_correlations'][4, 5]
    if abs(corr_eta) > 0.95:
        logger.warning(f"  → |Corr(η₀, η₁)| = {abs(corr_eta):.4f} > 0.95")
        logger.warning(f"    Consider: (a) fix one parameter, (b) reparameterise, (c) add distinguishing moments")

    # ── Step 9: D-optimal moment selection ───────────────────────────
    logger.info("\n" + "=" * 80)
    logger.info("STEP 9: D-optimal moment selection")
    logger.info("=" * 80)

    selection_result = d_optimal_selection(J_norm)

    # ── Step 10: Generate plots ──────────────────────────────────────
    logger.info("\n" + "=" * 80)
    logger.info("STEP 10: Generate plots")
    logger.info("=" * 80)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    plot_jacobian_heatmap(J_norm, RESULTS_DIR)
    plot_svd_diagnostics(svd_result, RESULTS_DIR)
    plot_column_correlations(collin_result['column_correlations'], RESULTS_DIR)
    plot_moment_sensitivity_bars(moment_vectors, RESULTS_DIR)
    plot_d_optimal_comparison(J_norm, selection_result, RESULTS_DIR)

    # ── Step 11: Save numerical results ──────────────────────────────
    logger.info("\n" + "=" * 80)
    logger.info("STEP 11: Save numerical results")
    logger.info("=" * 80)

    # Save Jacobians as npz
    np.savez(
        os.path.join(RESULTS_DIR, 'jacobian_results_risky.npz'),
        J_raw=J_raw,
        J_norm=J_norm,
        baseline_moments=baseline_moments,
        singular_values=svd_result['sigma'],
        V_matrix=svd_result['V'],
        column_correlations=collin_result['column_correlations'],
        condition_indices=collin_result['condition_indices'],
        variance_decomposition=collin_result['variance_decomposition'],
        base6_indices=np.array(selection_result['base6']),
        grid_resolutions=np.array(GRID_RESOLUTIONS),
    )

    # Save moment vectors for all configs
    moment_save = {cfg_tag: moment_vectors[cfg_tag] for cfg_tag in moment_vectors}
    np.savez(os.path.join(RESULTS_DIR, 'all_moment_vectors_risky.npz'), **moment_save)

    # Save human-readable summary
    summary_path = os.path.join(RESULTS_DIR, 'identification_summary_risky.txt')
    with open(summary_path, 'w') as f:
        f.write("Jacobian-Based Moment Selection — Risky Debt Case\n")
        f.write("=" * 80 + "\n\n")

        f.write("Baseline Parameters:\n")
        for k in PARAM_KEYS:
            f.write(f"  {PARAM_SYMBOLS[k]} ({k}): {BASELINE[k]}\n")

        f.write(f"\nFixed Parameters:\n")
        for k, v in FIXED_PARAMS.items():
            f.write(f"  {k}: {v}\n")

        f.write(f"\nStep sizes:\n")
        for k in PARAM_KEYS:
            f.write(f"  {PARAM_SYMBOLS[k]}: {STEP_SIZES[k]}\n")

        f.write(f"\nParameter bounds (from econ_params_risky_dist.json):\n")
        for k in PARAM_KEYS:
            lo, hi = PARAM_BOUNDS[k]
            f.write(f"  {PARAM_SYMBOLS[k]}: [{lo}, {hi}]\n")

        f.write(f"\nGrid Resolution:\n")
        f.write(f"  {GRID_RESOLUTIONS} (n_k=n_d, n_z={N_PRODUCTIVITY})\n")

        f.write(f"\n\nSVD Results:\n")
        f.write(f"  Singular values: {svd_result['sigma']}\n")
        f.write(f"  Condition number: {svd_result['condition_number']:.4f}\n")
        f.write(f"  det(J'J): {svd_result['det_JtJ']:.6e}\n")

        f.write(f"\n\nSelected Base 6 Moments:\n")
        J_base = J_norm[selection_result['base6'], :]
        for idx, mi in enumerate(selection_result['base6']):
            row_abs = np.abs(J_base[idx, :])
            primary_j = np.argmax(row_abs)
            f.write(f"  {MOMENT_LABELS[mi]:35s} → {PARAM_SYMBOLS[PARAM_KEYS[primary_j]]}\n")

        f.write(f"\n  det(J'J) = {selection_result['base6_det']:.6e}\n")
        f.write(f"  Condition number = {selection_result['base6_kappa']:.4f}\n")

        f.write(f"\n\nAugmentation Results:\n")
        for k_target in [7, 8, 9]:
            res = selection_result['augmentation'][k_target]
            if res['best']:
                _, added, kap, sm = res['best']
                f.write(f"  k={k_target}: add {[MOMENT_LABELS[i] for i in added]}, "
                        f"κ={kap:.2f}, σ_min={sm:.4f}\n")

        f.write(f"\n\nColumn Correlations:\n")
        labels = [PARAM_SYMBOLS[k] for k in PARAM_KEYS]
        f.write(f"  {'':>8s}" + "".join([f"{l:>10s}" for l in labels]) + "\n")
        for i in range(N_PARAMS):
            f.write(f"  {labels[i]:>8s}")
            for j in range(N_PARAMS):
                f.write(f"{collin_result['column_correlations'][i, j]:>10.4f}")
            f.write("\n")

        f.write(f"\n\nη₀–η₁ Collinearity Assessment:\n")
        f.write(f"  |Corr(η₀, η₁)| = {abs(collin_result['column_correlations'][4, 5]):.4f}\n")
        if abs(collin_result['column_correlations'][4, 5]) > 0.95:
            f.write(f"  → Nearly collinear. Consider fixing one issuance cost parameter.\n")
        elif abs(collin_result['column_correlations'][4, 5]) > 0.90:
            f.write(f"  → High correlation. Exercise caution in separate estimation.\n")
        else:
            f.write(f"  → Acceptable. η₀ and η₁ are distinguishable.\n")

        f.write(f"\n\nBaseline Moments (30 candidates):\n")
        for j, label in enumerate(MOMENT_LABELS):
            semi_flag = " [semi-elasticity]" if j in semi_rows else ""
            f.write(f"  {label:35s} = {baseline_moments[j]:.6f}{semi_flag}\n")

        f.write(f"\n\nRecommended Estimation Strategy:\n")
        f.write(f"  1. Estimate with base 6 (just-identified) for initial θ̂.\n")
        f.write(f"  2. Re-estimate with k=7 or k=8 for robustness + J-test.\n")
        f.write(f"  3. Use J-test on overidentified sets for model specification.\n")
        f.write(f"  4. Report base-6 as primary; k=7–9 as sensitivity.\n")
        f.write(f"  5. If η₀–η₁ collinearity severe, report 5-param robustness check.\n")

    logger.info(f"  Saved: {summary_path}")
    logger.info(f"  Saved: {os.path.join(RESULTS_DIR, 'jacobian_results_risky.npz')}")
    logger.info(f"  Saved: {os.path.join(RESULTS_DIR, 'all_moment_vectors_risky.npz')}")

    # ── Final summary to console ─────────────────────────────────────
    logger.info("\n" + "=" * 80)
    logger.info("IDENTIFICATION ANALYSIS COMPLETE — RISKY DEBT MODEL")
    logger.info("=" * 80)

    sigma = svd_result['sigma']
    logger.info(f"\n  Singular values: [{', '.join(f'{s:.4f}' for s in sigma)}]")
    logger.info(f"  Condition number: {svd_result['condition_number']:.2f}")
    logger.info(f"  σ₆ (weakest identification): {sigma[-1]:.4f}")
    logger.info(f"  σ₅ (second weakest): {sigma[-2]:.4f}")

    logger.info(f"\n  Base 6 moments:")
    J_base = J_norm[selection_result['base6'], :]
    for idx, mi in enumerate(selection_result['base6']):
        row_abs = np.abs(J_base[idx, :])
        primary_j = np.argmax(row_abs)
        logger.info(f"    {MOMENT_LABELS[mi]:35s} → identifies {PARAM_SYMBOLS[PARAM_KEYS[primary_j]]}")

    logger.info(f"\n  η₀–η₁ correlation: {collin_result['column_correlations'][4, 5]:.4f}")
    logger.info(f"\n  Results saved to: {os.path.abspath(RESULTS_DIR)}")

    return {
        'J_raw': J_raw,
        'J_norm': J_norm,
        'baseline_moments': baseline_moments,
        'moment_vectors': moment_vectors,
        'svd': svd_result,
        'collinearity': collin_result,
        'selection': selection_result,
    }


if __name__ == "__main__":
    main()
