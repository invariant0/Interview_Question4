#!/usr/bin/env python3
"""Generalised Method of Moments (GMM) estimation for the basic model.

Implements the redesigned GMM framework (see docs/GMM_revised_blueprint.tex):

  - All moment conditions expressed in terms of **observables** only
    (sales Y, capital K); unobservable TFP Z is backed out as
    z_{i,t} = y_{i,t} − θ k_{i,t}.
  - Two-step efficient GMM with identity → optimal weighting
  - 3 AR(1) productivity moments + 6 Euler equation instruments (q = 9)
  - Investment Euler residual uses the observable ratio θ·Y_{t+1}/K_{t+1}
    rather than directly referencing unobservable Z (blueprint eq. 14)
  - Firm-clustered long-run covariance matrix
  - Nelder-Mead from multiple Sobol starts
  - Hansen J-test for overidentifying restrictions (9 − 3 = 6 d.f.)
  - Two treatments: (a) known ψ₁; (b) ψ₁ = 0 misspecification
  - Monte Carlo experiment framework for bias/RMSE/coverage analysis
  - Structured result serialization (CSV, JSON, LaTeX tables)

Works directly on VFI-simulated panel data (no DL surrogate needed).

Usage::

    # Single estimation, both treatments
    python basic_GMM.py

    # Monte Carlo with 50 replications
    python basic_GMM.py --mc-replications 50

    # Single treatment only
    python basic_GMM.py --treatments known_psi1
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2
from scipy.stats.qmc import Sobol

from src.econ_models.config.economic_params import EconomicParams
from src.econ_models.config.bond_config import BondsConfig
from src.econ_models.io.file_utils import load_json_file
from src.econ_models.simulator import synthetic_data_generator
from src.econ_models.simulator.vfi.basic import VFISimulator_basic

from basic_common import (
    BASE_DIR,
    BASELINE,
    FIXED_PARAMS,
    PARAM_KEYS,
    PARAM_SYMBOLS,
    apply_burn_in,
    to_python_float,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# =========================================================================
#  §0  Parameter Definitions (Report Table 2, Section 5)
# =========================================================================

TRUE_PARAMS: Dict[str, float] = {
    "productivity_persistence": 0.600,
    "productivity_std_dev":     0.175,
    "adjustment_cost_convex":   1.005,
    "adjustment_cost_fixed":    0.030,   # calibrated, NOT estimated by GMM
}
"""True parameter values aligned with the report's Table 2 baseline."""

# ── Recovery parameters (3 only — ψ₁ is not estimable by GMM) ──
GMM_PARAM_ORDER: List[str] = [
    "productivity_persistence",      # ρ
    "productivity_std_dev",          # σ
    "adjustment_cost_convex",        # ψ₀
]
GMM_PARAM_LABELS: List[str] = ["ρ", "σ", "ψ₀"]
P_GMM: int = 3

GMM_SEARCH_BOUNDS: Dict[str, Tuple[float, float]] = {
    "productivity_persistence": (0.40, 0.80),
    "productivity_std_dev":     (0.05, 0.30),
    "adjustment_cost_convex":   (0.01, 2.00),
}
"""Search bounds for the 3 GMM-estimable parameters."""


# =========================================================================
#  §1  Panel Dimensions (Report Section 5.4, Table 34)
# =========================================================================

N_FIRMS: int = 3000
"""Number of firms in the simulated panel."""

T_PERIODS_EFF: int = 200
"""Effective time periods after burn-in."""

T_BURN: int = 200
"""Burn-in periods discarded before moment computation."""

T_RAW: int = T_PERIODS_EFF + T_BURN + 1   # = 401
"""Raw periods passed to synthetic_data_generator."""

INACTION_EPSILON: float = 1e-4
"""Threshold detecting inaction-region observations."""


# =========================================================================
#  §2  Moment Condition Counts
# =========================================================================

N_AR1_MOMENTS: int = 3
"""AR(1) moment conditions: g₁, g₂, g₃."""

Q_EULER_BASIC: int = 6
"""Euler equation instruments for basic model (eq. 17)."""

Q_TOTAL_BASIC: int = N_AR1_MOMENTS + Q_EULER_BASIC   # = 9
"""Total moment conditions."""

DF_OVERID_BASIC: int = Q_TOTAL_BASIC - P_GMM          # = 6
"""Overidentifying restrictions."""


# =========================================================================
#  §3  Treatment Configuration
# =========================================================================

@dataclasses.dataclass
class GMMTreatment:
    """Define a GMM estimation treatment.

    Attributes
    ----------
    name : str
        Short identifier (e.g. ``"known_psi1"``).
    psi1_value : float
        Fixed adjustment cost ψ₁ used in this treatment.
    condition_on_adjustment : bool
        Whether to condition the sample on active adjustment.
    description : str
        Human-readable description for logging.
    """

    name: str
    psi1_value: float
    condition_on_adjustment: bool
    description: str


TREATMENTS: Dict[str, GMMTreatment] = {
    "known_psi1": GMMTreatment(
        name="known_psi1",
        psi1_value=0.030,
        condition_on_adjustment=True,
        description="Treatment (a): ψ₁ fixed at true value, conditioned on adjustment",
    ),
    "psi1_zero_misspec": GMMTreatment(
        name="psi1_zero_misspec",
        psi1_value=0.000,
        condition_on_adjustment=False,
        description="Treatment (b): ψ₁ = 0 misspecification, all observations used",
    ),
}


# =========================================================================
#  §4  GMM Replication Result
# =========================================================================

@dataclasses.dataclass
class GMMReplicationResult:
    """Store results from a single GMM estimation replication.

    Attributes
    ----------
    replication_id : int
        Zero-based Monte Carlo replication index.
    treatment_name : str
        Treatment label (matches :class:`GMMTreatment.name`).
    theta_hat : np.ndarray
        Second-step parameter estimates, shape ``(3,)``.
    se : np.ndarray
        Asymptotic standard errors, shape ``(3,)``.
    Q_min : float
        Minimized second-step GMM objective.
    J_stat : float
        Hansen J-statistic.
    J_pvalue : float
        p-value of the J-test.
    J_df : int
        Degrees of freedom for the J-test.
    n_obs_ar1 : int
        Number of AR(1) moment observations.
    n_obs_euler : int
        Number of Euler-equation observations (after conditioning).
    wall_time : float
        Total elapsed time in seconds.
    theta_hat_step1 : np.ndarray
        First-step parameter estimates, shape ``(3,)``.
    Q_step1 : float
        First-step minimized objective.
    """

    replication_id: int
    treatment_name: str
    theta_hat: np.ndarray
    se: np.ndarray
    Q_min: float
    J_stat: float
    J_pvalue: float
    J_df: int
    n_obs_ar1: int
    n_obs_euler: int
    wall_time: float
    theta_hat_step1: np.ndarray
    Q_step1: float


# =========================================================================
#  §5  VFI Path Helpers (reused from basic_SMM)
# =========================================================================

def _golden_vfi_path() -> str:
    """Canonical path for the golden VFI solution at TRUE_PARAMS."""
    p = TRUE_PARAMS
    return (
        f"./ground_truth_basic/golden_vfi_results_"
        f"{p['productivity_persistence']}_{p['productivity_std_dev']}_"
        f"{p['adjustment_cost_convex']}_{p['adjustment_cost_fixed']}.npz"
    )


def _ensure_golden_vfi(econ_params: EconomicParams, bounds: Dict) -> str:
    """Generate the golden VFI solution if it does not exist.

    Parameters
    ----------
    econ_params : EconomicParams
        Economic parameters at the true values.
    bounds : dict
        State-space bounds (k_min, k_max, z_min, z_max).

    Returns
    -------
    str
        File path to the golden VFI ``.npz`` archive.
    """
    path = _golden_vfi_path()
    if os.path.exists(path):
        logger.info("Golden VFI already exists: %s", path)
        return path

    logger.info("Golden VFI not found — solving VFI at n_capital=3000 ...")
    from src.econ_models.config.vfi_config import load_grid_config
    from src.econ_models.vfi.basic import BasicModelVFI
    from src.econ_models.io.artifacts import save_vfi_results
    import dataclasses as dc

    config = load_grid_config(
        os.path.join(BASE_DIR, "hyperparam/prefixed/vfi_params.json"), "basic"
    )
    config = dc.replace(config, n_capital=3000, n_productivity=50)
    model = BasicModelVFI(
        econ_params, config,
        k_bounds=(bounds["k_min"], bounds["k_max"]),
    )
    vfi_solution = model.solve()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_vfi_results(vfi_solution, path)
    logger.info("Golden VFI saved: %s", path)
    return path


# =========================================================================
#  Step 1: Panel Data Generation
# =========================================================================

def generate_gmm_panel(
    econ_params: EconomicParams,
    bonds_config: Dict,
    vfi_solution: Any,
) -> Dict[str, np.ndarray]:
    """Generate VFI-simulated panel for GMM estimation.

    Parameters
    ----------
    econ_params : EconomicParams
        Structural economic parameters.
    bonds_config : dict
        Validated bonds/bounds configuration.
    vfi_solution : Any
        Pre-loaded VFI solution arrays.

    Returns
    -------
    dict
        Keys ``K_curr``, ``K_next``, ``Z_curr``, ``Z_next`` (each N × T_eff).
    """
    data_gen = synthetic_data_generator(
        econ_params_benchmark=econ_params,
        sample_bonds_config=bonds_config,
        batch_size=N_FIRMS,
        T_periods=T_RAW,
    )
    initial_states, shock_sequence = data_gen.gen()

    simulator = VFISimulator_basic(econ_params)
    simulator.load_solved_vfi_solution(vfi_solution)
    results = simulator.simulate(
        tuple(s.numpy() if hasattr(s, "numpy") else s for s in initial_states),
        shock_sequence.numpy() if hasattr(shock_sequence, "numpy") else shock_sequence,
    )
    stationary = apply_burn_in(results, T_BURN)

    logger.info(
        "Panel generated: N=%d, T_eff=%d, shape=%s",
        stationary["K_curr"].shape[0], stationary["K_curr"].shape[1],
        stationary["K_curr"].shape,
    )
    return stationary


# =========================================================================
#  Step 2: Variable Construction
# =========================================================================

def construct_gmm_variables(
    panel: Dict[str, np.ndarray],
    econ_params: EconomicParams,
) -> Dict[str, np.ndarray]:
    """Construct observable-only derived variables for GMM.

    Following the revised blueprint (§Moment Conditions), all variables are
    expressed in terms of observables: output Y_{i,t} = Z_{i,t} K_{i,t}^θ
    and capital K_{i,t}.  Log-productivity is backed out as
    z_{i,t} = y_{i,t} − θ k_{i,t}  (y=ln Y, k=ln K).

    Parameters
    ----------
    panel : dict
        Raw simulation panel with ``K_curr``, ``K_next``, ``Z_curr`` arrays.
    econ_params : EconomicParams
        Structural parameters (depreciation rate and capital share used).

    Returns
    -------
    dict
        ``(N, T)`` arrays: ``inv_rate``, ``backed_out_ln_z``, ``YK_ratio``,
        ``Y_curr``, ``K_curr``, ``K_next``.
    """
    delta = econ_params.depreciation_rate
    theta = econ_params.capital_share

    K_curr = panel["K_curr"]
    K_next = panel["K_next"]
    Z_curr = panel["Z_curr"]

    K_safe = np.maximum(K_curr, 1e-8)

    # Investment rate: i_t = (K_{t+1} - (1-δ)K_t) / K_t
    inv_rate = (K_next - (1.0 - delta) * K_curr) / K_safe

    # Observable output: Y_{i,t} = Z_{i,t} K_{i,t}^θ
    Y_curr = Z_curr * np.power(K_safe, theta)

    # Backed-out log-productivity from observables (blueprint §Productivity)
    # z_{i,t} = y_{i,t} − θ k_{i,t}  where y = ln Y, k = ln K
    ln_Y = np.log(np.maximum(Y_curr, 1e-12))
    ln_K = np.log(K_safe)
    backed_out_ln_z = ln_Y - theta * ln_K

    # Revenue-to-capital ratio: Y_t / K_t  (observable)
    YK_ratio = Y_curr / K_safe

    return {
        "inv_rate": inv_rate,
        "backed_out_ln_z": backed_out_ln_z,
        "YK_ratio": YK_ratio,
        "Y_curr": Y_curr,
        "K_curr": K_curr,
        "K_next": K_next,
    }


# =========================================================================
#  Step 3: Sample Conditioning (3-period window)
# =========================================================================

def condition_sample(
    variables: Dict[str, np.ndarray],
    treatment: GMMTreatment,
) -> Tuple[Dict[str, np.ndarray], int, np.ndarray]:
    """Align time windows and condition on adjustment status.

    Uses a 3-period window: (t-1, t, t+1) for instruments needing i_{t-1}.

    Parameters
    ----------
    variables : dict
        Derived observable arrays from :func:`construct_gmm_variables`.
    treatment : GMMTreatment
        Treatment specifying whether to condition on active adjustment.

    Returns
    -------
    aligned : dict
        Flattened 1-D arrays with suffixes ``_tm1``, ``_t``, ``_tp1``.
    n_obs : int
        Number of valid (firm, time) observations.
    firm_ids : np.ndarray
        Firm index for each valid observation (for clustering).
    """
    N, T = variables["inv_rate"].shape

    if treatment.condition_on_adjustment:
        active = np.abs(variables["inv_rate"]) >= INACTION_EPSILON
        # Need 3 consecutive active periods for instruments
        mask = active[:, :-2] & active[:, 1:-1] & active[:, 2:]  # (N, T-2)
    else:
        mask = np.ones((N, T - 2), dtype=bool)

    n_obs = int(mask.sum())
    logger.info(
        "Sample conditioning [%s]: %d / %d observations retained (%.1f%%)",
        treatment.name, n_obs, N * (T - 2),
        100.0 * n_obs / max(N * (T - 2), 1),
    )

    aligned: Dict[str, np.ndarray] = {}
    for key, arr in variables.items():
        aligned[key + "_tm1"] = arr[:, :-2][mask]   # t-1
        aligned[key + "_t"]   = arr[:, 1:-1][mask]  # t
        aligned[key + "_tp1"] = arr[:, 2:][mask]    # t+1

    # Firm indices for clustering
    firm_ids = np.broadcast_to(
        np.arange(N)[:, None], (N, T - 2)
    )[mask]

    return aligned, n_obs, firm_ids


# =========================================================================
#  Step 4: AR(1) Moment Conditions
# =========================================================================

def compute_ar1_moments(
    backed_out_ln_z: np.ndarray,
    rho: float,
    sigma: float,
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    """Compute the 3 AR(1) moment conditions on backed-out productivity.

    Following the revised blueprint (§Productivity Process Moments), the
    structural innovation is expressed using observable log-sales (y) and
    log-capital (k):

        u_{i,t+1}(ρ) = (y_{i,t+1} − θ k_{i,t+1}) − ρ (y_{i,t} − θ k_{i,t})

    Moment conditions:
        g₁: E[u_{t+1}] = 0                          (zero mean)
        g₂: E[u_{t+1} · (y_t − θ k_t)] = 0          (orthogonality)
        g₃: E[u_{t+1}² − σ²] = 0                    (variance)

    Uses ALL observations (no conditioning on adjustment).

    Parameters
    ----------
    backed_out_ln_z : (N, T)
        Backed-out log-productivity z_{i,t} = y_{i,t} − θ k_{i,t}.
    rho, sigma : candidate parameter values

    Returns
    -------
    g_bar_ar1 : (3,) averaged moments
    g_obs : (g1, g2, g3) each (N, T-1)
    firm_ids_ar1 : (n_obs_ar1,)  firm index for each obs
    """
    N, T = backed_out_ln_z.shape
    z_t   = backed_out_ln_z[:, :-1]       # (N, T-1)
    z_tp1 = backed_out_ln_z[:, 1:]        # (N, T-1)

    # Innovation: u_{i,t+1}(ρ) = z_{i,t+1} − ρ z_{i,t}
    u_tp1 = z_tp1 - rho * z_t

    g1 = u_tp1                     # E[u_{t+1}] = 0
    g2 = u_tp1 * z_t               # E[u_{t+1} · z_t] = 0  (orthogonality)
    g3 = u_tp1**2 - sigma**2       # E[u²_{t+1} − σ²] = 0  (variance)

    g_bar_ar1 = np.array([
        np.mean(g1),
        np.mean(g2),
        np.mean(g3),
    ])

    # Firm IDs for clustering (all obs used)
    firm_ids_ar1 = np.broadcast_to(
        np.arange(N)[:, None], (N, T - 1)
    ).ravel()

    return g_bar_ar1, (g1, g2, g3), firm_ids_ar1


# =========================================================================
#  Step 5: Euler Equation Residuals
# =========================================================================

def compute_euler_residuals_basic(
    aligned: Dict[str, np.ndarray],
    psi0: float,
    econ_params: EconomicParams,
    treatment: GMMTreatment,
) -> np.ndarray:
    """Compute Euler equation residuals e_{t+1}(ψ₀) using observables.

    Blueprint equation (14):
      e_{t+1} = β [θ · Y_{t+1}/K_{t+1} + (1−δ)(1 + ψ₀ i_{t+1})
                    + (ψ₀/2) i_{t+1}²] − (1 + ψ₀ i_t)

    All terms use observable quantities only (Y, K, investment rate).

    Parameters
    ----------
    aligned : dict
        Flattened observation arrays with time suffixes.
    psi0 : float
        Convex adjustment cost parameter.
    econ_params : EconomicParams
        Structural parameters (β, δ, θ).
    treatment : GMMTreatment
        Treatment configuration (provides ψ₁ offset).

    Returns
    -------
    np.ndarray
        Euler residuals, shape ``(n_obs,)``.
    """
    beta  = econ_params.discount_factor
    delta = econ_params.depreciation_rate
    theta = econ_params.capital_share

    i_t   = aligned["inv_rate_t"]
    i_tp1 = aligned["inv_rate_tp1"]

    # MPK at t+1 using observable ratio: θ · Y_{t+1} / K_{t+1}
    Y_tp1 = aligned["Y_curr_tp1"]
    K_tp1 = np.maximum(aligned["K_curr_tp1"], 1e-8)
    MPK_tp1 = theta * Y_tp1 / K_tp1

    RHS = beta * (
        MPK_tp1
        + (1.0 - delta) * (1.0 + psi0 * i_tp1)
        + (psi0 / 2.0) * i_tp1**2
    )
    LHS = 1.0 + psi0 * i_t

    e_tp1 = RHS - LHS

    # Treatment (a): account for known ψ₁ in the envelope condition
    if treatment.psi1_value > 0.0 and treatment.condition_on_adjustment:
        e_tp1 = e_tp1 - beta * treatment.psi1_value

    return e_tp1


# =========================================================================
#  Step 6: Instrument Matrix
# =========================================================================

def build_instruments_basic(
    aligned: Dict[str, np.ndarray],
) -> np.ndarray:
    """Construct instrument matrix z_t for the Euler equation.

    z_t = (1, i_t, i_{t-1}, Y_t/K_t, Y_{t-1}/K_{t-1}, ln K_t)ᵀ

    Parameters
    ----------
    aligned : dict
        Flattened observation arrays with time suffixes.

    Returns
    -------
    np.ndarray
        Instrument matrix, shape ``(n_obs, 6)``.
    """
    n_obs = len(aligned["inv_rate_t"])
    z = np.column_stack([
        np.ones(n_obs),                                         # constant
        aligned["inv_rate_t"],                                   # i_t
        aligned["inv_rate_tm1"],                                 # i_{t-1}
        aligned["YK_ratio_t"],                                   # Y_t/K_t
        aligned["YK_ratio_tm1"],                                 # Y_{t-1}/K_{t-1}
        np.log(np.maximum(aligned["K_curr_t"], 1e-8)),           # ln K_t
    ])
    return z


# =========================================================================
#  Step 7: Stacked Moment Conditions
# =========================================================================

def compute_stacked_moments(
    theta: np.ndarray,
    panel: Dict[str, np.ndarray],
    treatment: GMMTreatment,
    econ_params: EconomicParams,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Compute all 9 stacked moment conditions ḡ(θ).

    Parameters
    ----------
    theta : [rho, sigma, psi0]
    panel : raw simulation panel (N, T) arrays
    treatment : GMMTreatment
    econ_params : for fixed structural params

    Returns
    -------
    g_bar : (9,) average moment conditions
    info : dict with per-observation data for covariance computation
    """
    rho, sigma, psi0 = theta

    # ── Construct observable-only variables ──
    variables = construct_gmm_variables(panel, econ_params)
    backed_out_ln_z = variables["backed_out_ln_z"]

    # ── AR(1) moments (ALL observations, backed-out z) ──
    g_bar_ar1, g_obs_ar1, firm_ids_ar1 = compute_ar1_moments(
        backed_out_ln_z, rho, sigma
    )
    n_obs_ar1 = backed_out_ln_z[:, :-1].size

    # ── Euler moments (conditioned sample) ──
    aligned, n_obs_euler, firm_ids_euler = condition_sample(variables, treatment)

    e_tp1 = compute_euler_residuals_basic(aligned, psi0, econ_params, treatment)
    z_instruments = build_instruments_basic(aligned)

    # Per-observation Euler moment contributions: e_{t+1} ⊗ z_t
    g_euler_obs = e_tp1[:, None] * z_instruments   # (n_obs_euler, 6)
    g_bar_euler = np.mean(g_euler_obs, axis=0)     # (6,)

    # ── Stack ──
    g_bar = np.concatenate([g_bar_ar1, g_bar_euler])   # (9,)

    info = {
        "g_obs_ar1": g_obs_ar1,
        "g_euler_obs": g_euler_obs,
        "firm_ids_ar1": firm_ids_ar1,
        "firm_ids_euler": firm_ids_euler,
        "n_obs_ar1": n_obs_ar1,
        "n_obs_euler": n_obs_euler,
        "aligned": aligned,
    }
    return g_bar, info


# =========================================================================
#  Step 8: Firm-Clustered Covariance
# =========================================================================

def compute_clustered_covariance(
    g_obs_ar1: Tuple[np.ndarray, np.ndarray, np.ndarray],
    g_euler_obs: np.ndarray,
    firm_ids_ar1: np.ndarray,
    firm_ids_euler: np.ndarray,
    N_firms: int,
) -> np.ndarray:
    """Compute firm-clustered long-run covariance Ŝ of moment conditions.

    Parameters
    ----------
    g_obs_ar1 : tuple of np.ndarray
        Per-observation AR(1) moment arrays ``(g1, g2, g3)``.
    g_euler_obs : np.ndarray
        Per-observation Euler moment matrix, shape ``(n_obs_euler, 6)``.
    firm_ids_ar1 : np.ndarray
        Firm index for each AR(1) observation.
    firm_ids_euler : np.ndarray
        Firm index for each Euler-equation observation.
    N_firms : int
        Total number of firms.

    Returns
    -------
    np.ndarray
        Clustered covariance matrix Ŝ, shape ``(9, 9)``.
    """
    g1, g2, g3 = g_obs_ar1

    # Flatten AR(1) per-observation moments
    g1_flat = g1.ravel()
    g2_flat = g2.ravel()
    g3_flat = g3.ravel()

    # Per-firm sums (vectorized via bincount)
    g_firm_ar1 = np.zeros((N_firms, 3))
    counts_ar1 = np.bincount(firm_ids_ar1, minlength=N_firms).astype(float)
    g_firm_ar1[:, 0] = np.bincount(firm_ids_ar1, weights=g1_flat, minlength=N_firms)
    g_firm_ar1[:, 1] = np.bincount(firm_ids_ar1, weights=g2_flat, minlength=N_firms)
    g_firm_ar1[:, 2] = np.bincount(firm_ids_ar1, weights=g3_flat, minlength=N_firms)

    g_firm_euler = np.zeros((N_firms, Q_EULER_BASIC))
    counts_euler = np.bincount(firm_ids_euler, minlength=N_firms).astype(float)
    for j in range(Q_EULER_BASIC):
        g_firm_euler[:, j] = np.bincount(
            firm_ids_euler, weights=g_euler_obs[:, j], minlength=N_firms
        )

    # Per-firm average moment vector (9-dim)
    g_firm = np.zeros((N_firms, Q_TOTAL_BASIC))
    safe_ar1 = np.maximum(counts_ar1, 1.0)
    safe_euler = np.maximum(counts_euler, 1.0)
    g_firm[:, :3] = g_firm_ar1 / safe_ar1[:, None]
    g_firm[:, 3:] = g_firm_euler / safe_euler[:, None]

    # Cluster-robust covariance
    g_bar_firm = np.mean(g_firm, axis=0)
    demeaned = g_firm - g_bar_firm
    S_hat = (demeaned.T @ demeaned) / N_firms

    return S_hat


# =========================================================================
#  Step 9: GMM Objective
# =========================================================================

def gmm_objective(
    theta: np.ndarray,
    panel: Dict[str, np.ndarray],
    treatment: GMMTreatment,
    econ_params: EconomicParams,
    W: np.ndarray,
) -> float:
    """Evaluate the GMM quadratic objective Q(θ) = ḡ(θ)ᵀ W ḡ(θ).

    Includes soft bound enforcement via quadratic penalty.

    Parameters
    ----------
    theta : np.ndarray
        Candidate parameter vector ``[ρ, σ, ψ₀]``.
    panel : dict
        Raw simulation panel arrays.
    treatment : GMMTreatment
        Active estimation treatment.
    econ_params : EconomicParams
        Fixed structural parameters.
    W : np.ndarray
        Weighting matrix, shape ``(9, 9)``.

    Returns
    -------
    float
        Objective value (lower is better).
    """
    bounds_lo = np.array([GMM_SEARCH_BOUNDS[k][0] for k in GMM_PARAM_ORDER])
    bounds_hi = np.array([GMM_SEARCH_BOUNDS[k][1] for k in GMM_PARAM_ORDER])
    theta_clipped = np.clip(theta, bounds_lo, bounds_hi)
    penalty = 1e4 * np.sum((theta - theta_clipped) ** 2)

    if theta_clipped[1] <= 0:
        return 1e10

    try:
        g_bar, _ = compute_stacked_moments(
            theta_clipped, panel, treatment, econ_params
        )
    except Exception as e:
        logger.warning("Moment evaluation failed at θ=%s: %s", theta, e)
        return 1e10

    Q = float(g_bar @ W @ g_bar) + penalty
    return Q


# =========================================================================
#  Step 10: Two-Step Efficient GMM
# =========================================================================

def two_step_gmm(
    panel: Dict[str, np.ndarray],
    treatment: GMMTreatment,
    econ_params: EconomicParams,
    n_starts: int = 10,
) -> Dict[str, Any]:
    """Run two-step efficient GMM estimation.

    Step 1: Identity weighting matrix → first-step estimate θ̂₁.
    Step 2: Optimal W = Ŝ⁻¹ from θ̂₁ → second-step θ̂₂.

    Parameters
    ----------
    panel : dict
        Raw simulation panel arrays.
    treatment : GMMTreatment
        Active estimation treatment.
    econ_params : EconomicParams
        Fixed structural parameters.
    n_starts : int, optional
        Number of Sobol starting points for Nelder-Mead (default 10).

    Returns
    -------
    dict
        Keys: ``theta_hat``, ``Q_min``, ``W_optimal``, ``S_hat``,
        ``theta_hat_step1``, ``Q_step1``, ``info_final``, ``g_bar_final``.
    """
    bounds_lo = np.array([GMM_SEARCH_BOUNDS[k][0] for k in GMM_PARAM_ORDER])
    bounds_hi = np.array([GMM_SEARCH_BOUNDS[k][1] for k in GMM_PARAM_ORDER])

    # ==== FIRST STEP: Identity W ====
    W1 = np.eye(Q_TOTAL_BASIC)

    # Sobol starting points
    sobol = Sobol(d=P_GMM, scramble=True)
    sobol_pts = bounds_lo + sobol.random(n_starts) * (bounds_hi - bounds_lo)

    # Also include true values as sanity check
    true_start = np.array([TRUE_PARAMS[k] for k in GMM_PARAM_ORDER])
    starts = np.vstack([true_start, sobol_pts])

    best_Q1 = np.inf
    best_theta1 = None

    logger.info("Step 1: Identity W, %d starting points", len(starts))
    for i, x0 in enumerate(starts):
        result = minimize(
            gmm_objective,
            x0,
            args=(panel, treatment, econ_params, W1),
            method="Nelder-Mead",
            options={"xatol": 1e-6, "fatol": 1e-10, "maxiter": 5000, "adaptive": True},
        )
        Q_val = result.fun
        marker = ""
        if Q_val < best_Q1:
            best_Q1 = Q_val
            best_theta1 = np.clip(result.x, bounds_lo, bounds_hi)
            marker = " ★"
        logger.info(
            "  Start %2d/%d: Q=%.6e  θ=[%.4f, %.4f, %.4f]%s",
            i + 1, len(starts), Q_val, *np.clip(result.x, bounds_lo, bounds_hi), marker,
        )

    if best_theta1 is None:
        raise RuntimeError("All first-step optimizations failed.")

    logger.info(
        "Step 1 estimate: θ̂₁=[%.5f, %.5f, %.5f], Q₁=%.6e",
        *best_theta1, best_Q1,
    )

    # ==== COMPUTE OPTIMAL WEIGHTING MATRIX ====
    g_bar1, info1 = compute_stacked_moments(
        best_theta1, panel, treatment, econ_params
    )

    S_hat = compute_clustered_covariance(
        info1["g_obs_ar1"], info1["g_euler_obs"],
        info1["firm_ids_ar1"], info1["firm_ids_euler"],
        N_FIRMS,
    )

    # Regularize if ill-conditioned
    cond = np.linalg.cond(S_hat)
    if cond > 1e6:
        eps = 1e-6 * np.trace(S_hat) / Q_TOTAL_BASIC
        S_hat += eps * np.eye(Q_TOTAL_BASIC)
        logger.warning("Ŝ regularized (cond was %.1e, ε=%.2e)", cond, eps)

    W2 = np.linalg.inv(S_hat)
    logger.info("Optimal W computed (cond(Ŝ)=%.1f)", np.linalg.cond(S_hat))

    # ==== SECOND STEP: Optimal W ====
    logger.info("Step 2: Optimal W, refining from θ̂₁")
    result2 = minimize(
        gmm_objective,
        best_theta1,
        args=(panel, treatment, econ_params, W2),
        method="Nelder-Mead",
        options={"xatol": 1e-8, "fatol": 1e-12, "maxiter": 10000, "adaptive": True},
    )

    theta_hat_2 = np.clip(result2.x, bounds_lo, bounds_hi)
    Q_min = result2.fun

    logger.info(
        "Step 2 estimate: θ̂₂=[%.5f, %.5f, %.5f], Q₂=%.6e",
        *theta_hat_2, Q_min,
    )

    # Recompute info at final estimate
    g_bar_final, info_final = compute_stacked_moments(
        theta_hat_2, panel, treatment, econ_params
    )

    return {
        "theta_hat": theta_hat_2,
        "Q_min": Q_min,
        "W_optimal": W2,
        "S_hat": S_hat,
        "theta_hat_step1": best_theta1,
        "Q_step1": best_Q1,
        "info_final": info_final,
        "g_bar_final": g_bar_final,
    }


# =========================================================================
#  Step 11: Inference — Standard Errors and J-Test
# =========================================================================

def compute_gmm_inference(
    theta_hat: np.ndarray,
    panel: Dict[str, np.ndarray],
    treatment: GMMTreatment,
    econ_params: EconomicParams,
    W: np.ndarray,
    S_hat: np.ndarray,
    n_obs: int,
    h_rel: float = 0.01,
) -> Dict[str, Any]:
    """Compute GMM standard errors (sandwich formula) and Hansen J-test.

    Parameters
    ----------
    theta_hat : np.ndarray
        Second-step parameter estimates, shape ``(3,)``.
    panel : dict
        Raw simulation panel arrays.
    treatment : GMMTreatment
        Active estimation treatment.
    econ_params : EconomicParams
        Fixed structural parameters.
    W : np.ndarray
        Optimal weighting matrix, shape ``(9, 9)``.
    S_hat : np.ndarray
        Clustered covariance of moment conditions, shape ``(9, 9)``.
    n_obs : int
        Effective observation count for variance scaling.
    h_rel : float, optional
        Relative step size for central-difference Jacobian (default 0.01).

    Returns
    -------
    dict
        Keys: ``se``, ``Sigma``, ``G``, ``J_stat``, ``J_pvalue``, ``J_df``.
    """
    # ── Jacobian G = ∂ḡ/∂θᵀ via central differences ──
    G = np.zeros((Q_TOTAL_BASIC, P_GMM))

    for j in range(P_GMM):
        h_j = max(h_rel * abs(theta_hat[j]), 1e-5)
        theta_plus = theta_hat.copy()
        theta_minus = theta_hat.copy()
        theta_plus[j]  += h_j
        theta_minus[j] -= h_j

        g_plus, _  = compute_stacked_moments(theta_plus, panel, treatment, econ_params)
        g_minus, _ = compute_stacked_moments(theta_minus, panel, treatment, econ_params)
        G[:, j] = (g_plus - g_minus) / (2.0 * h_j)

    logger.info("Jacobian G computed: shape %s", G.shape)

    # ── Standard errors: Var(θ̂) = (1/n) (GᵀŴG)⁻¹ ──
    GtWG = G.T @ W @ G
    try:
        GtWG_inv = np.linalg.inv(GtWG)
    except np.linalg.LinAlgError:
        logger.warning("G'WG singular — using pseudo-inverse")
        GtWG_inv = np.linalg.pinv(GtWG)

    Sigma = (1.0 / n_obs) * GtWG_inv
    se = np.sqrt(np.abs(np.diag(Sigma)))
    logger.info("Standard errors: %s", np.array2string(se, precision=6))

    # ── Hansen J-test: J = n · ḡ(θ̂)ᵀ Ŵ ḡ(θ̂) ~ χ²(q - p) ──
    g_bar_hat, _ = compute_stacked_moments(theta_hat, panel, treatment, econ_params)
    J_stat = float(n_obs * g_bar_hat @ W @ g_bar_hat)
    df = DF_OVERID_BASIC
    p_value = 1.0 - chi2.cdf(J_stat, df)

    logger.info("J-test: J=%.4f, df=%d, p-value=%.4f", J_stat, df, p_value)

    return {
        "se": se,
        "Sigma": Sigma,
        "G": G,
        "J_stat": J_stat,
        "J_pvalue": p_value,
        "J_df": df,
    }


# =========================================================================
#  Step 12: Single Estimation Run
# =========================================================================

def run_single_gmm_estimation(
    econ_params: EconomicParams,
    bonds_config: Dict,
    vfi_solution: Any,
    treatment: GMMTreatment,
    replication_id: int = 0,
    n_starts: int = 10,
) -> GMMReplicationResult:
    """Run one complete GMM estimation (data generation + two-step + inference).

    Parameters
    ----------
    econ_params : EconomicParams
        Structural parameters at true values.
    bonds_config : dict
        Validated bonds/bounds configuration.
    vfi_solution : Any
        Pre-loaded VFI solution arrays.
    treatment : GMMTreatment
        Treatment specification.
    replication_id : int, optional
        Monte Carlo replication index (default 0).
    n_starts : int, optional
        Sobol starting-point count for Nelder-Mead (default 10).

    Returns
    -------
    GMMReplicationResult
        Complete result object with estimates, SEs, and test statistics.
    """
    t0 = time.perf_counter()

    logger.info(
        "\n[Rep %d | %s] Starting GMM estimation ...",
        replication_id, treatment.name,
    )
    logger.info("  %s", treatment.description)

    # Step 1: Generate panel
    panel = generate_gmm_panel(econ_params, bonds_config, vfi_solution)

    # Steps 2–10: Two-step GMM
    gmm_result = two_step_gmm(panel, treatment, econ_params, n_starts=n_starts)
    theta_hat = gmm_result["theta_hat"]

    # Determine effective observation count for inference
    # Use the Euler obs count (the conditioned sample)
    n_obs = gmm_result["info_final"]["n_obs_euler"]

    # Step 11: Inference
    inference = compute_gmm_inference(
        theta_hat, panel, treatment, econ_params,
        gmm_result["W_optimal"], gmm_result["S_hat"], n_obs,
    )

    wall_time = time.perf_counter() - t0

    # ── Log parameter recovery ──
    logger.info("\n" + "=" * 72)
    logger.info("GMM PARAMETER RECOVERY (Rep %d, %s)", replication_id, treatment.name)
    logger.info(
        "%-10s  %10s  %10s  %10s  %10s  %10s",
        "Param", "True", "Estimate", "SE", "Error", "Error%",
    )
    logger.info("-" * 72)
    for j, key in enumerate(GMM_PARAM_ORDER):
        true_val = TRUE_PARAMS[key]
        est_val = theta_hat[j]
        err = est_val - true_val
        pct_err = 100.0 * err / true_val if abs(true_val) > 1e-10 else 0.0
        logger.info(
            "%-10s  %10.5f  %10.5f  %10.5f  %+10.5f  %+9.2f%%",
            GMM_PARAM_LABELS[j], true_val, est_val, inference["se"][j], err, pct_err,
        )
    logger.info("Q(θ̂) = %.6e", gmm_result["Q_min"])
    logger.info(
        "J-stat = %.4f  (df=%d, p=%.4f)",
        inference["J_stat"], inference["J_df"], inference["J_pvalue"],
    )
    logger.info("Wall time = %.1f s", wall_time)
    logger.info("=" * 72)

    return GMMReplicationResult(
        replication_id=replication_id,
        treatment_name=treatment.name,
        theta_hat=theta_hat,
        se=inference["se"],
        Q_min=gmm_result["Q_min"],
        J_stat=inference["J_stat"],
        J_pvalue=inference["J_pvalue"],
        J_df=inference["J_df"],
        n_obs_ar1=gmm_result["info_final"]["n_obs_ar1"],
        n_obs_euler=gmm_result["info_final"]["n_obs_euler"],
        wall_time=wall_time,
        theta_hat_step1=gmm_result["theta_hat_step1"],
        Q_step1=gmm_result["Q_step1"],
    )


# =========================================================================
#  Step 13: Monte Carlo Aggregation
# =========================================================================

def aggregate_gmm_results(
    results: List[GMMReplicationResult],
) -> Dict[str, Any]:
    """Aggregate Monte Carlo replications into summary statistics.

    Parameters
    ----------
    results : list of GMMReplicationResult
        Individual replication results.

    Returns
    -------
    dict
        Summary with bias, RMSE, coverage, J-test rejection rate, etc.
    """
    R = len(results)
    thetas = np.array([r.theta_hat for r in results])     # (R, 3)
    ses = np.array([r.se for r in results])                # (R, 3)
    true_vals = np.array([TRUE_PARAMS[k] for k in GMM_PARAM_ORDER])

    median_theta = np.median(thetas, axis=0)
    mean_se = np.mean(ses, axis=0)
    sd_theta = np.std(thetas, axis=0, ddof=1) if R > 1 else np.zeros(P_GMM)

    bias = np.mean(thetas, axis=0) - true_vals
    bias_pct = 100.0 * bias / true_vals

    rmse = np.sqrt(np.mean((thetas - true_vals) ** 2, axis=0))
    rmse_pct = 100.0 * rmse / true_vals

    # 95% CI coverage
    coverage = np.zeros(P_GMM)
    for j in range(P_GMM):
        lo = thetas[:, j] - 1.96 * ses[:, j]
        hi = thetas[:, j] + 1.96 * ses[:, j]
        coverage[j] = np.mean((true_vals[j] >= lo) & (true_vals[j] <= hi))

    # J-test rejection rate
    j_pvalues = np.array([r.J_pvalue for r in results])
    rejection_rate = float(np.mean(j_pvalues < 0.05))

    mean_Q = float(np.mean([r.Q_min for r in results]))
    mean_time = float(np.mean([r.wall_time for r in results]))
    treatment_name = results[0].treatment_name

    summary: Dict[str, Any] = {
        "R": R,
        "treatment": treatment_name,
        "param_labels": GMM_PARAM_LABELS,
        "param_keys": GMM_PARAM_ORDER,
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
        "mean_wall_time": mean_time,
    }

    # Log summary
    logger.info("\n" + "=" * 80)
    logger.info("GMM MONTE CARLO SUMMARY  [%s]  (%d replications)", treatment_name, R)
    logger.info("=" * 80)
    logger.info(
        "%-10s  %8s  %8s  %8s  %8s  %8s  %8s  %8s",
        "Param", "True", "Median", "SE_avg", "SD(θ̂)",
        "Bias%", "RMSE%", "Cover95",
    )
    logger.info("-" * 80)
    for j in range(P_GMM):
        logger.info(
            "%-10s  %8.4f  %8.4f  %8.4f  %8.4f  %+7.2f%%  %7.2f%%  %7.1f%%",
            GMM_PARAM_LABELS[j], true_vals[j], median_theta[j],
            mean_se[j], sd_theta[j], bias_pct[j], rmse_pct[j],
            100.0 * coverage[j],
        )
    logger.info("-" * 80)
    logger.info("J-test rejection rate (5%%): %.1f%%", 100.0 * rejection_rate)
    logger.info("Mean Q(θ̂): %.6e", mean_Q)
    logger.info("Mean wall time: %.1f s", mean_time)
    logger.info("=" * 80)

    return summary


# =========================================================================
#  Step 14: Moment Condition Diagnostics
# =========================================================================

def compute_moment_diagnostics(
    theta_hat: np.ndarray,
    panel: Dict[str, np.ndarray],
    treatment: GMMTreatment,
    econ_params: EconomicParams,
    W: np.ndarray,
) -> Dict[str, Any]:
    """Compute per-moment diagnostics at θ̂.

    Parameters
    ----------
    theta_hat : np.ndarray
        Estimated parameter vector, shape ``(3,)``.
    panel : dict
        Raw simulation panel arrays.
    treatment : GMMTreatment
        Active estimation treatment.
    econ_params : EconomicParams
        Fixed structural parameters.
    W : np.ndarray
        Weighting matrix, shape ``(9, 9)``.

    Returns
    -------
    dict
        Per-moment names, values, and absolute values.
    """
    g_bar, _ = compute_stacked_moments(theta_hat, panel, treatment, econ_params)

    moment_names = [
        "AR1: E[u]",
        "AR1: E[u·z_t]",
        "AR1: E[u²−σ²]",
        "Euler×1",
        "Euler×i_t",
        "Euler×i_{t-1}",
        "Euler×(Y/K)_t",
        "Euler×(Y/K)_{t-1}",
        "Euler×lnK_t",
    ]

    diagnostics = {
        "moment_names": moment_names,
        "g_bar": g_bar.tolist(),
        "g_bar_abs": np.abs(g_bar).tolist(),
    }

    logger.info("\nMOMENT CONDITION VALUES AT θ̂:")
    logger.info("%-25s  %12s", "Moment", "ḡ(θ̂)")
    logger.info("-" * 40)
    for i, name in enumerate(moment_names):
        logger.info("%-25s  %+12.6e", name, g_bar[i])

    return diagnostics


# =========================================================================
#  Step 15: Serialization and Output
# =========================================================================

def save_replication_csv(
    results: List[GMMReplicationResult],
    output_path: str,
) -> None:
    """Save per-replication results to CSV.

    Parameters
    ----------
    results : list of GMMReplicationResult
        Individual replication results to serialize.
    output_path : str
        Destination CSV file path.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fieldnames = [
        "replication_id", "treatment",
        "rho_hat", "sigma_hat", "psi0_hat",
        "se_rho", "se_sigma", "se_psi0",
        "Q_min", "J_stat", "J_pvalue", "J_df",
        "n_obs_ar1", "n_obs_euler", "wall_time",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "replication_id": r.replication_id,
                "treatment": r.treatment_name,
                "rho_hat": r.theta_hat[0],
                "sigma_hat": r.theta_hat[1],
                "psi0_hat": r.theta_hat[2],
                "se_rho": r.se[0],
                "se_sigma": r.se[1],
                "se_psi0": r.se[2],
                "Q_min": r.Q_min,
                "J_stat": r.J_stat,
                "J_pvalue": r.J_pvalue,
                "J_df": r.J_df,
                "n_obs_ar1": r.n_obs_ar1,
                "n_obs_euler": r.n_obs_euler,
                "wall_time": r.wall_time,
            })
    logger.info("Replication CSV saved: %s", output_path)


def save_summary_json(summary: Dict[str, Any], output_path: str) -> None:
    """Save Monte Carlo summary dictionary to a JSON file.

    Parameters
    ----------
    summary : dict
        Aggregated summary statistics.
    output_path : str
        Destination JSON file path.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary JSON saved: %s", output_path)


# ── LaTeX Table Generators ──

def generate_latex_table35(summary: Dict[str, Any]) -> str:
    r"""Generate LaTeX for Table 35: Basic Model GMM Recovery with Known ψ₁.

    Parameters
    ----------
    summary : dict
        Aggregated MC summary for the *known_psi1* treatment.

    Returns
    -------
    str
        Complete LaTeX table source.
    """
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{GMM Parameter Recovery --- Basic Model, Known $\psi_1$}",
        r"\label{tab:gmm_basic_known_psi1}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Parameter & True & Median $\hat{\theta}$ & SE & Bias (\%) & RMSE (\%) & Coverage (95\%) \\",
        r"\midrule",
    ]
    for j in range(P_GMM):
        lines.append(
            f"${GMM_PARAM_LABELS[j]}$ & "
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


def generate_latex_table36(summary: Dict[str, Any]) -> str:
    r"""Generate LaTeX for Table 36: GMM Recovery under ψ₁ = 0 Misspecification.

    Parameters
    ----------
    summary : dict
        Aggregated MC summary for the *psi1_zero_misspec* treatment.

    Returns
    -------
    str
        Complete LaTeX table source.
    """
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{GMM Parameter Recovery --- Basic Model, $\psi_1 = 0$ Misspecification}",
        r"\label{tab:gmm_basic_psi1_zero}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Parameter & True & Median $\hat{\theta}$ & SE & Bias (\%) & RMSE (\%) & Coverage (95\%) \\",
        r"\midrule",
    ]
    for j in range(P_GMM):
        lines.append(
            f"${GMM_PARAM_LABELS[j]}$ & "
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


def generate_latex_table34(summaries: Dict[str, Dict]) -> str:
    r"""Generate LaTeX for Table 34: GMM Monte Carlo Experimental Design.

    Parameters
    ----------
    summaries : dict
        Mapping of treatment names to their MC summary dicts.

    Returns
    -------
    str
        Complete LaTeX table source.
    """
    R = max(s["R"] for s in summaries.values())
    mean_time = np.mean([s["mean_wall_time"] for s in summaries.values()])
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{GMM Monte Carlo Experimental Design --- Basic Model}",
        r"\label{tab:gmm_basic_mc_design}",
        r"\begin{tabular}{lc}",
        r"\toprule",
        r"Setting & Value \\",
        r"\midrule",
        f"$N$ (firms) & {N_FIRMS} \\\\",
        f"$T$ (effective periods) & {T_PERIODS_EFF} \\\\",
        f"Burn-in & {T_BURN} \\\\",
        f"Recovery parameters & {P_GMM} ($\\rho$, $\\sigma$, $\\psi_0$) \\\\",
        f"Total moment conditions & {Q_TOTAL_BASIC} \\\\",
        f"Overid. restrictions & {DF_OVERID_BASIC} \\\\",
        f"Treatments & {len(summaries)} \\\\",
        f"MC replications ($R$) & {R} \\\\",
        f"Mean wall time per run (s) & {mean_time:.1f} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def generate_latex_table39(
    gmm_summary: Dict[str, Any],
    smm_summary: Dict[str, Any],
) -> str:
    r"""Generate LaTeX for Table 39: GMM versus SMM Recovery comparison.

    Parameters
    ----------
    gmm_summary : dict
        Aggregated GMM MC summary (known ψ₁ treatment).
    smm_summary : dict
        Aggregated SMM MC summary.

    Returns
    -------
    str
        Complete LaTeX table source.
    """
    # Shared parameters: ρ, σ, ψ₀ (indices 0, 1, 2 in both orderings)
    # In SMM, param order is [ρ, σ, ξ, F] → shared indices [0, 1, 2]
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{GMM vs.\ SMM Recovery of Shared Parameters --- Basic Model}",
        r"\label{tab:gmm_vs_smm_basic}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Parameter & True & GMM Median & SMM Median & $|\text{Bias}_{\text{GMM}}|$ (\%) & $|\text{Bias}_{\text{SMM}}|$ (\%) \\",
        r"\midrule",
    ]
    for j in range(P_GMM):
        true_val = gmm_summary["true_values"][j]
        gmm_med = gmm_summary["median_theta"][j]
        smm_med = smm_summary["median_theta"][j]
        gmm_bias = abs(gmm_summary["bias_pct"][j])
        smm_bias = abs(smm_summary["bias_pct"][j])
        lines.append(
            f"${GMM_PARAM_LABELS[j]}$ & "
            f"{true_val:.4f} & "
            f"{gmm_med:.4f} & "
            f"{smm_med:.4f} & "
            f"{gmm_bias:.2f} & "
            f"{smm_bias:.2f} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def save_gmm_results(
    all_results: Dict[str, List[GMMReplicationResult]],
    all_summaries: Dict[str, Dict[str, Any]],
    output_dir: str,
    smm_summary: Optional[Dict[str, Any]] = None,
) -> None:
    """Save all GMM results: CSVs, JSONs, and LaTeX tables.

    Parameters
    ----------
    all_results : dict
        Mapping of treatment name to list of :class:`GMMReplicationResult`.
    all_summaries : dict
        Mapping of treatment name to aggregated summary dict.
    output_dir : str
        Directory in which to write output files.
    smm_summary : dict, optional
        SMM summary for cross-method comparison tables.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Per-treatment: CSV and summary JSON
    for treatment_name, results in all_results.items():
        save_replication_csv(
            results,
            os.path.join(output_dir, f"replications_{treatment_name}.csv"),
        )

    for treatment_name, summary in all_summaries.items():
        save_summary_json(
            summary,
            os.path.join(output_dir, f"summary_{treatment_name}.json"),
        )

    # Combined summary JSON
    combined = {name: summ for name, summ in all_summaries.items()}
    save_summary_json(combined, os.path.join(output_dir, "summary_all.json"))

    # LaTeX tables
    latex_parts = [
        "% Auto-generated LaTeX tables — basic model GMM\n",
        "% Generated by basic_GMM.py\n\n",
    ]

    # Table 34: MC Design
    latex_parts.append("% Table 34: MC Design\n")
    latex_parts.append(generate_latex_table34(all_summaries) + "\n\n")

    # Table 35: Known ψ₁
    if "known_psi1" in all_summaries:
        latex_parts.append("% Table 35: GMM Recovery — Known ψ₁\n")
        latex_parts.append(generate_latex_table35(all_summaries["known_psi1"]) + "\n\n")

    # Table 36: ψ₁ = 0 Misspecification
    if "psi1_zero_misspec" in all_summaries:
        latex_parts.append("% Table 36: GMM Recovery — ψ₁ = 0 Misspecification\n")
        latex_parts.append(generate_latex_table36(all_summaries["psi1_zero_misspec"]) + "\n\n")

    # Table 39: GMM vs SMM comparison
    if smm_summary is not None and "known_psi1" in all_summaries:
        latex_parts.append("% Table 39: GMM vs SMM Comparison\n")
        latex_parts.append(
            generate_latex_table39(all_summaries["known_psi1"], smm_summary) + "\n"
        )

    latex_path = os.path.join(output_dir, "latex_tables.tex")
    with open(latex_path, "w") as f:
        f.write("".join(latex_parts))
    logger.info("LaTeX tables saved: %s", latex_path)

    # GMM vs SMM comparison JSON
    if smm_summary is not None and "known_psi1" in all_summaries:
        gmm_s = all_summaries["known_psi1"]
        comparison = {
            "param_labels": GMM_PARAM_LABELS,
            "true_values": gmm_s["true_values"],
            "gmm_median_theta": gmm_s["median_theta"],
            "smm_median_theta": smm_summary["median_theta"][:P_GMM],
            "gmm_bias_pct": gmm_s["bias_pct"],
            "smm_bias_pct": smm_summary["bias_pct"][:P_GMM],
            "gmm_rmse_pct": gmm_s["rmse_pct"],
            "smm_rmse_pct": smm_summary["rmse_pct"][:P_GMM],
            "gmm_j_rejection": gmm_s["j_rejection_rate"],
            "smm_j_rejection": smm_summary.get("j_rejection_rate", None),
        }
        comp_path = os.path.join(output_dir, "gmm_vs_smm_comparison.json")
        with open(comp_path, "w") as f:
            json.dump(comparison, f, indent=2)
        logger.info("GMM vs SMM comparison saved: %s", comp_path)


# =========================================================================
#  Main
# =========================================================================

def main() -> None:
    """Run the complete GMM estimation pipeline."""
    parser = argparse.ArgumentParser(
        description="GMM estimation via Euler equations — basic model"
    )
    parser.add_argument(
        "--mc-replications", type=int, default=1,
        help="Number of Monte Carlo replications (default: 1)",
    )
    parser.add_argument(
        "--treatments", nargs="+",
        choices=["known_psi1", "psi1_zero_misspec", "both"],
        default=["both"],
        help="Which treatments to run",
    )
    parser.add_argument(
        "--n-starts", type=int, default=10,
        help="Number of Sobol starting points for Nelder-Mead (default: 10)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./results/gmm_basic",
        help="Output directory",
    )
    parser.add_argument(
        "--seed", type=int, default=2026,
        help="Master random seed",
    )
    parser.add_argument(
        "--smm-results-dir", type=str, default="./results/smm_basic",
        help="Path to SMM results for comparison table",
    )
    args = parser.parse_args()

    master_rng = np.random.default_rng(args.seed)

    # Resolve treatments
    if "both" in args.treatments:
        selected_treatments = list(TREATMENTS.values())
    else:
        selected_treatments = [TREATMENTS[t] for t in args.treatments]

    logger.info("=" * 72)
    logger.info("GMM ESTIMATION — BASIC MODEL (Report Section 5)")
    logger.info("=" * 72)
    logger.info("  MC replications  : %d", args.mc_replications)
    logger.info("  Treatments       : %s", [t.name for t in selected_treatments])
    logger.info("  Sobol starts     : %d", args.n_starts)
    logger.info("  Panel            : N=%d, T_eff=%d (T_raw=%d, burn=%d)",
                N_FIRMS, T_PERIODS_EFF, T_RAW, T_BURN)
    logger.info("  Moments          : q=%d (AR1=%d + Euler=%d), p=%d, overid=%d",
                Q_TOTAL_BASIC, N_AR1_MOMENTS, Q_EULER_BASIC, P_GMM, DF_OVERID_BASIC)
    logger.info("  Output dir       : %s", args.output_dir)
    logger.info("  TRUE_PARAMS      : %s", TRUE_PARAMS)
    logger.info("  GMM_SEARCH_BOUNDS: %s", GMM_SEARCH_BOUNDS)

    # ─── Load economic parameters ───────────────────────────────────
    p = TRUE_PARAMS
    econ_path = os.path.join(
        BASE_DIR,
        f"hyperparam/prefixed/econ_params_basic_{p['productivity_persistence']}_"
        f"{p['productivity_std_dev']}_{p['adjustment_cost_convex']}_"
        f"{p['adjustment_cost_fixed']}.json",
    )
    econ_params = EconomicParams(**load_json_file(econ_path))

    bounds_path = os.path.join(
        BASE_DIR,
        f"hyperparam/autogen/bounds_basic_{p['productivity_persistence']}_"
        f"{p['productivity_std_dev']}_{p['adjustment_cost_convex']}_"
        f"{p['adjustment_cost_fixed']}.json",
    )
    bonds_config = BondsConfig.validate_and_load(
        bounds_file=bounds_path, current_params=econ_params,
    )

    # ─── Load golden VFI solution ───────────────────────────────────
    _ensure_golden_vfi(econ_params, bonds_config)
    golden_path = _golden_vfi_path()
    vfi_solution = np.load(golden_path)
    logger.info("Golden VFI solution loaded: %s", golden_path)

    # ─── Monte Carlo loop per treatment ─────────────────────────────
    all_results: Dict[str, List[GMMReplicationResult]] = {}
    all_summaries: Dict[str, Dict[str, Any]] = {}

    for treatment in selected_treatments:
        logger.info(
            "\n%s\n  TREATMENT: %s\n  %s\n%s",
            "=" * 72, treatment.name, treatment.description, "=" * 72,
        )

        treatment_results: List[GMMReplicationResult] = []

        for rep in range(args.mc_replications):
            logger.info(
                "\n── MC REPLICATION %d / %d (%s) ──",
                rep + 1, args.mc_replications, treatment.name,
            )
            # Use different random state per replication via fresh data generation
            result = run_single_gmm_estimation(
                econ_params=econ_params,
                bonds_config=bonds_config,
                vfi_solution=vfi_solution,
                treatment=treatment,
                replication_id=rep,
                n_starts=args.n_starts,
            )
            treatment_results.append(result)

        summary = aggregate_gmm_results(treatment_results)
        all_results[treatment.name] = treatment_results
        all_summaries[treatment.name] = summary

    # ─── Load SMM results for comparison (if available) ─────────────
    smm_summary = None
    smm_json_path = os.path.join(args.smm_results_dir, "summary.json")
    if os.path.exists(smm_json_path):
        try:
            with open(smm_json_path) as f:
                smm_summary = json.load(f)
            logger.info("SMM summary loaded from %s", smm_json_path)
        except Exception as e:
            logger.warning("Failed to load SMM summary: %s", e)

    # ─── Save everything ────────────────────────────────────────────
    save_gmm_results(all_results, all_summaries, args.output_dir, smm_summary)

    # ─── Moment diagnostics for the last replication of each treatment ──
    for treatment in selected_treatments:
        if all_results[treatment.name]:
            last_result = all_results[treatment.name][-1]
            # Re-generate panel for diagnostics (same seed not guaranteed,
            # but useful for a snapshot)
            panel = generate_gmm_panel(econ_params, bonds_config, vfi_solution)
            diag = compute_moment_diagnostics(
                last_result.theta_hat, panel, treatment, econ_params,
                np.eye(Q_TOTAL_BASIC),  # Just for display purposes
            )
            diag_path = os.path.join(
                args.output_dir, f"moment_diagnostics_{treatment.name}.json"
            )
            with open(diag_path, "w") as f:
                json.dump(diag, f, indent=2)
            logger.info("Moment diagnostics saved: %s", diag_path)

    # ─── Cross-treatment comparison ─────────────────────────────────
    if len(all_summaries) == 2:
        logger.info("\n" + "=" * 72)
        logger.info("CROSS-TREATMENT COMPARISON")
        logger.info("=" * 72)
        s_a = all_summaries.get("known_psi1", {})
        s_b = all_summaries.get("psi1_zero_misspec", {})
        if s_a and s_b:
            logger.info(
                "%-10s  %10s  %10s  %10s",
                "Param", "True", "T(a) Med", "T(b) Med",
            )
            logger.info("-" * 50)
            for j in range(P_GMM):
                logger.info(
                    "%-10s  %10.4f  %10.4f  %10.4f",
                    GMM_PARAM_LABELS[j],
                    s_a["true_values"][j],
                    s_a["median_theta"][j],
                    s_b["median_theta"][j],
                )
            logger.info(
                "\nJ-test rejection: T(a)=%.1f%%  T(b)=%.1f%%",
                100 * s_a["j_rejection_rate"],
                100 * s_b["j_rejection_rate"],
            )

    logger.info("\n" + "=" * 72)
    logger.info("ALL DONE — results saved to %s", args.output_dir)
    logger.info("=" * 72)


if __name__ == "__main__":
    main()
