#!/usr/bin/env python3
"""Generalised Method of Moments (GMM) estimation for the risky debt model.

Implements the redesigned GMM framework (see docs/GMM_revised_blueprint.tex):

  - All moment conditions in terms of observables (sales Y, capital K, debt B)
  - Observable log-productivity: z_{i,t} = y_{i,t} − θ k_{i,t}
  - Risky Euler equation (blueprint eq. 16): includes (1−τ) tax wedge and
    reduced-form financing-friction shadow cost λ_t = γ₀ + γ₁(B/K) + γ₂(Y/K)
  - Two-step efficient GMM with identity → optimal weighting
  - 3 AR(1) moments + 6 Euler instruments = 9 total moments
  - 6 estimated params: ρ, σ, ψ₀ (structural) + γ₀, γ₁, γ₂ (nuisance)
  - Hansen J-test (9 − 6 = 3 overid. restrictions)
  - Two treatments: (a) known ψ₁; (b) ψ₁ = 0 misspecification
  - Monte Carlo framework for bias/RMSE/coverage analysis

Usage::

    python risky_GMM.py
    python risky_GMM.py --mc-replications 50
    python risky_GMM.py --treatments known_psi1
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
from src.econ_models.simulator.vfi.risky import VFISimulator_risky

from basic_common import apply_burn_in, to_python_float

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# =========================================================================
#  §0  Parameter Definitions — Risky Debt Model
# =========================================================================

FIXED_PARAMS: Dict[str, float] = {
    "discount_factor":              0.96,
    "capital_share":                0.60,
    "depreciation_rate":            0.15,
    "risk_free_rate":               0.04,
    "default_cost_proportional":    0.30,
    "corporate_tax_rate":           0.20,
    "collateral_recovery_fraction": 0.50,
}
"""Fixed (calibrated) structural parameters for the risky debt model."""

TRUE_PARAMS: Dict[str, float] = {
    "productivity_persistence":      0.600,
    "productivity_std_dev":          0.175,
    "adjustment_cost_convex":        1.005,
    "adjustment_cost_fixed":         0.030,    # calibrated, NOT estimated
    "equity_issuance_cost_fixed":    0.105,    # calibrated, NOT estimated
    "equity_issuance_cost_linear":   0.105,    # calibrated, NOT estimated
}
"""True parameter values — Table 2 baseline for risky model."""

# ── GMM-estimable structural parameters (3) ──
GMM_STRUCTURAL_ORDER: List[str] = [
    "productivity_persistence",      # ρ
    "productivity_std_dev",          # σ
    "adjustment_cost_convex",        # ψ₀
]
GMM_STRUCTURAL_LABELS: List[str] = ["ρ", "σ", "ψ₀"]
P_STRUCTURAL: int = 3

# ── Nuisance parameters: financing-friction wedge (3) ──
# λ_t = γ₀ + γ₁(B_t/K_t) + γ₂(Y_t/K_t)
GMM_NUISANCE_LABELS: List[str] = ["γ₀", "γ₁", "γ₂"]
P_NUISANCE: int = 3

# Full GMM parameter vector: θ = [ρ, σ, ψ₀, γ₀, γ₁, γ₂]
P_GMM: int = P_STRUCTURAL + P_NUISANCE
GMM_PARAM_LABELS: List[str] = GMM_STRUCTURAL_LABELS + GMM_NUISANCE_LABELS

GMM_SEARCH_BOUNDS_STRUCTURAL: Dict[str, Tuple[float, float]] = {
    "productivity_persistence": (0.40, 0.80),
    "productivity_std_dev":     (0.05, 0.30),
    "adjustment_cost_convex":   (0.01, 2.00),
}
"""Search bounds for the 3 structural GMM parameters."""

NUISANCE_BOUNDS: List[Tuple[float, float]] = [
    (-1.0, 1.0),    # γ₀  (intercept)
    (-1.0, 1.0),    # γ₁  (leverage coefficient)
    (-1.0, 1.0),    # γ₂  (profitability coefficient)
]
"""Search bounds for nuisance financing-wedge parameters."""


# =========================================================================
#  §1  Panel Dimensions
# =========================================================================

N_FIRMS: int = 3000
"""Number of firms in the simulated panel."""

T_PERIODS_EFF: int = 200
"""Effective time periods after burn-in."""

T_BURN: int = 200
"""Burn-in periods discarded before moment computation."""

T_RAW: int = T_PERIODS_EFF + T_BURN + 1
"""Raw periods passed to synthetic_data_generator."""

INACTION_EPSILON: float = 1e-4
"""Threshold detecting inaction-region observations."""


# =========================================================================
#  §2  Moment Condition Counts
# =========================================================================

N_AR1_MOMENTS: int = 3
"""AR(1) moment conditions: g₁, g₂, g₃."""

Q_EULER_RISKY: int = 6
"""Euler equation instruments for the risky model."""

Q_TOTAL_RISKY: int = N_AR1_MOMENTS + Q_EULER_RISKY   # = 9
"""Total moment conditions."""

DF_OVERID_RISKY: int = Q_TOTAL_RISKY - P_GMM          # = 3
"""Overidentifying restrictions (9 moments − 6 params)."""


# =========================================================================
#  §3  Treatment Configuration
# =========================================================================

@dataclasses.dataclass
class GMMTreatment:
    """Defines a GMM estimation treatment."""
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
#  §4  Replication Result
# =========================================================================

@dataclasses.dataclass
class GMMReplicationResult:
    """Results from a single risky-model GMM estimation replication."""
    replication_id: int
    treatment_name: str
    theta_hat: np.ndarray        # [ρ, σ, ψ₀, γ₀, γ₁, γ₂]
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
#  §5  VFI Path Helpers
# =========================================================================

BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))


def _build_full_params(
    overrides: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Merge fixed, baseline, and optional overrides."""
    params: Dict[str, float] = {}
    params.update(FIXED_PARAMS)
    params.update(TRUE_PARAMS)
    if overrides:
        params.update(overrides)
    return params


def _param_tag() -> str:
    """6-element filename tag: ρ_σ_ψ₀_ψ₁_η₀_η₁."""
    p = TRUE_PARAMS
    return (
        f"{p['productivity_persistence']}_{p['productivity_std_dev']}_"
        f"{p['adjustment_cost_convex']}_{p['adjustment_cost_fixed']}_"
        f"{p['equity_issuance_cost_fixed']}_{p['equity_issuance_cost_linear']}"
    )


def _golden_vfi_path() -> str:
    """Canonical path for the golden VFI solution at TRUE_PARAMS."""
    tag = _param_tag()
    return f"./ground_truth_risky/golden_vfi_risky_{tag}_560_560.npz"


def _ensure_golden_vfi(econ_params: EconomicParams, bounds: Dict) -> str:
    """Generate the golden VFI solution if it does not exist."""
    path = _golden_vfi_path()
    if os.path.exists(path):
        logger.info("Golden VFI already exists: %s", path)
        return path

    logger.info("Golden VFI not found — solving VFI at 560×560 grid ...")
    from src.econ_models.config.vfi_config import load_grid_config
    from src.econ_models.vfi.risky import RiskyModelVFI
    from src.econ_models.io.artifacts import save_vfi_results
    import dataclasses as dc

    config = load_grid_config(
        os.path.join(BASE_DIR, "hyperparam/prefixed/vfi_params.json"), "risky"
    )
    config = dc.replace(config, n_capital=560, n_debt=560, n_productivity=12)
    model = RiskyModelVFI(
        econ_params, config,
        k_bounds=(bounds["k_min"], bounds["k_max"]),
        d_bounds=(bounds.get("d_min", 0.0), bounds.get("d_max", bounds["k_max"])),
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
    """Generate VFI-simulated panel for risky-model GMM estimation.

    Returns dict with keys K_curr, K_next, Z_curr, Z_next,
    B_curr, B_next, equity_issuance, bond_price  (each N×T_eff).
    """
    data_gen = synthetic_data_generator(
        econ_params_benchmark=econ_params,
        sample_bonds_config=bonds_config,
        batch_size=N_FIRMS,
        T_periods=T_RAW,
        include_debt=True,
    )
    initial_states, shock_sequence = data_gen.gen()

    simulator = VFISimulator_risky(econ_params)
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
#  Step 2: Variable Construction (observables only)
# =========================================================================

def construct_gmm_variables(
    panel: Dict[str, np.ndarray],
    econ_params: EconomicParams,
) -> Dict[str, np.ndarray]:
    """Construct observable-only derived variables for risky-model GMM.

    Following the revised blueprint, all variables are expressed in terms of
    observables: output Y_{i,t} = Z_{i,t} K_{i,t}^θ, capital K_{i,t}, and
    debt B_{i,t}.  Log-productivity is backed out as
    z_{i,t} = y_{i,t} − θ k_{i,t}  (y=ln Y, k=ln K).

    Returns dict of (N, T) arrays: inv_rate, backed_out_ln_z, YK_ratio,
    BK_ratio, Y_curr, K_curr, K_next.
    """
    delta = econ_params.depreciation_rate
    theta = econ_params.capital_share

    K_curr = panel["K_curr"]
    K_next = panel["K_next"]
    Z_curr = panel["Z_curr"]
    B_curr = panel["B_curr"]

    K_safe = np.maximum(K_curr, 1e-8)

    # Investment rate: i_t = (K_{t+1} − (1−δ)K_t) / K_t
    inv_rate = (K_next - (1.0 - delta) * K_curr) / K_safe

    # Observable output: Y_{i,t} = Z_{i,t} K_{i,t}^θ
    Y_curr = Z_curr * np.power(K_safe, theta)

    # Backed-out log-productivity from observables (blueprint §Productivity)
    # z_{i,t} = y_{i,t} − θ k_{i,t}  where y = ln Y, k = ln K
    ln_Y = np.log(np.maximum(Y_curr, 1e-12))
    ln_K = np.log(K_safe)
    backed_out_ln_z = ln_Y - theta * ln_K

    # Observable ratios
    YK_ratio = Y_curr / K_safe        # revenue-to-capital
    BK_ratio = B_curr / K_safe        # leverage

    return {
        "inv_rate": inv_rate,
        "backed_out_ln_z": backed_out_ln_z,
        "YK_ratio": YK_ratio,
        "BK_ratio": BK_ratio,
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

    Returns
    -------
    aligned : dict   Flattened 1-D arrays with suffixes _tm1, _t, _tp1.
    n_obs : int      Number of valid observations.
    firm_ids : ndarray  Firm index for clustering.
    """
    N, T = variables["inv_rate"].shape

    if treatment.condition_on_adjustment:
        active = np.abs(variables["inv_rate"]) >= INACTION_EPSILON
        mask = active[:, :-2] & active[:, 1:-1] & active[:, 2:]
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
        aligned[key + "_tm1"] = arr[:, :-2][mask]
        aligned[key + "_t"]   = arr[:, 1:-1][mask]
        aligned[key + "_tp1"] = arr[:, 2:][mask]

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

    Following the revised blueprint (§Productivity Process Moments):

        u_{i,t+1}(ρ) = z_{i,t+1} − ρ z_{i,t}
        where  z_{i,t} = y_{i,t} − θ k_{i,t}  (observable)

    Moment conditions:
        g₁: E[u_{t+1}] = 0                     (zero mean)
        g₂: E[u_{t+1} · z_t] = 0               (orthogonality)
        g₃: E[u_{t+1}² − σ²] = 0               (variance)

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
    firm_ids_ar1 : (n_obs_ar1,)
    """
    N, T = backed_out_ln_z.shape
    z_t   = backed_out_ln_z[:, :-1]
    z_tp1 = backed_out_ln_z[:, 1:]

    u_tp1 = z_tp1 - rho * z_t

    g1 = u_tp1                     # E[u_{t+1}] = 0
    g2 = u_tp1 * z_t               # E[u_{t+1} · z_t] = 0
    g3 = u_tp1**2 - sigma**2       # E[u²_{t+1} − σ²] = 0

    g_bar_ar1 = np.array([np.mean(g1), np.mean(g2), np.mean(g3)])

    firm_ids_ar1 = np.broadcast_to(
        np.arange(N)[:, None], (N, T - 1)
    ).ravel()

    return g_bar_ar1, (g1, g2, g3), firm_ids_ar1


# =========================================================================
#  Step 5: Risky Euler Equation Residuals
# =========================================================================

def compute_euler_residuals_risky(
    aligned: Dict[str, np.ndarray],
    psi0: float,
    gamma: np.ndarray,
    econ_params: EconomicParams,
    treatment: GMMTreatment,
) -> np.ndarray:
    """Compute risky Euler residuals using observables (blueprint eq. 16).

    e_{t+1} = β[(1−τ)·θ·Y_{t+1}/K_{t+1} + (1−δ)(1+ψ₀ i_{t+1})
                  + (ψ₀/2) i_{t+1}²]
              − (1 + ψ₀ i_t + λ_t)

    where the reduced-form shadow cost is parameterized as:
        λ_t = γ₀ + γ₁(B_t/K_t) + γ₂(Y_t/K_t)

    All terms use observable quantities only (Y, K, B, investment rate).

    Returns shape (n_obs,).
    """
    beta  = econ_params.discount_factor
    delta = econ_params.depreciation_rate
    theta = econ_params.capital_share
    tau   = econ_params.corporate_tax_rate

    i_t   = aligned["inv_rate_t"]
    i_tp1 = aligned["inv_rate_tp1"]

    # MPK at t+1 using observable ratio: (1−τ)·θ·Y_{t+1}/K_{t+1}
    Y_tp1 = aligned["Y_curr_tp1"]
    K_tp1 = np.maximum(aligned["K_curr_tp1"], 1e-8)
    MPK_tp1 = (1.0 - tau) * theta * Y_tp1 / K_tp1

    RHS = beta * (
        MPK_tp1
        + (1.0 - delta) * (1.0 + psi0 * i_tp1)
        + (psi0 / 2.0) * i_tp1**2
    )

    # Shadow cost: λ_t = γ₀ + γ₁(B_t/K_t) + γ₂(Y_t/K_t)
    gamma0, gamma1, gamma2 = gamma
    lambda_t = (
        gamma0
        + gamma1 * aligned["BK_ratio_t"]
        + gamma2 * aligned["YK_ratio_t"]
    )

    LHS = 1.0 + psi0 * i_t + lambda_t

    e_tp1 = RHS - LHS

    # Treatment (a): account for known ψ₁ in the envelope condition
    if treatment.psi1_value > 0.0 and treatment.condition_on_adjustment:
        e_tp1 = e_tp1 - beta * treatment.psi1_value

    return e_tp1


# =========================================================================
#  Step 6: Instrument Matrix
# =========================================================================

def build_instruments_risky(
    aligned: Dict[str, np.ndarray],
) -> np.ndarray:
    """Construct instrument matrix z_t for the risky Euler equation.

    z_t = (1, i_t, i_{t-1}, Y_t/K_t, Y_{t-1}/K_{t-1}, ln K_t)ᵀ

    Returns shape (n_obs, 6).
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
    theta_vec: np.ndarray,
    panel: Dict[str, np.ndarray],
    treatment: GMMTreatment,
    econ_params: EconomicParams,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Compute all 9 stacked moment conditions ḡ(θ).

    Parameters
    ----------
    theta_vec : [ρ, σ, ψ₀, γ₀, γ₁, γ₂]
    panel : raw simulation panel (N, T) arrays
    treatment : GMMTreatment
    econ_params : for fixed structural parameters

    Returns
    -------
    g_bar : (9,) average moment conditions
    info : dict with per-observation data for covariance computation
    """
    rho, sigma, psi0 = theta_vec[:3]
    gamma = theta_vec[3:]

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

    e_tp1 = compute_euler_residuals_risky(
        aligned, psi0, gamma, econ_params, treatment
    )
    z_instruments = build_instruments_risky(aligned)

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
    """Compute firm-clustered long-run covariance Ŝ.  Returns shape (9, 9)."""
    g1, g2, g3 = g_obs_ar1

    g1_flat = g1.ravel()
    g2_flat = g2.ravel()
    g3_flat = g3.ravel()

    # Per-firm sums (vectorized via bincount)
    g_firm_ar1 = np.zeros((N_firms, 3))
    counts_ar1 = np.bincount(firm_ids_ar1, minlength=N_firms).astype(float)
    g_firm_ar1[:, 0] = np.bincount(firm_ids_ar1, weights=g1_flat, minlength=N_firms)
    g_firm_ar1[:, 1] = np.bincount(firm_ids_ar1, weights=g2_flat, minlength=N_firms)
    g_firm_ar1[:, 2] = np.bincount(firm_ids_ar1, weights=g3_flat, minlength=N_firms)

    q_euler = g_euler_obs.shape[1]
    g_firm_euler = np.zeros((N_firms, q_euler))
    counts_euler = np.bincount(firm_ids_euler, minlength=N_firms).astype(float)
    for j in range(q_euler):
        g_firm_euler[:, j] = np.bincount(
            firm_ids_euler, weights=g_euler_obs[:, j], minlength=N_firms
        )

    # Per-firm average moment vector (9-dim)
    q_total = 3 + q_euler
    g_firm = np.zeros((N_firms, q_total))
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

def _get_search_bounds() -> Tuple[np.ndarray, np.ndarray]:
    """Return (bounds_lo, bounds_hi) arrays for the 6-dim parameter vector."""
    struct_lo = np.array([GMM_SEARCH_BOUNDS_STRUCTURAL[k][0]
                          for k in GMM_STRUCTURAL_ORDER])
    struct_hi = np.array([GMM_SEARCH_BOUNDS_STRUCTURAL[k][1]
                          for k in GMM_STRUCTURAL_ORDER])
    nuis_lo = np.array([b[0] for b in NUISANCE_BOUNDS])
    nuis_hi = np.array([b[1] for b in NUISANCE_BOUNDS])
    return np.concatenate([struct_lo, nuis_lo]), np.concatenate([struct_hi, nuis_hi])


def gmm_objective(
    theta: np.ndarray,
    panel: Dict[str, np.ndarray],
    treatment: GMMTreatment,
    econ_params: EconomicParams,
    W: np.ndarray,
) -> float:
    """Evaluate Q(θ) = ḡ(θ)ᵀ W ḡ(θ) with soft bound penalty."""
    bounds_lo, bounds_hi = _get_search_bounds()
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
    n_starts: int = 12,
) -> Dict[str, Any]:
    """Run two-step efficient GMM for the risky debt model.

    Step 1: Identity weighting matrix → first-step estimate θ̂₁
    Step 2: Optimal W = Ŝ⁻¹ from θ̂₁ → second-step θ̂₂

    Returns dict with theta_hat, Q_min, W_optimal, S_hat, etc.
    """
    bounds_lo, bounds_hi = _get_search_bounds()

    # ==== FIRST STEP: Identity W ====
    W1 = np.eye(Q_TOTAL_RISKY)

    # Sobol starting points
    sobol = Sobol(d=P_GMM, scramble=True)
    sobol_pts = bounds_lo + sobol.random(n_starts) * (bounds_hi - bounds_lo)

    # True structural values + γ = 0 as a starting point
    true_start = np.zeros(P_GMM)
    true_start[:P_STRUCTURAL] = [TRUE_PARAMS[k] for k in GMM_STRUCTURAL_ORDER]
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
            options={"xatol": 1e-6, "fatol": 1e-10, "maxiter": 8000, "adaptive": True},
        )
        Q_val = result.fun
        marker = ""
        if Q_val < best_Q1:
            best_Q1 = Q_val
            best_theta1 = np.clip(result.x, bounds_lo, bounds_hi)
            marker = " ★"
        if i < 5 or marker:
            clipped = np.clip(result.x, bounds_lo, bounds_hi)
            logger.info(
                "  Start %2d/%d: Q=%.6e  struct=[%.4f, %.4f, %.4f]  γ=[%.4f, %.4f, %.4f]%s",
                i + 1, len(starts), Q_val,
                *clipped[:3], *clipped[3:], marker,
            )

    if best_theta1 is None:
        raise RuntimeError("All first-step optimizations failed.")

    logger.info(
        "Step 1 estimate: struct=[%.5f, %.5f, %.5f], γ=[%.5f, %.5f, %.5f], Q₁=%.6e",
        *best_theta1[:3], *best_theta1[3:], best_Q1,
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
        eps = 1e-6 * np.trace(S_hat) / Q_TOTAL_RISKY
        S_hat += eps * np.eye(Q_TOTAL_RISKY)
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
        options={"xatol": 1e-8, "fatol": 1e-12, "maxiter": 15000, "adaptive": True},
    )

    theta_hat_2 = np.clip(result2.x, bounds_lo, bounds_hi)
    Q_min = result2.fun

    logger.info(
        "Step 2 estimate: struct=[%.5f, %.5f, %.5f], γ=[%.5f, %.5f, %.5f], Q₂=%.6e",
        *theta_hat_2[:3], *theta_hat_2[3:], Q_min,
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

    Returns dict with se, Sigma, G, J_stat, J_pvalue, J_df.
    """
    # ── Jacobian G = ∂ḡ/∂θᵀ via central differences ──
    G = np.zeros((Q_TOTAL_RISKY, P_GMM))

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

    # ── Hansen J-test: J = n · ḡ(θ̂)ᵀ Ŵ ḡ(θ̂) ~ χ²(q − p) ──
    g_bar_hat, _ = compute_stacked_moments(theta_hat, panel, treatment, econ_params)
    J_stat = float(n_obs * g_bar_hat @ W @ g_bar_hat)
    df = DF_OVERID_RISKY
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
    n_starts: int = 12,
) -> GMMReplicationResult:
    """Run one complete risky-model GMM estimation."""
    t0 = time.perf_counter()

    logger.info(
        "\n[Rep %d | %s] Starting risky GMM estimation ...",
        replication_id, treatment.name,
    )
    logger.info("  %s", treatment.description)

    # Step 1: Generate panel
    panel = generate_gmm_panel(econ_params, bonds_config, vfi_solution)

    # Steps 2–10: Two-step GMM
    gmm_result = two_step_gmm(panel, treatment, econ_params, n_starts=n_starts)
    theta_hat = gmm_result["theta_hat"]

    n_obs = gmm_result["info_final"]["n_obs_euler"]

    # Step 11: Inference
    inference = compute_gmm_inference(
        theta_hat, panel, treatment, econ_params,
        gmm_result["W_optimal"], gmm_result["S_hat"], n_obs,
    )

    wall_time = time.perf_counter() - t0

    # ── Log parameter recovery ──
    logger.info("\n" + "=" * 80)
    logger.info("RISKY GMM RECOVERY (Rep %d, %s)", replication_id, treatment.name)
    logger.info(
        "%-10s  %10s  %10s  %10s  %10s",
        "Param", "True", "Estimate", "SE", "Type",
    )
    logger.info("-" * 80)
    for j in range(P_STRUCTURAL):
        key = GMM_STRUCTURAL_ORDER[j]
        true_val = TRUE_PARAMS[key]
        est_val = theta_hat[j]
        err = est_val - true_val
        pct_err = 100.0 * err / true_val if abs(true_val) > 1e-10 else 0.0
        logger.info(
            "%-10s  %10.5f  %10.5f  %10.5f  structural  err=%+.2f%%",
            GMM_STRUCTURAL_LABELS[j], true_val, est_val, inference["se"][j], pct_err,
        )
    for j in range(P_NUISANCE):
        idx = P_STRUCTURAL + j
        logger.info(
            "%-10s  %10s  %10.5f  %10.5f  nuisance",
            GMM_NUISANCE_LABELS[j], "---", theta_hat[idx], inference["se"][idx],
        )
    logger.info("Q(θ̂) = %.6e", gmm_result["Q_min"])
    logger.info(
        "J-stat = %.4f  (df=%d, p=%.4f)",
        inference["J_stat"], inference["J_df"], inference["J_pvalue"],
    )
    logger.info("Wall time = %.1f s", wall_time)
    logger.info("=" * 80)

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
    """Aggregate MC replications into summary statistics.

    Structural parameters (ρ, σ, ψ₀): full bias/RMSE/coverage statistics.
    Nuisance parameters (γ₀, γ₁, γ₂): descriptive statistics only (no true
    values available for comparison).
    """
    R = len(results)
    thetas = np.array([r.theta_hat for r in results])     # (R, 6)
    ses = np.array([r.se for r in results])                # (R, 6)
    true_structural = np.array([TRUE_PARAMS[k] for k in GMM_STRUCTURAL_ORDER])

    # ── Structural parameters: full MC statistics ──
    struct_thetas = thetas[:, :P_STRUCTURAL]
    struct_ses = ses[:, :P_STRUCTURAL]

    median_struct = np.median(struct_thetas, axis=0)
    mean_se_struct = np.mean(struct_ses, axis=0)
    sd_struct = np.std(struct_thetas, axis=0, ddof=1) if R > 1 else np.zeros(P_STRUCTURAL)

    bias = np.mean(struct_thetas, axis=0) - true_structural
    bias_pct = 100.0 * bias / true_structural
    rmse = np.sqrt(np.mean((struct_thetas - true_structural) ** 2, axis=0))
    rmse_pct = 100.0 * rmse / true_structural

    coverage = np.zeros(P_STRUCTURAL)
    for j in range(P_STRUCTURAL):
        lo = struct_thetas[:, j] - 1.96 * struct_ses[:, j]
        hi = struct_thetas[:, j] + 1.96 * struct_ses[:, j]
        coverage[j] = np.mean((true_structural[j] >= lo) & (true_structural[j] <= hi))

    # ── Nuisance parameters: descriptive statistics only ──
    nuis_thetas = thetas[:, P_STRUCTURAL:]
    nuis_ses = ses[:, P_STRUCTURAL:]
    median_nuis = np.median(nuis_thetas, axis=0)
    mean_se_nuis = np.mean(nuis_ses, axis=0)
    sd_nuis = np.std(nuis_thetas, axis=0, ddof=1) if R > 1 else np.zeros(P_NUISANCE)

    # J-test rejection rate
    j_pvalues = np.array([r.J_pvalue for r in results])
    rejection_rate = float(np.mean(j_pvalues < 0.05))

    mean_Q = float(np.mean([r.Q_min for r in results]))
    mean_time = float(np.mean([r.wall_time for r in results]))
    treatment_name = results[0].treatment_name

    summary: Dict[str, Any] = {
        "R": R,
        "treatment": treatment_name,
        "structural_labels": GMM_STRUCTURAL_LABELS,
        "nuisance_labels": GMM_NUISANCE_LABELS,
        "true_structural": true_structural.tolist(),
        "median_structural": median_struct.tolist(),
        "mean_se_structural": mean_se_struct.tolist(),
        "sd_structural": sd_struct.tolist(),
        "bias": bias.tolist(),
        "bias_pct": bias_pct.tolist(),
        "rmse": rmse.tolist(),
        "rmse_pct": rmse_pct.tolist(),
        "coverage_95": coverage.tolist(),
        "median_nuisance": median_nuis.tolist(),
        "mean_se_nuisance": mean_se_nuis.tolist(),
        "sd_nuisance": sd_nuis.tolist(),
        "j_rejection_rate": rejection_rate,
        "mean_Q": mean_Q,
        "mean_wall_time": mean_time,
    }

    # ── Log summary ──
    logger.info("\n" + "=" * 80)
    logger.info("RISKY GMM MC SUMMARY  [%s]  (%d replications)", treatment_name, R)
    logger.info("=" * 80)
    logger.info("── Structural Parameters ──")
    logger.info(
        "%-10s  %8s  %8s  %8s  %8s  %8s  %8s  %8s",
        "Param", "True", "Median", "SE_avg", "SD(θ̂)", "Bias%", "RMSE%", "Cover95",
    )
    logger.info("-" * 80)
    for j in range(P_STRUCTURAL):
        logger.info(
            "%-10s  %8.4f  %8.4f  %8.4f  %8.4f  %+7.2f%%  %7.2f%%  %7.1f%%",
            GMM_STRUCTURAL_LABELS[j], true_structural[j], median_struct[j],
            mean_se_struct[j], sd_struct[j], bias_pct[j], rmse_pct[j],
            100.0 * coverage[j],
        )
    logger.info("── Nuisance Parameters (financing-friction wedge) ──")
    logger.info(
        "%-10s  %8s  %8s  %8s",
        "Param", "Median", "SE_avg", "SD(θ̂)",
    )
    for j in range(P_NUISANCE):
        logger.info(
            "%-10s  %+8.4f  %8.4f  %8.4f",
            GMM_NUISANCE_LABELS[j], median_nuis[j], mean_se_nuis[j], sd_nuis[j],
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
    """Compute per-moment diagnostics at θ̂."""
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
    """Save per-replication results to CSV."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fieldnames = [
        "replication_id", "treatment",
        "rho_hat", "sigma_hat", "psi0_hat",
        "gamma0_hat", "gamma1_hat", "gamma2_hat",
        "se_rho", "se_sigma", "se_psi0",
        "se_gamma0", "se_gamma1", "se_gamma2",
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
                "gamma0_hat": r.theta_hat[3],
                "gamma1_hat": r.theta_hat[4],
                "gamma2_hat": r.theta_hat[5],
                "se_rho": r.se[0],
                "se_sigma": r.se[1],
                "se_psi0": r.se[2],
                "se_gamma0": r.se[3],
                "se_gamma1": r.se[4],
                "se_gamma2": r.se[5],
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
    """Save MC summary to JSON."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary JSON saved: %s", output_path)


# ── LaTeX Table Generator ──

def generate_latex_table_risky_recovery(summary: Dict[str, Any]) -> str:
    r"""LaTeX table: Risky model GMM structural + nuisance parameter recovery."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{GMM Parameter Recovery --- Risky Debt Model, "
        + summary["treatment"] + r"}",
        r"\label{tab:gmm_risky_" + summary["treatment"] + r"}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Parameter & True & Median $\hat{\theta}$ & SE & Bias (\%) & RMSE (\%) & Coverage (95\%) \\",
        r"\midrule",
        r"\multicolumn{7}{l}{\textit{Structural Parameters}} \\[2pt]",
    ]
    for j in range(P_STRUCTURAL):
        lines.append(
            f"${GMM_STRUCTURAL_LABELS[j]}$ & "
            f"{summary['true_structural'][j]:.4f} & "
            f"{summary['median_structural'][j]:.4f} & "
            f"{summary['mean_se_structural'][j]:.4f} & "
            f"{summary['bias_pct'][j]:+.2f} & "
            f"{summary['rmse_pct'][j]:.2f} & "
            f"{100 * summary['coverage_95'][j]:.1f}\\% \\\\"
        )
    lines.append(r"\midrule")
    lines.append(
        r"\multicolumn{7}{l}{\textit{Nuisance Parameters "
        r"(financing-friction wedge $\lambda_t$)}} \\[2pt]"
    )
    for j in range(P_NUISANCE):
        lines.append(
            f"${GMM_NUISANCE_LABELS[j]}$ & "
            f"--- & "
            f"{summary['median_nuisance'][j]:+.4f} & "
            f"{summary['mean_se_nuisance'][j]:.4f} & "
            f"--- & --- & --- \\\\"
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


def save_gmm_results(
    all_results: Dict[str, List[GMMReplicationResult]],
    all_summaries: Dict[str, Dict[str, Any]],
    output_dir: str,
) -> None:
    """Save all risky-model GMM results: CSVs, JSONs, and LaTeX tables."""
    os.makedirs(output_dir, exist_ok=True)

    # Per-treatment outputs
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

    # Combined summary
    combined = {name: summ for name, summ in all_summaries.items()}
    save_summary_json(combined, os.path.join(output_dir, "summary_all.json"))

    # LaTeX tables
    latex_parts = [
        "% Auto-generated LaTeX tables — risky debt model GMM\n",
        "% Generated by risky_GMM.py\n\n",
    ]
    for treatment_name, summary in all_summaries.items():
        latex_parts.append(f"% Recovery table: {treatment_name}\n")
        latex_parts.append(generate_latex_table_risky_recovery(summary) + "\n\n")

    latex_path = os.path.join(output_dir, "latex_tables.tex")
    with open(latex_path, "w") as f:
        f.write("".join(latex_parts))
    logger.info("LaTeX tables saved: %s", latex_path)


# =========================================================================
#  Main
# =========================================================================

def main() -> None:
    """Run the complete risky-model GMM estimation pipeline."""
    parser = argparse.ArgumentParser(
        description="GMM estimation via Euler equations — risky debt model"
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
        "--n-starts", type=int, default=12,
        help="Number of Sobol starting points (default: 12)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./results/gmm_risky",
        help="Output directory",
    )
    parser.add_argument(
        "--seed", type=int, default=2026,
        help="Master random seed",
    )
    args = parser.parse_args()

    master_rng = np.random.default_rng(args.seed)

    # Resolve treatments
    if "both" in args.treatments:
        selected_treatments = list(TREATMENTS.values())
    else:
        selected_treatments = [TREATMENTS[t] for t in args.treatments]

    logger.info("=" * 72)
    logger.info("GMM ESTIMATION — RISKY DEBT MODEL (blueprint §Risky)")
    logger.info("=" * 72)
    logger.info("  MC replications  : %d", args.mc_replications)
    logger.info("  Treatments       : %s", [t.name for t in selected_treatments])
    logger.info("  Sobol starts     : %d", args.n_starts)
    logger.info("  Panel            : N=%d, T_eff=%d (T_raw=%d, burn=%d)",
                N_FIRMS, T_PERIODS_EFF, T_RAW, T_BURN)
    logger.info("  Moments          : q=%d (AR1=%d + Euler=%d), p=%d, overid=%d",
                Q_TOTAL_RISKY, N_AR1_MOMENTS, Q_EULER_RISKY, P_GMM, DF_OVERID_RISKY)
    logger.info("  Output dir       : %s", args.output_dir)
    logger.info("  TRUE_PARAMS      : %s", TRUE_PARAMS)

    # ─── Load economic parameters ───────────────────────────────────
    tag = _param_tag()
    econ_path = os.path.join(
        BASE_DIR, f"hyperparam/prefixed/econ_params_risky_{tag}.json"
    )
    econ_params = EconomicParams(**load_json_file(econ_path))

    bounds_path = os.path.join(
        BASE_DIR, f"hyperparam/autogen/bounds_risky_{tag}.json"
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

    # ─── Save everything ────────────────────────────────────────────
    save_gmm_results(all_results, all_summaries, args.output_dir)

    # ─── Moment diagnostics for the last replication ────────────────
    for treatment in selected_treatments:
        if all_results[treatment.name]:
            last_result = all_results[treatment.name][-1]
            panel = generate_gmm_panel(econ_params, bonds_config, vfi_solution)
            diag = compute_moment_diagnostics(
                last_result.theta_hat, panel, treatment, econ_params,
                np.eye(Q_TOTAL_RISKY),
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
        logger.info("CROSS-TREATMENT COMPARISON (Risky)")
        logger.info("=" * 72)
        s_a = all_summaries.get("known_psi1", {})
        s_b = all_summaries.get("psi1_zero_misspec", {})
        if s_a and s_b:
            logger.info(
                "%-10s  %10s  %10s  %10s",
                "Param", "True", "T(a) Med", "T(b) Med",
            )
            logger.info("-" * 50)
            for j in range(P_STRUCTURAL):
                logger.info(
                    "%-10s  %10.4f  %10.4f  %10.4f",
                    GMM_STRUCTURAL_LABELS[j],
                    s_a["true_structural"][j],
                    s_a["median_structural"][j],
                    s_b["median_structural"][j],
                )
            for j in range(P_NUISANCE):
                logger.info(
                    "%-10s  %10s  %+10.4f  %+10.4f",
                    GMM_NUISANCE_LABELS[j], "---",
                    s_a["median_nuisance"][j],
                    s_b["median_nuisance"][j],
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
