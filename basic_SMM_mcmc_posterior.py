#!/usr/bin/env python3
"""Phase 2–4 — MVN Prior Fitting & MCMC Posterior Sampling (Basic Model).

Loads pre-computed prior moment samples (from ``basic_SMM_mcmc_prior.py``),
fits a multivariate normal distribution over the moment space, computes
golden (observed) moments via VFI, then runs MCMC posterior sampling.

Workflow:
  1. Load prior_cache.npz (moments + params from Phase 1)
  2. Fit MVN N(μ, Σ) over the moment samples
  3. Compute golden moments m* via VFI at the true parameters
  4. Build un-normalised log-posterior using DL simulator as forward model
  5. Run MCMC (adaptive random-walk Metropolis)
  6. Save posterior summary, samples, and diagnostic plots

Usage::

    # Standard run (reads prior_cache.npz from results dir)
    python basic_SMM_mcmc_posterior.py \\
        --prior-cache ./results/smm_mcmc_basic/prior_cache.npz

    # Custom MCMC settings
    python basic_SMM_mcmc_posterior.py \\
        --prior-cache ./results/smm_mcmc_basic/prior_cache.npz \\
        --n-mcmc-samples 4000 --n-chains 4 --gpu 0

    # Quick test run
    python basic_SMM_mcmc_posterior.py \\
        --prior-cache ./results/smm_mcmc_basic/prior_cache.npz \\
        --n-mcmc-samples 500 --n-mcmc-burnin 200 --n-chains 2
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import logging
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

# ── GPU must be configured BEFORE importing TensorFlow ─────────────────
def _configure_gpu_before_import() -> int:
    """Parse ``--gpu`` from sys.argv and set ``CUDA_VISIBLE_DEVICES``.

    Must be called before TensorFlow is imported, since TF locks the
    GPU list at import time.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gpu", type=int, default=0)
    known, _ = parser.parse_known_args()
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    if known.gpu < 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(known.gpu)
    return known.gpu

_GPU_ID = _configure_gpu_before_import()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
from scipy import stats as sp_stats
from scipy.stats import chi2
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

from src.econ_models.config.dl_config import load_dl_config
from src.econ_models.config.economic_params import EconomicParams
from src.econ_models.config.bond_config import BondsConfig
from src.econ_models.io.file_utils import load_json_file
from src.econ_models.simulator import (
    DLSimulatorBasicFinal_dist,
    VFISimulator_basic,
)
from src.econ_models.simulator import synthetic_data_generator
from src.econ_models.moment_calculator.compute_derived_quantities import (
    compute_all_derived_quantities,
)
from src.econ_models.moment_calculator.compute_autocorrelation import (
    compute_autocorrelation_lags_1_to_5,
)

from basic_common import (
    BASE_DIR,
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
#  Constants & Configuration
# =========================================================================

TRUE_PARAMS: Dict[str, float] = {
    "productivity_persistence": 0.600,
    "productivity_std_dev": 0.175,
    "adjustment_cost_convex": 1.005,
    "adjustment_cost_fixed": 0.030,
}

SEARCH_BOUNDS: Dict[str, Tuple[float, float]] = {
    "productivity_persistence": (0.40, 0.80),
    "productivity_std_dev": (0.05, 0.30),
    "adjustment_cost_convex": (0.01, 2.00),
    "adjustment_cost_fixed": (0.01, 0.05),
}

PARAM_ORDER: List[str] = [
    "productivity_persistence",
    "productivity_std_dev",
    "adjustment_cost_convex",
    "adjustment_cost_fixed",
]

PARAM_LABELS: List[str] = ["ρ", "σ", "ξ", "F"]

BOUNDS_LO = np.array([SEARCH_BOUNDS[k][0] for k in PARAM_ORDER])
BOUNDS_HI = np.array([SEARCH_BOUNDS[k][1] for k in PARAM_ORDER])

# Panel dimensions
N_DATA: int = 3000
T_DATA_EFF: int = 200
T_BURN: int = 200
T_DATA_RAW: int = T_DATA_EFF + T_BURN + 1

N_SIM: int = 10000
T_SIM_EFF: int = 500
T_SIM_RAW: int = T_SIM_EFF + T_BURN + 1

# Moment setup
K_MOMENTS: int = 6
P_PARAMS: int = 4

MOMENT_NAMES: List[str] = [
    "AC[I/K]",
    "Skew[I/K]",
    "P(I/K < 0)",
    "E[Y/K]",
    "SD[Y/K]",
    "Corr(I/K, lag Y/K)",
]

# MCMC defaults
DEFAULT_N_MCMC_SAMPLES: int = 4000
DEFAULT_N_MCMC_BURNIN: int = 1000
DEFAULT_N_CHAINS: int = 4
DEFAULT_HMC_STEP_SIZE: float = 0.005
DEFAULT_HMC_LEAPFROG_STEPS: int = 30
DEFAULT_DL_EPOCH: int = 1200

# File paths
DL_CONFIG_PATH: str = "./hyperparam_dist/prefixed/dl_params_dist.json"
DL_CHECKPOINT_DIR: str = "./checkpoints_final_dist/basic"
BONDS_FILE_DIST: str = os.path.join(
    BASE_DIR, "hyperparam_dist/autogen/bounds_basic_dist.json"
)
ECON_PARAMS_FILE_DIST: str = os.path.join(
    BASE_DIR, "hyperparam_dist/prefixed/econ_params_basic_dist.json"
)


# =========================================================================
#  Moment Computation
# =========================================================================

def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Return the Pearson correlation between two arrays, handling NaN values."""
    x_flat = np.asarray(x).flatten()
    y_flat = np.asarray(y).flatten()
    mask = np.isfinite(x_flat) & np.isfinite(y_flat)
    xv, yv = x_flat[mask], y_flat[mask]
    if len(xv) < 3:
        return 0.0
    r = np.corrcoef(xv, yv)[0, 1]
    return float(np.clip(r, -1.0, 1.0))


def compute_identification_moments(
    sim_results: Dict[str, np.ndarray],
    delta: float,
    alpha: float,
) -> np.ndarray:
    """Compute the 6-moment identification vector from post-burn-in panel."""
    derived = compute_all_derived_quantities(sim_results, delta, alpha)
    inv_rate = derived["investment_rate"]
    capital = derived["capital"]
    output = derived["output"]
    capital_safe = np.maximum(capital, 1e-5)
    revenue_capital = output / capital_safe

    ir_2d = np.asarray(inv_rate)
    rc_2d = np.asarray(revenue_capital)
    flat_ir = ir_2d.flatten()
    valid_ir = flat_ir[np.isfinite(flat_ir)]

    flat_rc = rc_2d.flatten()
    valid_rc = flat_rc[np.isfinite(flat_rc)]

    moments = np.zeros(K_MOMENTS)
    moments[0] = to_python_float(
        compute_autocorrelation_lags_1_to_5(inv_rate)["lag_1"]
    )
    moments[1] = (
        float(sp_stats.skew(valid_ir, nan_policy="omit"))
        if len(valid_ir) > 2 else 0.0
    )
    if len(valid_ir) > 0:
        moments[2] = float(np.sum(valid_ir < 0.0)) / len(valid_ir)
    if len(valid_rc) > 0:
        moments[3] = float(np.mean(valid_rc))
    if len(valid_rc) > 1:
        moments[4] = float(np.std(valid_rc, ddof=1))
    if ir_2d.ndim == 2 and ir_2d.shape[1] > 1:
        moments[5] = _safe_corr(ir_2d[:, 1:], rc_2d[:, :-1])

    return moments


# =========================================================================
#  DL Simulator Setup
# =========================================================================

def create_dl_simulator(epoch: int) -> DLSimulatorBasicFinal_dist:
    """Create and load the distributional DL simulator."""
    dl_config = load_dl_config(DL_CONFIG_PATH, "basic_final")
    econ_dist = EconomicParams(**load_json_file(ECON_PARAMS_FILE_DIST))
    bonds_dist = BondsConfig.validate_and_load(
        bounds_file=BONDS_FILE_DIST, current_params=econ_dist,
    )
    dl_config.capital_max = bonds_dist["k_max"]
    dl_config.capital_min = bonds_dist["k_min"]
    dl_config.productivity_max = bonds_dist["z_max"]
    dl_config.productivity_min = bonds_dist["z_min"]

    sim = DLSimulatorBasicFinal_dist(dl_config, bonds_dist)
    cap_path = os.path.join(
        DL_CHECKPOINT_DIR, f"basic_capital_policy_net_{epoch}.weights.h5"
    )
    inv_path = os.path.join(
        DL_CHECKPOINT_DIR, f"basic_investment_policy_net_{epoch}.weights.h5"
    )
    sim.load_solved_dl_solution(cap_path, inv_path)
    logger.info("DL simulator loaded (epoch %d)", epoch)
    return sim


def generate_fixed_shock_panel(
    econ_params: EconomicParams,
    bonds_config: Dict,
    n_firms: int,
    t_raw: int,
) -> Tuple[Any, Any]:
    """Generate a single shock-panel realisation (CRN, shared across evals)."""
    gen = synthetic_data_generator(
        econ_params_benchmark=econ_params,
        sample_bonds_config=bonds_config,
        batch_size=n_firms,
        T_periods=t_raw,
    )
    return gen.gen()


# =========================================================================
#  MVN Prior Fitting
# =========================================================================

def fit_moment_mvn_prior(
    moments_matrix: np.ndarray,
    max_condition: float = 1e4,
) -> Tuple[tfd.MultivariateNormalFullCovariance, np.ndarray, np.ndarray, np.ndarray]:
    r"""Fit a multivariate normal N(μ, Σ) to the sampled moment vectors.

    Returns
    -------
    prior_mvn : tfd.MultivariateNormalFullCovariance
    mu : np.ndarray, shape (6,)
    Sigma : np.ndarray, shape (6, 6)
    moment_scales : np.ndarray, shape (6,)
    """
    mu = np.mean(moments_matrix, axis=0)
    Sigma_raw = np.cov(moments_matrix, rowvar=False)
    cond_raw = np.linalg.cond(Sigma_raw)

    # Per-moment scale factors
    moment_scales = np.sqrt(np.diag(Sigma_raw))
    moment_scales = np.maximum(moment_scales, 1e-10)

    # Normalised covariance (correlation matrix)
    D_inv = np.diag(1.0 / moment_scales)
    Sigma_norm = D_inv @ Sigma_raw @ D_inv

    # Eigenvalue-floor regularisation
    eigvals, eigvecs = np.linalg.eigh(Sigma_norm)
    lambda_max = eigvals[-1]
    lambda_floor = lambda_max / max_condition
    eigvals_clamped = np.maximum(eigvals, lambda_floor)
    Sigma_norm_reg = eigvecs @ np.diag(eigvals_clamped) @ eigvecs.T
    Sigma_norm_reg = 0.5 * (Sigma_norm_reg + Sigma_norm_reg.T)

    cond_reg = np.linalg.cond(Sigma_norm_reg)

    # Map back to raw scale
    D = np.diag(moment_scales)
    Sigma = D @ Sigma_norm_reg @ D

    logger.info("Moment covariance regularisation:")
    logger.info("  Raw cond(Σ)        : %.1f", cond_raw)
    logger.info("  Eigenvalue floor   : %.2e  (λ_max=%.2e)", lambda_floor, lambda_max)
    logger.info("  Clamped eigenvalues: %s", np.array2string(eigvals_clamped, precision=4))
    logger.info("  Regularised cond(Σ): %.1f  (target ≤ %.0f)", cond_reg, max_condition)

    prior_mvn = tfd.MultivariateNormalFullCovariance(
        loc=tf.constant(mu, dtype=tf.float64),
        covariance_matrix=tf.constant(Sigma, dtype=tf.float64),
    )

    test_logp = prior_mvn.log_prob(tf.constant(mu, dtype=tf.float64))
    logger.info("Fitted MVN prior over moments:")
    logger.info("  μ = %s", np.array2string(mu, precision=4))
    logger.info("  diag(Σ) = %s", np.array2string(np.diag(Sigma), precision=6))
    logger.info("  cond(Σ) = %.1f", np.linalg.cond(Sigma))
    logger.info("  moment_scales = %s", np.array2string(moment_scales, precision=6))
    logger.info("  log p(μ) = %.4f  (sanity check; should be finite)", test_logp.numpy())

    return prior_mvn, mu, Sigma, moment_scales


# =========================================================================
#  Golden (Observed) Moments via VFI
# =========================================================================

def _golden_vfi_path() -> str:
    """Return the canonical path for the golden VFI solution at TRUE_PARAMS."""
    p = TRUE_PARAMS
    return (
        f"./ground_truth_basic/golden_vfi_results_"
        f"{p['productivity_persistence']}_{p['productivity_std_dev']}_"
        f"{p['adjustment_cost_convex']}_{p['adjustment_cost_fixed']}.npz"
    )


def _ensure_golden_vfi(econ_params: EconomicParams, bounds: Dict) -> str:
    """Generate the golden VFI solution if it does not already exist."""
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


def compute_golden_moments(
    econ_params: EconomicParams,
    bonds_config: Dict,
    vfi_solution: Optional[Any] = None,
) -> np.ndarray:
    """Compute golden moments via VFI simulation on the data panel."""
    if vfi_solution is None:
        golden_path = _golden_vfi_path()
        if not os.path.exists(golden_path):
            _ensure_golden_vfi(econ_params, bonds_config)
        vfi_solution = np.load(golden_path)

    vfi_sim = VFISimulator_basic(econ_params)
    vfi_sim.load_solved_vfi_solution(vfi_solution)

    data_gen = synthetic_data_generator(
        econ_params_benchmark=econ_params,
        sample_bonds_config=bonds_config,
        batch_size=N_DATA,
        T_periods=T_DATA_RAW,
    )
    data_init, data_shocks = data_gen.gen()

    results = vfi_sim.simulate(
        tuple(s.numpy() if hasattr(s, "numpy") else s for s in data_init),
        data_shocks.numpy() if hasattr(data_shocks, "numpy") else data_shocks,
    )
    stationary = apply_burn_in(results, T_BURN)

    delta = econ_params.depreciation_rate
    alpha = econ_params.capital_share
    golden = compute_identification_moments(stationary, delta, alpha)

    logger.info("Golden (observed) moments:")
    for name, val in zip(MOMENT_NAMES, golden):
        logger.info("  %s: %.6f", name, val)

    return golden


# =========================================================================
#  Posterior Construction
# =========================================================================

def build_log_posterior_fn(
    dl_simulator: DLSimulatorBasicFinal_dist,
    econ_params_template: EconomicParams,
    sim_initial_states: Any,
    sim_shock_sequence: Any,
    golden_moments: np.ndarray,
    Sigma_inv: np.ndarray,
    moment_scales: Optional[np.ndarray] = None,
) -> Callable:
    r"""Build the un-normalised log-posterior function.

    .. math::

        \log p(\theta \mid m^*) \propto
            -\tfrac{1}{2}\,\tilde d^\top \tilde\Sigma^{-1} \tilde d
            \;+\; \log \mathbb{1}[\theta \in \text{bounds}]

    The DL simulator acts as the forward model m(θ).
    """
    delta = econ_params_template.depreciation_rate
    alpha = econ_params_template.capital_share
    m_star = golden_moments.copy()
    S_inv = Sigma_inv.copy()
    scales = moment_scales.copy() if moment_scales is not None else None

    _eval_count = [0]
    _t_start = [time.perf_counter()]
    _oob_count = [0]

    def log_posterior(theta: np.ndarray) -> float:
        """Evaluate un-normalised log-posterior at θ."""
        _eval_count[0] += 1

        # Uniform prior: reject if outside bounds
        if np.any(theta < BOUNDS_LO) or np.any(theta > BOUNDS_HI):
            _oob_count[0] += 1
            return -1e30

        params = dataclasses.replace(
            econ_params_template,
            productivity_persistence=float(theta[0]),
            productivity_std_dev=float(theta[1]),
            adjustment_cost_convex=float(theta[2]),
            adjustment_cost_fixed=float(theta[3]),
        )

        try:
            dl_results = dl_simulator.simulate(
                sim_initial_states, sim_shock_sequence, params
            )
            stationary = apply_burn_in(dl_results, T_BURN)
            m_theta = compute_identification_moments(stationary, delta, alpha)
        except Exception as e:
            logger.warning("DL simulation failed at θ=%s: %s", theta, e)
            return -1e30

        if not np.all(np.isfinite(m_theta)):
            return -1e30

        diff = m_theta - m_star
        if scales is not None:
            diff = diff / scales
        log_lik = -0.5 * diff @ S_inv @ diff

        if _eval_count[0] % 50 == 0:
            elapsed = time.perf_counter() - _t_start[0]
            oob_pct = 100.0 * _oob_count[0] / _eval_count[0]
            logger.info(
                "  [eval %4d | %.0fs | OOB %.0f%%]  log p = %.4f  θ=[%.4f, %.4f, %.4f, %.4f]",
                _eval_count[0], elapsed, oob_pct, log_lik, *theta,
            )

        return float(log_lik)

    log_posterior.eval_count = _eval_count  # type: ignore[attr-defined]
    log_posterior.oob_count = _oob_count  # type: ignore[attr-defined]
    return log_posterior


# =========================================================================
#  MCMC Sampling
# =========================================================================

def run_mcmc_sampling(
    log_posterior_fn: Callable,
    n_samples: int = DEFAULT_N_MCMC_SAMPLES,
    n_burnin: int = DEFAULT_N_MCMC_BURNIN,
    n_chains: int = DEFAULT_N_CHAINS,
    step_size: float = DEFAULT_HMC_STEP_SIZE,
    n_leapfrog: int = DEFAULT_HMC_LEAPFROG_STEPS,
    use_nuts: bool = True,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run MCMC sampling from the posterior distribution.

    Uses adaptive random-walk Metropolis with Robbins-Monro step-size
    adaptation during burn-in.
    """
    rng = np.random.default_rng(seed)

    total_steps = n_burnin + n_samples
    all_samples = np.zeros((n_chains, n_samples, P_PARAMS))
    all_log_probs = np.zeros((n_chains, n_samples))
    acceptance_rates = np.zeros(n_chains)

    logger.info("=" * 72)
    logger.info("MCMC SAMPLING")
    logger.info("  Algorithm      : %s", "NUTS (adaptive MH)" if use_nuts else "Random-Walk MH")
    logger.info("  Chains         : %d", n_chains)
    logger.info("  Samples/chain  : %d  (burn-in: %d)", n_samples, n_burnin)
    logger.info("  Total evals    : ~%d", n_chains * total_steps)
    logger.info("=" * 72)

    for chain_id in range(n_chains):
        logger.info("\n── Chain %d/%d ──", chain_id + 1, n_chains)

        # Initialize near the center of the prior
        theta_current = BOUNDS_LO + rng.random(P_PARAMS) * (BOUNDS_HI - BOUNDS_LO)
        log_p_current = log_posterior_fn(theta_current)

        # Per-parameter adaptive proposal scales
        proposal_scale = step_size * (BOUNDS_HI - BOUNDS_LO)
        accept_count_total = 0
        accept_count_sampling = 0
        chain_samples = []
        chain_log_probs = []

        target_accept = 0.234

        t_chain_start = time.perf_counter()

        for step in range(total_steps):
            # Propose: componentwise random walk
            noise = rng.normal(size=P_PARAMS)
            theta_proposed = theta_current + proposal_scale * noise

            log_p_proposed = log_posterior_fn(theta_proposed)

            # Metropolis acceptance
            log_alpha = log_p_proposed - log_p_current
            u = rng.random()
            accepted = np.log(u) < log_alpha
            if accepted:
                theta_current = theta_proposed
                log_p_current = log_p_proposed
                accept_count_total += 1
                if step >= n_burnin:
                    accept_count_sampling += 1

            # Store post-burn-in samples
            if step >= n_burnin:
                chain_samples.append(theta_current.copy())
                chain_log_probs.append(log_p_current)

            # Adapt proposal scale during burn-in (Robbins-Monro)
            if step < n_burnin:
                gamma = 1.0 / (step + 1) ** 0.6
                accept_prob = min(1.0, np.exp(min(log_alpha, 0.0)))
                log_scale = np.log(proposal_scale + 1e-30)
                log_scale += gamma * (accept_prob - target_accept)
                proposal_scale = np.exp(log_scale)
                min_scale = 1e-6 * (BOUNDS_HI - BOUNDS_LO)
                max_scale = 0.5 * (BOUNDS_HI - BOUNDS_LO)
                proposal_scale = np.clip(proposal_scale, min_scale, max_scale)

            # Progress reporting
            if (step + 1) % 200 == 0:
                elapsed = time.perf_counter() - t_chain_start
                phase = "burn-in" if step < n_burnin else "sampling"
                ar = accept_count_total / (step + 1)
                logger.info(
                    "  Chain %d  step %4d/%d  [%s]  accept_rate=%.3f  "
                    "log_p=%.4f  θ=[%.4f, %.4f, %.4f, %.4f]  "
                    "scale=[%.4f, %.4f, %.4f, %.4f]  (%.0fs)",
                    chain_id + 1, step + 1, total_steps, phase, ar,
                    log_p_current, *theta_current, *proposal_scale, elapsed,
                )

        all_samples[chain_id] = np.array(chain_samples)
        all_log_probs[chain_id] = np.array(chain_log_probs)
        acceptance_rates[chain_id] = accept_count_sampling / n_samples

        logger.info(
            "  Chain %d done: sampling_accept_rate=%.3f, overall_accept_rate=%.3f, final log_p=%.4f",
            chain_id + 1, acceptance_rates[chain_id],
            accept_count_total / total_steps, log_p_current,
        )

    logger.info("\nMCMC complete. Mean acceptance rate: %.3f", acceptance_rates.mean())

    return {
        "samples": all_samples,
        "log_probs": all_log_probs,
        "acceptance_rates": acceptance_rates,
    }


# =========================================================================
#  MCMC Diagnostics & Summary Statistics
# =========================================================================

def compute_rhat(samples: np.ndarray) -> np.ndarray:
    """Compute the Gelman-Rubin R-hat diagnostic."""
    n_chains, n_samples, n_params = samples.shape
    if n_chains < 2:
        return np.ones(n_params) * np.nan

    rhat = np.zeros(n_params)
    for j in range(n_params):
        chain_means = np.mean(samples[:, :, j], axis=1)
        chain_vars = np.var(samples[:, :, j], axis=1, ddof=1)

        W = np.mean(chain_vars)
        B = n_samples * np.var(chain_means, ddof=1)

        var_hat = (1 - 1.0 / n_samples) * W + B / n_samples
        rhat[j] = np.sqrt(var_hat / W) if W > 0 else np.nan

    return rhat


def compute_effective_sample_size(samples: np.ndarray) -> np.ndarray:
    """Compute effective sample size (ESS) using auto-correlation."""
    n_chains, n_samples, n_params = samples.shape
    ess = np.zeros(n_params)

    for j in range(n_params):
        total_ess = 0.0
        for c in range(n_chains):
            x = samples[c, :, j]
            x_centered = x - np.mean(x)
            n = len(x_centered)
            fft_x = np.fft.fft(x_centered, n=2 * n)
            acf = np.fft.ifft(fft_x * np.conj(fft_x)).real[:n] / n
            if acf[0] > 0:
                acf /= acf[0]
            else:
                total_ess += n
                continue
            tau = 1.0
            for lag in range(1, n):
                if acf[lag] < 0:
                    break
                tau += 2.0 * acf[lag]
            total_ess += n / tau
        ess[j] = total_ess

    return ess


def summarize_posterior(
    mcmc_result: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute posterior summary statistics."""
    samples = mcmc_result["samples"]
    n_chains, n_samples, _ = samples.shape

    pooled = samples.reshape(-1, P_PARAMS)

    posterior_mean = np.mean(pooled, axis=0)
    posterior_median = np.median(pooled, axis=0)
    posterior_std = np.std(pooled, axis=0, ddof=1)
    ci_lo = np.percentile(pooled, 2.5, axis=0)
    ci_hi = np.percentile(pooled, 97.5, axis=0)

    rhat = compute_rhat(samples)
    ess = compute_effective_sample_size(samples)

    true_vals = np.array([TRUE_PARAMS[k] for k in PARAM_ORDER])
    bias = posterior_mean - true_vals
    bias_pct = 100.0 * bias / true_vals
    rmse = np.sqrt(np.mean((pooled - true_vals) ** 2, axis=0))
    rmse_pct = 100.0 * rmse / true_vals

    coverage = (true_vals >= ci_lo) & (true_vals <= ci_hi)

    summary = {
        "n_chains": n_chains,
        "n_samples_per_chain": n_samples,
        "n_total_samples": n_chains * n_samples,
        "param_labels": PARAM_LABELS,
        "param_keys": PARAM_ORDER,
        "true_values": true_vals.tolist(),
        "posterior_mean": posterior_mean.tolist(),
        "posterior_median": posterior_median.tolist(),
        "posterior_std": posterior_std.tolist(),
        "ci_95_lo": ci_lo.tolist(),
        "ci_95_hi": ci_hi.tolist(),
        "bias": bias.tolist(),
        "bias_pct": bias_pct.tolist(),
        "rmse": rmse.tolist(),
        "rmse_pct": rmse_pct.tolist(),
        "coverage_95": coverage.tolist(),
        "rhat": rhat.tolist(),
        "ess": ess.tolist(),
        "acceptance_rates": mcmc_result["acceptance_rates"].tolist(),
        "mean_acceptance_rate": float(mcmc_result["acceptance_rates"].mean()),
    }

    logger.info("\n" + "=" * 80)
    logger.info("POSTERIOR SUMMARY")
    logger.info("=" * 80)
    logger.info(
        "%-6s  %8s  %8s  %8s  %8s  %16s  %8s  %6s  %8s",
        "Param", "True", "Mean", "Median", "SD",
        "95% CI", "Bias%", "R-hat", "ESS",
    )
    logger.info("-" * 100)
    for j in range(P_PARAMS):
        logger.info(
            "%-6s  %8.4f  %8.4f  %8.4f  %8.4f  [%7.4f, %7.4f]  %+7.2f%%  %6.3f  %8.1f",
            PARAM_LABELS[j], true_vals[j],
            posterior_mean[j], posterior_median[j], posterior_std[j],
            ci_lo[j], ci_hi[j], bias_pct[j], rhat[j], ess[j],
        )
    logger.info("-" * 100)
    logger.info(
        "Mean acceptance rate: %.3f",
        mcmc_result["acceptance_rates"].mean(),
    )
    logger.info(
        "Coverage (95%%): %s",
        ", ".join(
            f"{PARAM_LABELS[j]}={'✓' if coverage[j] else '✗'}"
            for j in range(P_PARAMS)
        ),
    )
    logger.info("=" * 80)

    return summary


# =========================================================================
#  Visualization
# =========================================================================

def plot_posterior_traces(
    mcmc_result: Dict[str, Any],
    output_dir: str,
) -> None:
    """Plot trace plots and marginal posterior histograms."""
    samples = mcmc_result["samples"]
    n_chains = samples.shape[0]
    true_vals = np.array([TRUE_PARAMS[k] for k in PARAM_ORDER])

    fig, axes = plt.subplots(P_PARAMS, 2, figsize=(14, 3 * P_PARAMS))

    for j in range(P_PARAMS):
        ax_trace = axes[j, 0]
        for c in range(n_chains):
            ax_trace.plot(
                samples[c, :, j], alpha=0.6, linewidth=0.5,
                label=f"Chain {c + 1}",
            )
        ax_trace.axhline(
            true_vals[j], color="red", linestyle="--",
            linewidth=1.5, label="True",
        )
        ax_trace.set_ylabel(PARAM_LABELS[j], fontsize=12)
        ax_trace.set_title(f"Trace: {PARAM_LABELS[j]}", fontsize=11)
        if j == 0:
            ax_trace.legend(fontsize=8, ncol=n_chains + 1)

        ax_hist = axes[j, 1]
        pooled_j = samples[:, :, j].flatten()
        ax_hist.hist(
            pooled_j, bins=60, density=True,
            alpha=0.7, color="steelblue", edgecolor="white",
        )
        ax_hist.axvline(
            true_vals[j], color="red", linestyle="--",
            linewidth=1.5, label="True",
        )
        ax_hist.axvline(
            np.mean(pooled_j), color="blue", linestyle="-",
            linewidth=1.5, label="Post. mean",
        )
        lo, hi = np.percentile(pooled_j, [2.5, 97.5])
        ax_hist.axvspan(lo, hi, alpha=0.15, color="blue", label="95% CI")
        ax_hist.set_title(f"Posterior: {PARAM_LABELS[j]}", fontsize=11)
        if j == 0:
            ax_hist.legend(fontsize=8)

    axes[-1, 0].set_xlabel("MCMC iteration", fontsize=11)
    axes[-1, 1].set_xlabel("Parameter value", fontsize=11)

    plt.tight_layout()
    path = os.path.join(output_dir, "mcmc_traces_and_posteriors.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Trace & posterior plots saved: %s", path)


def plot_pairwise_posteriors(
    mcmc_result: Dict[str, Any],
    output_dir: str,
) -> None:
    """Plot pairwise posterior scatter plots (corner plot)."""
    samples = mcmc_result["samples"]
    pooled = samples.reshape(-1, P_PARAMS)
    true_vals = np.array([TRUE_PARAMS[k] for k in PARAM_ORDER])

    fig, axes = plt.subplots(P_PARAMS, P_PARAMS, figsize=(12, 12))

    for i in range(P_PARAMS):
        for j in range(P_PARAMS):
            ax = axes[i, j]
            if i == j:
                ax.hist(
                    pooled[:, i], bins=50, density=True,
                    alpha=0.7, color="steelblue",
                )
                ax.axvline(true_vals[i], color="red", linestyle="--", linewidth=1.5)
            elif i > j:
                ax.scatter(
                    pooled[:, j], pooled[:, i],
                    s=1, alpha=0.15, c="steelblue", rasterized=True,
                )
                ax.plot(
                    true_vals[j], true_vals[i],
                    "r*", markersize=10, zorder=5,
                )
            else:
                ax.set_visible(False)

            if i == P_PARAMS - 1:
                ax.set_xlabel(PARAM_LABELS[j], fontsize=10)
            if j == 0 and i > 0:
                ax.set_ylabel(PARAM_LABELS[i], fontsize=10)

    plt.suptitle("Pairwise Posterior Distributions", fontsize=13, y=1.01)
    plt.tight_layout()
    path = os.path.join(output_dir, "mcmc_pairwise_posteriors.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Pairwise posterior plot saved: %s", path)


def plot_xi_F_credible_ellipse(
    mcmc_result: Dict[str, Any],
    output_dir: str,
) -> None:
    """Plot 95% credible ellipse for the (ξ, F) parameter pair."""
    pooled = mcmc_result["samples"].reshape(-1, P_PARAMS)
    xi_samples = pooled[:, 2]
    F_samples = pooled[:, 3]

    xi_mean, F_mean = np.mean(xi_samples), np.mean(F_samples)
    cov_sub = np.cov(xi_samples, F_samples)

    eigenvalues, eigenvectors = np.linalg.eigh(cov_sub)
    chi2_val = chi2.ppf(0.95, 2)
    angle = np.degrees(np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1]))
    width = 2.0 * np.sqrt(chi2_val * max(eigenvalues[1], 0))
    height = 2.0 * np.sqrt(chi2_val * max(eigenvalues[0], 0))

    fig, ax = plt.subplots(figsize=(8, 6))

    ellipse = Ellipse(
        xy=(xi_mean, F_mean), width=width, height=height, angle=angle,
        edgecolor="blue", facecolor="lightblue", alpha=0.3, linewidth=2,
        label="95% credible ellipse",
    )
    ax.add_patch(ellipse)

    ax.scatter(
        xi_samples[::10], F_samples[::10],
        s=3, alpha=0.2, c="gray", label="Posterior samples", rasterized=True,
    )
    ax.plot(
        TRUE_PARAMS["adjustment_cost_convex"],
        TRUE_PARAMS["adjustment_cost_fixed"],
        "r*", markersize=15, label=r"True ($\xi^*$, $F^*$)", zorder=5,
    )
    ax.plot(
        xi_mean, F_mean,
        "b^", markersize=10, label=r"Post. mean ($\hat\xi$, $\hat F$)", zorder=5,
    )

    ax.set_xlabel(r"$\xi$ (convex adjustment cost)", fontsize=12)
    ax.set_ylabel("F (fixed adjustment cost)", fontsize=12)
    ax.set_title(r"Posterior ($\xi$, F) 95% Credible Ellipse", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "mcmc_xi_F_credible_ellipse.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("(ξ, F) credible ellipse saved: %s", path)


# =========================================================================
#  Serialization
# =========================================================================

def save_mcmc_samples_csv(
    mcmc_result: Dict[str, Any],
    output_path: str,
) -> None:
    """Save pooled posterior samples to CSV."""
    samples = mcmc_result["samples"].reshape(-1, P_PARAMS)
    log_probs = mcmc_result["log_probs"].flatten()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rho", "sigma", "xi", "F", "log_posterior"])
        for i in range(len(samples)):
            writer.writerow([
                f"{samples[i, 0]:.6f}",
                f"{samples[i, 1]:.6f}",
                f"{samples[i, 2]:.6f}",
                f"{samples[i, 3]:.6f}",
                f"{log_probs[i]:.6f}",
            ])
    logger.info("Posterior samples CSV saved: %s", output_path)


def generate_latex_posterior_table(summary: Dict[str, Any]) -> str:
    """Generate LaTeX table for posterior summary."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{MCMC Posterior Summary --- Basic Model}",
        r"\label{tab:mcmc_basic_posterior}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Parameter & True & Post.\ Mean & Post.\ SD & 95\% CI & Bias (\%) & $\hat{R}$ \\",
        r"\midrule",
    ]
    for j in range(P_PARAMS):
        ci_str = (
            f"[{summary['ci_95_lo'][j]:.4f}, {summary['ci_95_hi'][j]:.4f}]"
        )
        lines.append(
            f"${PARAM_LABELS[j]}$ & "
            f"{summary['true_values'][j]:.4f} & "
            f"{summary['posterior_mean'][j]:.4f} & "
            f"{summary['posterior_std'][j]:.4f} & "
            f"{ci_str} & "
            f"{summary['bias_pct'][j]:+.2f} & "
            f"{summary['rhat'][j]:.3f} \\\\"
        )
    lines += [
        r"\midrule",
        (
            r"\multicolumn{7}{l}{Mean acceptance rate: "
            f"{summary['mean_acceptance_rate']:.3f}}} \\\\"
        ),
        (
            f"\\multicolumn{{7}}{{l}}{{Chains: {summary['n_chains']}, "
            f"Samples/chain: {summary['n_samples_per_chain']}}} \\\\"
        ),
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# =========================================================================
#  Main
# =========================================================================

def main() -> None:
    """Run MVN fitting + MCMC posterior sampling pipeline."""
    parser = argparse.ArgumentParser(
        description=(
            "Phase 2-4: MVN prior fitting & MCMC posterior sampling "
            "for Bayesian SMM (basic model)"
        )
    )
    parser.add_argument(
        "--prior-cache", type=str, required=True,
        help="Path to prior_cache.npz from basic_SMM_mcmc_prior.py (required)",
    )
    parser.add_argument(
        "--gpu", type=int, default=0,
        help="GPU device ID to use (default: 0). Set to -1 for CPU only.",
    )
    parser.add_argument(
        "--dl-epoch", type=int, default=DEFAULT_DL_EPOCH,
        help=f"DL checkpoint epoch (default: {DEFAULT_DL_EPOCH})",
    )
    parser.add_argument(
        "--n-mcmc-samples", type=int, default=DEFAULT_N_MCMC_SAMPLES,
        help=f"MCMC posterior samples per chain (default: {DEFAULT_N_MCMC_SAMPLES})",
    )
    parser.add_argument(
        "--n-mcmc-burnin", type=int, default=DEFAULT_N_MCMC_BURNIN,
        help=f"MCMC burn-in steps per chain (default: {DEFAULT_N_MCMC_BURNIN})",
    )
    parser.add_argument(
        "--n-chains", type=int, default=DEFAULT_N_CHAINS,
        help=f"Number of MCMC chains (default: {DEFAULT_N_CHAINS})",
    )
    parser.add_argument(
        "--step-size", type=float, default=DEFAULT_HMC_STEP_SIZE,
        help=f"Initial proposal step size (default: {DEFAULT_HMC_STEP_SIZE})",
    )
    parser.add_argument(
        "--no-nuts", action="store_true",
        help="Use basic random-walk MH instead of adaptive variant",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./results/smm_mcmc_basic",
        help="Output directory for results",
    )
    parser.add_argument(
        "--seed", type=int, default=2026,
        help="Master random seed",
    )
    args = parser.parse_args()

    # GPU was already configured before TF import (see _configure_gpu_before_import)
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info("GPU %d selected — visible devices: %s", args.gpu, gpus)
    else:
        logger.info("No GPU visible — running on CPU")

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    logger.info("=" * 72)
    logger.info("MCMC-BASED BAYESIAN SMM — BASIC MODEL (Phase 2–4)")
    logger.info("=" * 72)
    logger.info("  Prior cache      : %s", args.prior_cache)
    logger.info("  GPU              : %s", args.gpu)
    logger.info("  MCMC samples     : %d per chain", args.n_mcmc_samples)
    logger.info("  MCMC burn-in     : %d", args.n_mcmc_burnin)
    logger.info("  Chains           : %d", args.n_chains)
    logger.info("  Step size        : %.4f", args.step_size)
    logger.info("  DL epoch         : %d", args.dl_epoch)
    logger.info("  Output dir       : %s", args.output_dir)
    logger.info("  TRUE_PARAMS      : %s", TRUE_PARAMS)
    logger.info("=" * 72)

    # ─── Load prior cache ──────────────────────────────────────────
    if not os.path.exists(args.prior_cache):
        raise FileNotFoundError(
            f"Prior cache not found: {args.prior_cache}\n"
            "Run basic_SMM_mcmc_prior.py first to generate it."
        )

    logger.info("Loading prior cache from %s", args.prior_cache)
    cache = np.load(args.prior_cache)
    moments_matrix = cache["moments"]
    valid_params = cache["params"]
    logger.info(
        "Loaded %d prior samples (%d moments each)",
        moments_matrix.shape[0], moments_matrix.shape[1],
    )

    # ─── Load economic parameters ──────────────────────────────────
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

    # ─── Fit MVN prior ─────────────────────────────────────────────
    logger.info("\n" + "=" * 72)
    logger.info("PHASE 2: Fitting MVN Prior Over Moments")
    logger.info("=" * 72)

    prior_mvn, mu_prior, Sigma_prior, moment_scales = fit_moment_mvn_prior(
        moments_matrix,
    )

    # Compute precision matrix in normalised space
    D_inv = np.diag(1.0 / moment_scales)
    Sigma_norm = D_inv @ Sigma_prior @ D_inv
    Sigma_norm_inv = np.linalg.inv(Sigma_norm)
    Sigma_norm_inv = 0.5 * (Sigma_norm_inv + Sigma_norm_inv.T)
    logger.info(
        "Normalised precision matrix cond: %.1f",
        np.linalg.cond(Sigma_norm_inv),
    )

    # ─── Golden (Observed) Moments ─────────────────────────────────
    logger.info("\n" + "=" * 72)
    logger.info("PHASE 3: Computing Golden (Observed) Moments via VFI")
    logger.info("=" * 72)

    _ensure_golden_vfi(econ_params, bonds_config)
    golden_path = _golden_vfi_path()
    vfi_solution = np.load(golden_path)

    golden_moments = compute_golden_moments(
        econ_params, bonds_config, vfi_solution=vfi_solution,
    )

    # ─── Load DL simulator (needed for posterior evaluation) ───────
    dl_simulator = create_dl_simulator(args.dl_epoch)

    # ─── Generate fixed shock panel for DL evaluations (CRN) ──────
    logger.info(
        "Generating fixed simulation shock panel (N=%d, T=%d) ...",
        N_SIM, T_SIM_RAW,
    )
    sim_init, sim_shocks = generate_fixed_shock_panel(
        econ_params, bonds_config, N_SIM, T_SIM_RAW,
    )

    # ─── Warm up DL simulator ─────────────────────────────────────
    logger.info("Warming up DL simulator (@tf.function tracing) ...")
    _t0 = time.perf_counter()
    _ = dl_simulator.simulate(sim_init, sim_shocks, econ_params)
    logger.info("Warm-up done (%.1f s)", time.perf_counter() - _t0)

    # ─── Build Posterior ───────────────────────────────────────────
    logger.info("\n" + "=" * 72)
    logger.info("PHASE 4: Constructing Posterior Distribution")
    logger.info("=" * 72)

    log_posterior_fn = build_log_posterior_fn(
        dl_simulator, econ_params,
        sim_init, sim_shocks,
        golden_moments, Sigma_norm_inv,
        moment_scales=moment_scales,
    )

    # Quick test evaluation at true parameters
    true_theta = np.array([TRUE_PARAMS[k] for k in PARAM_ORDER])
    log_p_true = log_posterior_fn(true_theta)
    logger.info("Log-posterior at θ* (true): %.4f", log_p_true)

    # ─── MCMC Sampling ─────────────────────────────────────────────
    logger.info("\n" + "=" * 72)
    logger.info("PHASE 5: MCMC Sampling")
    logger.info("=" * 72)

    t_mcmc = time.perf_counter()
    mcmc_result = run_mcmc_sampling(
        log_posterior_fn,
        n_samples=args.n_mcmc_samples,
        n_burnin=args.n_mcmc_burnin,
        n_chains=args.n_chains,
        step_size=args.step_size,
        use_nuts=not args.no_nuts,
        seed=args.seed,
    )
    mcmc_wall_time = time.perf_counter() - t_mcmc
    logger.info("MCMC wall time: %.1f s", mcmc_wall_time)

    # ─── Posterior Summary ─────────────────────────────────────────
    summary = summarize_posterior(mcmc_result)
    summary["mcmc_wall_time"] = mcmc_wall_time
    summary["n_prior_samples"] = moments_matrix.shape[0]

    # ─── Serialization ─────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)

    # Summary JSON
    summary_path = os.path.join(args.output_dir, "mcmc_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary JSON saved: %s", summary_path)

    # Posterior samples CSV
    save_mcmc_samples_csv(
        mcmc_result,
        os.path.join(args.output_dir, "posterior_samples.csv"),
    )

    # Prior info
    prior_info = {
        "mu_prior": mu_prior.tolist(),
        "Sigma_prior": Sigma_prior.tolist(),
        "Sigma_prior_cond": float(np.linalg.cond(Sigma_prior)),
        "moment_scales": moment_scales.tolist(),
        "Sigma_norm_cond": float(np.linalg.cond(Sigma_norm)),
        "golden_moments": golden_moments.tolist(),
        "moment_names": MOMENT_NAMES,
        "n_prior_samples": int(moments_matrix.shape[0]),
    }
    with open(os.path.join(args.output_dir, "prior_info.json"), "w") as f:
        json.dump(prior_info, f, indent=2)

    # ─── Visualization ─────────────────────────────────────────────
    plot_posterior_traces(mcmc_result, args.output_dir)
    plot_pairwise_posteriors(mcmc_result, args.output_dir)
    plot_xi_F_credible_ellipse(mcmc_result, args.output_dir)

    # ─── LaTeX table ───────────────────────────────────────────────
    latex = generate_latex_posterior_table(summary)
    latex_path = os.path.join(args.output_dir, "mcmc_latex_table.tex")
    with open(latex_path, "w") as f:
        f.write("% Auto-generated LaTeX table — MCMC Bayesian SMM\n")
        f.write("% Generated by basic_SMM_mcmc_posterior.py\n\n")
        f.write(latex + "\n")
    logger.info("LaTeX table saved: %s", latex_path)

    # ─── Final summary ─────────────────────────────────────────────
    logger.info("\n" + "=" * 72)
    logger.info("ALL DONE — results saved to %s", args.output_dir)
    logger.info("  Prior samples loaded  : %d", moments_matrix.shape[0])
    logger.info("  MCMC samples          : %d total (%d chains × %d)",
                summary["n_total_samples"], summary["n_chains"],
                summary["n_samples_per_chain"])
    logger.info("  MCMC wall time        : %.1f s", mcmc_wall_time)
    logger.info("  Mean acceptance rate   : %.3f", summary["mean_acceptance_rate"])
    logger.info("=" * 72)


if __name__ == "__main__":
    main()
