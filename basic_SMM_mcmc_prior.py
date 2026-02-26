#!/usr/bin/env python3
"""Phase 1 — Prior Moment Generation for MCMC-Based Bayesian SMM (Basic Model).

Samples N parameter vectors from the prior (Sobol quasi-random within
economic bounds), runs each through the DL simulator on a fixed shock
panel (common random numbers), and saves the resulting moment matrix
and valid parameter array to disk for downstream MCMC estimation.

Usage::

    # Quick test (200 samples, GPU 0)
    python basic_SMM_mcmc_prior.py --n-prior-samples 200 --gpu 0

    # Full run (2000 samples, GPU 3)
    python basic_SMM_mcmc_prior.py --n-prior-samples 2000 --gpu 3

    # Custom output directory
    python basic_SMM_mcmc_prior.py --n-prior-samples 2000 --gpu 0 \
        --output-dir ./results/smm_mcmc_basic

Outputs (saved to --output-dir):
    prior_cache.npz  — keys: ``moments`` (N_valid, 6), ``params`` (N_valid, 4)
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import sys
import time
from typing import Any, Dict, List, Tuple

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

import numpy as np
from scipy import stats as sp_stats
from scipy.stats.qmc import Sobol
import tensorflow as tf

from src.econ_models.config.dl_config import load_dl_config
from src.econ_models.config.economic_params import EconomicParams
from src.econ_models.config.bond_config import BondsConfig
from src.econ_models.io.file_utils import load_json_file
from src.econ_models.simulator import (
    DLSimulatorBasicFinal_dist,
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

# Panel dimensions (matching basic_SMM.py)
T_BURN: int = 200
T_SIM_EFF: int = 500
T_SIM_RAW: int = T_SIM_EFF + T_BURN + 1
N_SIM: int = 10000

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

# Defaults
DEFAULT_N_PRIOR_SAMPLES: int = 2000
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
#  Prior Sampling & Moment Computation
# =========================================================================

def sample_prior_parameters(
    n_samples: int,
    seed: int = 42,
) -> np.ndarray:
    """Generate quasi-random parameter samples via Sobol sequence.

    Returns
    -------
    params_array : np.ndarray, shape (n_samples, 4)
        Each row is [ρ, σ, ξ, F] uniformly sampled within SEARCH_BOUNDS.
    """
    sampler = Sobol(d=P_PARAMS, scramble=True, seed=seed)
    m = int(np.ceil(np.log2(max(n_samples, 1))))
    raw = sampler.random(2**m)[:n_samples]

    params_array = BOUNDS_LO + raw * (BOUNDS_HI - BOUNDS_LO)

    logger.info("Sampled %d parameter vectors (Sobol, seed=%d):", n_samples, seed)
    for j, key in enumerate(PARAM_ORDER):
        logger.info(
            "  %s: bounds [%.4f, %.4f], sample range [%.4f, %.4f]",
            PARAM_LABELS[j], BOUNDS_LO[j], BOUNDS_HI[j],
            params_array[:, j].min(), params_array[:, j].max(),
        )
    return params_array


def batch_compute_moments(
    params_array: np.ndarray,
    dl_simulator: DLSimulatorBasicFinal_dist,
    econ_params_template: EconomicParams,
    initial_states: Any,
    shock_sequence: Any,
    burn_in: int = T_BURN,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute DL-simulated moments for every row in params_array.

    Uses a fixed shock panel (common random numbers) across all parameter
    evaluations.  Filters out samples that produce non-finite moments.

    Returns
    -------
    moments_matrix : np.ndarray, shape (N_valid, 6)
    valid_params : np.ndarray, shape (N_valid, 4)
    """
    delta = econ_params_template.depreciation_rate
    alpha = econ_params_template.capital_share
    n_samples = params_array.shape[0]

    all_moments = np.zeros((n_samples, K_MOMENTS))
    valid_mask = np.ones(n_samples, dtype=bool)

    t0 = time.perf_counter()

    for i in range(n_samples):
        theta = params_array[i]
        modified_params = dataclasses.replace(
            econ_params_template,
            productivity_persistence=float(theta[0]),
            productivity_std_dev=float(theta[1]),
            adjustment_cost_convex=float(theta[2]),
            adjustment_cost_fixed=float(theta[3]),
        )

        try:
            dl_results = dl_simulator.simulate(
                initial_states, shock_sequence, modified_params
            )
            stationary = apply_burn_in(dl_results, burn_in)
            moments = compute_identification_moments(stationary, delta, alpha)

            if not np.all(np.isfinite(moments)):
                valid_mask[i] = False
                logger.warning("  Sample %d: non-finite moments, skipping", i)
            else:
                all_moments[i] = moments
        except Exception as e:
            valid_mask[i] = False
            logger.warning("  Sample %d failed: %s", i, e)

        if (i + 1) % 50 == 0:
            elapsed = time.perf_counter() - t0
            rate = (i + 1) / elapsed
            eta = (n_samples - i - 1) / rate
            logger.info(
                "  Prior sampling: %d/%d done (%.1f samples/s, ETA %.0fs)",
                i + 1, n_samples, rate, eta,
            )

    n_valid = int(valid_mask.sum())
    logger.info(
        "Prior moment computation complete: %d/%d valid samples (%.1f%%)",
        n_valid, n_samples, 100.0 * n_valid / n_samples,
    )

    return all_moments[valid_mask], params_array[valid_mask]


# =========================================================================
#  Main
# =========================================================================

def main() -> None:
    """Generate prior moment samples and save to disk."""
    parser = argparse.ArgumentParser(
        description=(
            "Phase 1: Generate prior moment samples via DL simulator "
            "for MCMC-based Bayesian SMM (basic model)"
        )
    )
    parser.add_argument(
        "--gpu", type=int, default=0,
        help="GPU device ID to use (default: 0). Set to -1 for CPU only.",
    )
    parser.add_argument(
        "--n-prior-samples", type=int, default=DEFAULT_N_PRIOR_SAMPLES,
        help=f"Number of prior parameter samples (default: {DEFAULT_N_PRIOR_SAMPLES})",
    )
    parser.add_argument(
        "--dl-epoch", type=int, default=DEFAULT_DL_EPOCH,
        help=f"DL checkpoint epoch (default: {DEFAULT_DL_EPOCH})",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./results/smm_mcmc_basic",
        help="Output directory for prior cache (default: ./results/smm_mcmc_basic)",
    )
    parser.add_argument(
        "--seed", type=int, default=2026,
        help="Random seed (default: 2026)",
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
    logger.info("PHASE 1 — PRIOR MOMENT GENERATION (Basic Model)")
    logger.info("=" * 72)
    logger.info("  GPU              : %s", args.gpu)
    logger.info("  Prior samples    : %d", args.n_prior_samples)
    logger.info("  DL epoch         : %d", args.dl_epoch)
    logger.info("  Output dir       : %s", args.output_dir)
    logger.info("  Seed             : %d", args.seed)
    logger.info("  TRUE_PARAMS      : %s", TRUE_PARAMS)
    logger.info("=" * 72)

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

    # ─── Load DL simulator ────────────────────────────────────────
    dl_simulator = create_dl_simulator(args.dl_epoch)

    # ─── Generate fixed shock panel (CRN) ─────────────────────────
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

    # ─── Sample parameters & compute moments ──────────────────────
    params_array = sample_prior_parameters(
        args.n_prior_samples, seed=args.seed,
    )

    t_phase1 = time.perf_counter()
    moments_matrix, valid_params = batch_compute_moments(
        params_array, dl_simulator, econ_params,
        sim_init, sim_shocks,
    )
    elapsed = time.perf_counter() - t_phase1
    logger.info("Prior moment computation done in %.1f s", elapsed)

    # ─── Save results ─────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    cache_path = os.path.join(args.output_dir, "prior_cache.npz")
    np.savez(
        cache_path,
        moments=moments_matrix,
        params=valid_params,
    )

    logger.info("=" * 72)
    logger.info("PRIOR GENERATION COMPLETE")
    logger.info("  Valid samples : %d / %d", moments_matrix.shape[0], args.n_prior_samples)
    logger.info("  Moments shape : %s", moments_matrix.shape)
    logger.info("  Saved to      : %s", cache_path)
    logger.info("  Wall time     : %.1f s", elapsed)
    logger.info("=" * 72)

    # ─── Quick summary statistics ─────────────────────────────────
    logger.info("Moment summary (across valid prior samples):")
    for j, name in enumerate(MOMENT_NAMES):
        col = moments_matrix[:, j]
        logger.info(
            "  %s: mean=%.6f  std=%.6f  min=%.6f  max=%.6f",
            name, np.mean(col), np.std(col), np.min(col), np.max(col),
        )


if __name__ == "__main__":
    main()
