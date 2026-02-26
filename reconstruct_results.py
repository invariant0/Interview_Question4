#!/usr/bin/env python3
"""Reconstruct SMM result files from log-extracted data.

This script uses the converged θ̂ from a stuck Nelder-Mead run
and re-computes all post-estimation diagnostics (SEs, J-test,
moment fit, DL bias, condition numbers, LaTeX tables, confidence
ellipse) without re-running the full optimization.

Usage:
    python reconstruct_results.py
"""

from __future__ import annotations

import csv
import dataclasses
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
from scipy.stats import chi2

# ── Import project modules ──────────────────────────────────────────
from src.econ_models.config.economic_params import EconomicParams
from src.econ_models.config.bond_config import BondsConfig
from src.econ_models.io.file_utils import load_json_file
from src.econ_models.simulator import (
    DLSimulatorRiskyFinal_dist,
    VFISimulator_risky,
)
from src.econ_models.simulator import synthetic_data_generator

# Import from risky_SMM.py
from risky_SMM import (
    TRUE_PARAMS, SEARCH_BOUNDS, PARAM_ORDER, PARAM_LABELS,
    MOMENT_NAMES, K_MOMENTS, P_PARAMS, DF_OVERID,
    N_DATA, T_DATA_EFF, T_DATA_RAW, N_SIM, T_SIM_EFF, T_SIM_RAW,
    T_BURN, J_RATIO,
    DEFAULT_DL_EPOCH, DEFAULT_N_BOOTSTRAP, DEFAULT_N_SOBOL_STARTS,
    DEFAULT_CMA_MAX_EVALS,
    VFI_N_K, VFI_N_D,
    compute_identification_moments,
    apply_burn_in,
    compute_smm_jacobian,
    compute_smm_standard_errors,
    compute_j_test,
    compute_moment_fit_table,
    compute_condition_number,
    compute_golden_moments,
    bootstrap_weighting_matrix,
    create_dl_simulator,
    _econ_params_path,
    _bounds_path,
    _golden_vfi_path,
    plot_confidence_ellipse,
    generate_latex_table_recovery,
    generate_latex_table_moment_fit,
    generate_latex_table_design,
    save_replication_csv,
    save_summary_json,
    MCReplicationResult,
    aggregate_mc_results,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# =========================================================================
#  Data extracted from log output
# =========================================================================

# Best θ̂ from Nelder-Mead Stage 2 (converged around eval 580)
THETA_HAT = np.array([0.6859, 0.1478, 1.7640, 0.0100, 0.0808, 0.0209])

Q_MIN = 4.023013e+05

# Golden moments from log
GOLDEN_MOMENTS = np.array([
    0.088465,   # AC[I/K]
    -0.793069,  # Skew[I/K]
    0.087490,   # SD[Y/K]
    0.088365,   # Corr(I/K,lagY/K)
    0.119649,   # P(equity iss.)
    0.107398,   # E[|e|/K|e<0]
    0.117291,   # E[D/K|D>0]
    -0.267113,  # Corr(ΔB/K,Y/K)
    5.279336,   # Median[B/K]
])

# Total evals: 10 CMA-ES starts × 207 each + ~5840 NM evals
# (NM still running but we treat convergence at ~590)
TOTAL_EVALS = 10 * 207 + 590  # effective evals until convergence

# Wall time: started 03:32:55 (CMA-ES), stage1 done 05:51:44,
# NM converged ~06:31 (eval 590)
WALL_TIME_SECONDS = (6*3600 + 31*60) - (3*3600 + 32*60 + 55)  # ~10775 s

OUTPUT_DIR = "./results/smm_risky"
SEED = 2026


def main() -> None:
    """Reconstruct all result files from the log-extracted data."""
    logger.info("=" * 72)
    logger.info("RECONSTRUCTING SMM RESULTS FROM LOG DATA")
    logger.info("=" * 72)
    logger.info("  θ̂ = %s", THETA_HAT)
    logger.info("  Q(θ̂) = %.6e", Q_MIN)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Load economic parameters ────────────────────────────────────
    econ_path = _econ_params_path()
    logger.info("Loading econ params: %s", econ_path)
    econ_params = EconomicParams(**load_json_file(econ_path))

    bounds_path_ = _bounds_path()
    logger.info("Loading bounds: %s", bounds_path_)
    bonds_config = BondsConfig.validate_and_load(
        bounds_file=bounds_path_, current_params=econ_params,
    )

    # Load golden VFI solution
    golden_path = _golden_vfi_path()
    vfi_solution = np.load(golden_path, allow_pickle=True)
    logger.info("Golden VFI solution loaded: %s", golden_path)

    # Load DL simulator
    dl_simulator = create_dl_simulator(DEFAULT_DL_EPOCH)
    logger.info("DL simulator loaded (epoch %d)", DEFAULT_DL_EPOCH)

    delta = econ_params.depreciation_rate
    alpha = econ_params.capital_share
    r_f = econ_params.risk_free_rate

    # ── Regenerate simulation panel with same seed ──────────────────
    master_rng = np.random.default_rng(SEED)
    rep_rng = np.random.default_rng(master_rng.integers(0, 2**32))

    # Data panel (same as the run)
    logger.info("Generating data panel (N=%d, T=%d) ...", N_DATA, T_DATA_EFF)
    data_gen = synthetic_data_generator(
        econ_params_benchmark=econ_params,
        sample_bonds_config=bonds_config,
        batch_size=N_DATA,
        T_periods=T_DATA_RAW,
        include_debt=True,
    )
    data_init, data_shocks = data_gen.gen()

    golden_moments_computed, stationary_data = compute_golden_moments(
        econ_params, data_init, data_shocks, bounds=bonds_config,
        vfi_solution=vfi_solution,
    )
    logger.info("Golden moments computed: %s", golden_moments_computed)

    # Bootstrap weighting matrix
    W, Omega_hat = bootstrap_weighting_matrix(
        stationary_data, delta, alpha, r_f=r_f,
        B=DEFAULT_N_BOOTSTRAP, rng=rep_rng,
    )

    # Simulation panel
    logger.info("Generating simulation panel (N=%d, T=%d) ...", N_SIM, T_SIM_EFF)
    sim_gen = synthetic_data_generator(
        econ_params_benchmark=econ_params,
        sample_bonds_config=bonds_config,
        batch_size=N_SIM,
        T_periods=T_SIM_RAW,
        include_debt=True,
    )
    sim_init, sim_shocks = sim_gen.gen()

    # Warm up DL simulator
    logger.info("Warming up DL simulator ...")
    warmup_results = dl_simulator.simulate(
        sim_init, sim_shocks, econ_params, jit_compile=True,
    )
    logger.info("Warm-up done")

    # DL moments at true θ*
    warmup_stationary = apply_burn_in(warmup_results, T_BURN)
    dl_moments_at_true = compute_identification_moments(
        warmup_stationary, delta, alpha, r_f=r_f,
    )

    # ── Post-estimation diagnostics using the converged θ̂ ──────────
    theta_hat = THETA_HAT
    N_total = N_DATA * T_DATA_EFF

    # Use the computed golden moments (should match the log)
    golden_moments = golden_moments_computed
    logger.info("Using golden moments from VFI computation")
    for i, name in enumerate(MOMENT_NAMES):
        logger.info("  %s: %.6f (log: %.6f, diff: %.2e)",
                     name, golden_moments[i], GOLDEN_MOMENTS[i],
                     abs(golden_moments[i] - GOLDEN_MOMENTS[i]))

    # §4: Jacobian
    logger.info("\nComputing Jacobian G ...")
    G = compute_smm_jacobian(
        theta_hat, dl_simulator, econ_params,
        sim_init, sim_shocks,
    )

    # §4: Standard errors
    se, Sigma = compute_smm_standard_errors(theta_hat, G, W, N_total, J_RATIO)

    # §5: J-test
    j_result = compute_j_test(Q_MIN, N_total, J_RATIO)

    # §6: Moment fit table
    fit_result = compute_moment_fit_table(
        theta_hat, golden_moments, dl_simulator, econ_params,
        sim_init, sim_shocks, Omega_hat,
    )

    # §9: Condition number at θ̂
    kappa_hat = compute_condition_number(G, theta_hat, golden_moments)

    # ── Log parameter recovery ──────────────────────────────────────
    true_theta = np.array([TRUE_PARAMS[k] for k in PARAM_ORDER])
    logger.info("\n" + "=" * 72)
    logger.info("PARAMETER RECOVERY")
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
    logger.info("Q(θ̂) = %.6e", Q_MIN)
    logger.info("J-stat = %.4f (p = %.4f)", j_result["T_J"], j_result["p_value"])
    logger.info("κ(E) = %.2f", kappa_hat)
    logger.info("=" * 72)

    # ── Build MCReplicationResult ───────────────────────────────────
    mc_result = MCReplicationResult(
        replication_id=0,
        theta_hat=theta_hat,
        se=se,
        Q_min=Q_MIN,
        j_stat=j_result["T_J"],
        j_pvalue=j_result["p_value"],
        n_evals=TOTAL_EVALS,
        wall_time=WALL_TIME_SECONDS,
        golden_moments=golden_moments,
        model_moments=fit_result["model_moments"],
    )

    # ── Aggregate (single replication) ──────────────────────────────
    all_results = [mc_result]
    summary = aggregate_mc_results(all_results)

    # ── Save all output files ───────────────────────────────────────

    # 1. replications.csv
    save_replication_csv(
        all_results, os.path.join(OUTPUT_DIR, "replications.csv")
    )

    # 2. summary.json
    save_summary_json(
        summary, os.path.join(OUTPUT_DIR, "summary.json")
    )

    # 3. Confidence ellipse
    if Sigma is not None:
        plot_confidence_ellipse(
            theta_hat=theta_hat,
            Sigma=Sigma,
            output_path=os.path.join(
                OUTPUT_DIR, "eta0_eta1_confidence_ellipse.png"
            ),
        )

    # 4. LaTeX tables
    latex_recovery = generate_latex_table_recovery(summary)
    latex_design = generate_latex_table_design(summary)
    latex_fit = generate_latex_table_moment_fit(
        fit_result, j_result
    )

    latex_path = os.path.join(OUTPUT_DIR, "latex_tables.tex")
    with open(latex_path, "w") as f:
        f.write("% Auto-generated LaTeX tables — risky debt model SMM\n")
        f.write("% Reconstructed from log output by reconstruct_results.py\n\n")
        f.write("% MC Design\n")
        f.write(latex_design + "\n\n")
        f.write("% Parameter Recovery\n")
        f.write(latex_recovery + "\n\n")
        f.write("% Moment Fit\n")
        f.write(latex_fit + "\n")
    logger.info("LaTeX tables saved: %s", latex_path)

    # 5. DL approximation bias at θ*
    logger.info("\n§9: DL approximation bias at θ* ...")
    vfi_sim = VFISimulator_risky(econ_params)
    vfi_sim.load_solved_vfi_solution(vfi_solution)
    vfi_results_star = vfi_sim.simulate(
        tuple(
            s.numpy() if hasattr(s, "numpy") else s for s in sim_init
        ),
        (
            sim_shocks.numpy()
            if hasattr(sim_shocks, "numpy") else sim_shocks
        ),
    )
    vfi_stationary_star = apply_burn_in(vfi_results_star, T_BURN)
    vfi_moments_star = compute_identification_moments(
        vfi_stationary_star, delta, alpha, r_f=r_f,
    )

    dl_bias = dl_moments_at_true - vfi_moments_star
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
            name, vfi_moments_star[i], dl_moments_at_true[i], dl_bias[i], pct,
        )

    dl_bias_info = {
        "moment_names": MOMENT_NAMES,
        "vfi_moments_at_true": vfi_moments_star.tolist(),
        "dl_moments_at_true": dl_moments_at_true.tolist(),
        "dl_bias": dl_bias.tolist(),
        "dl_bias_pct": [
            100.0 * abs(dl_bias[i]) / max(abs(vfi_moments_star[i]), 1e-10)
            for i in range(K_MOMENTS)
        ],
    }
    with open(os.path.join(OUTPUT_DIR, "dl_bias.json"), "w") as f:
        json.dump(dl_bias_info, f, indent=2)
    logger.info("DL bias saved: %s", os.path.join(OUTPUT_DIR, "dl_bias.json"))

    # 6. Condition number comparison
    G_star = compute_smm_jacobian(
        true_theta, dl_simulator, econ_params,
        sim_init, sim_shocks,
    )
    kappa_star = compute_condition_number(G_star, true_theta, vfi_moments_star)

    logger.info("\nCONDITION NUMBER COMPARISON:")
    logger.info("  κ(E) at θ*: %.2f", kappa_star)
    logger.info("  κ(E) at θ̂:  %.2f", kappa_hat)

    condition_info = {
        "kappa_at_true": float(kappa_star),
        "kappa_at_estimate": float(kappa_hat),
    }
    with open(os.path.join(OUTPUT_DIR, "condition_numbers.json"), "w") as f:
        json.dump(condition_info, f, indent=2)
    logger.info("Condition numbers saved")

    # ── Save the log output ─────────────────────────────────────────
    logger.info("\n" + "=" * 72)
    logger.info("ALL DONE — results reconstructed and saved to %s", OUTPUT_DIR)
    logger.info("=" * 72)

    # Print summary of files
    logger.info("\nGenerated files:")
    for fname in sorted(os.listdir(OUTPUT_DIR)):
        fpath = os.path.join(OUTPUT_DIR, fname)
        size = os.path.getsize(fpath)
        logger.info("  %s (%d bytes)", fname, size)


if __name__ == "__main__":
    main()
