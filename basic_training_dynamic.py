#!/usr/bin/env python3
"""Visualize deep-learning training convergence for the basic model.

For a sequence of training epochs this script evaluates three diagnostics:

1. **Bellman residual** — mean absolute Bellman residual on the
   initial-state batch.
2. **Value-function error** — mean absolute percentage error between
   the DL value network and the VFI value function on the VFI grid.
3. **Lifetime reward** — average discounted lifetime reward compared
   with the VFI benchmark.

The result is a single three-panel figure saved to
``results/effectiveness_basic/basic_training_dynamics.png``.

Usage::

    python basic_training_dynamic.py [--econ-id 0]
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from src.econ_models.config.dl_config import load_dl_config
from src.econ_models.config.economic_params import EconomicParams
from src.econ_models.simulator import DLSimulatorBasicFinal, VFISimulator_basic

from basic_common import (
    DEFAULT_ECON_LIST,
    get_golden_vfi_path,
    load_bonds_config,
    load_econ_params,
    setup_simulation_data,
)

# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------

RESULTS_DIR: str = "./results/effectiveness_basic"
CHECKPOINT_DIR: str = "./checkpoints_final/basic"

EPOCHS: List[int] = [20 * i for i in range(1, 11)]
"""Training epochs to evaluate (1 000 through 9 000)."""


# ---------------------------------------------------------------------------
#  DL simulator setup
# ---------------------------------------------------------------------------

def create_dl_simulator(
    econ_params: EconomicParams,
    bonds_config: Dict,
) -> DLSimulatorBasicFinal:
    """Create and configure the basic DL simulator.

    Parameters
    ----------
    econ_params:
        Benchmark economic parameters.
    bonds_config:
        Validated bonds/bounds configuration.

    Returns
    -------
    DLSimulatorBasicFinal
    """
    dl_config = load_dl_config("./hyperparam/prefixed/dl_params.json", "basic")
    dl_config.capital_max = bonds_config["k_max"]
    dl_config.capital_min = bonds_config["k_min"]
    dl_config.productivity_max = bonds_config["z_max"]
    dl_config.productivity_min = bonds_config["z_min"]
    return DLSimulatorBasicFinal(dl_config, econ_params)


# ---------------------------------------------------------------------------
#  Per-epoch evaluation
# ---------------------------------------------------------------------------

def evaluate_epoch(
    dl_simulator: DLSimulatorBasicFinal,
    epoch: int,
    initial_states,
    shock_sequence,
    vfi_solution: Dict,
) -> Dict[str, float]:
    """Load a DL checkpoint and compute convergence metrics.

    Parameters
    ----------
    dl_simulator:
        Configured DL simulator.
    epoch:
        Training epoch to evaluate.
    initial_states:
        Initial-state tensors for simulation.
    shock_sequence:
        Shock/innovation sequence tensor.
    vfi_solution:
        Loaded VFI ``.npz`` archive.

    Returns
    -------
    dict
        Keys: ``bellman_mae``, ``lifetime_reward_mean``, ``value_mape``.
    """
    dl_simulator.load_solved_dl_solution(
        os.path.join(CHECKPOINT_DIR, f"basic_capital_policy_net_{epoch}.weights.h5"),
        os.path.join(CHECKPOINT_DIR, f"basic_investment_policy_net_{epoch}.weights.h5"),
        os.path.join(CHECKPOINT_DIR, f"basic_value_net_{epoch}.weights.h5"),
    )

    # Bellman residual
    bellman = dl_simulator.simulate_bellman_residual(initial_states)
    bellman_mae = float(np.mean(np.abs(bellman["absolute_error"])))

    # Lifetime reward
    lifetime = dl_simulator.simulate_life_time_reward(initial_states, shock_sequence)
    lifetime_mean = float(np.mean(lifetime.numpy()))

    # Value-function gap on VFI grid
    v_vfi = np.maximum(vfi_solution["V_adjust"], vfi_solution["V_wait"])
    k_grid, z_grid = np.meshgrid(vfi_solution["K"], vfi_solution["Z"], indexing="ij")
    gap = dl_simulator.compute_value_function_gap(
        grid_points=(tf.constant(k_grid), tf.constant(z_grid)),
        value_labels=tf.constant(v_vfi),
    )
    value_mape = float(np.mean(gap["mape"]))

    return {
        "bellman_mae": bellman_mae,
        "lifetime_reward_mean": lifetime_mean,
        "value_mape": value_mape,
    }


# ---------------------------------------------------------------------------
#  Plotting
# ---------------------------------------------------------------------------

def plot_training_dynamics(
    epochs: List[int],
    bellman_maes: List[float],
    value_mapes: List[float],
    dl_reward_means: List[float],
    vfi_reward_mean: float,
    econ_id: int,
) -> plt.Figure:
    """Create the 1x3 training-convergence figure.

    Parameters
    ----------
    epochs:
        List of evaluated training epochs.
    bellman_maes:
        Mean absolute Bellman residual per epoch.
    value_mapes:
        Mean absolute percentage value-function error per epoch.
    dl_reward_means:
        Mean lifetime reward per epoch.
    vfi_reward_mean:
        VFI benchmark lifetime reward.
    econ_id:
        Economy index for the figure title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    plt.rcParams.update({"font.size": 12})
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(f"Training Convergence & Benchmarking (Econ ID: {econ_id})", fontsize=16)

    # Subplot 1: Bellman residual
    axes[0].plot(epochs, bellman_maes, marker="o", linewidth=2, color="tab:blue")
    axes[0].set_title("Bellman Residual Dynamics")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Mean Absolute Bellman Residual")
    axes[0].set_yscale("log")
    axes[0].grid(True, which="both", ls="-", alpha=0.2)

    # Subplot 2: Value-function error
    axes[1].plot(epochs, value_mapes, marker="s", linewidth=2, color="tab:red")
    axes[1].set_title("Value Function Error (DL vs VFI)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Mean Absolute Percentage Error")
    axes[1].set_yscale("log")
    axes[1].grid(True, which="both", ls="-", alpha=0.2)

    # Subplot 3: Lifetime reward
    axes[2].plot(epochs, dl_reward_means, marker="^", linewidth=2, color="tab:green", label="Deep Learning")
    axes[2].axhline(y=vfi_reward_mean, color="black", linestyle="--", linewidth=2, label="VFI Benchmark")
    axes[2].set_title("Lifetime Reward Comparison")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Average Lifetime Reward")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Evaluate training dynamics and generate convergence figure."""
    parser = argparse.ArgumentParser(description="Basic model training dynamics")
    parser.add_argument(
        "--econ-id", type=int, default=0, choices=range(len(DEFAULT_ECON_LIST)),
        help="Index into the default economy list (default: 0)",
    )
    args = parser.parse_args()

    econ_id: int = args.econ_id
    econ_spec = DEFAULT_ECON_LIST[econ_id]

    # --- Setup ---
    econ_params = load_econ_params(econ_spec)
    bonds_config = load_bonds_config(econ_spec, econ_params)
    initial_states, shock_sequence = setup_simulation_data(econ_params, bonds_config)

    # VFI benchmark lifetime reward
    vfi_solution = np.load(get_golden_vfi_path(econ_spec))
    vfi_simulator = VFISimulator_basic(econ_params)
    vfi_simulator.load_solved_vfi_solution(vfi_solution)
    vfi_lifetime = vfi_simulator.simulate_life_time_reward(
        tuple(s.numpy() for s in initial_states),
        shock_sequence.numpy(),
    )
    vfi_reward_mean = float(np.mean(vfi_lifetime))

    # DL epoch-by-epoch evaluation
    dl_simulator = create_dl_simulator(econ_params, bonds_config)
    bellman_maes: List[float] = []
    value_mapes: List[float] = []
    dl_reward_means: List[float] = []

    for epoch in EPOCHS:
        print(f"Processing epoch: {epoch}")
        metrics = evaluate_epoch(dl_simulator, epoch, initial_states, shock_sequence, vfi_solution)
        bellman_maes.append(metrics["bellman_mae"])
        value_mapes.append(metrics["value_mape"])
        dl_reward_means.append(metrics["lifetime_reward_mean"])

    # --- Generate figure ---
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fig = plot_training_dynamics(EPOCHS, bellman_maes, value_mapes, dl_reward_means, vfi_reward_mean, econ_id)
    fig.savefig(os.path.join(RESULTS_DIR, "basic_training_dynamics.png"))
    print("Training dynamics plot saved.")


if __name__ == "__main__":
    main()
