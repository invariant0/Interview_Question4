import os
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
from scipy import stats

from src.econ_models.config.dl_config import load_dl_config
from src.econ_models.simulator import DLSimulatorRiskyFinal
from src.econ_models.simulator import VFISimulator_risky
from src.econ_models.config.economic_params import EconomicParams
from src.econ_models.config.bond_config import BondsConfig
from src.econ_models.core.types import TENSORFLOW_DTYPE
from src.econ_models.simulator import synthetic_data_generator
from src.econ_models.io.file_utils import load_json_file

tfd = tfp.distributions

# ============================================================
#  1. Setup Paths and Config
# ============================================================

econ_list = [
    [0.6, 0.17, 1.0, 0.02, 0.1, 0.08],
    [0.6, 0.17, 1.0, 0.02, 0.03, 0.01],
]
econ_id = 0
econ_tag = '_'.join(str(x) for x in econ_list[econ_id])

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname('./')))
ECON_PARAMS_FILE = os.path.join(
    BASE_DIR, f"hyperparam/prefixed/econ_params_risky_{econ_tag}.json"
)
BOUNDARY_FILE = os.path.join(
    BASE_DIR, f"hyperparam/autogen/bounds_risky_{econ_tag}.json"
)

# Golden VFI solution
N_CAPITAL = 560
N_DEBT = 560
VFI_FILE = f'./ground_truth_risky/golden_vfi_risky_{econ_tag}_{N_CAPITAL}_{N_DEBT}.npz'

# CHECKPOINT_DIR = './checkpoints_final/risky'
CHECKPOINT_DIR = './checkpoints_final/risky'
RESULTS_DIR = './results/effectiveness_risky'
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
#  2. Load Parameters and Generate Shared Data
# ============================================================

econ_params = EconomicParams(**load_json_file(ECON_PARAMS_FILE))

sample_bonds_config = BondsConfig.validate_and_load(
    bounds_file=BOUNDARY_FILE,
    current_params=econ_params,
)

synthetic_data_gen = synthetic_data_generator(
    econ_params_benchmark=econ_params,
    sample_bonds_config=sample_bonds_config,
    batch_size=10000,
    T_periods=1000,
    include_debt=True,
)
initial_states, innovation_sequence = synthetic_data_gen.gen()

# ============================================================
#  3. Load VFI Solution for Value Function Benchmark
# ============================================================

print(f"Loading VFI solution from {VFI_FILE} ...")
solved_result_matrixs = np.load(VFI_FILE, allow_pickle=True)

# VFI value function: V = max(V_adjust, V_wait), shape (n_K, n_B, n_Z)
V_vfi = np.maximum(solved_result_matrixs['V_adjust'], solved_result_matrixs['V_wait'])

# Build 3D meshgrid for value function comparison
K_1d = solved_result_matrixs['K']  # (560,)
B_1d = solved_result_matrixs['B']  # (560,)
Z_1d = solved_result_matrixs['Z']  # (40,)
K_grid, B_grid, Z_grid = np.meshgrid(K_1d, B_1d, Z_1d, indexing='ij')

# --- Run VFI Lifetime Reward Benchmark ---
vfi_simulator = VFISimulator_risky(econ_params)
vfi_simulator.load_solved_vfi_solution(solved_result_matrixs)
vfi_life_time_reward_benchmark = vfi_simulator.simulate_life_time_reward(
    tuple(s.numpy() for s in initial_states),
    innovation_sequence.numpy()
)
print(f"VFI lifetime reward benchmark: {vfi_life_time_reward_benchmark:.4f}")

# ============================================================
#  4. Initialize DL Simulator
# ============================================================

dl_config = load_dl_config('./hyperparam/prefixed/dl_params.json', 'risky_final')
dl_config.capital_max = sample_bonds_config['k_max']
dl_config.capital_min = sample_bonds_config['k_min']
dl_config.productivity_max = sample_bonds_config['z_max']
dl_config.productivity_min = sample_bonds_config['z_min']
dl_config.debt_max = sample_bonds_config['b_max']
dl_config.debt_min = sample_bonds_config['b_min']

dl_simulator = DLSimulatorRiskyFinal(dl_config, econ_params)

# ============================================================
#  5. Loop Over Epochs: Compute Training Dynamics
# ============================================================

bellman_training_dynamics = []
life_time_reward_dynamics = []
value_pred_mae_dynamics = []
value_pred_mape_dynamics = []
    
epochs_list = [200 * i for i in range(1, 11)]  # [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

for epoch in epochs_list:
    print(f'Processing epoch: {epoch}')
    dl_simulator.load_solved_dl_solution(
        capital_policy_filepath=os.path.join(
            CHECKPOINT_DIR, f'risky_capital_policy_net_{epoch}.weights.h5'
        ),
        debt_filepath=os.path.join(
            CHECKPOINT_DIR, f'risky_debt_policy_net_{epoch}.weights.h5'
        ),
        investment_policy_filepath=os.path.join(
            CHECKPOINT_DIR, f'risky_investment_decision_net_{epoch}.weights.h5'
        ),
        default_policy_filepath=os.path.join(
            CHECKPOINT_DIR, f'risky_default_policy_net_{epoch}.weights.h5'
        ),
        value_function_filepath=os.path.join(
            CHECKPOINT_DIR, f'risky_value_net_{epoch}.weights.h5'
        ),
        equity_issuance_invest_filepath=os.path.join(
            CHECKPOINT_DIR, f'risky_equity_issuance_net_{epoch}.weights.h5'
        ),
        equity_issuance_noinvest_filepath=os.path.join(
            CHECKPOINT_DIR, f'risky_equity_issuance_net_noinvest_{epoch}.weights.h5'
        ),
    )

    # --- Bellman Residual ---
    bellman_residuals_dl = dl_simulator.simulate_bellman_residual(initial_states)

    # --- Lifetime Reward ---
    life_time_rewards_dl = dl_simulator.simulate_life_time_reward(
        initial_states, innovation_sequence
    )

    # --- Value Function Gap (DL vs VFI on 3D grid) ---
    gap_results = dl_simulator.compute_value_function_gap(
        grid_points=(tf.constant(K_grid), tf.constant(B_grid), tf.constant(Z_grid)),
        value_labels=tf.constant(V_vfi),
    )

    bellman_training_dynamics.append(bellman_residuals_dl['absolute_error'])
    life_time_reward_dynamics.append(float(life_time_rewards_dl.numpy()))
    value_pred_mae_dynamics.append(gap_results["mae"])
    value_pred_mape_dynamics.append(gap_results["mape"])

    print(f'  Bellman |res|={bellman_residuals_dl["absolute_error"]:.4f}  '
          f'MAPE={gap_results["mape"]:.4f}  '
          f'LTR={float(life_time_rewards_dl.numpy()):.4f}')

# ============================================================
#  6. Figure: Training Dynamics (3 subplots)
# ============================================================

# --- Data Processing ---
bellman_means = [b for b in bellman_training_dynamics]
value_error_means = [v for v in value_pred_mape_dynamics]
dl_reward_means = [r for r in life_time_reward_dynamics]

# --- Plotting ---
plt.rcParams.update({'font.size': 12})
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle(f'Risky Model: Training Convergence & Benchmarking (Econ ID: {econ_id})', fontsize=16)

# Subplot 1: Bellman Residual Dynamics
axes[0].plot(epochs_list, bellman_means, marker='o', linestyle='-', linewidth=2, color='tab:blue')
axes[0].set_title('Bellman Residual Dynamics')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Mean Absolute Bellman Residual')
axes[0].set_yscale('log')
axes[0].grid(True, which="both", ls="-", alpha=0.2)

# Subplot 2: Value Function Comparison (MAPE)
axes[1].plot(epochs_list, value_error_means, marker='s', linestyle='-', linewidth=2, color='tab:red')
axes[1].set_title('Value Function Error (DL vs VFI)')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Mean Absolute Percentage Error')
axes[1].set_yscale('log')
axes[1].grid(True, which="both", ls="-", alpha=0.2)

# Subplot 3: Lifetime Reward Dynamics
axes[2].plot(epochs_list, dl_reward_means, marker='^', linestyle='-', linewidth=2, color='tab:green', label='Deep Learning')
axes[2].axhline(y=vfi_life_time_reward_benchmark, color='black', linestyle='--', linewidth=2, label='VFI Benchmark')
axes[2].set_title('Lifetime Reward Comparison')
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('Average Lifetime Reward')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(RESULTS_DIR, 'risky_training_dynamics.png'), dpi=150)
print(f"\nFigure saved to {RESULTS_DIR}/risky_training_dynamics.png")
