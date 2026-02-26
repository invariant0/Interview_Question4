# Economic Models: VFI and Deep Learning Solvers

This repository provides a comprehensive framework for solving dynamic corporate finance models using **Value Function Iteration (VFI)** and **Deep Learning (DL)** approaches. It supports both **Basic Investment Models** and **Risky Debt Investment Models**.

## Implementation Overview

The repository features two main Deep Learning implementation strategies:

1.  **Single Economy Version**: Trains a model on a single set of economic parameters.
2.  **Distributed Economy Version (Dist)**: Leveraging distributed computing to solve for a parameterized method of moments or handle multiple economies simultaneously.

### Pretrain-Finetuning Paradigm

Both implementation versions (Single and Distributed) adopt a robust **Pretrain-Finetuning** paradigm to ensure convergence and stability:

*   **Pretraining Phase**: Utilizes First-Order Condition (FOC) constraints to guide the neural networks towards a reasonable initial solution.
    *   For the Basic Model: Uses the `basic` configuration.
    *   For the Risky Model: Uses a `risk_free` approximation as a starting point.
*   **Finetuning Phase**: Implemented in `_final.py` modules (`basic_final`, `risky_final`). This phase uses an **Actor-Critic style** approach, loading the FOC-pretrained checkpoints and refining the solution (e.g., handling non-differentiable kinks in the Risky model via direct maximization).

---

## Installation and Setup

### Prerequisites

- Python 3.8+
- TensorFlow 2.x
- TensorFlow Probability

### Setup

```bash
# Clone the repository
git clone https://github.com/invariant0/Interview_Question4.git
cd Interview_Question4

# Create virtual environment
python -m venv tf
source tf/bin/activate  # On Windows: venv\Scripts\activate

# Install package in development mode
pip install -e .
```

---

## Usage

### 1. Single Economy Workflow

The single economy workflow focuses on solving the model for a specific set of parameters.

#### Step 1: Boundary Discovery (VFI)
First, use the VFI solver to automatically discover the appropriate state space boundaries.

```bash
# Find bounds for Basic Model
solve-vfi --model basic --find_bounds

# Find bounds for Risky Model
solve-vfi --model risky --find_bounds
```

#### Step 2: Compute Golden Ground Truth
Generate high-resolution VFI solutions to serve as the ground truth for validation.

```bash
# For Basic Model (specify economy ID, e.g., 0 or 1)
python basic_golden_vfi_finder.py -econ_id 0

# For Risky Model
python risky_golden_vfi_finder.py
```

#### Step 3: Deep Learning Training
Train the neural networks using the two-stage pretrain-finetune process.

**Basic Model:**
```bash
# Stage 1: Pretrain (FOC-based)
train-dl --model basic

# Stage 2: Finetune (Actor-Critic style)
train-dl --model basic_final
```

**Risky Model:**
```bash
# Stage 1: Pretrain (Risk-free approximation)
train-dl --model risk_free

# Stage 2: Finetune (Actor-Critic style with direct maximization)
train-dl --model risky_final
```


### 2. Distributed Economy Workflow (Dist)

The distributed workflow is designed for efficiency when dealing with parameter distributions or larger scale experiments.

#### Step 1: VFI Solver (Dist)
Run the distributed VFI solver.

```bash
solve-vfi-dist --model basic 
solve-vfi-dist --model risky
```

#### Step 2: Deep Learning Training (Dist)
Train the distributed deep learning models.

```bash
# Basic Model (Dist)
train-dl-dist --model basic          # Pretrain
train-dl-dist --model basic_final    # Finetune

# Risky Model (Dist)
train-dl-dist --model risk_free      # Pretrain
train-dl-dist --model risky_final    # Finetune
```

## Checkpoints Downloads

All checkpoints can be downloaded here. After downloading, place each checkpoint folder and ground truth folder in the project root directory (`econ-dl/`) to run simulation without training.

**Download link:** https://drive.google.com/drive/folders/10f6vb8CBqFftReS5yQlszonI2CTXU6Zo?usp=drive_link

### Expected Folder Layout After Download

After downloading and extracting, the project root should contain the following folders:

```
econ-dl/
├── checkpoints_pretrain/          # Pretrained (FOC-based) DL weights — Single Economy
│   ├── basic/                     #   basic_policy_net_*.weights.h5, basic_value_net_*.weights.h5
│   └── risk_free/                 #   risk_free_{capital_policy,debt_policy,default_policy,value}_net_*.weights.h5
│
├── checkpoints_final/             # Finetuned (Actor-Critic) DL weights — Single Economy
│   ├── basic/                     #   basic_{capital_policy,investment_policy,value}_net_*.weights.h5
│   └── risky/                     #   risky_{capital_policy,continuous,debt_policy,debt_policy_invest,
│                                  #          debt_policy_noinvest,default_policy,equity_issuance,
│                                  #          equity_issuance_noinvest,investment_decision,value}_net_*.weights.h5
│
├── checkpoints_pretrain_dist/     # Pretrained DL weights — Distributed Economy
│   ├── basic/                     #   basic_{policy,value}_net_*.weights.h5
│   └── risk_free/                 #   risk_free_{capital_policy,debt_policy,default_policy,value}_net_*.weights.h5
│
├── checkpoints_final_dist/        # Finetuned DL weights — Distributed Economy
│   ├── basic/                     #   basic_{capital_policy,investment_policy,value}_net_*.weights.h5
│   └── risky/                     #   risky_{capital_policy,continuous,debt_policy,default_policy,
│                                  #          equity_issuance,equity_issuance_noinvest,
│                                  #          investment_decision,value}_net_*.weights.h5
│
├── ground_truth_basic/            # High-resolution VFI solutions for the Basic model (.npz)
│                                  #   golden_vfi_basic_{alpha}_{delta}_{adj}_{r}_*.npz
│
├── ground_truth_risky/            # High-resolution VFI solutions for the Risky model (.npz)
│                                  #   golden_vfi_risky_{alpha}_{delta}_{adj}_{r}_{ef}_{el}_*.npz
│
├── hyperparam/                    # Hyperparameters & bounds — Single Economy
│   ├── autogen/                   #   bounds_basic_*.json, bounds_risky_*.json
│   └── prefixed/                  #   econ_params_basic_*.json, econ_params_risky_*.json,
│                                  #   dl_params.json, vfi_params.json
│
└── hyperparam_dist/               # Hyperparameters & bounds — Distributed Economy
    ├── autogen/                   #   bounds_basic_dist.json, bounds_risky_dist.json
    └── prefixed/                  #   econ_params_*_dist.json, dl_params_dist.json, vfi_params_dist.json
```

**Notes:**
- Weight files follow the naming pattern `{model}_{net_name}_{epoch}.weights.h5`, where the epoch number indicates the training checkpoint.
- The `hyperparam/autogen/` folder contains VFI-discovered state-space bounds (generated by `solve-vfi --find_bounds`).
- The `hyperparam/prefixed/` folder contains the economic parameter configurations and DL/VFI solver settings used for training.
- Ground truth files encode the economic parameters in their filename (e.g., `golden_vfi_basic_0.6_0.175_1.005_0.03_3000.npz` corresponds to $\alpha=0.6, \delta=0.175, \theta=1.005, r=0.03$ with grid size 3000).

---

## Project Structure

```bash
econ-dl/
├── src/
│   └── econ_models/
│       ├── cli/               # Command-line interfaces (solve_vfi, train_dl, etc.)
│       ├── dl/                # Single Economy Deep Learning implementations
│       │   ├── basic.py       # Pretraining logic
│       │   ├── basic_final.py # Finetuning logic (Actor-Critic)
│       │   ├── risk_free.py   # Pretraining logic for Risky model
│       │   └── risky_final.py # Finetuning logic for Risky model
│       ├── dl_dist/           # Distributed Economy implementations
│       │   ├── basic_final.py
│       │   └── ...
│       ├── vfi/               # Value Function Iteration solvers
│       └── ...
├── basic_golden_vfi_finder.py # Ground truth finder for Basic model
├── risky_golden_vfi_finder.py # Ground truth finder for Risky model
└── ...
```

---

## Scripts Usage Explaination

### Basic Model


For simulation test on basic model and basic_dist model run 
```bash
python basic_simulation.py
python basic_simulation_dist.py
```

it will generate a single econ param and two econ params situation for both version using the final version model under basic econ model setting

### Risky Model 

For simulation test on risky model and risky_dist model run 

```bash
python risky_simulation.py
python risky_simulation_dist.py
```

it will generate a single econ param and two econ params situation for both version using the final version model under risky econ model setting

### GMM estimation

run 
```bash
python basic_GMM.py
```

it will generate the GMM results and saved in result folder gmm_basic, only basic GMM is implemented

### SMM estimation

run 
```bash
python basic_SMM.py
python risky_SMM.py
```

The SMM estimation process will first load from the checkpoints and doing SMM estimation and saved the result in result folder smm_basic or smm_risky


### MCMC version of structural estimation

for basic model we also implement its MCMC version, it will first load the checkpoints and then construct the mcmc process in two step:

step 1: moment prior generation
step 2: fit the saved prior sample to multi-variable normal distribution and using posterial distribution conditioned on the bencmark the moments to aceept and reject samples according to the likelihood ratio

## Testing

The project uses **pytest** for testing. All tests run on CPU (no GPU required). All test dependencies are included in the main install:

```bash
pip install -e .
```

### Run All Tests

```bash
pytest
```

### Run Unit Tests Only

```bash
pytest tests/unit/
```

### Run Integration Tests Only

```bash
pytest tests/integration/
```

### Run a Specific Test File

```bash
# Example: run grid builder tests
pytest tests/unit/test_grid_builder.py

# Example: run DL config tests
pytest tests/unit/dl/test_config.py
```

### Run with Verbose Output

```bash
pytest -v
```

### Test Suite Overview

**Unit Tests** (`tests/unit/`)

| File | Module Under Test |
|---|---|
| `test_adjust_kernels.py` | VFI adjust-tile kernel (shapes, dtypes, index bounds, constraints) |
| `test_bellman_kernels.py` | Bellman operators: `compute_ev`, `bellman_update`, `sup_norm_diff` |
| `test_bond_price_kernels.py` | Bond price update kernel (risk-free pricing, default scenarios) |
| `test_chunk_accumulate.py` | Tile-index remapping and running-best accumulation |
| `test_flows.py` | Cash-flow builders: `build_adjust_flow_part1`, `build_debt_components` |
| `test_grid_builder.py` | Grid construction (productivity, capital, debt grids) |
| `test_grid_utils.py` | 1-D batch linear interpolation and grid utilities |
| `test_policies.py` | Policy extraction for basic and risky models |
| `test_basic_simulator.py` | Basic model simulator (shapes, bounds, convergence, reproducibility) |
| `test_risky_simulator.py` | Risky model simulator (default/no-default scenarios, depreciation map) |
| `test_tile_executor.py` | Tile executor: correct tiling coverage with mock kernels |
| `test_tile_strategy.py` | Optimal chunk-size computation respecting VRAM budget |
| `test_wait_kernels.py` | Wait-branch flow computation and branch reduction |
| `dl/test_config.py` | `EconomicParams` and `DeepLearningConfig` validation |
| `dl/test_normalizers.py` | State-space normalizers (round-trip, boundary, missing-field checks) |
| `dl/test_econ_functions.py` | Core economic function utilities |
| `dl/test_fischer_burmeister.py` | Fischer-Burmeister complementarity smoothing |
| `dl/test_neural_net_factory.py` | Neural network factory construction |
| `dl/test_state_sampler.py` | State sampling for DL training |
| `dl/test_transitions.py` | Productivity transition logic |
| `dl/test_risky_simulator_alignment.py` | Risky simulator alignment checks |

**Integration Tests** (`tests/integration/`)

| File | What It Tests |
|---|---|
| `test_basic_integration.py` | End-to-end basic model VFI solve on a tiny grid (convergence, monotonicity) |
| `test_risky_integration.py` | End-to-end risky-debt model VFI solve (bond-price bounds, default regions) |
| `test_roundtrip.py` | Full pipeline: solve risky model → simulate → verify results |
| `test_model_smoke.py` | DL model construction and single training-step smoke tests |

