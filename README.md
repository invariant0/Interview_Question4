# Economic Models: VFI and Deep Learning Solvers

This repository provides a comprehensive framework for solving dynamic corporate finance models using **Value Function Iteration (VFI)** and **Deep Learning (DL)** approaches. It supports both **Basic Investment Models** and **Risky Debt Investment Models**.

## Implementation Overview

The repository features two main Deep Learning implementation strategies:

1.  **Single Economy Version**: Trains a model on a single set of economic parameters.
2.  **Distributed Economy Version (Dist)**: Augment the input dimension of the deep learning model to incorporate econ param input and train a surrogate deep learning solution for a vector space of econ params which need estimation.

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
Then, use the VFI solver to automatically discover the appropriate state space boundaries.

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
Train the neural networks using the two-stage pretrain-finetune process. For both the Basic Model and the Risky Model, we prepare two testing econ parameter sets, named econ 0 and econ 1. You can try both of them, but since this is the single model version, each deep learning run is dedicated to one setting. The default pretraining epochs are set to 1500 for the Basic Model and 480 for the Risky Model. You can certainly try multiple starting points.

**Basic Model:**
```bash
# Stage 1: Pretrain (FOC-based)
train-dl --model basic --econ_id 0

# Stage 2: Finetune (Actor-Critic style)
train-dl --model basic_final --econ_id 0 --pretrained_epoch 1500
```

**Risky Model:**
```bash
# Stage 1: Pretrain (Risk-free approximation)
train-dl --model risk_free --econ_id 0

# Stage 2: Finetune (Actor-Critic style with direct maximization)
train-dl --model risky_final --econ_id 0 --pretrained_epoch 480
```


### 2. Distributed Economy Workflow (Dist)

The distributed workflow is designed for efficiency when dealing with parameter distributions or larger scale experiments.

#### Step 1: VFI Solver (Dist)
Run the distributed VFI solver. This will automatically test corner areas of the state space and determine the global maximum and minimum of the state value range for the deep learning dist version to train on. The global boundary will be saved in the hyperparam_dist folder

```bash
solve-vfi-dist --model basic 
solve-vfi-dist --model risky
```
#### Step 2: Deep Learning Training (Dist)
Train the distributed deep learning models. This will load the previously found global maximum and minimal boundary and train by cross-sampling of both econ param and states. The pretraining epochs are set to 6200 for the Basic Model and 2500 for the Risky Model. You can certainly try multiple starting points. 

```bash
# Basic Model (Dist)
train-dl-dist --model basic                                 # Pretrain
train-dl-dist --model basic_final  --pretrained_epoch 6200  # Finetune

# Risky Model (Dist)
train-dl-dist --model risk_free                             # Pretrain
train-dl-dist --model risky_final  --pretrained_epoch 2500  # Finetune
```

#### Deep Learning Configuration

All deep learning and VFI configurations are located in the /hyperparam and /hyperparam_dist folders. While hyperparameter optimization is time-consuming, you can leverage multiple GPUs to run parallel experiments by creating multiple JSON configuration files to efficiently identify the best combinations.

The final version of the risky model is the most difficult to train. For your reference, here are some of the most impactful hyperparameters:

*   **equity_fb_weight**: Setting this value too low causes payoff leakage, meaning the model receives a free negative payoff due to insufficient forward-backward (FB) constraints on the equity issuance network. Experiments show that a value of 20 works well.
*   **Learning rate scheduler**: This is arguably the most consequential hyperparameter to tune. After experimentation, we found success setting the overall learning rate to 0.001. The policy learning rate should match the value network (a policy scale of 1.0); setting the policy rate too low will cause the model to converge to a local minimum.
*   **polyak_averaging_decay**: This is another critical factor. We found that starting the decay at 0.99 and ending at 0.999 yields the best results. The key takeaway is that the initial decay rate must be low enough to allow for the deconstruction of the pretrained value function, while the final decay rate must be high enough to stabilize the reconstructed value function.

## Checkpoints Downloads

Models can be retrained. However, to run a simulation comparison with VFI methods, the ground truth folder must be downloaded first, or you will need to rerun the golden VFI finder to train it, which takes too much time. 

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

## Scripts Usage Explanation

### Basic Model


For simulation tests on the Basic Model and Basic Dist Model, run:
```bash
python basic_simulation.py
python basic_simulation_dist.py
```

It will generate a single econ param and two econ params scenarios for both versions using the final version model under the Basic Model setting.

### Risky Model 

For simulation tests on the Risky Model and Risky Dist Model, run:

```bash
python risky_simulation.py
python risky_simulation_dist.py
```

It will generate a single econ param and two econ params scenarios for both versions using the final version model under the Risky Model setting.

### GMM estimation

run 
```bash
python basic_GMM.py
```

It will generate the GMM results and save them in the result folder gmm_basic. Only the Basic Model GMM is implemented.

### SMM estimation

run 
```bash
python basic_SMM.py
python risky_SMM.py
```

The SMM estimation process will first load from the checkpoints and perform SMM estimation, saving the result in the result folder smm_basic or smm_risky.


### MCMC version of structural estimation

For the Basic Model, we also implement its MCMC version. It will first load the checkpoints and then construct the MCMC process in two steps:

Step 1: Moment prior generation
Step 2: Fit the saved prior sample to a multivariate normal distribution and use the posterior distribution conditioned on the benchmark moments to accept and reject samples according to the likelihood ratio.

Prior moments sample generation, saving cached result to results/smm_mcmc_basic folder:
```bash
python basic_SMM_mcmc_prior.py --n-prior-samples 200 --gpu 0 
```
Load saved cached result and construct posterior distribution for MCMC inference:

```bash
python basic_SMM_mcmc_posterior.py \\
        --prior-cache ./results/smm_mcmc_basic/prior_cache.npz
```

## Testing

The project includes **235 automated tests** organized into unit tests and integration tests. All tests run on **CPU only** (no GPU required) and use **pytest** as the test framework.

### Quick Start

```bash
# Install the package (includes pytest as a dependency)
pip install -e .

# Run all tests
pytest

# Run with verbose output
pytest -v
```

### Running Specific Test Subsets

```bash
# Unit tests only
pytest tests/unit/

# Integration tests only
pytest tests/integration/

# A specific test file
pytest tests/unit/test_grid_builder.py

# A specific test class or method
pytest tests/unit/test_bellman_kernels.py::TestComputeEV::test_shape_2d

# DL-specific unit tests
pytest tests/unit/dl/
```

### Test Directory Layout

```
tests/
├── __init__.py
├── unit/                              # Fast, isolated tests (~seconds)
│   ├── conftest.py                    # Shared fixtures (make_test_params, etc.)
│   ├── test_adjust_kernels.py         # VFI adjust-tile kernel
│   ├── test_basic_simulator.py        # Basic model simulator
│   ├── test_bellman_kernels.py        # Bellman operators
│   ├── test_bond_price_kernels.py     # Bond price update kernel
│   ├── test_chunk_accumulate.py       # Tile-index remapping & accumulation
│   ├── test_flows.py                  # Cash-flow builders
│   ├── test_grid_builder.py           # Grid construction
│   ├── test_grid_utils.py             # Interpolation & grid utilities
│   ├── test_policies.py               # Policy extraction
│   ├── test_risky_simulator.py        # Risky model simulator
│   ├── test_tile_executor.py          # Tile executor
│   ├── test_tile_strategy.py          # Chunk-size computation
│   ├── test_wait_kernels.py           # Wait-branch kernels
│   └── dl/                            # Deep Learning module tests
│       ├── test_config.py             # EconomicParams & DeepLearningConfig
│       ├── test_econ_functions.py     # Core economic functions
│       ├── test_fischer_burmeister.py # Fischer-Burmeister smoothing
│       ├── test_neural_net_factory.py # Neural network factory
│       ├── test_normalizers.py        # State-space normalizers
│       ├── test_risky_simulator_alignment.py # Simulator alignment
│       ├── test_state_sampler.py      # GPU-side state sampling
│       └── test_transitions.py        # AR(1) productivity transitions
└── integration/                       # End-to-end tests (~10-20s each)
    ├── test_basic_integration.py      # Basic VFI solve on tiny grid
    ├── test_risky_integration.py      # Risky VFI solve on tiny grid
    ├── test_roundtrip.py              # Solve → simulate round-trip
    └── test_model_smoke.py            # DL model smoke tests
```

### Test Suite Reference

#### Unit Tests (`tests/unit/`)

| File | Module Under Test | Key Checks |
|---|---|---|
| `test_adjust_kernels.py` | `vfi.kernels.adjust_kernels` | Output shapes/dtypes, index bounds, constraint enforcement, no NaN |
| `test_bellman_kernels.py` | `vfi.kernels.bellman_kernels` | `compute_ev` (2-D/3-D), `bellman_update` (adjust/wait/default), `sup_norm_diff` |
| `test_bond_price_kernels.py` | `vfi.kernels.bond_price_kernels` | Risk-free pricing, default scenarios, relaxation blending, price floor |
| `test_chunk_accumulate.py` | `vfi.kernels.chunk_accumulate` | Single/multi-tile global indexing, no-improvement passthrough, offset arithmetic |
| `test_flows.py` | `vfi.flows.adjust_flow`, `debt_components` | Output shapes, no NaN, productivity monotonicity, zero-debt components |
| `test_grid_builder.py` | `vfi.grids.grid_builder` | Productivity/capital/debt grid shapes, transition matrix stochasticity, sorting |
| `test_grid_utils.py` | `vfi.grids.grid_utils` | Interpolation at grid points/midpoints, extrapolation clamping, batch dims |
| `test_policies.py` | `vfi.policies` | Basic/risky policy extraction, default masking, adjust-vs-wait indicator, flat-index decomposition |
| `test_basic_simulator.py` | `vfi.simulation.basic_simulator` | History shapes, capital bounds, convergence, seed reproducibility |
| `test_risky_simulator.py` | `vfi.simulation.risky_simulator` | Never-default/always-default scenarios, depreciation map, history shapes |
| `test_tile_executor.py` | `vfi.chunking.tile_executor` | Output shapes, single-tile/multi-tile kernel call counts, no residual `-inf` |
| `test_tile_strategy.py` | `vfi.chunking.tile_strategy` | Small/medium/large grids, VRAM budget compliance, symmetric choice dims |
| `test_wait_kernels.py` | `vfi.kernels.wait_kernels` | Wait-flow shapes, no NaN, constraint penalty, optimal debt index selection |
| `dl/test_config.py` | `config.economic_params`, `config.dl_config` | Parameter validation, config loading |
| `dl/test_normalizers.py` | `core.standardize` | Round-trip normalisation, boundary values, missing-field errors |
| `dl/test_econ_functions.py` | `econ.*` | Production, adjustment cost, cash-flow, collateral, issuance cost functions |
| `dl/test_fischer_burmeister.py` | `core.math` | Fischer-Burmeister complementarity smoothing |
| `dl/test_neural_net_factory.py` | `core.nets` | MLP construction with various activations and layer norms |
| `dl/test_state_sampler.py` | `core.sampling.state_sampler` | GPU-side uniform sampling within bounds |
| `dl/test_transitions.py` | `core.sampling.transitions` | Log-AR(1) transition, mean reversion, shock scaling |
| `dl/test_risky_simulator_alignment.py` | `simulator.dl.risky_final` | DL vs VFI simulator alignment |

#### Integration Tests (`tests/integration/`)

| File | What It Tests | Key Checks |
|---|---|---|
| `test_basic_integration.py` | `BasicModelVFI.solve()` on a 10×3 grid | Result keys, V-shape, non-negativity, monotonicity in K and Z, policy bounds, transition matrix |
| `test_risky_integration.py` | `RiskyDebtModelVFI.solve()` on a 8×6×3 grid | Result keys, V-shape, non-negativity, bond-price bounds [0, q_rf], default region, policy index validity |
| `test_roundtrip.py` | Solve risky VFI → `RiskySimulator.run()` | Simulation completes, history shapes, K/B within grid bounds, finite stats, no NaN |
| `test_model_smoke.py` | `BasicModelDL` and `RiskFreeModelDL` | Construction, single training-step finiteness, network output shapes |
