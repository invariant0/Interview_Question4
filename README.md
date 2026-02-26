# Economic Models: Value Function Iteration and Deep Learning Solvers

A Python framework for solving dynamic corporate finance models using both traditional Value Function Iteration (VFI) and modern deep learning approaches. This repository implements solvers for the Basic Investment Model and the Risky Debt Investment Model with endogenous default.

---

## Table of Contents

- [Overview](#overview)
- [Models](#models)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Workflow](#workflow)
- [Usage](#usage)
- [Testing and Validation](#testing-and-validation)
- [Configuration](#configuration)
- [License](#license)

---

## Overview

This repository provides two complementary approaches for solving infinite-horizon dynamic programming problems in corporate finance:

**Value Function Iteration (VFI)**: A traditional discrete-grid method that provides high-accuracy "ground truth" solutions through Bellman equation iteration. The VFI solver also performs automatic boundary discovery to ensure the computational domain captures equilibrium dynamics.

**Deep Learning with AiO Loss**: Our neural network training follows the deep learning framework of Maliar, Maliar, and Winant (2021). We leverage their All-in-One (AiO) expectation operator, which provides unbiased estimation of nested expectations through the product of two independent samples—a technique that reduces computational cost from $O(n^2)$ to $O(n)$ for integration. The key implementation challenge, as noted by Maliar et al. (2021), is enforcing maximization on the right-hand side of the Bellman equation. Their original work employs FOC constraints. We adopt FOC for basic modeling and extend this to risky debt modeling by also implementing direct grid search maximization, providing a complementary approach that avoids derivative computation.

---

## Models

### Basic Investment Model

A standard basic investment model with capital adjustment costs.

| Component | Description |
|-----------|-------------|
| State Space | Capital (K), Productivity (Z) |
| Choice Variable | Investment Rate (I) |
| Production | Cobb-Douglas: Y = Z × K^θ |
| Features | Convex adjustment costs, AR(1) productivity shocks, smooth objective enabling FOC-based optimization |

### Investment Model with Risky Debt

A corporate finance model with risky debt and endogenous default options.

| Component | Description |
|-----------|-------------|
| State Space | Capital (K), Debt (B), Productivity (Z) |
| Choice Variables | Next-period capital (K'), next-period debt (B') |
| Bond Pricing | Bond price are not fixed but dynamic with competitive lending |
| Features | Collateral constraints, equity issuance costs, tax shields, non-differentiable kinks at default boundaries requiring smooth approximation |

---

## Installation

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
asdfsdf
## Project Structure

```bash
econ-dl/
├── src/
│   └── econ_models/
│       ├── cli/
│       │   ├── solve_vfi.py
│       │   └── train_dl.py
│       │
│       ├── config/
│       │   ├── dl_config.py
│       │   ├── economic_params.py
│       │   └── vfi_config.py
│       │
│       ├── core/
│       │   ├── sampling/
│       │   │   ├── candidate_sampler.py
│       │   │   ├── curriculum.py
│       │   │   ├── state_sampler.py
│       │   │   └── transitions.py
│       │   ├── math.py
│       │   ├── nets.py
│       │   ├── standardize.py
│       │   └── types.py
│       │
│       ├── dl/
│       │   ├── training/
│       │   │   ├── dataset_builder.py
│       │   │   └── loss_calculator.py
│       │   ├── basic.py
│       │   └── risky.py
│       │   └── risky_upgrade.py
│       │
│       ├── econ/
│       │   ├── adjustment_costs.py
│       │   ├── bond_pricing.py
│       │   ├── cash_flow.py
│       │   ├── collateral.py
│       │   ├── production.py
│       │   └── steady_state.py
│       │
│       ├── grids/
│       │   └── tauchen.py
│       │
│       ├── io/
│       │   ├── artifacts.py
│       │   ├── checkpoints.py
│       │   └── file_utils.py
│       │
│       └── vfi/
│           ├── grids/
│           │   └── grid_builder.py
│           ├── simulation/
│           │   └── simulator.py
│           ├── basic.py
│           ├── bounds.py
│           ├── engine.py
│           └── risky_debt.py
│
├── tests/
│   ├── unit/
│   └── integration/
│
├── hyperparam/
│   ├── prefixed/
│   └── autogen/
│
└── ground_truth/
    ├── basic_model_vfi_results.npz
    └── risky_debt_model_vfi_results.npz
```

## Workflow
The recommended workflow ensures consistent and reproducible results:
```bash
┌─────────────────────────────────────────────────────────────────┐
│                    1. CONFIGURE PARAMETERS                       │
│  Define economic parameters and solver settings in JSON files   │
│  Location: hyperparam/prefixed/                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   2. RUN VFI SOLVER                              │
│  Discover state space boundaries and compute ground truth       │
│  Command: solve_vfi --model <type>    │
│  Output: boundaries → hyperparam/autogen/                        │
│          solutions → ground_truth/                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 3. TRAIN DEEP LEARNING MODEL                     │
│  Train neural networks using validated boundaries               │
│  Command: train_dl --model <type>     │
│  Output: checkpoints → checkpoints/                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   4. VALIDATE RESULTS                            │
│  Compare DL solutions against VFI ground truth                  │
│  Scripts: effectiveness_dl_basic.py, effectiveness_dl_risky.py │
└─────────────────────────────────────────────────────────────────┘
```


### Important Notes
VFI must run first: The VFI solver determines economically reasonable state space boundaries. These boundaries are saved to hyperparam/autogen/ and are required for deep learning training.

Parameter consistency: The validation scripts check that boundaries were computed with the same economic parameters currently in use. If parameters change, re-run the VFI solver.

Ground truth generation: VFI solutions serve as benchmarks for evaluating deep learning model quality.

## Usage
### Step 1: Setting parameters value for economic modeling
Create or modify the JSON configuration files in hyperparam/prefixed/econ_params_*.json:

```bash
# basic model economic settings
{
    "discount_factor": 0.96,
    "capital_share": 0.70,
    "depreciation_rate": 0.15,
    "productivity_persistence": 0.70,
    "productivity_std_dev": 0.15,
    "adjustment_cost_convex": 0.05,
    "adjustment_cost_fixed": 0.0,
    "equity_issuance_cost_fixed": 0.0,
    "equity_issuance_cost_linear": 0.0,
    "default_cost_proportional": 0.0,
    "corporate_tax_rate": 0.0,
    "risk_free_rate": 0.04,
    "collateral_recovery_fraction": 0.5
}
# risky model economic settings
{
    "discount_factor": 0.96,
    "capital_share": 0.6,
    "depreciation_rate": 0.15,
    "productivity_persistence": 0.7,
    "productivity_std_dev": 0.15,
    "adjustment_cost_convex": 0.4,
    "adjustment_cost_fixed": 0.05,
    "equity_issuance_cost_fixed": 0.08,
    "equity_issuance_cost_linear": 0.028,
    "default_cost_proportional": 0.3,
    "corporate_tax_rate": 0.2,
    "risk_free_rate": 0.04,
    "collateral_recovery_fraction": 0.5
}
```

### Step 2: Solve with VFI

Modify the VFI configurations if needed in hyperparam/prefixed/vfi_params.json

Run the VFI solver to discover boundaries and generate ground truth solutions:

```bash
# Solve Basic RBC Model
solve-vfi --model basic

# Solve Risky Debt Model
solve-vfi --model risky
```
This will:

1. Automatically discover state space boundaries via simulation

2. Save boundaries to hyperparam/autogen/bounds_<model>.json
Compute and save ground truth value functions to ground_truth/

Alternatively you can download the ground_truth and put it in the ./ground_truth directory of the project here: https://drive.google.com/drive/folders/1QDq4vC87LEuFycMqK76y6bkvZx4UcpPM?usp=drive_link

### Step 3: Train Deep Learning Models
Train neural network approximations using the validated boundaries:

```bash
# Train Basic Model
train-dl --model basic

# Train Risky Debt Model
train-dl --model risky

# Train Risky Debt Model with FB function
train-dl --model risky_upgrade
```

Alternatively you can download the checkpoint and put it in the ./ground_truth directory of the project here: https://drive.google.com/drive/folders/14y0NWKdKzn-BYPP4wDecb7Z1oYXCUZpv?usp=drive_link

### Training features:

**For Basic Model**  
  We employ the AiO (All-in-One) expectation operator from Maliar et al. (2021), minimizing both Bellman residuals and FOC residuals jointly. We use separate value and policy networks to avoid gradient conflicts during optimization—the value network learns the continuation value while the policy network learns optimal investment decisions, with each network receiving clean gradient signals for its respective objective.

**For Risky Model**  
  We use direct grid search maximization with Monte Carlo simulation (sample size 30) to approximate the maximization operator on the right-hand side of the Bellman equation. To stabilize training, we employ: (1) a target network with gradual (soft) updates to prevent oscillation from bootstrapping, and (2) curriculum learning that initially focuses on states far from the default boundary—where the value function is smooth and gradients are well-defined—then gradually expands coverage to include states near default. Implementation details and rationale are provided in the full report.

**For Risky Model upgrade**  
  FB function enforce the limited liability, continouse in value function.

### Step 4: Validate Results
Evaluate deep learning solution quality against VFI ground truth:

## Testing and Validation
### Ground Truth Validation
The VFI solutions provide benchmark for value functions and company economic behavior under optimal strategy. Validation includes:

| Check | Description |
|-------|-------------|
| Convergence Diagnostics | Report iterations to convergence, tolerance achieved |
| Grid Size Convergence Check | Verify current grid size is sufficient |
| Boundary Hit Analysis | Verify the current boundary setting are sufficient to include all equilibrium states value |
| Economic decision analysis | Verify the policy from the value function make economic sense |
| 3D value surface | Visual check on value variation across state space |


### Deep Learning Effectiveness Assessment
The effectiveness of the deep learning solution is defined with the following metrics:

| Metric | Description |
|--------|-------------|
| Mean Absolute Error | Average absolute deviation from VFI solution |
| Max Absolute Error | Worst-case deviation from VFI solution across state space |
| Mean Absolute Percentage Error | DL solution deviation divided by states value |
| Total life time reward deviation | Simulation results deviation from theoretical value  |
| Bellman Residual | Violation of Bellman equation |
| Euler Residual | For basic model using FOC constraint Euler residual is enabled |
| Recovery rate of default states | For risky model f1 score is used to measure accuracy of identifying default states |
| Risky Bound Price Error | For risky model we test whether dl produced Risky Bond Price align with VFI benchmark |
| Ergodic Moment Comparison | Compare mean(K), std(K), autocorr(K) under both policies |
| Simulation based Comparison | Compare mean(K), std(K), autocorr(K) under both policies |


## Running Tests

```bash
# Run all validation tests
python -m unittest discover -s tests/unit
python -m unittest discover -s tests/integration

# validate benchmark ground truth
python ./validate_basic_model.py
python ./validate_risky_model.py

# evaluate effectivenss of dl solution
bash ./effectiveness_dl_basic.sh
bash ./effectiveness_dl_risky.sh
bash ./effectiveness_dl_risky_upgrade.sh

```