# Economic Models: Value Function Iteration and Deep Learning Solvers

A Python framework for solving dynamic corporate finance models using both traditional Value Function Iteration (VFI) and modern deep learning approaches. This repository implements solvers for the Basic RBC Model and the Risky Debt Model with endogenous default.

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

**Deep Learning with AiO Loss**: A neural network approach using the All-in-One (AiO) loss function, which combines Bellman residuals and Euler equation errors. This method enables efficient policy and value function approximation with curriculum learning for stable training.

---

## Models

### Basic RBC Model

A standard Real Business Cycle model with capital adjustment costs.

| Component | Description |
|-----------|-------------|
| State Space | Capital (K), Productivity (Z) |
| Choice Variable | Next-period capital (K') |
| Production | Cobb-Douglas: Y = Z × K^θ |
| Features | Convex adjustment costs, AR(1) productivity shocks |

### Risky Debt Model

A corporate finance model with risky debt and endogenous default options.

| Component | Description |
|-----------|-------------|
| State Space | Capital (K), Debt (B), Productivity (Z) |
| Choice Variables | Next-period capital (K'), next-period debt (B') |
| Bond Pricing | Risk-neutral pricing with endogenous default |
| Features | Collateral constraints, equity issuance costs, tax shields |

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
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package in development mode
pip install -e .

```

## Project Structure
```bash
econ_models/
├── cli/                              # Command-line interfaces
│   ├── solve_vfi.py                  # VFI solver CLI
│   └── train_dl.py                   # Deep learning training CLI
│
├── config/                           # Configuration classes
│   ├── economic_params.py            # Economic parameter definitions
│   ├── vfi_config.py                 # VFI grid configuration
│   └── dl_config.py                  # Deep learning configuration
│
├── core/                             # Core utilities
│   ├── types.py                      # Type definitions and precision
│   ├── nets.py                       # Neural network factory
│   ├── standardize.py                # State space normalization
│   ├── math.py                       # Math utilities wrapper
│   ├── grids/
│   │   └── tauchen.py                # Tauchen discretization
│   ├── econ/                         # Economic calculations
│   │   ├── production.py
│   │   ├── adjustment_costs.py
│   │   ├── cash_flow.py
│   │   ├── bond_pricing.py
│   │   ├── collateral.py
│   │   └── steady_state.py
│   └── sampling/                     # Sampling utilities
│       ├── transitions.py
│       ├── state_sampler.py
│       ├── candidate_sampler.py
│       └── curriculum.py
│
├── vfi/                              # Value Function Iteration
│   ├── engine.py                     # Generic VFI engine
│   ├── basic.py                      # Basic model solver
│   ├── risky_debt.py                 # Risky debt model solver
│   ├── bounds.py                     # Automatic boundary discovery
│   ├── grids/
│   │   └── grid_builder.py
│   └── simulation/
│       └── simulator.py
│
├── dl/                               # Deep Learning solvers
│   ├── basic.py                      # Basic model DL solver
│   ├── risky.py                      # Risky debt model DL solver
│   └── training/
│       └── dataset_builder.py
│
├── io/                               # I/O utilities
│   ├── file_utils.py                 # JSON file operations
│   ├── artifacts.py                  # NumPy artifact storage
│   └── checkpoints.py                # Model checkpoint utilities
│
├── tests/                            # Testing scripts
│   ├── unit    # unit test
│   └── integration    # integration test
│
├── hyperparam/                       # Configuration files
│   ├── prefixed/                     # User-defined parameters
│   │   ├── econ_params_basic.json
│   │   ├── econ_params_risky.json
│   │   ├── vfi_params.json
│   │   └── dl_params.json
│   └── autogen/                      # Auto-generated boundaries
│       ├── bounds_basic.json
│       └── bounds_risky.json
│
└── ground_truth/                     # VFI solutions
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
### Step 1: Configure Parameters
Create or modify the JSON configuration files in hyperparam/prefixed/:

Economic Parameters (econ_params_basic.json or econ_params_risky.json)

### Step 2: Solve with VFI
Run the VFI solver to discover boundaries and generate ground truth solutions:

```bash
# Solve Basic RBC Model
solve_vfi --model basic

# Solve Risky Debt Model
solve_vfi --model risky
```
This will:

Automatically discover state space boundaries via simulation
Save boundaries to hyperparam/autogen/bounds_<model>.json
Compute and save ground truth value functions to ground_truth/

### Step 3: Train Deep Learning Models
Train neural network approximations using the validated boundaries:

```bash
# Train Basic Model
train-dl --model basic

# Train Risky Debt Model
train-dl --model risky
```

Training features:

Curriculum Learning: Gradually expands sampling domain from steady state
Target Networks: Polyak averaging for training stability
AiO Loss: Combines Bellman and Euler residuals (basic model)
Automatic Checkpointing: Saves weights periodically

### Step 4: Validate Results
Evaluate deep learning solution quality against VFI ground truth:

```bash
# Validate Basic Model
python effectiveness_dl_basic.py

# Validate Risky Debt Model
python effectiveness_dl_risky.py
```

## Testing and Validation
### Ground Truth Validation
The VFI solutions provide benchmark value functions and policy functions. Validation includes:

| Check | Description |
|-------|-------------|
| Convergence Check | Verify VFI has converged within tolerance |
| Boundary Hit Analysis | Ensure simulated paths stay within the grid |
| 3D value surface | Check value variation across state space |


### Deep Learning Effectiveness Assessment
The effectiveness scripts evaluate:

| Metric | Description |
|--------|-------------|
| Mean Absolute Error | Average absolute deviation from VFI solution |
| Max Absolute Error | Worst-case deviation from VFI solution across state space |
| Mean Absolute Percentage Error | DL solution deviation divided by value |
| Total life time reward diviation | Simulation results deviation from theoratical value  |
| Bellman Residual | Violation of Bellman equation |
| Euler Residual | For basic model using FOC constrain Euler residual is enabled |
| Recovery rate of default states | For risky model f1 score is used to measure accuracy of identifying default states |


## Running Tests

```bash
# Run all validation tests
python -m pytest tests/

# Run specific effectiveness evaluation
python tests/effectiveness_dl_basic.py
python tests/effectiveness_dl_risky.py

```