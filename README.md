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

## Testing and Validation

### Ground Truth Validation
The VFI solutions provide a benchmark for value functions and company economic behavior. Validation includes convergence diagnostics, boundary analysis, and checking economic logic.

### Deep Learning Effectiveness
Key metrics compared against VFI benchmarks include Mean/Max Absolute Error, Bellman Residuals, and Euler Residuals.

```bash
# Run all validation tests
python -m unittest discover -s tests/unit
python -m unittest discover -s tests/integration

# Validate benchmark ground truth
python ./validate_basic_model.py
python ./validate_risky_model.py

# Evaluate effectiveness of DL solution
bash ./effectiveness_dl_basic.sh
bash ./effectiveness_dl_risky.sh
bash ./effectiveness_dl_risky_upgrade.sh
```
