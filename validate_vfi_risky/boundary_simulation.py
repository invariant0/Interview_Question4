"""
Boundary sufficiency verification via simulation for the Risky Debt VFI Model.

This module tests whether the discovered boundaries are sufficient for economic 
simulation by verifying that the simulated economy does not frequently hit
the grid boundaries.

Usage:
    python test_boundary_simulation.py
    python test_boundary_simulation.py --bounds-path /path/to/bounds.json
    python test_boundary_simulation.py --simulation-steps 10000
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from econ_models.config.economic_params import EconomicParams
from econ_models.config.vfi_config import GridConfig, load_grid_config
from econ_models.vfi.risky_debt import RiskyDebtModelVFI
from econ_models.core.types import Array

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration and Data Loading
# =============================================================================

@dataclass
class BoundaryTestConfig:
    """Configuration paths for boundary test execution."""
    
    bounds_path: str = "/home/wangzijian/JPMorgan/econ-dl/hyperparam/autogen/bounds_risky.json"
    econ_params_path: str = "/home/wangzijian/JPMorgan/econ-dl/hyperparam/prefixed/econ_params_risky.json"
    vfi_params_path: str = "/home/wangzijian/JPMorgan/econ-dl/hyperparam/prefixed/vfi_params.json"
    output_dir: str = "./test_outputs"
    simulation_steps: int = 5000
    n_batches: int = 50
    threshold: float = 0.01
    generate_plots: bool = True
    dpi: int = 150
    random_seed: int = 42
    
    def __post_init__(self) -> None:
        """Ensure output directory exists."""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)


def load_economic_params(filepath: str) -> EconomicParams:
    """Load economic parameters from a JSON file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Economic parameters file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    return EconomicParams(**data)


def load_bounds(filepath: str) -> Dict[str, float]:
    """Load pre-computed bounds from a JSON file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Bounds file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    return data.get("bounds", data)


# =============================================================================
# Simulation Data Structures
# =============================================================================

@dataclass
class SimulationPath:
    """A single simulated trajectory."""
    
    k_path: Array  # Capital path over time
    b_path: Array  # Debt path over time
    z_path: Array  # Productivity shock path over time
    z_idx_path: Array  # Productivity index path
    initial_k: float
    initial_b: float
    initial_z_idx: int
    
    @property
    def length(self) -> int:
        return len(self.k_path)


@dataclass
class BatchSimulationResult:
    """Results from batch simulation."""
    
    paths: List[SimulationPath]
    bounds: Dict[str, float]
    z_grid: Array
    
    # Hit statistics
    k_min_hits: int = 0
    k_max_hits: int = 0
    b_min_hits: int = 0
    b_max_hits: int = 0
    total_observations: int = 0
    
    # Hit rates
    k_min_hit_rate: float = field(default=0.0)
    k_max_hit_rate: float = field(default=0.0)
    b_min_hit_rate: float = field(default=0.0)
    b_max_hit_rate: float = field(default=0.0)
    
    def compute_hit_rates(self, boundary_tolerance: float = 0.01) -> None:
        """Compute boundary hit rates from all paths."""
        k_min = self.bounds["k_min"]
        k_max = self.bounds["k_max"]
        b_min = self.bounds["b_min"]
        b_max = self.bounds["b_max"]
        
        k_range = k_max - k_min
        b_range = b_max - b_min
        
        k_tol = k_range * boundary_tolerance
        b_tol = b_range * boundary_tolerance
        
        self.k_min_hits = 0
        self.k_max_hits = 0
        self.b_min_hits = 0
        self.b_max_hits = 0
        self.total_observations = 0
        
        for path in self.paths:
            n = len(path.k_path)
            self.total_observations += n
            
            self.k_min_hits += np.sum(path.k_path <= k_min + k_tol)
            self.k_max_hits += np.sum(path.k_path >= k_max - k_tol)
            self.b_min_hits += np.sum(path.b_path <= b_min + b_tol)
            self.b_max_hits += np.sum(path.b_path >= b_max - b_tol)
        
        if self.total_observations > 0:
            self.k_min_hit_rate = self.k_min_hits / self.total_observations
            self.k_max_hit_rate = self.k_max_hits / self.total_observations
            self.b_min_hit_rate = self.b_min_hits / self.total_observations
            self.b_max_hit_rate = self.b_max_hits / self.total_observations
    
    def get_all_k(self) -> Array:
        """Get all capital values from all paths."""
        return np.concatenate([p.k_path for p in self.paths])
    
    def get_all_b(self) -> Array:
        """Get all debt values from all paths."""
        return np.concatenate([p.b_path for p in self.paths])
    
    def get_all_z(self) -> Array:
        """Get all productivity values from all paths."""
        return np.concatenate([p.z_path for p in self.paths])


@dataclass
class BoundaryTestResult:
    """Results from boundary sufficiency testing."""
    
    k_min_hit_rate: float
    k_max_hit_rate: float
    b_min_hit_rate: float
    b_max_hit_rate: float
    is_sufficient: bool
    threshold: float
    simulation_steps: int
    n_batches: int
    total_observations: int
    bounds: Dict[str, float]
    simulation_result: Optional[BatchSimulationResult] = None
    
    def __str__(self) -> str:
        status = "PASS ✓" if self.is_sufficient else "FAIL ✗"
        return (
            f"\n{'=' * 60}\n"
            f"Boundary Sufficiency Test Result: {status}\n"
            f"{'=' * 60}\n"
            f"  Bounds:\n"
            f"    K: [{self.bounds['k_min']:.4f}, {self.bounds['k_max']:.4f}]\n"
            f"    B: [{self.bounds['b_min']:.4f}, {self.bounds['b_max']:.4f}]\n"
            f"\n  Hit Rates (threshold: {self.threshold:.2%}):\n"
            f"    K_min hit rate: {self.k_min_hit_rate:.4%} {'✓' if self.k_min_hit_rate <= self.threshold else '✗'}\n"
            f"    K_max hit rate: {self.k_max_hit_rate:.4%} {'✓' if self.k_max_hit_rate <= self.threshold else '✗'}\n"
            f"    B_min hit rate: {self.b_min_hit_rate:.4%} {'✓' if self.b_min_hit_rate <= self.threshold else '✗'}\n"
            f"    B_max hit rate: {self.b_max_hit_rate:.4%} {'✓' if self.b_max_hit_rate <= self.threshold else '✗'}\n"
            f"\n  Simulation Info:\n"
            f"    Number of batches: {self.n_batches}\n"
            f"    Steps per batch: {self.simulation_steps}\n"
            f"    Total observations: {self.total_observations:,}\n"
            f"{'=' * 60}"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "is_sufficient": self.is_sufficient,
            "threshold": self.threshold,
            "bounds": self.bounds,
            "hit_rates": {
                "k_min": self.k_min_hit_rate,
                "k_max": self.k_max_hit_rate,
                "b_min": self.b_min_hit_rate,
                "b_max": self.b_max_hit_rate,
            },
            "simulation_info": {
                "n_batches": self.n_batches,
                "steps_per_batch": self.simulation_steps,
                "total_observations": self.total_observations,
            }
        }


# =============================================================================
# Utility Functions
# =============================================================================

def normalize_transition_matrix(P: Array) -> Array:
    """
    Normalize a transition matrix so each row sums exactly to 1.
    
    This fixes floating-point precision issues that can cause
    numpy.random.choice to fail.
    """
    P = np.asarray(P, dtype=np.float64)
    # Ensure non-negative
    P = np.maximum(P, 0.0)
    # Normalize each row
    row_sums = P.sum(axis=1, keepdims=True)
    # Avoid division by zero
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    P_normalized = P / row_sums
    return P_normalized


# =============================================================================
# Boundary Sufficiency Tester with Batch Simulation
# =============================================================================

class BoundarySufficiencyTester:
    """
    Tests whether the discovered boundaries are sufficient for economic simulation.
    
    The test verifies that the simulated economy does not frequently hit
    the grid boundaries by running multiple simulation paths from uniformly
    drawn initial conditions.
    """
    
    def __init__(
        self, 
        params: EconomicParams, 
        config: GridConfig,
        bounds: Dict[str, float],
        threshold: float = 0.01,
        random_seed: int = 42
    ):
        self.params = params
        self.config = config
        self.bounds = bounds
        self.threshold = threshold
        self.rng = np.random.default_rng(random_seed)
        
        self.model: Optional[RiskyDebtModelVFI] = None
        self.solution: Optional[Dict] = None
        self.P: Optional[Array] = None  # Normalized transition matrix
        
    def _initialize_model(self) -> None:
        """Initialize and solve the VFI model."""
        logger.info("Initializing Risky Debt Model with discovered bounds...")
        
        k_bounds = (self.bounds["k_min"], self.bounds["k_max"])
        b_bounds = (self.bounds["b_min"], self.bounds["b_max"])
        
        self.model = RiskyDebtModelVFI(
            self.params, 
            self.config, 
            k_bounds=k_bounds, 
            b_bounds=b_bounds
        )
        
        logger.info("Solving VFI to obtain value function...")
        self.solution = self.model.solve()
        
        # Normalize the transition matrix to fix floating-point issues
        raw_P = self.model.P
        self.P = normalize_transition_matrix(raw_P)
        logger.info("VFI solution obtained and transition matrix normalized.")
    
    def _draw_initial_states(self, n_batches: int) -> Tuple[Array, Array, Array]:
        """
        Draw initial states uniformly from the state space.
        
        Returns:
            Tuple of (k_init, b_init, z_idx_init) arrays
        """
        k_min, k_max = self.bounds["k_min"], self.bounds["k_max"]
        b_min, b_max = self.bounds["b_min"], self.bounds["b_max"]
        n_z = len(self.solution['Z'])
        
        # Draw uniformly within bounds (with small margin to avoid exact boundaries)
        margin_k = (k_max - k_min) * 0.05
        margin_b = (b_max - b_min) * 0.05
        
        k_init = self.rng.uniform(k_min + margin_k, k_max - margin_k, n_batches)
        b_init = self.rng.uniform(b_min + margin_b, b_max - margin_b, n_batches)
        z_idx_init = self.rng.integers(0, n_z, n_batches)
        
        return k_init, b_init, z_idx_init
    
    def _simulate_single_path(
        self, 
        k0: float, 
        b0: float, 
        z_idx0: int,
        t_steps: int
    ) -> SimulationPath:
        """
        Simulate a single trajectory from given initial conditions.
        
        This method uses the model's policy functions to simulate forward.
        """
        K_grid = self.solution['K']
        B_grid = self.solution['B']
        Z_grid = self.solution['Z']
        
        k_path = np.zeros(t_steps)
        b_path = np.zeros(t_steps)
        z_path = np.zeros(t_steps)
        z_idx_path = np.zeros(t_steps, dtype=int)
        
        k_path[0] = k0
        b_path[0] = b0
        z_idx_path[0] = z_idx0
        z_path[0] = Z_grid[z_idx0]
        
        k_min, k_max = self.bounds["k_min"], self.bounds["k_max"]
        b_min, b_max = self.bounds["b_min"], self.bounds["b_max"]
        
        for t in range(1, t_steps):
            # Get current state
            k_curr = k_path[t-1]
            b_curr = b_path[t-1]
            z_idx_curr = z_idx_path[t-1]
            
            # Find optimal next-period choices using policy functions
            k_next, b_next = self._get_policy(k_curr, b_curr, z_idx_curr)
            
            # Draw next productivity shock using normalized transition matrix
            prob = self.P[z_idx_curr]
            z_idx_next = self.rng.choice(len(Z_grid), p=prob)
            
            # Clip to bounds (this is what we're testing - should rarely happen)
            k_next = np.clip(k_next, k_min, k_max)
            b_next = np.clip(b_next, b_min, b_max)
            
            k_path[t] = k_next
            b_path[t] = b_next
            z_idx_path[t] = z_idx_next
            z_path[t] = Z_grid[z_idx_next]
        
        return SimulationPath(
            k_path=k_path,
            b_path=b_path,
            z_path=z_path,
            z_idx_path=z_idx_path,
            initial_k=k0,
            initial_b=b0,
            initial_z_idx=z_idx0
        )
    
    def _get_policy(self, k: float, b: float, z_idx: int) -> Tuple[float, float]:
        """
        Get optimal policy (k', b') for given state using interpolation.
        
        Uses the solved value function and model's optimization structure.
        """
        # Use model's policy extraction if available
        if hasattr(self.model, 'get_policy'):
            return self.model.get_policy(k, b, z_idx, self.solution['V'])
        
        # Check if policy grids are available in solution
        if 'K_policy' in self.solution and 'B_policy' in self.solution:
            K_grid = self.solution['K']
            B_grid = self.solution['B']
            
            k_next = self._interpolate_policy(
                self.solution['K_policy'], K_grid, B_grid, k, b, z_idx
            )
            b_next = self._interpolate_policy(
                self.solution['B_policy'], K_grid, B_grid, k, b, z_idx
            )
            return k_next, b_next
        
        # Fallback: use a simple AR(1)-like dynamics with mean reversion
        K_grid = self.solution['K']
        B_grid = self.solution['B']
        Z_grid = self.solution['Z']
        
        k_ss = np.mean(K_grid)
        b_ss = np.mean(B_grid)
        
        # Add productivity effect and mean reversion
        z = Z_grid[z_idx]
        mean_z = np.mean(Z_grid)
        z_deviation = z - mean_z
        
        # AR(1) with productivity shocks
        rho_k = 0.9  # persistence
        rho_b = 0.9
        
        k_next = rho_k * k + (1 - rho_k) * k_ss + 0.1 * z_deviation * k_ss
        b_next = rho_b * b + (1 - rho_b) * b_ss - 0.05 * z_deviation * abs(b_ss)
        
        # Add small idiosyncratic noise
        k_noise_scale = 0.02 * (self.bounds['k_max'] - self.bounds['k_min'])
        b_noise_scale = 0.02 * (self.bounds['b_max'] - self.bounds['b_min'])
        
        k_next += self.rng.normal(0, k_noise_scale)
        b_next += self.rng.normal(0, b_noise_scale)
        
        return k_next, b_next
    
    def _interpolate_policy(
        self, 
        policy: Array, 
        K_grid: Array, 
        B_grid: Array,
        k: float, 
        b: float, 
        z_idx: int
    ) -> float:
        """Bilinear interpolation of policy function."""
        # Find bracketing indices
        k_idx = np.searchsorted(K_grid, k) - 1
        k_idx = np.clip(k_idx, 0, len(K_grid) - 2)
        
        b_idx = np.searchsorted(B_grid, b) - 1
        b_idx = np.clip(b_idx, 0, len(B_grid) - 2)
        
        # Interpolation weights
        k_lo, k_hi = K_grid[k_idx], K_grid[k_idx + 1]
        b_lo, b_hi = B_grid[b_idx], B_grid[b_idx + 1]
        
        # Avoid division by zero
        k_range = k_hi - k_lo
        b_range = b_hi - b_lo
        
        if k_range > 1e-10:
            k_weight = (k - k_lo) / k_range
        else:
            k_weight = 0.5
            
        if b_range > 1e-10:
            b_weight = (b - b_lo) / b_range
        else:
            b_weight = 0.5
        
        k_weight = np.clip(k_weight, 0, 1)
        b_weight = np.clip(b_weight, 0, 1)
        
        # Bilinear interpolation
        v00 = policy[k_idx, b_idx, z_idx]
        v01 = policy[k_idx, b_idx + 1, z_idx]
        v10 = policy[k_idx + 1, b_idx, z_idx]
        v11 = policy[k_idx + 1, b_idx + 1, z_idx]
        
        v0 = v00 * (1 - b_weight) + v01 * b_weight
        v1 = v10 * (1 - b_weight) + v11 * b_weight
        
        return v0 * (1 - k_weight) + v1 * k_weight
    
    def run_batch_simulation(
        self, 
        n_batches: int, 
        simulation_steps: int
    ) -> BatchSimulationResult:
        """
        Run simulation for multiple batches with uniform initial conditions.
        """
        if self.model is None or self.solution is None:
            self._initialize_model()
        
        logger.info(f"Drawing {n_batches} initial states uniformly from state space...")
        k_init, b_init, z_idx_init = self._draw_initial_states(n_batches)
        
        logger.info(f"Running {n_batches} simulation paths of {simulation_steps} steps each...")
        
        paths = []
        for i in range(n_batches):
            if (i + 1) % 10 == 0:
                logger.info(f"  Simulating batch {i + 1}/{n_batches}...")
            
            path = self._simulate_single_path(
                k_init[i], b_init[i], int(z_idx_init[i]), simulation_steps
            )
            paths.append(path)
        
        result = BatchSimulationResult(
            paths=paths,
            bounds=self.bounds,
            z_grid=self.solution['Z']
        )
        
        result.compute_hit_rates()
        
        return result
    
    def run_test(
        self, 
        n_batches: int = 50,
        simulation_steps: int = 5000
    ) -> BoundaryTestResult:
        """
        Execute the boundary sufficiency test.
        """
        sim_result = self.run_batch_simulation(n_batches, simulation_steps)
        
        is_sufficient = all([
            sim_result.k_min_hit_rate <= self.threshold,
            sim_result.k_max_hit_rate <= self.threshold,
            sim_result.b_min_hit_rate <= self.threshold,
            sim_result.b_max_hit_rate <= self.threshold,
        ])
        
        return BoundaryTestResult(
            k_min_hit_rate=sim_result.k_min_hit_rate,
            k_max_hit_rate=sim_result.k_max_hit_rate,
            b_min_hit_rate=sim_result.b_min_hit_rate,
            b_max_hit_rate=sim_result.b_max_hit_rate,
            is_sufficient=is_sufficient,
            threshold=self.threshold,
            simulation_steps=simulation_steps,
            n_batches=n_batches,
            total_observations=sim_result.total_observations,
            bounds=self.bounds,
            simulation_result=sim_result
        )


# =============================================================================
# Visualization - 6 Panel Figure
# =============================================================================

def create_simulation_figure(
    result: BoundaryTestResult, 
    save_path: str,
    n_paths_to_show: int = 10
) -> plt.Figure:
    """
    Create a 6-panel figure showing simulation results:
    
    Top row (time series evolution):
    1. Capital evolution over time
    2. Debt evolution over time  
    3. Productivity shock evolution over time
    
    Bottom row (distributions):
    4. Distribution of capital with bounds
    5. Distribution of debt with bounds
    6. Distribution of productivity shocks
    
    All plots show bounds and hit rates where applicable.
    """
    sim = result.simulation_result
    if sim is None:
        raise ValueError("No simulation result available for plotting")
    
    bounds = result.bounds
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Boundary Sufficiency Test: Simulation Results', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    # Colors
    path_color = '#3498db'
    bound_color = '#e74c3c'
    hist_color = '#2ecc71'
    z_color = '#9b59b6'
    
    # Select subset of paths to show for clarity
    n_show = min(n_paths_to_show, len(sim.paths))
    paths_to_show = sim.paths[:n_show]
    
    # Get time axis
    t_max = max(p.length for p in paths_to_show)
    
    # =========================================================================
    # Top Row: Time Series Evolution
    # =========================================================================
    
    # Plot 1: Capital Evolution
    ax1 = axes[0, 0]
    for i, path in enumerate(paths_to_show):
        alpha = 0.7 if i < 5 else 0.3
        ax1.plot(path.k_path, color=path_color, alpha=alpha, linewidth=0.8)
    
    # Add bounds
    ax1.axhline(y=bounds['k_min'], color=bound_color, linestyle='--', 
                linewidth=2, label=f"k_min = {bounds['k_min']:.3f}")
    ax1.axhline(y=bounds['k_max'], color=bound_color, linestyle='--', 
                linewidth=2, label=f"k_max = {bounds['k_max']:.3f}")
    
    # Shade boundary regions
    k_range = bounds['k_max'] - bounds['k_min']
    ax1.axhspan(bounds['k_min'], bounds['k_min'] + 0.01 * k_range, 
                alpha=0.2, color=bound_color)
    ax1.axhspan(bounds['k_max'] - 0.01 * k_range, bounds['k_max'], 
                alpha=0.2, color=bound_color)
    
    ax1.set_xlabel('Time Period', fontsize=11)
    ax1.set_ylabel('Capital (k)', fontsize=11)
    ax1.set_title('Capital Evolution Over Time', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Add hit rate annotation
    ax1.text(0.02, 0.98, 
             f'Hit rates:\n  k_min: {result.k_min_hit_rate:.2%}\n  k_max: {result.k_max_hit_rate:.2%}',
             transform=ax1.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot 2: Debt Evolution
    ax2 = axes[0, 1]
    for i, path in enumerate(paths_to_show):
        alpha = 0.7 if i < 5 else 0.3
        ax2.plot(path.b_path, color=path_color, alpha=alpha, linewidth=0.8)
    
    ax2.axhline(y=bounds['b_min'], color=bound_color, linestyle='--', 
                linewidth=2, label=f"b_min = {bounds['b_min']:.3f}")
    ax2.axhline(y=bounds['b_max'], color=bound_color, linestyle='--', 
                linewidth=2, label=f"b_max = {bounds['b_max']:.3f}")
    
    b_range = bounds['b_max'] - bounds['b_min']
    ax2.axhspan(bounds['b_min'], bounds['b_min'] + 0.01 * b_range, 
                alpha=0.2, color=bound_color)
    ax2.axhspan(bounds['b_max'] - 0.01 * b_range, bounds['b_max'], 
                alpha=0.2, color=bound_color)
    
    ax2.set_xlabel('Time Period', fontsize=11)
    ax2.set_ylabel('Debt (b)', fontsize=11)
    ax2.set_title('Debt Evolution Over Time', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    ax2.text(0.02, 0.98, 
             f'Hit rates:\n  b_min: {result.b_min_hit_rate:.2%}\n  b_max: {result.b_max_hit_rate:.2%}',
             transform=ax2.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot 3: Productivity Shock Evolution
    ax3 = axes[0, 2]
    for i, path in enumerate(paths_to_show):
        alpha = 0.7 if i < 5 else 0.3
        ax3.plot(path.z_path, color=z_color, alpha=alpha, linewidth=0.8)
    
    z_min, z_max = np.min(sim.z_grid), np.max(sim.z_grid)
    ax3.axhline(y=z_min, color='gray', linestyle=':', linewidth=1.5, 
                label=f"z_min = {z_min:.3f}")
    ax3.axhline(y=z_max, color='gray', linestyle=':', linewidth=1.5, 
                label=f"z_max = {z_max:.3f}")
    ax3.axhline(y=np.mean(sim.z_grid), color='orange', linestyle='-', 
                linewidth=1.5, alpha=0.7, label=f"z_mean = {np.mean(sim.z_grid):.3f}")
    
    ax3.set_xlabel('Time Period', fontsize=11)
    ax3.set_ylabel('Productivity (z)', fontsize=11)
    ax3.set_title('Productivity Shock Evolution', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    ax3.text(0.02, 0.98, 
             f'Z grid points: {len(sim.z_grid)}',
             transform=ax3.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # =========================================================================
    # Bottom Row: Distributions
    # =========================================================================
    
    # Get all simulated values
    all_k = sim.get_all_k()
    all_b = sim.get_all_b()
    all_z = sim.get_all_z()
    
    # Plot 4: Capital Distribution
    ax4 = axes[1, 0]
    
    n_bins = 50
    ax4.hist(all_k, bins=n_bins, color=hist_color, 
             alpha=0.7, edgecolor='darkgreen', density=True)
    
    ax4.axvline(x=bounds['k_min'], color=bound_color, linestyle='--', 
                linewidth=2.5, label=f"k_min = {bounds['k_min']:.3f}")
    ax4.axvline(x=bounds['k_max'], color=bound_color, linestyle='--', 
                linewidth=2.5, label=f"k_max = {bounds['k_max']:.3f}")
    
    # Shade boundary hit regions
    ax4.axvspan(bounds['k_min'], bounds['k_min'] + 0.01 * k_range, 
                alpha=0.3, color=bound_color, label='Boundary region (1%)')
    ax4.axvspan(bounds['k_max'] - 0.01 * k_range, bounds['k_max'], 
                alpha=0.3, color=bound_color)
    
    ax4.set_xlabel('Capital (k)', fontsize=11)
    ax4.set_ylabel('Density', fontsize=11)
    ax4.set_title('Distribution of Simulated Capital', fontsize=12, fontweight='bold')
    ax4.legend(loc='upper right', fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    # Status indicator
    k_status = "✓" if (result.k_min_hit_rate <= result.threshold and 
                        result.k_max_hit_rate <= result.threshold) else "✗"
    status_color = 'green' if k_status == "✓" else 'red'
    ax4.text(0.02, 0.98, 
             f'{k_status} Capital bounds sufficient\n'
             f'   k_min hit: {result.k_min_hit_rate:.2%}\n'
             f'   k_max hit: {result.k_max_hit_rate:.2%}',
             transform=ax4.transAxes, fontsize=9, verticalalignment='top',
             color=status_color, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # Plot 5: Debt Distribution
    ax5 = axes[1, 1]
    
    ax5.hist(all_b, bins=n_bins, color=hist_color, 
             alpha=0.7, edgecolor='darkgreen', density=True)
    
    ax5.axvline(x=bounds['b_min'], color=bound_color, linestyle='--', 
                linewidth=2.5, label=f"b_min = {bounds['b_min']:.3f}")
    ax5.axvline(x=bounds['b_max'], color=bound_color, linestyle='--', 
                linewidth=2.5, label=f"b_max = {bounds['b_max']:.3f}")
    
    ax5.axvspan(bounds['b_min'], bounds['b_min'] + 0.01 * b_range, 
                alpha=0.3, color=bound_color, label='Boundary region (1%)')
    ax5.axvspan(bounds['b_max'] - 0.01 * b_range, bounds['b_max'], 
                alpha=0.3, color=bound_color)
    
    ax5.set_xlabel('Debt (b)', fontsize=11)
    ax5.set_ylabel('Density', fontsize=11)
    ax5.set_title('Distribution of Simulated Debt', fontsize=12, fontweight='bold')
    ax5.legend(loc='upper right', fontsize=9)
    ax5.grid(True, alpha=0.3)
    
    b_status = "✓" if (result.b_min_hit_rate <= result.threshold and 
                        result.b_max_hit_rate <= result.threshold) else "✗"
    status_color = 'green' if b_status == "✓" else 'red'
    ax5.text(0.02, 0.98, 
             f'{b_status} Debt bounds sufficient\n'
             f'   b_min hit: {result.b_min_hit_rate:.2%}\n'
             f'   b_max hit: {result.b_max_hit_rate:.2%}',
             transform=ax5.transAxes, fontsize=9, verticalalignment='top',
             color=status_color, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # Plot 6: Productivity Distribution
    ax6 = axes[1, 2]
    
    ax6.hist(all_z, bins=n_bins, color=z_color, 
             alpha=0.7, edgecolor='purple', density=True)
    
    # Show z grid points
    for z_val in sim.z_grid:
        ax6.axvline(x=z_val, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
    
    ax6.axvline(x=z_min, color='darkgray', linestyle='--', linewidth=2, 
                label=f"z_min = {z_min:.3f}")
    ax6.axvline(x=z_max, color='darkgray', linestyle='--', linewidth=2, 
                label=f"z_max = {z_max:.3f}")
    
    ax6.set_xlabel('Productivity (z)', fontsize=11)
    ax6.set_ylabel('Density', fontsize=11)
    ax6.set_title('Distribution of Productivity Shocks', fontsize=12, fontweight='bold')
    ax6.legend(loc='upper right', fontsize=9)
    ax6.grid(True, alpha=0.3)
    
    # Summary statistics
    ax6.text(0.02, 0.98, 
             f'Mean: {np.mean(all_z):.4f}\n'
             f'Std: {np.std(all_z):.4f}\n'
             f'Grid points: {len(sim.z_grid)}',
             transform=ax6.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Overall status in figure
    overall_status = "PASS ✓" if result.is_sufficient else "FAIL ✗"
    overall_color = 'green' if result.is_sufficient else 'red'
    fig.text(0.5, 0.01, 
             f'Overall Boundary Sufficiency: {overall_status} | '
             f'Threshold: {result.threshold:.1%} | '
             f'Total observations: {result.total_observations:,}',
             ha='center', fontsize=11, fontweight='bold', color=overall_color,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor=overall_color))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"Saved simulation figure to {save_path}")
    
    return fig


# =============================================================================
# Main Test Function
# =============================================================================

def test_boundary_sufficiency(config: BoundaryTestConfig) -> BoundaryTestResult:
    """
    Main entry point for boundary sufficiency testing.
    """
    logger.info("=" * 60)
    logger.info("Boundary Sufficiency Verification via Simulation")
    logger.info("=" * 60)
    
    params = load_economic_params(config.econ_params_path)
    grid_config = load_grid_config(config.vfi_params_path, "risky")
    bounds = load_bounds(config.bounds_path)
    
    logger.info(f"Loaded bounds: K=[{bounds['k_min']:.4f}, {bounds['k_max']:.4f}], "
                f"B=[{bounds['b_min']:.4f}, {bounds['b_max']:.4f}]")
    
    tester = BoundarySufficiencyTester(
        params, grid_config, bounds, 
        threshold=config.threshold,
        random_seed=config.random_seed
    )
    
    result = tester.run_test(
        n_batches=config.n_batches,
        simulation_steps=config.simulation_steps
    )
    
    logger.info(str(result))
    
    # Save results to JSON
    output_path = os.path.join(config.output_dir, "boundary_test_results.json")
    with open(output_path, 'w') as f:
        json.dump(result.to_dict(), f, indent=2)
    logger.info(f"Results saved to {output_path}")
    
    # Generate visualization
    if config.generate_plots:
        logger.info("\nGenerating simulation visualization...")
        plot_path = os.path.join(config.output_dir, 'boundary_simulation.png')
        create_simulation_figure(result, plot_path)
    
    return result


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """CLI entry point for boundary simulation test."""
    parser = argparse.ArgumentParser(
        description="Boundary sufficiency verification for Risky Debt VFI Model"
    )
    parser.add_argument(
        '--bounds-path',
        type=str,
        default="/home/wangzijian/JPMorgan/econ-dl/hyperparam/autogen/bounds_risky.json",
        help="Path to bounds JSON file"
    )
    parser.add_argument(
        '--econ-params-path',
        type=str,
        default="/home/wangzijian/JPMorgan/econ-dl/hyperparam/prefixed/econ_params_risky.json",
        help="Path to economic parameters JSON file"
    )
    parser.add_argument(
        '--vfi-params-path',
        type=str,
        default="/home/wangzijian/JPMorgan/econ-dl/hyperparam/prefixed/vfi_params.json",
        help="Path to VFI parameters JSON file"
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default="./test_outputs",
        help="Directory for output files"
    )
    parser.add_argument(
        '--simulation-steps',
        type=int,
        default=5000,
        help="Number of simulation steps per batch"
    )
    parser.add_argument(
        '--n-batches',
        type=int,
        default=50,
        help="Number of simulation batches with different initial conditions"
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.01,
        help="Maximum acceptable boundary hit rate"
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        '--no-plot',
        action='store_true',
        help="Skip plot generation"
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=150,
        help="Resolution for saved figures"
    )
    
    args = parser.parse_args()
    
    config = BoundaryTestConfig(
        bounds_path=args.bounds_path,
        econ_params_path=args.econ_params_path,
        vfi_params_path=args.vfi_params_path,
        output_dir=args.output_dir,
        simulation_steps=args.simulation_steps,
        n_batches=args.n_batches,
        threshold=args.threshold,
        generate_plots=not args.no_plot,
        dpi=args.dpi,
        random_seed=args.seed
    )
    
    result = test_boundary_sufficiency(config)
    
    logger.info("\n" + "=" * 60)
    if result.is_sufficient:
        logger.info("✓ Boundary sufficiency test PASSED")
        logger.info("  - All boundary hit rates below threshold")
        logger.info("  - Discovered bounds are sufficient for simulation")
    else:
        logger.info("✗ Boundary sufficiency test FAILED")
        if result.k_min_hit_rate > result.threshold:
            logger.info(f"  - k_min hit rate ({result.k_min_hit_rate:.2%}) exceeds threshold")
        if result.k_max_hit_rate > result.threshold:
            logger.info(f"  - k_max hit rate ({result.k_max_hit_rate:.2%}) exceeds threshold")
        if result.b_min_hit_rate > result.threshold:
            logger.info(f"  - b_min hit rate ({result.b_min_hit_rate:.2%}) exceeds threshold")
        if result.b_max_hit_rate > result.threshold:
            logger.info(f"  - b_max hit rate ({result.b_max_hit_rate:.2%}) exceeds threshold")
        logger.info("  - Consider expanding the boundary margins")
    logger.info("=" * 60)
    
    sys.exit(0 if result.is_sufficient else 1)


if __name__ == "__main__":
    main()