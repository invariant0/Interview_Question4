"""
Grid size validation and convergence analysis for the Risky Debt VFI Model.

This module tests the model at multiple grid resolutions to verify convergence
of the value function, ensuring that the chosen grid resolution is sufficient.

Usage:
    python test_grid_convergence.py
    python test_grid_convergence.py --convergence-threshold 0.005
    python test_grid_convergence.py --no-plot
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

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
# Configuration
# =============================================================================

GRID_SIZES: List[Tuple[int, int]] = [
    (50, 50),
    (60, 60),
    (70, 70),
    (80, 80),
    (90, 90),
]

@dataclass
class GridValidationConfig:
    """Configuration for grid validation test."""
    
    bounds_path: str = "/home/wangzijian/JPMorgan/econ-dl/hyperparam/autogen/bounds_risky.json"
    econ_params_path: str = "/home/wangzijian/JPMorgan/econ-dl/hyperparam/prefixed/econ_params_risky.json"
    vfi_params_path: str = "/home/wangzijian/JPMorgan/econ-dl/hyperparam/prefixed/vfi_params.json"
    convergence_threshold: float = 0.01
    output_dir: str = "./test_outputs"
    generate_plots: bool = True
    dpi: int = 150
    
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
# Data Classes for Results
# =============================================================================

@dataclass
class GridErrorMetrics:
    """Metrics for error of a grid relative to the reference (finest) grid."""
    
    grid_size: Tuple[int, int]
    mean_error: float
    median_error: float
    max_error: float
    std_error: float
    p95_error: float
    p99_error: float
    l2_error: float
    error_matrix: Array
    
    @property
    def grid_label(self) -> str:
        return f"({self.grid_size[0]}, {self.grid_size[1]})"
    
    def __str__(self) -> str:
        return (
            f"{self.grid_label}: "
            f"mean={self.mean_error:.6e}, median={self.median_error:.6e}, "
            f"max={self.max_error:.6e}"
        )


@dataclass
class GridMetrics:
    """Metrics for a single grid resolution run."""
    
    n_capital: int
    n_debt: int
    n_productivity: int
    value_function: Array
    k_grid: Array
    b_grid: Array
    z_grid: Array
    computation_time: float
    max_value: float = field(default=0.0)
    min_value: float = field(default=0.0)
    mean_value: float = field(default=0.0)
    
    def __post_init__(self):
        """Compute summary statistics."""
        self.max_value = float(np.nanmax(self.value_function))
        self.min_value = float(np.nanmin(self.value_function))
        self.mean_value = float(np.nanmean(self.value_function))
    
    @property
    def total_grid_points(self) -> int:
        return self.n_capital * self.n_debt * self.n_productivity
    
    @property
    def grid_size(self) -> Tuple[int, int]:
        return (self.n_capital, self.n_debt)
    
    @property
    def grid_size_label(self) -> str:
        return f"({self.n_capital}, {self.n_debt})"


@dataclass
class ConvergenceTestResult:
    """Results from grid convergence analysis."""
    
    metrics_list: List[GridMetrics]
    error_metrics_list: List[GridErrorMetrics]  # Error relative to finest grid
    is_converged: bool
    is_monotonically_decreasing: bool
    convergence_threshold: float
    reference_grid_size: Tuple[int, int]
    
    def __str__(self) -> str:
        status = "PASS" if self.is_converged else "FAIL"
        monotonic_status = "YES" if self.is_monotonically_decreasing else "NO"
        
        lines = [
            f"Grid Convergence Test Result: {status}",
            f"Reference (finest) grid: {self.reference_grid_size}",
            f"Convergence threshold: {self.convergence_threshold:.6f}",
            f"Errors monotonically decreasing: {monotonic_status}",
            "",
            f"{'Grid Size':<15} | {'Points':<10} | {'Time (s)':<10} | {'Mean MAE':<15} | {'Max MAE':<15}",
            "-" * 80
        ]
        
        for i, metrics in enumerate(self.metrics_list):
            grid_label = metrics.grid_size_label
            points = metrics.total_grid_points
            time_str = f"{metrics.computation_time:.2f}"
            
            if i < len(self.error_metrics_list):
                err = self.error_metrics_list[i]
                mean_err_str = f"{err.mean_error:.6e}"
                max_err_str = f"{err.max_error:.6e}"
            else:
                # This is the reference grid
                mean_err_str = "0 (reference)"
                max_err_str = "0 (reference)"
            
            lines.append(
                f"{grid_label:<15} | {points:<10} | {time_str:<10} | {mean_err_str:<15} | {max_err_str:<15}"
            )
        
        return "\n".join(lines)
    
    def get_convergence_summary(self) -> Dict[str, Any]:
        """Get convergence summary based on mean MAE."""
        if len(self.error_metrics_list) == 0:
            return {}
        
        mean_errors = [em.mean_error for em in self.error_metrics_list]
        
        # Check if errors decrease as grid size increases (excluding reference)
        monotonic = all(mean_errors[i] >= mean_errors[i+1] for i in range(len(mean_errors)-1))
        
        # The last non-reference grid should have small error
        final_mean = mean_errors[-1] if mean_errors else float('inf')
        converged = final_mean < self.convergence_threshold
        
        return {
            'metric': 'Mean MAE vs Reference',
            'threshold': self.convergence_threshold,
            'final_value': final_mean,
            'converged': converged,
            'monotonically_decreasing': monotonic,
            'reference_grid': self.reference_grid_size,
        }
    
    def print_convergence_summary(self) -> None:
        """Print a clean convergence summary."""
        summary = self.get_convergence_summary()
        
        if not summary:
            print("No convergence data available.")
            return
        
        print("\n" + "=" * 70)
        print("GRID SUFFICIENCY TEST SUMMARY")
        print("=" * 70)
        print(f"  Reference Grid:      {summary['reference_grid']}")
        print(f"  Metric:              {summary['metric']}")
        print(f"  Threshold:           {summary['threshold']}")
        print(f"  Second-finest Error: {summary['final_value']:.6f}")
        print(f"  Monotonic Decrease:  {'YES ✓' if summary['monotonically_decreasing'] else 'NO ✗'}")
        print(f"  Grid Sufficient:     {'YES ✓' if summary['converged'] else 'NO ✗'}")
        print("=" * 70 + "\n")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary for JSON serialization."""
        return {
            "is_converged": self.is_converged,
            "is_monotonically_decreasing": self.is_monotonically_decreasing,
            "convergence_threshold": self.convergence_threshold,
            "reference_grid_size": self.reference_grid_size,
            "grid_results": [
                {
                    "grid_size": m.grid_size,
                    "total_points": m.total_grid_points,
                    "time": m.computation_time,
                    "max_value": m.max_value,
                    "min_value": m.min_value,
                    "mean_value": m.mean_value,
                }
                for m in self.metrics_list
            ],
            "error_metrics": [
                {
                    "grid_size": em.grid_size,
                    "mean_error": em.mean_error,
                    "median_error": em.median_error,
                    "max_error": em.max_error,
                    "std_error": em.std_error,
                    "p95_error": em.p95_error,
                    "p99_error": em.p99_error,
                    "l2_error": em.l2_error,
                }
                for em in self.error_metrics_list
            ],
            "convergence_summary": self.get_convergence_summary(),
        }


# =============================================================================
# Grid Convergence Tester
# =============================================================================

class GridConvergenceTester:
    """
    Tests whether the grid resolution is sufficient via convergence analysis.
    
    The approach:
    1. Solve the model at multiple grid resolutions
    2. Use the finest grid as the reference ("ground truth")
    3. Interpolate each coarser grid solution to the reference grid
    4. Compute error metrics (MAE) relative to the reference
    5. Check that errors decrease as grid size increases
    """
    
    def __init__(
        self,
        params: EconomicParams,
        base_config: GridConfig,
        bounds: Dict[str, float],
        convergence_threshold: float = 0.01
    ):
        self.params = params
        self.base_config = base_config
        self.bounds = bounds
        self.convergence_threshold = convergence_threshold
        
    def _create_grid_config(self, n_capital: int, n_debt: int) -> GridConfig:
        """Create a GridConfig with specified grid sizes."""
        return GridConfig(
            n_capital=n_capital,
            n_productivity=self.base_config.n_productivity,
            n_debt=n_debt,
            tauchen_width=self.base_config.tauchen_width,
            tol_vfi=self.base_config.tol_vfi,
            max_iter_vfi=self.base_config.max_iter_vfi,
            tol_outer=self.base_config.tol_outer,
            max_outer=self.base_config.max_outer,
            v_default_eps=self.base_config.v_default_eps,
            b_eps=self.base_config.b_eps,
            q_min=self.base_config.q_min,
            relax_q=self.base_config.relax_q
        )
    
    def _interpolate_to_reference(
        self,
        v_source: Array,
        k_source: Array,
        b_source: Array,
        z_source: Array,
        k_ref: Array,
        b_ref: Array,
        z_ref: Array
    ) -> Array:
        """Interpolate value function from source grid to reference grid."""
        interpolator = RegularGridInterpolator(
            (k_source, b_source, z_source),
            v_source,
            method='linear',
            bounds_error=False,
            fill_value=None
        )
        
        k_mesh, b_mesh, z_mesh = np.meshgrid(
            k_ref, b_ref, z_ref, indexing='ij'
        )
        
        points = np.stack(
            [k_mesh.ravel(), b_mesh.ravel(), z_mesh.ravel()], 
            axis=-1
        )
        
        v_interp = interpolator(points).reshape(
            len(k_ref), len(b_ref), len(z_ref)
        )
        
        return v_interp
    
    def run_test(self) -> ConvergenceTestResult:
        """Execute the grid convergence test across all predefined grid sizes."""
        k_bounds = (self.bounds["k_min"], self.bounds["k_max"])
        b_bounds = (self.bounds["b_min"], self.bounds["b_max"])
        
        metrics_list: List[GridMetrics] = []
        
        # Solve at all grid sizes
        for n_capital, n_debt in GRID_SIZES:
            config = self._create_grid_config(n_capital, n_debt)
            
            logger.info(
                f"Solving with grid: K={config.n_capital}, "
                f"B={config.n_debt}, Z={config.n_productivity}"
            )
            
            model = RiskyDebtModelVFI(
                self.params, config, k_bounds=k_bounds, b_bounds=b_bounds
            )
            
            start_time = time.time()
            solution = model.solve()
            elapsed = time.time() - start_time
            
            metrics = GridMetrics(
                n_capital=config.n_capital,
                n_debt=config.n_debt,
                n_productivity=config.n_productivity,
                value_function=solution['V'],
                k_grid=solution['K'],
                b_grid=solution['B'],
                z_grid=solution['Z'],
                computation_time=elapsed
            )
            
            metrics_list.append(metrics)
            logger.info(f"  Solved in {elapsed:.2f} seconds, {metrics.total_grid_points} grid points")
        
        # Compute errors relative to the finest (last) grid
        # error_metrics_list = self._compute_errors_vs_reference(metrics_list)
        error_metrics_list = self._compute_errors_via_sampling(metrics_list)
        
        # Check convergence: errors should decrease as grid gets finer
        mean_errors = [em.mean_error for em in error_metrics_list]
        is_monotonically_decreasing = self._check_monotonic_decrease(mean_errors)
        
        # Grid is sufficient if the second-finest grid has small error
        is_converged = (
            len(mean_errors) > 0 and 
            mean_errors[-1] < self.convergence_threshold and
            is_monotonically_decreasing
        )
        
        reference_grid = metrics_list[-1].grid_size
        
        return ConvergenceTestResult(
            metrics_list=metrics_list,
            error_metrics_list=error_metrics_list,
            is_converged=is_converged,
            is_monotonically_decreasing=is_monotonically_decreasing,
            convergence_threshold=self.convergence_threshold,
            reference_grid_size=reference_grid
        )
    
    def _compute_errors_vs_reference(
        self, 
        metrics_list: List[GridMetrics]
    ) -> List[GridErrorMetrics]:
        """
        Compute error of each grid relative to the finest (reference) grid.
        
        The reference grid is the last (finest) grid. We don't compute error
        for the reference grid itself (it would be zero by definition).
        """
        if len(metrics_list) < 2:
            return []
        
        # Reference is the finest grid
        ref = metrics_list[-1]
        k_ref = ref.k_grid
        b_ref = ref.b_grid
        z_ref = ref.z_grid
        v_ref = ref.value_function
        
        error_metrics_list: List[GridErrorMetrics] = []
        
        logger.info("\n" + "=" * 70)
        logger.info("Computing Errors Relative to Reference Grid")
        logger.info(f"Reference grid: {ref.grid_size_label}")
        logger.info("=" * 70)
        
        # Compute error for each coarser grid (all except the last)
        for m in metrics_list[:-1]:
            logger.info(f"\nGrid {m.grid_size_label} vs Reference {ref.grid_size_label}")
            
            # Interpolate coarser grid to reference grid points
            v_interp = self._interpolate_to_reference(
                m.value_function, m.k_grid, m.b_grid, m.z_grid,
                k_ref, b_ref, z_ref
            )
            
            # Compute absolute error
            error_matrix = np.abs(v_interp - v_ref)
            
            mean_error = float(np.nanmean(error_matrix))
            median_error = float(np.nanmedian(error_matrix))
            max_error = float(np.nanmax(error_matrix))
            std_error = float(np.nanstd(error_matrix))
            l2_error = float(np.sqrt(np.nanmean(error_matrix ** 2)))
            p95_error = float(np.nanpercentile(error_matrix, 95))
            p99_error = float(np.nanpercentile(error_matrix, 99))
            
            err_metrics = GridErrorMetrics(
                grid_size=m.grid_size,
                mean_error=mean_error,
                median_error=median_error,
                max_error=max_error,
                std_error=std_error,
                p95_error=p95_error,
                p99_error=p99_error,
                l2_error=l2_error,
                error_matrix=error_matrix
            )
            
            error_metrics_list.append(err_metrics)
            
            logger.info(f"  Mean |error|:    {mean_error:.6e}")
            logger.info(f"  Median |error|:  {median_error:.6e}")
            logger.info(f"  Max |error|:     {max_error:.6e}")
        
        logger.info("=" * 70 + "\n")
        
        return error_metrics_list
    
    def _compute_errors_via_sampling(
        self,
        metrics_list: List[GridMetrics],
        n_samples: int = 10000,
        seed: int = 42
    ) -> List[GridErrorMetrics]:
        """Compute errors using random sampling in the state space."""
        
        if len(metrics_list) < 2:
            return []
        
        ref = metrics_list[-1]
        rng = np.random.default_rng(seed)
        
        # Sample random points within the overlapping domain
        k_min = max(m.k_grid.min() for m in metrics_list)
        k_max = min(m.k_grid.max() for m in metrics_list)
        b_min = max(m.b_grid.min() for m in metrics_list)
        b_max = min(m.b_grid.max() for m in metrics_list)
        z_min = max(m.z_grid.min() for m in metrics_list)
        z_max = min(m.z_grid.max() for m in metrics_list)
        
        # Generate random sample points
        k_samples = rng.uniform(k_min, k_max, n_samples)
        b_samples = rng.uniform(b_min, b_max, n_samples)
        z_samples = rng.uniform(z_min, z_max, n_samples)
        sample_points = np.stack([k_samples, b_samples, z_samples], axis=-1)
        
        # Build interpolator for reference grid
        ref_interp = RegularGridInterpolator(
            (ref.k_grid, ref.b_grid, ref.z_grid),
            ref.value_function,
            method='linear',
            bounds_error=False,
            fill_value=np.nan
        )
        v_ref_samples = ref_interp(sample_points)
        
        error_metrics_list = []
        
        for m in metrics_list[:-1]:
            # Build interpolator for this grid
            m_interp = RegularGridInterpolator(
                (m.k_grid, m.b_grid, m.z_grid),
                m.value_function,
                method='linear',
                bounds_error=False,
                fill_value=np.nan
            )
            v_m_samples = m_interp(sample_points)
            
            # Compute errors at sample points
            errors = np.abs(v_m_samples - v_ref_samples)
            valid_mask = ~np.isnan(errors)
            errors = errors[valid_mask]
            
            err_metrics = GridErrorMetrics(
                grid_size=m.grid_size,
                mean_error=float(np.mean(errors)),
                median_error=float(np.median(errors)),
                max_error=float(np.max(errors)),
                std_error=float(np.std(errors)),
                p95_error=float(np.percentile(errors, 95)),
                p99_error=float(np.percentile(errors, 99)),
                l2_error=float(np.sqrt(np.mean(errors ** 2))),
                error_matrix=errors  # Now a 1D array of sample errors
            )
            error_metrics_list.append(err_metrics)
    
        return error_metrics_list

    def _check_monotonic_decrease(self, errors: List[float]) -> bool:
        """Check if errors decrease as grid size increases."""
        if len(errors) < 2:
            return True
        
        for i in range(1, len(errors)):
            if errors[i] > errors[i - 1]:
                return False
        return True


# =============================================================================
# Visualization - Single Clean Figure with 4 Subplots
# =============================================================================

def create_convergence_figure(result: ConvergenceTestResult, save_path: str) -> plt.Figure:
    """
    Create a single convergence analysis figure with 4 subplots:
    1. Mean MAE vs grid size
    2. Median MAE vs grid size
    3. Max MAE vs grid size
    4. Computation time vs grid size
    """
    
    # Extract data
    grid_sizes = [m.grid_size for m in result.metrics_list]
    grid_labels = [f"{g[0]}×{g[1]}" for g in grid_sizes]
    times = [m.computation_time for m in result.metrics_list]
    
    # Error metrics are for all grids except the reference (finest)
    error_grids = [em.grid_size for em in result.error_metrics_list]
    error_labels = [f"{g[0]}×{g[1]}" for g in error_grids]
    
    mean_errors = [em.mean_error for em in result.error_metrics_list]
    median_errors = [em.median_error for em in result.error_metrics_list]
    max_errors = [em.max_error for em in result.error_metrics_list]
    
    # Add zero error for reference grid for plotting
    error_labels_with_ref = error_labels + [grid_labels[-1]]
    mean_errors_with_ref = mean_errors + [0.0]
    median_errors_with_ref = median_errors + [0.0]
    max_errors_with_ref = max_errors + [0.0]
    
    threshold = result.convergence_threshold
    
    # Colors
    mean_color = '#2ecc71'
    median_color = '#9b59b6'
    max_color = '#e74c3c'
    time_color = '#3498db'
    ref_color = '#95a5a6'
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Grid Sufficiency Analysis (Reference: {result.reference_grid_size[0]}×{result.reference_grid_size[1]})', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    x = np.arange(len(error_labels_with_ref))
    
    # Plot 1: Mean MAE vs Grid Size
    ax1 = axes[0, 0]
    bars1 = ax1.bar(x[:-1], mean_errors, color=mean_color, alpha=0.8, 
                    edgecolor='darkgreen', linewidth=1.5, label='Mean MAE')
    ax1.bar(x[-1], 0, color=ref_color, alpha=0.5, edgecolor='gray', 
            linewidth=1.5, hatch='//', label='Reference (0)')
    
    ax1.axhline(y=threshold, color='red', linestyle='--', linewidth=2, 
                alpha=0.7, label=f'Threshold ({threshold})')
    
    # Add value labels
    for i, v in enumerate(mean_errors):
        ax1.annotate(f'{v:.4f}', xy=(i, v), xytext=(0, 5), 
                     textcoords='offset points', ha='center', fontsize=9, fontweight='bold')
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(error_labels_with_ref, rotation=45, ha='right')
    ax1.set_xlabel('Grid Size', fontsize=11)
    ax1.set_ylabel('Mean MAE', fontsize=11)
    ax1.set_title('Mean MAE vs Grid Size', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Convergence status
    if mean_errors[-1] < threshold:
        ax1.text(0.02, 0.98, '✓ CONVERGED', transform=ax1.transAxes, 
                 fontsize=10, color='green', fontweight='bold', verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    else:
        ax1.text(0.02, 0.98, '✗ NOT CONVERGED', transform=ax1.transAxes, 
                 fontsize=10, color='red', fontweight='bold', verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # Plot 2: Median MAE vs Grid Size
    ax2 = axes[0, 1]
    ax2.bar(x[:-1], median_errors, color=median_color, alpha=0.8, 
            edgecolor='purple', linewidth=1.5, label='Median MAE')
    ax2.bar(x[-1], 0, color=ref_color, alpha=0.5, edgecolor='gray', 
            linewidth=1.5, hatch='//', label='Reference (0)')
    
    ax2.axhline(y=threshold, color='red', linestyle='--', linewidth=2, 
                alpha=0.7, label=f'Threshold ({threshold})')
    
    for i, v in enumerate(median_errors):
        ax2.annotate(f'{v:.4f}', xy=(i, v), xytext=(0, 5), 
                     textcoords='offset points', ha='center', fontsize=9, fontweight='bold')
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(error_labels_with_ref, rotation=45, ha='right')
    ax2.set_xlabel('Grid Size', fontsize=11)
    ax2.set_ylabel('Median MAE', fontsize=11)
    ax2.set_title('Median MAE vs Grid Size', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Max MAE vs Grid Size
    ax3 = axes[1, 0]
    ax3.bar(x[:-1], max_errors, color=max_color, alpha=0.8, 
            edgecolor='darkred', linewidth=1.5, label='Max MAE')
    ax3.bar(x[-1], 0, color=ref_color, alpha=0.5, edgecolor='gray', 
            linewidth=1.5, hatch='//', label='Reference (0)')
    
    ax3.axhline(y=threshold, color='red', linestyle='--', linewidth=2, 
                alpha=0.7, label=f'Threshold ({threshold})')
    
    for i, v in enumerate(max_errors):
        ax3.annotate(f'{v:.2f}', xy=(i, v), xytext=(0, 5), 
                     textcoords='offset points', ha='center', fontsize=9, fontweight='bold')
    
    ax3.set_xticks(x)
    ax3.set_xticklabels(error_labels_with_ref, rotation=45, ha='right')
    ax3.set_xlabel('Grid Size', fontsize=11)
    ax3.set_ylabel('Max MAE', fontsize=11)
    ax3.set_title('Max MAE vs Grid Size', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Computation Time vs Grid Size
    ax4 = axes[1, 1]
    x_time = np.arange(len(grid_sizes))
    ax4.bar(x_time, times, color=time_color, alpha=0.7, 
            edgecolor='navy', linewidth=1.5)
    
    for i, (t, g) in enumerate(zip(times, grid_labels)):
        ax4.annotate(f'{t:.1f}s', xy=(i, t), xytext=(0, 5), 
                     textcoords='offset points', ha='center', fontsize=10, fontweight='bold')
    
    # Compute scaling exponent
    grid_nums = [g[0] for g in grid_sizes]
    log_n = np.log(grid_nums)
    log_t = np.log(times)
    slope, intercept = np.polyfit(log_n, log_t, 1)
    
    ax4.set_xticks(x_time)
    ax4.set_xticklabels(grid_labels, rotation=45, ha='right')
    ax4.set_xlabel('Grid Size', fontsize=11)
    ax4.set_ylabel('Computation Time (seconds)', fontsize=11)
    ax4.set_title(f'Computation Time vs Grid Size (O(n^{slope:.2f}))', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"Saved convergence analysis figure to {save_path}")
    
    return fig


# =============================================================================
# Main Test Function
# =============================================================================

def test_grid_convergence(config: GridValidationConfig) -> ConvergenceTestResult:
    """Main entry point for grid convergence testing."""
    logger.info("=" * 60)
    logger.info("Grid Size Sufficiency Test")
    logger.info("=" * 60)
    logger.info(f"Testing grid sizes: {GRID_SIZES}")
    logger.info(f"Reference (finest) grid: {GRID_SIZES[-1]}")
    
    params = load_economic_params(config.econ_params_path)
    base_config = load_grid_config(config.vfi_params_path, "risky")
    bounds = load_bounds(config.bounds_path)
    
    tester = GridConvergenceTester(
        params, base_config, bounds, 
        convergence_threshold=config.convergence_threshold
    )
    result = tester.run_test()
    
    result.print_convergence_summary()
    
    logger.info("\n" + str(result))
    
    output_path = os.path.join(config.output_dir, "grid_convergence_results.json")
    with open(output_path, 'w') as f:
        json.dump(result.to_dict(), f, indent=2)
    logger.info(f"Results saved to {output_path}")
    
    if config.generate_plots:
        logger.info("\nGenerating convergence plot...")
        plot_path = os.path.join(config.output_dir, 'convergence_analysis.png')
        create_convergence_figure(result, plot_path)
    
    return result


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """CLI entry point for grid convergence test."""
    parser = argparse.ArgumentParser(
        description="Grid size validation for Risky Debt VFI Model"
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
        '--convergence-threshold',
        type=float,
        default=0.01,
        help="Convergence threshold for Mean MAE"
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default="./test_outputs",
        help="Directory for output files"
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
    
    config = GridValidationConfig(
        bounds_path=args.bounds_path,
        econ_params_path=args.econ_params_path,
        vfi_params_path=args.vfi_params_path,
        convergence_threshold=args.convergence_threshold,
        output_dir=args.output_dir,
        generate_plots=not args.no_plot,
        dpi=args.dpi
    )
    
    result = test_grid_convergence(config)
    
    logger.info("\n" + "=" * 60)
    if result.is_converged:
        logger.info("✓ Grid sufficiency test PASSED")
        logger.info("  - Errors decrease monotonically as grid size increases")
        logger.info("  - Second-finest grid has Mean MAE below threshold")
        logger.info(f"  - Grid size {GRID_SIZES[-2]} is sufficient for threshold {result.convergence_threshold}")
    else:
        logger.info("✗ Grid sufficiency test FAILED")
        summary = result.get_convergence_summary()
        if not summary.get('monotonically_decreasing', True):
            logger.info("  - Errors do NOT decrease monotonically with grid size")
        if summary.get('final_value', float('inf')) >= result.convergence_threshold:
            logger.info(f"  - Mean MAE ({summary.get('final_value', 0):.4f}) >= threshold ({result.convergence_threshold})")
        logger.info("  - Consider using finer grids")
    logger.info("=" * 60)
    
    sys.exit(0 if result.is_converged else 1)


if __name__ == "__main__":
    main()