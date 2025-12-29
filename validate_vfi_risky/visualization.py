"""
Bond price and value function 3D visualization for the Risky Debt VFI Model.

This module generates comprehensive visualizations including:
- 3D surface plots of value functions across productivity levels
- Default rate analysis with respect to state variables (K, B, Z)
- Bond price visualizations
- Default boundary and probability heatmaps

Usage:
    python test_visualization.py
    python test_visualization.py --output-dir ./plots
    python test_visualization.py --productivity-level high
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import Normalize

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

@dataclass
class VisualizationConfig:
    """Configuration for visualization generation."""
    
    bounds_path: str = "/home/wangzijian/JPMorgan/econ-dl/hyperparam/autogen/bounds_risky.json"
    econ_params_path: str = "/home/wangzijian/JPMorgan/econ-dl/hyperparam/prefixed/econ_params_risky.json"
    vfi_params_path: str = "/home/wangzijian/JPMorgan/econ-dl/hyperparam/prefixed/vfi_params.json"
    output_dir: str = "./test_outputs"
    default_threshold: float = 1e-10
    dpi: int = 150
    
    def __post_init__(self) -> None:
        """Ensure output directory exists."""
        Path(self.output_dir).mkdir(exist_ok=True)


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
# Bond Price Visualizer
# =============================================================================

class BondPriceVisualizer:
    """
    Creates 3D visualizations of bond prices and default states.
    
    Generates surface plots showing the relationship between capital,
    debt, and bond prices, as well as the default boundary.
    """
    
    def __init__(
        self,
        solution: Dict[str, Array],
        output_dir: str,
        default_threshold: float = 1e-10,
        dpi: int = 150
    ):
        """
        Initialize the visualizer.
        
        Args:
            solution: VFI solution dictionary containing V, Q, K, B, Z arrays.
            output_dir: Directory to save visualization outputs.
            default_threshold: Value threshold below which default is assumed.
            dpi: Resolution for saved figures.
        """
        self.V = solution['V']
        self.Q = solution['Q']
        self.K = solution['K']
        self.B = solution['B']
        self.Z = solution['Z']
        self.output_dir = output_dir
        self.default_threshold = default_threshold
        self.dpi = dpi
        
        # Compute default indicator: 1 if default, 0 if continuation
        self.is_default = (self.V <= self.default_threshold).astype(float)
        
        Path(output_dir).mkdir(exist_ok=True)
        
    def _get_productivity_indices(self) -> Dict[str, int]:
        """Get indices for low, medium, and high productivity states."""
        n_z = len(self.Z)
        return {
            'low': 0,
            'medium': n_z // 2,
            'high': n_z - 1
        }
    
    def plot_value_and_default_summary(self, save: bool = True) -> plt.Figure:
        """
        Create a comprehensive 2x3 figure:
        
        Row 1: Value function 3D surfaces for low, medium, high productivity
        Row 2: Default rate vs K, Default rate vs B, Default rate vs Z
        
        Args:
            save: Whether to save the figure to file.
            
        Returns:
            matplotlib Figure object.
        """
        z_indices = self._get_productivity_indices()
        
        fig = plt.figure(figsize=(18, 12))
        
        # Create grid meshes
        K_mesh, B_mesh = np.meshgrid(self.K, self.B, indexing='ij')
        
        # Color settings
        value_cmap = 'plasma'
        default_color = '#e74c3c'
        line_colors = {'low': '#3498db', 'medium': '#2ecc71', 'high': '#e74c3c'}
        
        # =====================================================================
        # Row 1: Value Function 3D Surfaces
        # =====================================================================
        
        # Find global value range for consistent colorbar
        v_min = np.min(self.V[self.V > self.default_threshold]) if np.any(self.V > self.default_threshold) else np.min(self.V)
        v_max = np.max(self.V)
        
        for idx, (level, z_idx) in enumerate(z_indices.items()):
            ax = fig.add_subplot(2, 3, idx + 1, projection='3d')
            
            v_slice = self.V[:, :, z_idx].copy()
            
            # Create surface plot
            surf = ax.plot_surface(
                K_mesh, B_mesh, v_slice,
                cmap=value_cmap,
                alpha=0.85,
                edgecolor='none',
                vmin=v_min,
                vmax=v_max
            )
            
            # Mark default region with different color overlay
            default_mask = self.is_default[:, :, z_idx]
            if np.any(default_mask):
                # Create a separate scatter for default points
                k_default = K_mesh[default_mask > 0.5]
                b_default = B_mesh[default_mask > 0.5]
                v_default = v_slice[default_mask > 0.5]
                if len(k_default) > 0:
                    ax.scatter(k_default, b_default, v_default, 
                              c=default_color, s=2, alpha=0.5, label='Default')
            
            ax.set_xlabel('Capital (K)', fontsize=10, labelpad=5)
            ax.set_ylabel('Debt (B)', fontsize=10, labelpad=5)
            ax.set_zlabel('V(K, B, Z)', fontsize=10, labelpad=5)
            
            # Calculate default rate for this z level
            default_rate_z = np.mean(default_mask) * 100
            
            ax.set_title(
                f'{level.capitalize()} Productivity\n'
                f'Z = {self.Z[z_idx]:.4f} | Default: {default_rate_z:.1f}%',
                fontsize=11, fontweight='bold'
            )
            
            # Adjust viewing angle for better visualization
            ax.view_init(elev=25, azim=-60)
            
            # Add colorbar only to the last subplot in the row
            if idx == 2:
                cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=15, pad=0.1)
                cbar.set_label('Value Function', fontsize=10)
        
        # =====================================================================
        # Row 2: Default Rate Analysis
        # =====================================================================
        
        # Plot 4: Default Rate vs Capital (K)
        ax4 = fig.add_subplot(2, 3, 4)
        
        # Compute default rate for each K (averaged over B and Z)
        default_rate_by_k = np.mean(self.is_default, axis=(1, 2)) * 100
        
        # Also show for different Z levels
        for level, z_idx in z_indices.items():
            default_rate_k_z = np.mean(self.is_default[:, :, z_idx], axis=1) * 100
            ax4.plot(self.K, default_rate_k_z, 
                    color=line_colors[level], linewidth=2, 
                    label=f'Z={level}', alpha=0.8)
        
        # Overall average
        ax4.plot(self.K, default_rate_by_k, 
                color='black', linewidth=2.5, linestyle='--',
                label='Average', alpha=0.9)
        
        ax4.fill_between(self.K, 0, default_rate_by_k, alpha=0.2, color='gray')
        
        ax4.set_xlabel('Capital (K)', fontsize=11)
        ax4.set_ylabel('Default Rate (%)', fontsize=11)
        ax4.set_title('Default Rate vs Capital', fontsize=12, fontweight='bold')
        ax4.legend(loc='upper right', fontsize=9)
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(bottom=0)
        
        # Add annotation for key insight
        k_at_50pct = None
        if np.any(default_rate_by_k >= 50) and np.any(default_rate_by_k < 50):
            cross_idx = np.argmin(np.abs(default_rate_by_k - 50))
            k_at_50pct = self.K[cross_idx]
            ax4.axvline(x=k_at_50pct, color='orange', linestyle=':', alpha=0.7)
            ax4.annotate(f'50% at K≈{k_at_50pct:.2f}', 
                        xy=(k_at_50pct, 50), xytext=(k_at_50pct + 0.1*(self.K[-1]-self.K[0]), 60),
                        fontsize=9, arrowprops=dict(arrowstyle='->', color='orange'))
        
        # Plot 5: Default Rate vs Debt (B)
        ax5 = fig.add_subplot(2, 3, 5)
        
        # Compute default rate for each B (averaged over K and Z)
        default_rate_by_b = np.mean(self.is_default, axis=(0, 2)) * 100
        
        for level, z_idx in z_indices.items():
            default_rate_b_z = np.mean(self.is_default[:, :, z_idx], axis=0) * 100
            ax5.plot(self.B, default_rate_b_z, 
                    color=line_colors[level], linewidth=2, 
                    label=f'Z={level}', alpha=0.8)
        
        ax5.plot(self.B, default_rate_by_b, 
                color='black', linewidth=2.5, linestyle='--',
                label='Average', alpha=0.9)
        
        ax5.fill_between(self.B, 0, default_rate_by_b, alpha=0.2, color='gray')
        
        ax5.set_xlabel('Debt (B)', fontsize=11)
        ax5.set_ylabel('Default Rate (%)', fontsize=11)
        ax5.set_title('Default Rate vs Debt', fontsize=12, fontweight='bold')
        ax5.legend(loc='upper left', fontsize=9)
        ax5.grid(True, alpha=0.3)
        ax5.set_ylim(bottom=0)
        
        # Add annotation
        if np.any(default_rate_by_b >= 50) and np.any(default_rate_by_b < 50):
            cross_idx = np.argmin(np.abs(default_rate_by_b - 50))
            b_at_50pct = self.B[cross_idx]
            ax5.axvline(x=b_at_50pct, color='orange', linestyle=':', alpha=0.7)
            ax5.annotate(f'50% at B≈{b_at_50pct:.2f}', 
                        xy=(b_at_50pct, 50), xytext=(b_at_50pct - 0.2*(self.B[-1]-self.B[0]), 60),
                        fontsize=9, arrowprops=dict(arrowstyle='->', color='orange'))
        
        # Plot 6: Default Rate vs Productivity (Z)
        ax6 = fig.add_subplot(2, 3, 6)
        
        # Compute default rate for each Z (averaged over K and B)
        default_rate_by_z = np.mean(self.is_default, axis=(0, 1)) * 100
        
        # Bar plot for discrete Z values
        bar_colors = [line_colors['low'] if i < len(self.Z)//3 
                     else line_colors['high'] if i >= 2*len(self.Z)//3 
                     else line_colors['medium'] 
                     for i in range(len(self.Z))]
        
        bars = ax6.bar(range(len(self.Z)), default_rate_by_z, 
                      color=bar_colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Add line connecting the bars
        ax6.plot(range(len(self.Z)), default_rate_by_z, 
                color='black', linewidth=2, marker='o', markersize=6)
        
        # Set x-axis labels
        if len(self.Z) <= 10:
            ax6.set_xticks(range(len(self.Z)))
            ax6.set_xticklabels([f'{z:.3f}' for z in self.Z], rotation=45, ha='right')
        else:
            # Show fewer labels for large grids
            step = max(1, len(self.Z) // 5)
            ax6.set_xticks(range(0, len(self.Z), step))
            ax6.set_xticklabels([f'{self.Z[i]:.3f}' for i in range(0, len(self.Z), step)], 
                               rotation=45, ha='right')
        
        ax6.set_xlabel('Productivity (Z)', fontsize=11)
        ax6.set_ylabel('Default Rate (%)', fontsize=11)
        ax6.set_title('Default Rate vs Productivity', fontsize=12, fontweight='bold')
        ax6.grid(True, alpha=0.3, axis='y')
        ax6.set_ylim(bottom=0)
        
        # Add text annotations for min/max
        min_idx = np.argmin(default_rate_by_z)
        max_idx = np.argmax(default_rate_by_z)
        ax6.annotate(f'Min: {default_rate_by_z[min_idx]:.1f}%\nZ={self.Z[min_idx]:.3f}',
                    xy=(min_idx, default_rate_by_z[min_idx]),
                    xytext=(min_idx + 0.5, default_rate_by_z[min_idx] + 10),
                    fontsize=8, ha='left',
                    arrowprops=dict(arrowstyle='->', color='green', lw=1.5))
        ax6.annotate(f'Max: {default_rate_by_z[max_idx]:.1f}%\nZ={self.Z[max_idx]:.3f}',
                    xy=(max_idx, default_rate_by_z[max_idx]),
                    xytext=(max_idx - 0.5, default_rate_by_z[max_idx] + 10),
                    fontsize=8, ha='right',
                    arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
        
        # =====================================================================
        # Figure-level formatting
        # =====================================================================
        
        # Overall title
        overall_default_rate = np.mean(self.is_default) * 100
        fig.suptitle(
            f'Risky Debt Model: Value Function and Default Analysis\n'
            f'Grid: K∈[{self.K[0]:.2f}, {self.K[-1]:.2f}], '
            f'B∈[{self.B[0]:.2f}, {self.B[-1]:.2f}], '
            f'Z∈[{self.Z[0]:.3f}, {self.Z[-1]:.3f}] | '
            f'Overall Default Rate: {overall_default_rate:.1f}%',
            fontsize=13, fontweight='bold', y=0.98
        )
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        if save:
            filepath = os.path.join(self.output_dir, 'value_and_default_summary.png')
            fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight', facecolor='white')
            logger.info(f"Saved value and default summary to {filepath}")
        
        return fig
    
    def plot_bond_price_surface(
        self, 
        z_level: str = 'medium',
        save: bool = True
    ) -> plt.Figure:
        """
        Create 3D surface plot of bond prices.
        
        Args:
            z_level: Productivity level ('low', 'medium', 'high').
            save: Whether to save the figure to file.
            
        Returns:
            matplotlib Figure object.
        """
        z_indices = self._get_productivity_indices()
        z_idx = z_indices[z_level]
        
        q_slice = self.Q[:, :, z_idx]
        K_mesh, B_mesh = np.meshgrid(self.K, self.B, indexing='ij')
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        surf = ax.plot_surface(
            K_mesh, B_mesh, q_slice,
            cmap='viridis',
            alpha=0.8,
            edgecolor='none'
        )
        
        ax.set_xlabel('Capital (K)', fontsize=12)
        ax.set_ylabel('Debt (B)', fontsize=12)
        ax.set_zlabel('Bond Price (q)', fontsize=12)
        ax.set_title(
            f'Bond Price Surface\n(Productivity: {z_level}, Z={self.Z[z_idx]:.3f})',
            fontsize=14
        )
        
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Bond Price')
        
        if save:
            filepath = os.path.join(
                self.output_dir, 
                f'bond_price_surface_{z_level}.png'
            )
            fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved bond price surface plot to {filepath}")
        
        return fig
    
    def plot_value_function_surface(
        self, 
        z_level: str = 'medium',
        save: bool = True
    ) -> plt.Figure:
        """
        Create 3D surface plot of the value function.
        
        Args:
            z_level: Productivity level ('low', 'medium', 'high').
            save: Whether to save the figure to file.
            
        Returns:
            matplotlib Figure object.
        """
        z_indices = self._get_productivity_indices()
        z_idx = z_indices[z_level]
        
        v_slice = self.V[:, :, z_idx]
        K_mesh, B_mesh = np.meshgrid(self.K, self.B, indexing='ij')
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        surf = ax.plot_surface(
            K_mesh, B_mesh, v_slice,
            cmap='plasma',
            alpha=0.8,
            edgecolor='none'
        )
        
        ax.set_xlabel('Capital (K)', fontsize=12)
        ax.set_ylabel('Debt (B)', fontsize=12)
        ax.set_zlabel('Value Function V(K, B, Z)', fontsize=12)
        ax.set_title(
            f'Value Function Surface\n(Productivity: {z_level}, Z={self.Z[z_idx]:.3f})',
            fontsize=14
        )
        
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Value')
        
        if save:
            filepath = os.path.join(
                self.output_dir, 
                f'value_function_surface_{z_level}.png'
            )
            fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved value function surface plot to {filepath}")
        
        return fig
    
    def plot_default_boundary(
        self, 
        z_level: str = 'medium',
        save: bool = True
    ) -> plt.Figure:
        """
        Create 3D visualization of the default boundary.
        
        Args:
            z_level: Productivity level ('low', 'medium', 'high').
            save: Whether to save the figure to file.
            
        Returns:
            matplotlib Figure object.
        """
        z_indices = self._get_productivity_indices()
        z_idx = z_indices[z_level]
        
        v_slice = self.V[:, :, z_idx]
        is_default = (v_slice <= self.default_threshold).astype(float)
        
        K_mesh, B_mesh = np.meshgrid(self.K, self.B, indexing='ij')
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        surf = ax.plot_surface(
            K_mesh, B_mesh, v_slice,
            facecolors=plt.cm.RdYlGn(1 - is_default),
            alpha=0.8,
            edgecolor='none'
        )
        
        ax.set_xlabel('Capital (K)', fontsize=12)
        ax.set_ylabel('Debt (B)', fontsize=12)
        ax.set_zlabel('Value Function V(K, B, Z)', fontsize=12)
        ax.set_title(
            f'Value Function with Default Boundary\n'
            f'(Productivity: {z_level}, Z={self.Z[z_idx]:.3f})\n'
            f'Red = Default, Green = Continuation',
            fontsize=14
        )
        
        if save:
            filepath = os.path.join(
                self.output_dir, 
                f'default_boundary_{z_level}.png'
            )
            fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved default boundary plot to {filepath}")
        
        return fig
    
    def plot_bond_price_contours(self, save: bool = True) -> plt.Figure:
        """
        Create contour plots of bond prices across productivity levels.
        
        Args:
            save: Whether to save the figure to file.
            
        Returns:
            matplotlib Figure object.
        """
        z_indices = self._get_productivity_indices()
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        K_mesh, B_mesh = np.meshgrid(self.K, self.B, indexing='ij')
        
        for ax, (level, z_idx) in zip(axes, z_indices.items()):
            q_slice = self.Q[:, :, z_idx]
            
            contour = ax.contourf(K_mesh, B_mesh, q_slice, levels=20, cmap='viridis')
            ax.set_xlabel('Capital (K)', fontsize=11)
            ax.set_ylabel('Debt (B)', fontsize=11)
            ax.set_title(f'{level.capitalize()} Productivity\n(Z={self.Z[z_idx]:.3f})')
            
            fig.colorbar(contour, ax=ax, label='Bond Price')
        
        fig.suptitle('Bond Price Contours Across Productivity Levels', fontsize=14)
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, 'bond_price_contours.png')
            fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved bond price contour plot to {filepath}")
        
        return fig
    
    def plot_default_probability_heatmap(self, save: bool = True) -> plt.Figure:
        """
        Create heatmap showing default probability across states.
        
        The default probability is computed as the fraction of productivity
        states that lead to default for each (K, B) pair.
        
        Args:
            save: Whether to save the figure to file.
            
        Returns:
            matplotlib Figure object.
        """
        default_prob = np.mean(self.is_default, axis=2)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(
            default_prob.T,
            origin='lower',
            aspect='auto',
            extent=[self.K[0], self.K[-1], self.B[0], self.B[-1]],
            cmap='RdYlGn_r'
        )
        
        ax.set_xlabel('Capital (K)', fontsize=12)
        ax.set_ylabel('Debt (B)', fontsize=12)
        ax.set_title('Default Probability Heatmap\n(Averaged Across Productivity States)', fontsize=14)
        
        fig.colorbar(im, ax=ax, label='Default Probability')
        
        if save:
            filepath = os.path.join(self.output_dir, 'default_probability_heatmap.png')
            fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved default probability heatmap to {filepath}")
        
        return fig
    
    def plot_combined_3d_visualization(self, save: bool = True) -> plt.Figure:
        """
        Create a comprehensive 3D visualization combining bond prices and default states.
        
        Args:
            save: Whether to save the figure to file.
            
        Returns:
            matplotlib Figure object.
        """
        z_indices = self._get_productivity_indices()
        z_idx_med = z_indices['medium']
        
        fig = plt.figure(figsize=(16, 6))
        
        K_mesh, B_mesh = np.meshgrid(self.K, self.B, indexing='ij')
        
        # Subplot 1: Bond Price Surface
        ax1 = fig.add_subplot(131, projection='3d')
        q_slice = self.Q[:, :, z_idx_med]
        ax1.plot_surface(
            K_mesh, B_mesh, q_slice,
            cmap='viridis', alpha=0.8, edgecolor='none'
        )
        ax1.set_xlabel('K')
        ax1.set_ylabel('B')
        ax1.set_zlabel('q')
        ax1.set_title('Bond Price')
        
        # Subplot 2: Value Function
        ax2 = fig.add_subplot(132, projection='3d')
        v_slice = self.V[:, :, z_idx_med]
        ax2.plot_surface(
            K_mesh, B_mesh, v_slice,
            cmap='plasma', alpha=0.8, edgecolor='none'
        )
        ax2.set_xlabel('K')
        ax2.set_ylabel('B')
        ax2.set_zlabel('V')
        ax2.set_title('Value Function')
        
        # Subplot 3: Default Region
        ax3 = fig.add_subplot(133, projection='3d')
        is_default = self.is_default[:, :, z_idx_med]
        ax3.plot_surface(
            K_mesh, B_mesh, is_default,
            cmap='RdYlGn_r', alpha=0.8, edgecolor='none'
        )
        ax3.set_xlabel('K')
        ax3.set_ylabel('B')
        ax3.set_zlabel('Default')
        ax3.set_title('Default Indicator')
        
        fig.suptitle(
            f'Risky Debt Model: 3D Visualization (Z={self.Z[z_idx_med]:.3f})',
            fontsize=14
        )
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, 'combined_3d_visualization.png')
            fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved combined 3D visualization to {filepath}")
        
        return fig
    
    def generate_all_visualizations(self) -> List[str]:
        """
        Generate all available visualizations.
        
        Returns:
            List of file paths to generated visualizations.
        """
        generated_files = []
        
        # Main summary figure (2x3 grid)
        self.plot_value_and_default_summary()
        generated_files.append('value_and_default_summary.png')
        
        # Bond price surfaces and default boundaries for different productivity levels
        for level in ['low', 'medium', 'high']:
            # self.plot_bond_price_surface(z_level=level)
            # generated_files.append(f'bond_price_surface_{level}.png')
            
            # self.plot_value_function_surface(z_level=level)
            # generated_files.append(f'value_function_surface_{level}.png')
            
            self.plot_default_boundary(z_level=level)
            generated_files.append(f'default_boundary_{level}.png')
        
        # # Summary plots
        self.plot_bond_price_contours()
        generated_files.append('bond_price_contours.png')
        
        self.plot_default_probability_heatmap()
        generated_files.append('default_probability_heatmap.png')
        
        self.plot_combined_3d_visualization()
        generated_files.append('combined_3d_visualization.png')
        
        return [os.path.join(self.output_dir, f) for f in generated_files]


# =============================================================================
# Main Visualization Function
# =============================================================================

def generate_visualizations(config: VisualizationConfig) -> List[str]:
    """
    Main entry point for visualization generation.
    
    Args:
        config: Configuration containing file paths and settings.
        
    Returns:
        List of file paths to generated visualizations.
    """
    logger.info("=" * 60)
    logger.info("Bond Price and Value Function 3D Visualization")
    logger.info("=" * 60)
    
    params = load_economic_params(config.econ_params_path)
    grid_config = load_grid_config(config.vfi_params_path, "risky")
    bounds = load_bounds(config.bounds_path)
    
    k_bounds = (bounds["k_min"], bounds["k_max"])
    b_bounds = (bounds["b_min"], bounds["b_max"])
    
    logger.info("Solving VFI model for visualization...")
    model = RiskyDebtModelVFI(
        params, grid_config, k_bounds=k_bounds, b_bounds=b_bounds
    )
    solution = model.solve()
    
    logger.info("Generating visualizations...")
    visualizer = BondPriceVisualizer(
        solution, 
        output_dir=config.output_dir,
        default_threshold=config.default_threshold,
        dpi=config.dpi
    )
    
    output_files = visualizer.generate_all_visualizations()
    
    logger.info(f"Generated {len(output_files)} visualization files")
    logger.info(f"Files saved to: {config.output_dir}")
    
    return output_files


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """CLI entry point for visualization generation."""
    parser = argparse.ArgumentParser(
        description="Generate bond price and value function visualizations"
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
        help="Directory for output visualization files"
    )
    parser.add_argument(
        '--default-threshold',
        type=float,
        default=1e-10,
        help="Threshold for determining default states"
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=150,
        help="Resolution for saved figures"
    )
    
    args = parser.parse_args()
    
    config = VisualizationConfig(
        bounds_path=args.bounds_path,
        econ_params_path=args.econ_params_path,
        vfi_params_path=args.vfi_params_path,
        output_dir=args.output_dir,
        default_threshold=args.default_threshold,
        dpi=args.dpi
    )
    
    output_files = generate_visualizations(config)
    
    logger.info("\nGenerated files:")
    for f in output_files:
        logger.info(f"  - {f}")
    
    sys.exit(0)


if __name__ == "__main__":
    main()