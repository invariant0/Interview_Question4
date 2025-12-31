# econ_models/vfi/simulation/simulator.py
"""
Simulation engine for VFI model validation.

This module provides simulation capabilities for testing whether
computed policies respect grid boundaries.
"""

from typing import Dict, Tuple

import numpy as np

from econ_models.core.types import Array


class SimulationHistory:
    """Container for simulation history data."""
    
    def __init__(
        self,
        trajectories: Dict[str, np.ndarray],
        n_batches: int,
        n_steps: int
    ) -> None:
        """
        Initialize simulation history.
        
        Args:
            trajectories: Dictionary mapping state names to trajectory arrays.
            n_batches: Number of simulation batches.
            n_steps: Number of steps per batch.
        """
        self.trajectories = trajectories
        self.n_batches = n_batches
        self.n_steps = n_steps
    
    @property
    def total_observations(self) -> int:
        """Total number of observations across all batches."""
        return self.n_batches * self.n_steps


class Simulator:
    """
    Simulation engine for policy function validation.

    This class provides methods for simulating the economy under
    computed optimal policies to verify grid boundary conditions.
    """

    @staticmethod
    def simulate_basic_model(
        policy_map: np.ndarray,
        transition_matrix: np.ndarray,
        n_capital: int,
        n_productivity: int,
        n_steps: int = 1000,
        n_batches: int = 1000,
        seed: int | None = None
    ) -> Tuple[SimulationHistory, Dict[str, float]]:
        """
        Simulate basic model to check boundary hits.

        Args:
            policy_map: Policy function mapping (k, z) -> k' index.
            transition_matrix: Markov transition matrix for productivity.
            n_capital: Number of capital grid points.
            n_productivity: Number of productivity grid points.
            n_steps: Number of simulation periods per batch.
            n_batches: Number of independent simulation batches.
            seed: Random seed for reproducibility.

        Returns:
            Tuple of (SimulationHistory, statistics dictionary).
        """
        rng = np.random.default_rng(seed)
        
        # Allocate trajectory arrays: shape (n_batches, n_steps)
        k_history = np.zeros((n_batches, n_steps), dtype=np.int32)
        z_history = np.zeros((n_batches, n_steps), dtype=np.int32)
        
        # Sample initial states uniformly
        k_init = rng.integers(0, n_capital, size=n_batches)
        z_init = rng.integers(0, n_productivity, size=n_batches)
        
        # Initialize current state vectors
        k_idx = k_init.copy()
        z_idx = z_init.copy()
        
        for t in range(n_steps):
            # Record current state
            k_history[:, t] = k_idx
            z_history[:, t] = z_idx
            
            # Apply policy: vectorized lookup
            k_idx = policy_map[k_idx, z_idx]
            
            # Transition productivity: sample for each batch
            z_idx = np.array([
                rng.choice(n_productivity, p=transition_matrix[z])
                for z in z_idx
            ])
        
        # Build history object
        history = SimulationHistory(
            trajectories={
                "k_idx": k_history,
                "z_idx": z_history
            },
            n_batches=n_batches,
            n_steps=n_steps
        )
        
        # Calculate statistics from full history
        stats = Simulator._compute_basic_stats(k_history, n_capital)
        
        return history, stats
    
    @staticmethod
    def _compute_basic_stats(
        k_history: np.ndarray,
        n_capital: int
    ) -> Dict[str, float]:
        """
        Compute boundary hit statistics from capital history.
        
        Args:
            k_history: Capital index history, shape (n_batches, n_steps).
            n_capital: Number of capital grid points.
            
        Returns:
            Dictionary with boundary hit percentages.
        """
        total_obs = k_history.size
        hits_min = np.sum(k_history == 0)
        hits_max = np.sum(k_history == n_capital - 1)
        
        return {
            "min_hit_pct": hits_min / total_obs,
            "max_hit_pct": hits_max / total_obs,
            "total_observations": total_obs
        }

    @staticmethod
    def simulate_risky_model(
        pol_k: np.ndarray,
        pol_b: np.ndarray,
        transition_matrix: np.ndarray,
        n_capital: int,
        n_debt: int,
        n_productivity: int,
        n_steps: int = 1000,
        n_batches: int = 1000,
        seed: int | None = None
    ) -> Tuple[SimulationHistory, Dict[str, float]]:
        """
        Simulate risky debt model to check boundary hits.

        Args:
            pol_k: Policy function for capital, shape (n_k, n_b, n_z).
            pol_b: Policy function for debt, shape (n_k, n_b, n_z).
            transition_matrix: Markov transition matrix for productivity.
            n_capital: Number of capital grid points.
            n_debt: Number of debt grid points.
            n_productivity: Number of productivity grid points.
            n_steps: Number of simulation periods per batch.
            n_batches: Number of independent simulation batches.
            seed: Random seed for reproducibility.

        Returns:
            Tuple of (SimulationHistory, statistics dictionary).
        """
        rng = np.random.default_rng(seed)
        
        # Allocate trajectory arrays: shape (n_batches, n_steps)
        k_history = np.zeros((n_batches, n_steps), dtype=np.int32)
        b_history = np.zeros((n_batches, n_steps), dtype=np.int32)
        z_history = np.zeros((n_batches, n_steps), dtype=np.int32)
        
        # Sample initial states uniformly
        k_idx = rng.integers(0, n_capital, size=n_batches)
        b_idx = rng.integers(0, n_debt, size=n_batches)
        z_idx = rng.integers(0, n_productivity, size=n_batches)
        
        for t in range(n_steps):
            # Record current state
            k_history[:, t] = k_idx
            b_history[:, t] = b_idx
            z_history[:, t] = z_idx
            
            # Apply policies: vectorized lookup
            k_next = pol_k[k_idx, b_idx, z_idx]
            b_next = pol_b[k_idx, b_idx, z_idx]
            
            # Update state
            k_idx = k_next
            b_idx = b_next
            
            # Transition productivity
            z_idx = np.array([
                rng.choice(n_productivity, p=transition_matrix[z])
                for z in z_idx
            ])
        
        # Build history object
        history = SimulationHistory(
            trajectories={
                "k_idx": k_history,
                "b_idx": b_history,
                "z_idx": z_history
            },
            n_batches=n_batches,
            n_steps=n_steps
        )
        
        # Calculate statistics from full history
        stats = Simulator._compute_risky_stats(
            k_history, b_history, n_capital, n_debt
        )
        
        return history, stats
    
    @staticmethod
    def _compute_risky_stats(
        k_history: np.ndarray,
        b_history: np.ndarray,
        n_capital: int,
        n_debt: int
    ) -> Dict[str, float]:
        """
        Compute boundary hit statistics from capital and debt history.
        
        Args:
            k_history: Capital index history, shape (n_batches, n_steps).
            b_history: Debt index history, shape (n_batches, n_steps).
            n_capital: Number of capital grid points.
            n_debt: Number of debt grid points.
            
        Returns:
            Dictionary with boundary hit percentages for each state.
        """
        total_obs = k_history.size
        
        stats = {
            "k_min": np.sum(k_history == 0) / total_obs,
            "k_max": np.sum(k_history == n_capital - 1) / total_obs,
            "b_min": np.sum(b_history == 0) / total_obs,
            "b_max": np.sum(b_history == n_debt - 1) / total_obs,
            "total_observations": total_obs
        }
        
        return stats