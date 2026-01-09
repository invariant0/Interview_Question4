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
        Simulate risky debt model to check boundary hits and defaults.
        Handles termination index -1.
        """
        rng = np.random.default_rng(seed)
        
        # Allocate trajectory arrays
        # Initialize with a safe value, though they get overwritten immediately
        k_history = np.ones((n_batches, n_steps), dtype=np.int32) * -1
        b_history = np.ones((n_batches, n_steps), dtype=np.int32) * -1
        z_history = np.ones((n_batches, n_steps), dtype=np.int32) * -1
        
        # Sample initial states uniformly (ensure we don't start at -1)
        k_idx = rng.integers(0, n_capital, size=n_batches)
        b_idx = rng.integers(0, n_debt, size=n_batches)
        z_idx = rng.integers(0, n_productivity, size=n_batches)

        for t in range(n_steps):
            # Record current state
            
            k_history[:, t] = k_idx
            b_history[:, t] = b_idx
            z_history[:, t] = z_idx

            # Identify active agents (those who have not defaulted)
            # We assume -1 indicates a defaulted/terminated state
            active_mask = (k_idx != -1) & (b_idx != -1)
            
            # Prepare next state arrays (default to current state)
            # This ensures defaulted agents stay -1
            k_next_step = k_idx.copy()
            b_next_step = b_idx.copy()
            
            if np.any(active_mask):
                # Extract indices for active agents
                k_active = k_idx[active_mask]
                b_active = b_idx[active_mask]
                z_active = z_idx[active_mask]
                
                # Apply policies only for active agents
                # If pol returns -1, that value gets assigned to k_next_step
                k_next_vals = pol_k[k_active, b_active, z_active]
                b_next_vals = pol_b[k_active, b_active, z_active]
                
                k_next_step[active_mask] = k_next_vals
                b_next_step[active_mask] = b_next_vals
            
            # Update state
            k_idx = k_next_step
            b_idx = b_next_step
            
            # Transition productivity
            # (Optional: stop transitioning Z for defaulted agents, 
            # but keeping it running is harmless computationally)
            z_idx = np.array([
                rng.choice(n_productivity, p=transition_matrix[z])
                for z in z_idx
            ])

        history = SimulationHistory(
            trajectories={
                "k_idx": k_history,
                "b_idx": b_history,
                "z_idx": z_history
            },
            n_batches=n_batches,
            n_steps=n_steps
        )

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
        Compute boundary hit statistics, accounting for defaults (-1).
        """
        k_history = k_history[:, -500:]
        b_history = b_history[:, -500:]
        total_obs = k_history.size
        # Count defaults
        default_count = np.sum(k_history == -1)

        # Count boundary hits ONLY for non-defaulted observations
        # We don't want a default (-1) to count as a min boundary hit (0) or wrap around
        valid_mask = k_history != -1
        valid_obs = np.sum(valid_mask)
        
        if valid_obs == 0:
            return {
                "k_min": 0.0, "k_max": 0.0, "b_min": 0.0, "b_max": 0.0,
                "default_rate": 1.0, "total_observations": total_obs
            }

        stats = {
            # Percentage of TOTAL observations that are defaults
            "default_rate": default_count / total_obs,
            
            # Percentage of VALID observations hitting boundaries
            "k_min": np.sum((k_history == 0) & valid_mask) / valid_obs,
            "k_max": np.sum((k_history == n_capital - 1) & valid_mask) / valid_obs,
            "b_min": np.sum((b_history == 0) & valid_mask) / valid_obs,
            "b_max": np.sum((b_history == n_debt - 1) & valid_mask) / valid_obs,
            
            "total_observations": total_obs
        }
        
        return stats