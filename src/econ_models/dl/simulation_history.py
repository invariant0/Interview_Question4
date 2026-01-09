import numpy as np
from typing import Dict


class DLSimulationHistory:
    """Container for deep learning simulation history data."""
    
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
                Expected keys:
                - "k": Capital trajectories, shape (n_steps, n_batches)
                - "z": Productivity trajectories, shape (n_steps, n_batches)
                - "investment_rate": Investment rate trajectories, shape (n_steps, n_batches)
                - "investment": Investment amount trajectories, shape (n_steps, n_batches)
                - "steady_state_capital": Scalar steady state capital value
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
    
    @property
    def capital_history(self) -> np.ndarray:
        """Capital trajectories, shape (n_steps, n_batches)."""
        return self.trajectories["k"]
    
    @property
    def productivity_history(self) -> np.ndarray:
        """Productivity trajectories, shape (n_steps, n_batches)."""
        return self.trajectories["z"]
    
    @property
    def investment_rate_history(self) -> np.ndarray:
        """Investment rate trajectories, shape (n_steps, n_batches)."""
        return self.trajectories["investment_rate"]
    
    @property
    def investment_history(self) -> np.ndarray:
        """Investment amount trajectories, shape (n_steps, n_batches)."""
        return self.trajectories["investment"]
    
    @property
    def steady_state_capital(self) -> float:
        """Deterministic steady state capital level."""
        return self.trajectories["steady_state_capital"]
    
    def get_trajectory_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Compute summary statistics for all trajectories.
        
        Returns:
            Dictionary mapping trajectory names to their statistics.
        """
        stats = {}
        
        array_keys = ["k", "z", "investment_rate", "investment"]
        names = ["capital", "productivity", "investment_rate", "investment"]
        
        for key, name in zip(array_keys, names):
            if key in self.trajectories:
                data = self.trajectories[key]
                stats[name] = {
                    "mean": float(np.mean(data)),
                    "std": float(np.std(data)),
                    "min": float(np.min(data)),
                    "max": float(np.max(data)),
                }
        
        return stats
    
    def get_final_distribution(self) -> Dict[str, np.ndarray]:
        """
        Get the final period values across all trajectories.
        
        Returns:
            Dictionary with final period values.
        """
        return {
            "capital": self.capital_history[-1, :],
            "productivity": self.productivity_history[-1, :],
            "investment_rate": self.investment_rate_history[-1, :],
            "investment": self.investment_history[-1, :],
        }
    
    def __repr__(self) -> str:
        """String representation of the simulation history."""
        return (
            f"DLSimulationHistory("
            f"n_batches={self.n_batches}, "
            f"n_steps={self.n_steps}, "
            f"steady_state_capital={self.steady_state_capital:.4f})"
        )


class RiskyDLSimulationHistory:
    """Container for deep learning simulation history data (Risky Model)."""

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
                Expected keys: k, b, z, q, d (default), steady_state_capital
            n_batches: Number of simulation batches.
            n_steps: Number of steps per batch.
        """
        self.trajectories = trajectories
        self.n_batches = n_batches
        self.n_steps = n_steps

    @property
    def total_observations(self) -> int:
        return self.n_batches * self.n_steps

    @property
    def capital_history(self) -> np.ndarray:
        """Capital trajectories, shape (n_steps, n_batches)."""
        return self.trajectories["k"]

    @property
    def debt_history(self) -> np.ndarray:
        """Debt trajectories, shape (n_steps, n_batches)."""
        return self.trajectories["b"]

    @property
    def productivity_history(self) -> np.ndarray:
        """Productivity trajectories, shape (n_steps, n_batches)."""
        return self.trajectories["z"]

    @property
    def bond_price_history(self) -> np.ndarray:
        """Bond price trajectories, shape (n_steps, n_batches)."""
        return self.trajectories["q"]
    
    @property
    def default_history(self) -> np.ndarray:
        """
        Default status trajectories, shape (n_steps, n_batches).
        1.0 indicates default, 0.0 indicates repayment.
        """
        return self.trajectories["d"]

    def get_trajectory_stats(self) -> Dict[str, Dict[str, float]]:
        """Compute summary statistics for all trajectories (excluding defaults)."""
        stats = {}
        array_keys = ["k", "b", "z", "q"]
        names = ["capital", "debt", "productivity", "bond_price"]
        
        # Create a mask for valid (non-defaulted) observations
        # Assuming if d=1, the firm is dead.
        if "d" in self.trajectories:
            valid_mask = self.trajectories["d"] == 0.0
        else:
            valid_mask = np.ones_like(self.trajectories["k"], dtype=bool)

        for key, name in zip(array_keys, names):
            if key in self.trajectories:
                data = self.trajectories[key]
                # Filter by valid mask
                valid_data = data[valid_mask]
                
                if valid_data.size > 0:
                    stats[name] = {
                        "mean": float(np.mean(valid_data)),
                        "std": float(np.std(valid_data)),
                        "min": float(np.min(valid_data)),
                        "max": float(np.max(valid_data)),
                    }
                else:
                    stats[name] = {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        return stats
    
    def get_final_distribution(self) -> Dict[str, np.ndarray]:
        """Get the final period values across all trajectories."""
        return {
            "capital": self.capital_history[-1, :],
            "debt": self.debt_history[-1, :],
            "productivity": self.productivity_history[-1, :],
            "bond_price": self.bond_price_history[-1, :],
            "default": self.default_history[-1, :],
        }
    
# class RiskyDLSimulationHistory:
#     """Container for deep learning simulation history data (Risky Model)."""

#     def __init__(
#         self,
#         trajectories: Dict[str, np.ndarray],
#         n_batches: int,
#         n_steps: int
#     ) -> None:
#         """
#         Initialize simulation history.

#         Args:
#             trajectories: Dictionary mapping state names to trajectory arrays.
#                 Expected keys: k, b, z, q, d (default), steady_state_capital
#             n_batches: Number of simulation batches.
#             n_steps: Number of steps per batch.
#         """
#         self.trajectories = trajectories
#         self.n_batches = n_batches
#         self.n_steps = n_steps

#     @property
#     def total_observations(self) -> int:
#         return self.n_batches * self.n_steps

#     @property
#     def capital_history(self) -> np.ndarray:
#         """Capital trajectories, shape (n_steps, n_batches)."""
#         return self.trajectories["k"]

#     @property
#     def debt_history(self) -> np.ndarray:
#         """Debt trajectories, shape (n_steps, n_batches)."""
#         return self.trajectories["b"]

#     @property
#     def productivity_history(self) -> np.ndarray:
#         """Productivity trajectories, shape (n_steps, n_batches)."""
#         return self.trajectories["z"]

#     @property
#     def bond_price_history(self) -> np.ndarray:
#         """Bond price trajectories, shape (n_steps, n_batches)."""
#         return self.trajectories["q"]
    
#     @property
#     def default_history(self) -> np.ndarray:
#         """
#         Default status trajectories, shape (n_steps, n_batches).
#         1.0 indicates default, 0.0 indicates repayment.
#         """
#         return self.trajectories["d"]

#     def get_trajectory_stats(self) -> Dict[str, Dict[str, float]]:
#         """Compute summary statistics for all trajectories (excluding defaults)."""
#         stats = {}
#         array_keys = ["k", "b", "z", "q"]
#         names = ["capital", "debt", "productivity", "bond_price"]
        
#         if "d" in self.trajectories:
#             valid_mask = self.trajectories["d"] == 0.0
#         else:
#             valid_mask = np.ones_like(self.trajectories["k"], dtype=bool)

#         for key, name in zip(array_keys, names):
#             if key in self.trajectories:
#                 data = self.trajectories[key]
#                 valid_data = data[valid_mask]
                
#                 if valid_data.size > 0:
#                     stats[name] = {
#                         "mean": float(np.mean(valid_data)),
#                         "std": float(np.std(valid_data)),
#                         "min": float(np.min(valid_data)),
#                         "max": float(np.max(valid_data)),
#                     }
#                 else:
#                     stats[name] = {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
#         return stats
    
#     def get_final_distribution(self) -> Dict[str, np.ndarray]:
#         """Get the final period values across all trajectories."""
#         return {
#             "capital": self.capital_history[-1, :],
#             "debt": self.debt_history[-1, :],
#             "productivity": self.productivity_history[-1, :],
#             "bond_price": self.bond_price_history[-1, :],
#             "default": self.default_history[-1, :],
#         }
