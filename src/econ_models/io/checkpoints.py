# econ_models/io/checkpoints.py
"""
Utilities for saving and loading neural network model weights.

This module provides checkpoint management for deep learning models,
enabling training resumption and model persistence.

Example:
    >>> from econ_models.io.checkpoints import save_checkpoint_basic
    >>> save_checkpoint_basic(value_net, policy_net, epoch=100)
"""

from pathlib import Path
from typing import Optional
import tensorflow as tf


def save_checkpoint_basic(
    value_net: tf.keras.Model,
    policy_net: tf.keras.Model,
    epoch: int,
    save_dir: str = "checkpoints/basic"
) -> None:
    """
    Save checkpoints for basic model networks.

    Args:
        value_net: Value function neural network.
        policy_net: Policy function neural network.
        epoch: Current training epoch for naming.
        save_dir: Directory for checkpoint storage.
    """
    path = Path(save_dir)
    path.mkdir(parents=True, exist_ok=True)
    value_net.save_weights(path / f"basic_value_net_{epoch}.weights.h5")
    policy_net.save_weights(path / f"basic_policy_net_{epoch}.weights.h5")


def save_checkpoint_risky(
    value_net: tf.keras.Model,
    epoch: int,
    save_dir: str = "checkpoints/risky",
    suffix: str = ""
) -> None:
    """
    Save checkpoint for risky model value network.

    Args:
        value_net: Value function neural network.
        epoch: Current training epoch for naming.
        save_dir: Directory for checkpoint storage.
        suffix: Optional suffix for filename.
    """
    path = Path(save_dir)
    path.mkdir(parents=True, exist_ok=True)
    value_net.save_weights(path / f"risky_value_net_{epoch}{suffix}.weights.h5")

def save_checkpoint_risky_upgrade(
    value_net: tf.keras.Model,
    capital_policy_net: tf.keras.Model,
    debt_policy_net: tf.keras.Model,
    epoch: int,
    save_dir: str = "checkpoints/risky_upgrade",
    suffix: str = "",
    continuous_net: Optional[tf.keras.Model] = None,
) -> None:
    """
    Save checkpoint for risky upgrade model networks.

    Args:
        value_net: Value function neural network V(s).
        capital_policy_net: Capital policy network πk(s).
        debt_policy_net: Debt policy network πb(s).
        epoch: Current training epoch for naming.
        save_dir: Directory for checkpoint storage.
        suffix: Optional suffix for filename.
        continuous_net: Optional continuous value network Vcont(s).
    """
    path = Path(save_dir)
    path.mkdir(parents=True, exist_ok=True)
    
    # Save value network
    value_net.save_weights(path / f"risky_value_net_{epoch}{suffix}.weights.h5")
    
    # Save policy networks
    capital_policy_net.save_weights(path / f"risky_capital_policy_net_{epoch}{suffix}.weights.h5")
    debt_policy_net.save_weights(path / f"risky_debt_policy_net_{epoch}{suffix}.weights.h5")
    
    # Optionally save continuous network
    if continuous_net is not None:
        continuous_net.save_weights(path / f"risky_continuous_net_{epoch}{suffix}.weights.h5")