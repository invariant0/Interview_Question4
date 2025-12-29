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