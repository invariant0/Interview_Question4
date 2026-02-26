# src/econ_models/moment_calculator/compute_mean.py
"""Compute mean statistics."""

import tensorflow as tf
from src.econ_models.core.types import TENSORFLOW_DTYPE


def compute_global_mean(data: tf.Tensor) -> tf.Tensor:
    """
    Compute global mean across all dimensions.
    
    Args:
        data: Tensor of shape (batch_size, n_periods)
    
    Returns:
        Scalar tensor with global mean
    """
    valid_mask = tf.math.is_finite(data)
    data_casted = tf.cast(data, dtype=TENSORFLOW_DTYPE)
    valid_data = tf.where(valid_mask, data_casted, tf.zeros_like(data_casted, dtype=TENSORFLOW_DTYPE))

    n_valid = tf.cast(tf.reduce_sum(tf.cast(valid_mask, tf.int32)), TENSORFLOW_DTYPE)
    
    return tf.reduce_sum(valid_data) / tf.maximum(n_valid, 1.0)