# src/econ_models/moment_calculator/compute_std.py
"""Compute standard deviation statistics."""

import tensorflow as tf
from src.econ_models.core.types import TENSORFLOW_DTYPE
from .compute_mean import compute_global_mean


def compute_global_std(data: tf.Tensor) -> tf.Tensor:
    """
    Compute global standard deviation across all dimensions.
    
    Args:
        data: Tensor of shape (batch_size, n_periods)
    
    Returns:
        Scalar tensor with global standard deviation
    """
    valid_mask = tf.math.is_finite(data)
    mean = compute_global_mean(data)
    data = tf.cast(data, dtype=TENSORFLOW_DTYPE)
    mean = tf.cast(mean, dtype=TENSORFLOW_DTYPE)
    squared_diff = tf.where(
        valid_mask,
        tf.square(data - mean),
        tf.zeros_like(data, dtype=TENSORFLOW_DTYPE)
    )
    
    n_valid = tf.cast(tf.reduce_sum(tf.cast(valid_mask, tf.int32)), TENSORFLOW_DTYPE)
    variance = tf.reduce_sum(squared_diff) / tf.maximum(n_valid - 1.0, 1.0)
    
    return tf.sqrt(variance)