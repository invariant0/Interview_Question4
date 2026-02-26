# src/econ_models/moment_calculator/compute_inaction_rate.py
"""Compute inaction rate statistics."""

import tensorflow as tf
from src.econ_models.core.types import TENSORFLOW_DTYPE


def compute_inaction_rate(
    data: tf.Tensor,
    lower_threshold: float = -0.01,
    upper_threshold: float = 0.01,
) -> tf.Tensor:
    """
    Compute inaction rate - fraction of observations within threshold of zero.
    
    Inaction is defined as observations where the absolute value is below
    the threshold, indicating near-zero activity (e.g., near-zero investment).
    
    Args:
        data: Tensor of shape (batch_size, n_periods)
        lower_threshold: Lower bound for inaction region (default -0.01)
        upper_threshold: Upper bound for inaction region (default 0.01)
    
    Returns:
        Scalar tensor with inaction rate (fraction of inaction observations)
    """
    # Create valid mask for finite values
    valid_mask = tf.math.is_finite(data)
    
    # Create inaction mask: values within threshold bounds
    inaction_mask = (data >= lower_threshold) & (data <= upper_threshold)
    
    # Combine masks: only count valid observations
    valid_inaction = valid_mask & inaction_mask
    
    # Count valid and inaction observations
    n_valid = tf.cast(tf.reduce_sum(tf.cast(valid_mask, tf.int32)), TENSORFLOW_DTYPE)
    n_inaction = tf.cast(tf.reduce_sum(tf.cast(valid_inaction, tf.int32)), TENSORFLOW_DTYPE)
    
    # Compute inaction rate
    inaction_rate = n_inaction / tf.maximum(n_valid, 1.0)
    
    return inaction_rate
