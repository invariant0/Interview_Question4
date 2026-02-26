# src/econ_models/moment_calculator/compute_autocorrelation.py
"""Compute autocorrelation statistics."""

import tensorflow as tf
from src.econ_models.core.types import TENSORFLOW_DTYPE


def compute_autocorrelation(data: tf.Tensor, lag: int = 1) -> tf.Tensor:
    """
    Compute autocorrelation at specified lag by pooling across all observations.
    
    Args:
        data: Tensor of shape (batch_size, n_periods)
        lag: Autocorrelation lag (default 1)
    
    Returns:
        Scalar tensor with autocorrelation coefficient
    """
    if data.shape[1] <= lag:
        return tf.constant(0.0, dtype=TENSORFLOW_DTYPE)
    
    # Extract current and lagged values
    x_current = data[:, :-lag]
    x_lagged = data[:, lag:]
    
    # Flatten for pooled computation
    x_current_flat = tf.reshape(x_current, [-1])
    x_lagged_flat = tf.reshape(x_lagged, [-1])
    
    # Create valid mask
    valid_mask = tf.math.is_finite(x_current_flat) & tf.math.is_finite(x_lagged_flat)
    
    x_current_valid = tf.boolean_mask(x_current_flat, valid_mask)
    x_lagged_valid = tf.boolean_mask(x_lagged_flat, valid_mask)
    
    n_valid = tf.cast(tf.shape(x_current_valid)[0], TENSORFLOW_DTYPE)
    
    if n_valid < 10:
        return tf.constant(0.0, dtype=TENSORFLOW_DTYPE)
    
    # Compute means
    mean_current = tf.reduce_mean(x_current_valid)
    mean_lagged = tf.reduce_mean(x_lagged_valid)
    
    # Compute covariance and variances
    cov = tf.reduce_mean((x_current_valid - mean_current) * (x_lagged_valid - mean_lagged))
    var_current = tf.reduce_mean(tf.square(x_current_valid - mean_current))
    var_lagged = tf.reduce_mean(tf.square(x_lagged_valid - mean_lagged))
    
    # Compute correlation
    denominator = tf.sqrt(var_current * var_lagged) + 1e-8
    correlation = cov / denominator
    
    return tf.clip_by_value(correlation, -1.0, 1.0)


def compute_autocorrelation_lags_1_to_5(data: tf.Tensor) -> dict:
    """
    Compute autocorrelations for lags 1 through 5.
    
    Args:
        data: Tensor of shape (batch_size, n_periods)
    
    Returns:
        Dictionary with keys 'lag_1' through 'lag_5'
    """
    return {
        f'lag_{i}': compute_autocorrelation(data, lag=i)
        for i in range(1, 6)
    }