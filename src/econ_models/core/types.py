# econ_models/core/types.py
"""
Global type definitions for TensorFlow and NumPy precision.

This module establishes a single source of truth for numerical precision
across the entire codebase, ensuring consistency between TensorFlow
operations and NumPy array manipulations.

Example:
    >>> from econ_models.core.types import TENSORFLOW_DTYPE, NUMPY_DTYPE
    >>> import tensorflow as tf
    >>> tensor = tf.constant([1.0, 2.0], dtype=TENSORFLOW_DTYPE)
"""

import tensorflow as tf
import numpy as np
from typing import Union

# -----------------------------------------------------------------------------
# Global Precision Settings
# -----------------------------------------------------------------------------
# We use float32 for a balance between numerical stability and performance.
# For applications requiring higher precision, consider switching to float32.

TENSORFLOW_DTYPE = tf.float32
NUMPY_DTYPE = np.float32

# Configure Keras backend to match global precision
tf.keras.backend.set_floatx("float32")

# -----------------------------------------------------------------------------
# Type Aliases
# -----------------------------------------------------------------------------
# These aliases improve code readability and make type hints more expressive.

Tensor = tf.Tensor
Array = np.ndarray
Numeric = Union[float, np.float32, tf.Tensor]