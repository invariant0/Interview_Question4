"""Shared conftest for DL unit tests.

Patches tf.Tensor.__float__ to support float() on single-element tensors
with non-zero ndim (e.g. shape (1,1)), which broke in NumPy 2.0+.
"""

import tensorflow as tf

# The concrete tensor type in eager mode
_EagerTensor = type(tf.constant(0.0))
_orig_tensor_float = _EagerTensor.__float__


def _safe_tensor_float(self):
    """Convert a single-element TF tensor to Python float, any shape."""
    try:
        return _orig_tensor_float(self)
    except TypeError:
        return self.numpy().item()


_EagerTensor.__float__ = _safe_tensor_float
