# econ_models/core/sampling/__init__.py
"""
Sampling utilities for deep learning training.

This package provides functions for sampling state variables
and candidate actions during neural network training.
"""

from econ_models.core.sampling.state_sampler import StateSampler
from econ_models.core.sampling.candidate_sampler import CandidateSampler
from econ_models.core.sampling.transitions import TransitionFunctions

__all__ = ['StateSampler', 'CandidateSampler', 'TransitionFunctions']


import os
import random
import numpy as np
import tensorflow as tf
def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # For GPU determinism (optional, slower)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
set_seed(42)