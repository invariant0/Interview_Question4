# econ_models/core/sampling/__init__.py
"""
Sampling utilities for deep learning training.

This package provides functions for sampling state variables
and candidate actions during neural network training.
"""

from econ_models.core.sampling.state_sampler import StateSampler
from econ_models.core.sampling.transitions import TransitionFunctions

__all__ = ['StateSampler', 'TransitionFunctions']