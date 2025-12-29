# econ_models/dl/training/__init__.py
"""
Training utilities for deep learning models.

This package provides training loop components and loss calculations
for the neural network economic solvers.
"""

from econ_models.dl.training.dataset_builder import DatasetBuilder
from econ_models.dl.training.loss_calculator import LossCalculator

__all__ = ['DatasetBuilder', 'LossCalculator']