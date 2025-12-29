# econ_models/core/standardize.py
"""
State space normalization for neural network inputs.

This module provides min-max scaling of economic state variables
to the [0, 1] range, which improves neural network training stability.

Example:
    >>> normalizer = StateSpaceNormalizer(config)
    >>> k_norm = normalizer.normalize_capital(capital_values)
    >>> k_original = normalizer.denormalize_capital(k_norm)
"""

import tensorflow as tf

from econ_models.config.dl_config import DeepLearningConfig
from econ_models.core.types import TENSORFLOW_DTYPE, Tensor


class StateSpaceNormalizer:
    """
    Maps state variables between physical and normalized domains.

    This class performs min-max normalization, mapping values from
    their economic domain [min, max] to the normalized domain [0, 1],
    which is more suitable for neural network training.

    Attributes:
        k_min: Minimum capital value.
        k_range: Range of capital values (max - min).
        z_min: Minimum productivity value.
        z_range: Range of productivity values.
        b_min: Minimum debt value.
        b_range: Range of debt values.
    """

    def __init__(self, config: DeepLearningConfig) -> None:
        """
        Initialize normalizer with domain boundaries from configuration.

        Args:
            config: Configuration object containing state space boundaries.

        Raises:
            ValueError: If required boundaries are not set.
        """
        self._validate_boundaries(config)
        self._initialize_capital_constants(config)
        self._initialize_productivity_constants(config)
        self._initialize_debt_constants(config)

    def _validate_boundaries(self, config: DeepLearningConfig) -> None:
        """Validate that required boundaries are set."""
        if config.capital_min is None or config.capital_max is None:
            raise ValueError(
                "Capital boundaries (capital_min, capital_max) must be set."
            )
        if config.productivity_min is None or config.productivity_max is None:
            raise ValueError(
                "Productivity boundaries (productivity_min, productivity_max) "
                "must be set."
            )

    def _initialize_capital_constants(self, config: DeepLearningConfig) -> None:
        """Initialize capital normalization constants."""
        self.k_min = tf.constant(config.capital_min, dtype=TENSORFLOW_DTYPE)
        self.k_range = tf.constant(
            config.capital_max - config.capital_min,
            dtype=TENSORFLOW_DTYPE
        )

    def _initialize_productivity_constants(
        self,
        config: DeepLearningConfig
    ) -> None:
        """Initialize productivity normalization constants."""
        self.z_min = tf.constant(config.productivity_min, dtype=TENSORFLOW_DTYPE)
        self.z_range = tf.constant(
            config.productivity_max - config.productivity_min,
            dtype=TENSORFLOW_DTYPE
        )

    def _initialize_debt_constants(self, config: DeepLearningConfig) -> None:
        """Initialize debt normalization constants."""
        if config.debt_min is not None and config.debt_max is not None:
            self.b_min = tf.constant(config.debt_min, dtype=TENSORFLOW_DTYPE)
            self.b_range = tf.constant(
                config.debt_max - config.debt_min,
                dtype=TENSORFLOW_DTYPE
            )
        else:
            # Default values to prevent graph construction errors
            self.b_min = tf.constant(0.0, dtype=TENSORFLOW_DTYPE)
            self.b_range = tf.constant(1.0, dtype=TENSORFLOW_DTYPE)

    @tf.function
    def normalize_capital(self, k: Tensor) -> Tensor:
        """
        Scale capital from [k_min, k_max] to [0, 1].

        Args:
            k: Capital values in physical units.

        Returns:
            Normalized capital values in [0, 1].
        """
        return (k - self.k_min) / self.k_range

    @tf.function
    def denormalize_capital(self, k_norm: Tensor) -> Tensor:
        """
        Scale capital from [0, 1] back to [k_min, k_max].

        Args:
            k_norm: Normalized capital values.

        Returns:
            Capital values in physical units.
        """
        return k_norm * self.k_range + self.k_min

    @tf.function
    def normalize_productivity(self, z: Tensor) -> Tensor:
        """
        Scale productivity from [z_min, z_max] to [0, 1].

        Args:
            z: Productivity values in physical units.

        Returns:
            Normalized productivity values in [0, 1].
        """
        return (z - self.z_min) / self.z_range

    @tf.function
    def normalize_debt(self, b: Tensor) -> Tensor:
        """
        Scale debt from [b_min, b_max] to [0, 1].

        Args:
            b: Debt values in physical units.

        Returns:
            Normalized debt values in [0, 1].
        """
        return (b - self.b_min) / self.b_range

    @tf.function
    def denormalize_debt(self, b_norm: Tensor) -> Tensor:
        """
        Scale debt from [0, 1] back to [b_min, b_max].

        Args:
            b_norm: Normalized debt values.

        Returns:
            Debt values in physical units.
        """
        return b_norm * self.b_range + self.b_min