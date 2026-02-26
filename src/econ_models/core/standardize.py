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
    
class ParamSpaceNormalizer:
    """
    Maps parameter variables between physical and normalized domains.

    This class performs min-max normalization, mapping values from
    their economic domain [min, max] to the normalized domain [0, 1],
    which is more suitable for neural network training.
    """

    def __init__(self, bonds: dict) -> None:
        """Initialize normalizer with domain boundaries from configuration.

        Args:
            bonds: Dictionary containing parameter space boundaries,
                with keys such as ``rho_min``, ``rho_max``, etc.

        Raises:
            ValueError: If required boundaries are not set.
        """
        self._validate_boundaries(bonds)
        self._initialize_rho_constants(bonds)
        self._initialize_std_constants(bonds)
        self._initialize_convex_constants(bonds)
        self._initialize_fixed_constants(bonds)
        self._initialize_eta0_constants(bonds)
        self._initialize_eta1_constants(bonds)
    def _validate_boundaries(self, bonds: dict) -> None:
        """Validate that required boundaries are set."""
        if bonds.get("rho_min") is None or bonds.get("rho_max") is None:
            raise ValueError(
                "rho boundaries (rho_min, rho_max) must be set."
            )
        if bonds.get("std_min") is None or bonds.get("std_max") is None:
            raise ValueError(
                "std boundaries (std_min, std_max) "
                "must be set."
            )
        if bonds.get("convex_min") is None or bonds.get("convex_max") is None:
            raise ValueError(
                "convex boundaries (convex_min, convex_max) "
                "must be set."
            )
        if bonds.get("fixed_min") is None or bonds.get("fixed_max") is None:
            raise ValueError(
                "fixed cost boundaries (fixed_min, fixed_max) "
                "must be set."
            )
        # eta0 and eta1 are optional (only needed for risky dist model)
        # No validation error if missing
    def _initialize_rho_constants(self, bonds: dict) -> None:
        """Initialize rho normalization constants."""
        self.rho_min = tf.constant(bonds["rho_min"], dtype=TENSORFLOW_DTYPE)
        self.rho_range = tf.constant(
            bonds["rho_max"] - bonds["rho_min"],
            dtype=TENSORFLOW_DTYPE
        )
    def _initialize_std_constants(self, bonds: dict) -> None:
        """Initialize std normalization constants."""
        self.std_min = tf.constant(bonds["std_min"], dtype=TENSORFLOW_DTYPE)
        self.std_range = tf.constant(
            bonds["std_max"] - bonds["std_min"],
            dtype=TENSORFLOW_DTYPE
        )
    def _initialize_convex_constants(self, bonds: dict) -> None:
        """Initialize convexity normalization constants."""
        self.convex_min = tf.constant(bonds["convex_min"], dtype=TENSORFLOW_DTYPE)
        self.convex_range = tf.constant(
            bonds["convex_max"] - bonds["convex_min"],
            dtype=TENSORFLOW_DTYPE
        )
    def _initialize_fixed_constants(self, bonds: dict) -> None: 
        """Initialize fixed cost normalization constants."""
        self.fixed_min = tf.constant(bonds["fixed_min"], dtype=TENSORFLOW_DTYPE)
        self.fixed_range = tf.constant(
            bonds["fixed_max"] - bonds["fixed_min"],
            dtype=TENSORFLOW_DTYPE
        )
    def _initialize_eta0_constants(self, bonds: dict) -> None:
        """Initialize eta0 (equity issuance fixed cost) normalization constants."""
        if bonds.get("eta0_min") is not None and bonds.get("eta0_max") is not None:
            self.eta0_min = tf.constant(bonds["eta0_min"], dtype=TENSORFLOW_DTYPE)
            self.eta0_range = tf.constant(
                bonds["eta0_max"] - bonds["eta0_min"],
                dtype=TENSORFLOW_DTYPE
            )
        else:
            self.eta0_min = tf.constant(0.0, dtype=TENSORFLOW_DTYPE)
            self.eta0_range = tf.constant(1.0, dtype=TENSORFLOW_DTYPE)
    def _initialize_eta1_constants(self, bonds: dict) -> None:
        """Initialize eta1 (equity issuance linear cost) normalization constants."""
        if bonds.get("eta1_min") is not None and bonds.get("eta1_max") is not None:
            self.eta1_min = tf.constant(bonds["eta1_min"], dtype=TENSORFLOW_DTYPE)
            self.eta1_range = tf.constant(
                bonds["eta1_max"] - bonds["eta1_min"],
                dtype=TENSORFLOW_DTYPE
            )
        else:
            self.eta1_min = tf.constant(0.0, dtype=TENSORFLOW_DTYPE)
            self.eta1_range = tf.constant(1.0, dtype=TENSORFLOW_DTYPE)
    @tf.function
    def normalize_rho(self, rho: Tensor) -> Tensor:
        """
        Scale rho from [rho_min, rho_max] to [0, 1].

        Args:
            rho: rho values in physical units.

        Returns:
            Normalized rho values in [0, 1].
        """
        return (rho - self.rho_min) / self.rho_range
    @tf.function
    def denormalize_rho(self, rho_norm: Tensor) -> Tensor:
        """
        Scale rho from [0, 1] back to [rho_min, rho_max].

        Args:
            rho_norm: Normalized rho values.

        Returns:
            rho values in physical units.
        """
        return rho_norm * self.rho_range + self.rho_min
    @tf.function
    def normalize_std(self, std: Tensor) -> Tensor:
        """
        Scale std from [std_min, std_max] to [0, 1].

        Args:
            std: std values in physical units.

        Returns:
            Normalized std values in [0, 1].
        """
        return (std - self.std_min) / self.std_range
    @tf.function
    def denormalize_std(self, std_norm: Tensor) -> Tensor:
        """
        Scale std from [0, 1] back to [std_min, std_max].

        Args:
            std_norm: Normalized std values.

        Returns:
            std values in physical units.
        """
        return std_norm * self.std_range + self.std_min
    @tf.function
    def normalize_convex(self, convex: Tensor) -> Tensor:
        """
        Scale convex from [convex_min, convex_max] to [0, 1].

        Args:
            convex: convex values in physical units.

        Returns:
            Normalized convex values in [0, 1].
        """
        return (convex - self.convex_min) / self.convex_range
    @tf.function
    def denormalize_convex(self, convex_norm: Tensor) -> Tensor:
        """
        Scale convex from [0, 1] back to [convex_min, convex_max].

        Args:
            convex_norm: Normalized convex values.

        Returns:
            convex values in physical units.
        """
        return convex_norm * self.convex_range + self.convex_min
    @tf.function
    def normalize_fixed(self, fixed: Tensor) -> Tensor:
        """
        Scale fixed from [fixed_min, fixed_max] to [0, 1].

        Args:
            fixed: fixed values in physical units.

        Returns:
            Normalized fixed values in [0, 1].
        """
        return (fixed - self.fixed_min) / self.fixed_range
    @tf.function
    def denormalize_fixed(self, fixed_norm: Tensor) -> Tensor:
        """
        Scale fixed from [0, 1] back to [fixed_min, fixed_max].

        Args:
            fixed_norm: Normalized fixed values.

        Returns:
            fixed values in physical units.
        """
        return fixed_norm * self.fixed_range + self.fixed_min
    @tf.function
    def normalize_eta0(self, eta0: Tensor) -> Tensor:
        """
        Scale eta0 from [eta0_min, eta0_max] to [0, 1].

        Args:
            eta0: eta0 values in physical units.

        Returns:
            Normalized eta0 values in [0, 1].
        """
        return (eta0 - self.eta0_min) / self.eta0_range
    @tf.function
    def denormalize_eta0(self, eta0_norm: Tensor) -> Tensor:
        """
        Scale eta0 from [0, 1] back to [eta0_min, eta0_max].

        Args:
            eta0_norm: Normalized eta0 values.

        Returns:
            eta0 values in physical units.
        """
        return eta0_norm * self.eta0_range + self.eta0_min
    @tf.function
    def normalize_eta1(self, eta1: Tensor) -> Tensor:
        """
        Scale eta1 from [eta1_min, eta1_max] to [0, 1].

        Args:
            eta1: eta1 values in physical units.

        Returns:
            Normalized eta1 values in [0, 1].
        """
        return (eta1 - self.eta1_min) / self.eta1_range
    @tf.function
    def denormalize_eta1(self, eta1_norm: Tensor) -> Tensor:
        """
        Scale eta1 from [0, 1] back to [eta1_min, eta1_max].

        Args:
            eta1_norm: Normalized eta1 values.

        Returns:
            eta1 values in physical units.
        """
        return eta1_norm * self.eta1_range + self.eta1_min
        
   