# econ_models/config/economic_params.py
"""
Economic parameter definitions and loading utilities.

This module defines the fundamental economic parameters that govern
the behavior of both the basic RBC model and the risky debt model.
Parameters are immutable after initialization to prevent accidental
modification during model execution.

Example:
    >>> from econ_models.config.economic_params import load_economic_params
    >>> params = load_economic_params("config/params.json")
    >>> print(f"Discount factor: {params.discount_factor}")
"""

from dataclasses import dataclass
from typing import Optional
import sys
import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EconomicParams:
    """
    Immutable container for fundamental economic parameters.

    This dataclass holds all economic parameters required by the models.
    The frozen=True ensures immutability, preventing accidental modification
    during model execution.

    Attributes:
        discount_factor: Time preference parameter (beta), must be in (0, 1).
        capital_share: Output elasticity of capital (theta) in production.
        depreciation_rate: Capital depreciation rate (delta).
        productivity_persistence: AR(1) persistence for productivity (rho).
        productivity_std_dev: Standard deviation of productivity shocks (sigma).
        adjustment_cost_convex: Quadratic adjustment cost parameter (psi_0).
        adjustment_cost_fixed: Fixed adjustment cost parameter (psi_1).
        equity_issuance_cost_fixed: Fixed cost of equity issuance (eta_0).
        equity_issuance_cost_linear: Linear cost of equity issuance (eta_1).
        default_cost_proportional: Proportional loss in default (xi).
        corporate_tax_rate: Corporate tax rate (tau).
        risk_free_rate: Risk-free interest rate (r).
        collateral_recovery_fraction: Fraction of capital recoverable as collateral.

    Raises:
        ValueError: If discount_factor is not in the valid range (0, 1).
    """

    discount_factor: Optional[float] = None
    capital_share: Optional[float] = None
    depreciation_rate: Optional[float] = None
    productivity_persistence: Optional[float] = None
    productivity_std_dev: Optional[float] = None
    adjustment_cost_convex: Optional[float] = None
    adjustment_cost_fixed: Optional[float] = None
    equity_issuance_cost_fixed: Optional[float] = None
    equity_issuance_cost_linear: Optional[float] = None
    default_cost_proportional: Optional[float] = None
    corporate_tax_rate: Optional[float] = None
    risk_free_rate: Optional[float] = None
    collateral_recovery_fraction: Optional[float] = None
    estimate_param: Optional[dict] = None

    def __post_init__(self) -> None:
        """Validate parameters after initialization."""
        self._validate_discount_factor()

    def _validate_discount_factor(self) -> None:
        """Ensure discount factor is economically meaningful."""
        if not (0 < self.discount_factor < 1):
            raise ValueError(
                f"Discount factor must be in (0, 1), got {self.discount_factor}"
            )


# def load_economic_params(filename: str) -> EconomicParams:
#     """
#     Load economic parameters from a JSON file.

#     Args:
#         filename: Path to the JSON configuration file.

#     Returns:
#         Populated EconomicParams instance.

#     Raises:
#         SystemExit: If file cannot be read or parameters are invalid.
#     """
#     data = load_json_file(filename)
#     try:
#         return EconomicParams(**data)
#     except TypeError as e:
#         logger.error(f"Parameter mismatch in {filename}: {e}")
#         sys.exit(1)