# econ_models/core/econ/__init__.py
"""
Core economic logic module.

This package provides economic formulas and calculations used by both
VFI and deep learning solvers.
"""

from econ_models.econ.production import ProductionFunctions
from econ_models.econ.adjustment_costs import AdjustmentCostCalculator
from econ_models.econ.cash_flow import CashFlowCalculator
from econ_models.econ.bond_pricing import BondPricingCalculator
from econ_models.econ.steady_state import SteadyStateCalculator
from econ_models.econ.collateral import CollateralCalculator

# Backward compatibility alias
class EconomicLogic:
    """
    Facade providing backward-compatible access to economic formulas.
    
    This class delegates to specialized calculators while maintaining
    the original API for existing code.
    """
    
    production_function = staticmethod(ProductionFunctions.cobb_douglas)
    calculate_investment = staticmethod(ProductionFunctions.calculate_investment)
    adjustment_costs = staticmethod(AdjustmentCostCalculator.calculate)
    calculate_cash_flow = staticmethod(CashFlowCalculator.basic_cash_flow)
    risky_cash_flow = staticmethod(CashFlowCalculator.risky_cash_flow)
    recovery_value = staticmethod(BondPricingCalculator.recovery_value)
    bond_payoff = staticmethod(BondPricingCalculator.bond_payoff)
    bond_price_risk_neutral = staticmethod(BondPricingCalculator.risk_neutral_price)
    calculate_collateral_limit = staticmethod(CollateralCalculator.calculate_limit)
    calculate_steady_state_capital = staticmethod(SteadyStateCalculator.calculate_capital)


__all__ = [
    'ProductionFunctions',
    'AdjustmentCostCalculator', 
    'CashFlowCalculator',
    'BondPricingCalculator',
    'SteadyStateCalculator',
    'CollateralCalculator',
    'EconomicLogic',
]