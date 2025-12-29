# tests/unit/test_steady_state.py
"""
Unit tests for steady state calculations.
"""

import unittest
from unittest.mock import MagicMock

from econ_models.econ.steady_state import SteadyStateCalculator


class TestSteadyStateCalculator(unittest.TestCase):
    """Test cases for SteadyStateCalculator."""

    def setUp(self):
        """Set up test fixtures with mock parameters."""
        self.params = MagicMock()
        self.params.discount_factor = 0.96
        self.params.depreciation_rate = 0.1
        self.params.corporate_tax_rate = 0.2
        self.params.capital_share = 0.33

    def test_calculate_capital_returns_positive_value(self):
        """Test that steady state capital is positive."""
        k_ss = SteadyStateCalculator.calculate_capital(self.params)
        self.assertGreater(k_ss, 0.0)

    def test_calculate_capital_known_values(self):
        """Test steady state capital with known analytical solution."""
        # With these parameters:
        # r = 1/0.96 - 1 ≈ 0.04167
        # denom = 0.04167 + 0.1 = 0.14167
        # k_ss = ((0.8 * 0.33) / 0.14167)^(1/0.67)
        k_ss = SteadyStateCalculator.calculate_capital(self.params)
        
        # Manual calculation
        r_implied = (1.0 / 0.96) - 1.0
        denom = r_implied + 0.1
        expected = ((0.8 * 0.33) / denom) ** (1.0 / 0.67)
        
        self.assertAlmostEqual(k_ss, expected, places=6)

    def test_calculate_capital_higher_discount_factor_increases_capital(self):
        """Test that more patient agents accumulate more capital."""
        k_ss_base = SteadyStateCalculator.calculate_capital(self.params)
        
        # More patient (higher discount factor)
        self.params.discount_factor = 0.99
        k_ss_patient = SteadyStateCalculator.calculate_capital(self.params)
        
        self.assertGreater(k_ss_patient, k_ss_base)

    def test_calculate_capital_higher_depreciation_decreases_capital(self):
        """Test that higher depreciation reduces steady state capital."""
        k_ss_base = SteadyStateCalculator.calculate_capital(self.params)
        
        self.params.depreciation_rate = 0.2
        k_ss_high_dep = SteadyStateCalculator.calculate_capital(self.params)
        
        self.assertLess(k_ss_high_dep, k_ss_base)

    def test_calculate_capital_higher_tax_decreases_capital(self):
        """Test that higher corporate tax reduces steady state capital."""
        k_ss_base = SteadyStateCalculator.calculate_capital(self.params)
        
        self.params.corporate_tax_rate = 0.4
        k_ss_high_tax = SteadyStateCalculator.calculate_capital(self.params)
        
        self.assertLess(k_ss_high_tax, k_ss_base)

    def test_calculate_capital_zero_tax(self):
        """Test steady state with zero corporate tax."""
        self.params.corporate_tax_rate = 0.0
        k_ss = SteadyStateCalculator.calculate_capital(self.params)
        self.assertGreater(k_ss, 0.0)

    def test_calculate_capital_type_is_float(self):
        """Test that the return type is a float."""
        k_ss = SteadyStateCalculator.calculate_capital(self.params)
        self.assertIsInstance(k_ss, float)


if __name__ == "__main__":
    unittest.main()