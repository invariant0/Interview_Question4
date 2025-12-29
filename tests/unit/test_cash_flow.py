# tests/unit/test_cash_flow.py
"""
Unit tests for cash flow calculations.
"""

import unittest
from unittest.mock import MagicMock

import tensorflow as tf
import numpy as np

from econ_models.econ.cash_flow import CashFlowCalculator


class TestBasicCashFlow(unittest.TestCase):
    """Test cases for basic cash flow calculation."""

    def test_basic_cash_flow_positive(self):
        """Test positive cash flow scenario."""
        profit = tf.constant(10.0)
        investment = tf.constant(3.0)
        adj_cost = tf.constant(1.0)
        
        cash_flow = CashFlowCalculator.basic_cash_flow(profit, investment, adj_cost)
        
        # D = 10 - 3 - 1 = 6
        self.assertAlmostEqual(cash_flow.numpy(), 6.0, places=5)

    def test_basic_cash_flow_negative(self):
        """Test negative cash flow (equity issuance needed)."""
        profit = tf.constant(5.0)
        investment = tf.constant(8.0)
        adj_cost = tf.constant(2.0)
        
        cash_flow = CashFlowCalculator.basic_cash_flow(profit, investment, adj_cost)
        
        # D = 5 - 8 - 2 = -5
        self.assertAlmostEqual(cash_flow.numpy(), -5.0, places=5)

    def test_basic_cash_flow_zero(self):
        """Test zero cash flow scenario."""
        profit = tf.constant(10.0)
        investment = tf.constant(8.0)
        adj_cost = tf.constant(2.0)
        
        cash_flow = CashFlowCalculator.basic_cash_flow(profit, investment, adj_cost)
        
        self.assertAlmostEqual(cash_flow.numpy(), 0.0, places=5)

    def test_basic_cash_flow_batch_inputs(self):
        """Test basic cash flow with batch inputs."""
        profit = tf.constant([10.0, 20.0, 30.0])
        investment = tf.constant([3.0, 5.0, 10.0])
        adj_cost = tf.constant([1.0, 2.0, 3.0])
        
        cash_flow = CashFlowCalculator.basic_cash_flow(profit, investment, adj_cost)
        
        expected = [6.0, 13.0, 17.0]
        np.testing.assert_array_almost_equal(cash_flow.numpy(), expected, decimal=5)


class TestRiskyCashFlow(unittest.TestCase):
    """Test cases for risky cash flow with external finance costs."""

    def setUp(self):
        """Set up test fixtures."""
        self.params = MagicMock()
        self.params.corporate_tax_rate = 0.2
        self.params.risk_free_rate = 0.04
        self.params.equity_issuance_cost_fixed = 0.01
        self.params.equity_issuance_cost_linear = 0.05

    def test_risky_cash_flow_no_debt_positive_payout(self):
        """Test risky cash flow with no debt and positive payout."""
        revenue = tf.constant(10.0)
        investment = tf.constant(3.0)
        adj_cost = tf.constant(1.0)
        debt_curr = tf.constant(0.0)
        debt_next = tf.constant(0.0)
        bond_price = tf.constant(1.0 / 1.04)  # Risk-free price
        
        payout = CashFlowCalculator.risky_cash_flow(
            revenue, investment, adj_cost,
            debt_curr, debt_next, bond_price, self.params
        )
        
        # With no debt: e = 10 - 3 - 1 = 6, no issuance cost
        self.assertAlmostEqual(payout.numpy(), 6.0, places=4)

    def test_risky_cash_flow_negative_triggers_issuance_cost(self):
        """Test that negative payout triggers equity issuance cost."""
        revenue = tf.constant(5.0)
        investment = tf.constant(10.0)
        adj_cost = tf.constant(2.0)
        debt_curr = tf.constant(0.0)
        debt_next = tf.constant(0.0)
        bond_price = tf.constant(1.0 / 1.04)
        
        payout = CashFlowCalculator.risky_cash_flow(
            revenue, investment, adj_cost,
            debt_curr, debt_next, bond_price, self.params
        )
        
        # e_raw = 5 - 10 - 2 = -7
        # issuance_cost = (0.01 + 0.05 * 7) * 1 = 0.36
        # payout = -7 - 0.36 = -7.36
        self.assertLess(payout.numpy(), -7.0)

    def test_risky_cash_flow_debt_inflow(self):
        """Test that new debt provides cash inflow."""
        revenue = tf.constant(5.0)
        investment = tf.constant(10.0)
        adj_cost = tf.constant(0.0)
        debt_curr = tf.constant(0.0)
        debt_next = tf.constant(10.0)
        bond_price = tf.constant(0.9)
        
        payout = CashFlowCalculator.risky_cash_flow(
            revenue, investment, adj_cost,
            debt_curr, debt_next, bond_price, self.params
        )
        
        # Debt provides 0.9 * 10 = 9.0 cash inflow
        self.assertGreater(payout.numpy(), -5.0)

    def test_risky_cash_flow_debt_repayment(self):
        """Test that current debt requires repayment."""
        revenue = tf.constant(15.0)
        investment = tf.constant(3.0)
        adj_cost = tf.constant(1.0)
        debt_curr = tf.constant(5.0)
        debt_next = tf.constant(0.0)
        bond_price = tf.constant(1.0 / 1.04)
        
        payout_no_debt = CashFlowCalculator.risky_cash_flow(
            revenue, investment, adj_cost,
            tf.constant(0.0), debt_next, bond_price, self.params
        )
        payout_with_debt = CashFlowCalculator.risky_cash_flow(
            revenue, investment, adj_cost,
            debt_curr, debt_next, bond_price, self.params
        )
        
        # Debt repayment reduces payout by approximately 5.0
        self.assertLess(
            payout_with_debt.numpy(),
            payout_no_debt.numpy() - 4.0
        )


class TestIssuanceCost(unittest.TestCase):
    """Test cases for equity issuance cost calculation."""

    def setUp(self):
        """Set up test fixtures."""
        self.params = MagicMock()
        self.params.equity_issuance_cost_fixed = 0.02
        self.params.equity_issuance_cost_linear = 0.1

    def test_issuance_cost_zero_for_positive_payout(self):
        """Test that positive payout incurs no issuance cost."""
        payout = tf.constant(5.0)
        
        cost = CashFlowCalculator._calculate_issuance_cost(payout, self.params)
        
        self.assertAlmostEqual(cost.numpy(), 0.0, places=5)

    def test_issuance_cost_zero_for_zero_payout(self):
        """Test that zero payout incurs no issuance cost."""
        payout = tf.constant(0.0)
        
        cost = CashFlowCalculator._calculate_issuance_cost(payout, self.params)
        
        self.assertAlmostEqual(cost.numpy(), 0.0, places=5)

    def test_issuance_cost_positive_for_negative_payout(self):
        """Test that negative payout incurs issuance cost."""
        payout = tf.constant(-10.0)
        
        cost = CashFlowCalculator._calculate_issuance_cost(payout, self.params)
        
        # cost = (0.02 + 0.1 * 10) * 1 = 1.02
        self.assertAlmostEqual(cost.numpy(), 1.02, places=5)

    def test_issuance_cost_formula(self):
        """Test issuance cost formula: eta_0 + eta_1 * |e|."""
        payout = tf.constant(-5.0)
        
        cost = CashFlowCalculator._calculate_issuance_cost(payout, self.params)
        
        expected = 0.02 + 0.1 * 5.0
        self.assertAlmostEqual(cost.numpy(), expected, places=5)

    def test_issuance_cost_batch_inputs(self):
        """Test issuance cost with batch inputs."""
        payout = tf.constant([5.0, 0.0, -5.0, -10.0])
        
        cost = CashFlowCalculator._calculate_issuance_cost(payout, self.params)
        
        expected = [0.0, 0.0, 0.52, 1.02]
        np.testing.assert_array_almost_equal(cost.numpy(), expected, decimal=5)


if __name__ == "__main__":
    unittest.main()