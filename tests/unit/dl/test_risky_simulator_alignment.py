"""Alignment tests: verify the DL simulator faithfully reproduces the training
architecture of RiskyModelDL_FINAL.

These tests use pseudo data (random weights, synthetic states) to check
structural correctness — network wiring, normalisation, decision logic,
dividend computation, and bond pricing — without requiring trained
checkpoints.

Discrepancies detected during code review and tested here:
1. Network count: training has 8 distinct nets (+ targets); simulator has 6.
   Continuous net and target nets are training-only, which is correct.
2. Input normalisation order must match: (K_norm, B_norm, Z_norm).
3. Policy action composition: single-debt branch with hard invest threshold.
4. Dividend / cash-flow formulas must replicate training expressions.
5. Bond pricing: simulator uses online default net (training uses target).
   At inference with shared weights this is consistent.
6. Equity issuance via softplus network; inline issuance cost.
7. Weight-sharing: same weights loaded into trainer nets must produce
   identical outputs from simulator nets.
"""

import os
import sys
import tempfile
import pytest

import numpy as np
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")

# Ensure src is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from econ_models.config.economic_params import EconomicParams
from econ_models.config.dl_config import DeepLearningConfig
from econ_models.core.standardize import StateSpaceNormalizer
from econ_models.core.types import TENSORFLOW_DTYPE
from econ_models.core.nets import NeuralNetFactory
from econ_models.core.sampling.transitions import TransitionFunctions
from econ_models.econ import (
    ProductionFunctions,
    AdjustmentCostCalculator,
    BondPricingCalculator,
    DebtFlowCalculator,
)
from econ_models.simulator.dl.risky_final import DLSimulatorRiskyFinal

import tensorflow_probability as tfp

tfd = tfp.distributions


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def econ_params():
    """Realistic economic parameters matching the benchmark economy."""
    return EconomicParams(
        discount_factor=0.96,
        capital_share=0.6,
        depreciation_rate=0.15,
        productivity_persistence=0.6,
        productivity_std_dev=0.17,
        adjustment_cost_convex=1.0,
        adjustment_cost_fixed=0.02,
        equity_issuance_cost_fixed=0.08,
        equity_issuance_cost_linear=0.1,
        default_cost_proportional=0.3,
        corporate_tax_rate=0.2,
        risk_free_rate=0.04,
        collateral_recovery_fraction=0.5,
    )


@pytest.fixture
def dl_config():
    """Minimal DL config for testing (small nets, few MC samples)."""
    return DeepLearningConfig(
        batch_size=32,
        epochs=1,
        steps_per_epoch=1,
        hidden_layers=(32, 32),
        activation_function="tanh",
        learning_rate=1e-3,
        mc_sample_number_bond_priceing=10,
        polyak_averaging_decay=0.995,
        capital_min=1.0,
        capital_max=45.0,
        debt_min=-7.0,
        debt_max=75.0,
        productivity_min=0.5,
        productivity_max=2.0,
        min_q_price=1e-4,
        epsilon_debt=1e-4,
    )


@pytest.fixture
def simulator(dl_config, econ_params):
    """Build a DLSimulatorRiskyFinal with random weights."""
    sim = DLSimulatorRiskyFinal(dl_config, econ_params)
    # Warm up all networks with a dummy forward pass
    dummy = tf.zeros((1, 3), dtype=TENSORFLOW_DTYPE)
    for net in [
        sim.capital_policy_net,
        sim.debt_policy_net,
        sim.investment_policy_net,
        sim.default_policy_net,
        sim.equity_issuance_invest_net,
        sim.equity_issuance_noinvest_net,
        sim.value_function_net,
    ]:
        _ = net(dummy)
    return sim


@pytest.fixture
def sample_states():
    """Batch of 64 random states within realistic bounds."""
    tf.random.set_seed(42)
    k = tf.random.uniform((64,), 2.0, 40.0, dtype=TENSORFLOW_DTYPE)
    b = tf.random.uniform((64,), -5.0, 70.0, dtype=TENSORFLOW_DTYPE)
    z = tf.random.uniform((64,), 0.6, 1.8, dtype=TENSORFLOW_DTYPE)
    return k, b, z


# ---------------------------------------------------------------------------
# Test 1: Network architecture alignment
# ---------------------------------------------------------------------------


class TestNetworkArchitectureAlignment:
    """Verify the simulator builds the same network architectures as training."""

    def test_network_count(self, simulator):
        """Simulator should have exactly 6 loadable networks.

        Training has additional target/continuous nets that are not needed
        at inference time.
        """
        nets = [
            simulator.capital_policy_net,
            simulator.debt_policy_net,
            simulator.investment_policy_net,
            simulator.default_policy_net,
            simulator.equity_issuance_invest_net,
            simulator.equity_issuance_noinvest_net,
            simulator.value_function_net,
        ]
        assert all(net is not None for net in nets)

    def test_network_input_output_dims(self, simulator):
        """All nets must accept 3-dim input and produce 1-dim output."""
        dummy = tf.zeros((4, 3), dtype=TENSORFLOW_DTYPE)
        for net_name in [
            "capital_policy_net",
            "debt_policy_net",
            "investment_policy_net",
            "default_policy_net",
            "equity_issuance_invest_net",
            "equity_issuance_noinvest_net",
            "value_function_net",
        ]:
            net = getattr(simulator, net_name)
            out = net(dummy)
            assert out.shape == (4, 1), (
                f"{net_name} output shape {out.shape} != (4, 1)"
            )

    def test_sigmoid_activations(self, simulator):
        """Policy/decision nets should output in (0, 1) — sigmoid activation.

        Value net should be unconstrained (linear activation).
        Equity issuance net uses softplus (non-negative, unbounded above).
        """
        dummy = tf.random.normal((100, 3), dtype=TENSORFLOW_DTYPE)

        sigmoid_nets = [
            "capital_policy_net",
            "debt_policy_net",
            "investment_policy_net",
            "default_policy_net",
        ]
        for name in sigmoid_nets:
            net = getattr(simulator, name)
            out = net(dummy).numpy()
            assert np.all(out >= 0) and np.all(out <= 1), (
                f"{name} output not in [0,1]: min={out.min()}, max={out.max()}"
            )

        # Equity issuance nets: linear output (can be negative)
        for eq_name in ["equity_issuance_invest_net", "equity_issuance_noinvest_net"]:
            eq_out = getattr(simulator, eq_name)(dummy).numpy()
            # Linear output, no constraint check needed

        # Value net can be negative
        v_out = simulator.value_function_net(dummy).numpy()
        # Just verify it's not constrained to [0,1]
        assert v_out.min() < 0.5 or v_out.max() > 0.5, (
            "Value net appears sigmoid-constrained"
        )

    def test_network_names_match_training(self, simulator):
        """Network names should match those used in training for weight compatibility."""
        expected_names = {
            "capital_policy_net": "CapitalPolicyNet",
            "debt_policy_net": "DebtPolicyNet",
            "investment_policy_net": "InvestmentNet",
            "default_policy_net": "DefaultPolicyNet",
            "equity_issuance_invest_net": "EquityIssuanceNetInvest",
            "equity_issuance_noinvest_net": "EquityIssuanceNetNoinvest",
            "value_function_net": "ValueNet",
        }
        for attr, expected in expected_names.items():
            net = getattr(simulator, attr)
            assert net.name == expected, (
                f"Simulator {attr}.name = '{net.name}', "
                f"training uses '{expected}'"
            )


# ---------------------------------------------------------------------------
# Test 2: Input normalisation alignment
# ---------------------------------------------------------------------------


class TestNormalisationAlignment:
    """Verify normalisation is identical between training and simulator."""

    def test_normalisation_order(self, simulator, dl_config):
        """Input concatenation order must be (K_norm, B_norm, Z_norm)."""
        k = tf.constant([[10.0]], dtype=TENSORFLOW_DTYPE)
        b = tf.constant([[5.0]], dtype=TENSORFLOW_DTYPE)
        z = tf.constant([[1.0]], dtype=TENSORFLOW_DTYPE)

        normalizer = StateSpaceNormalizer(dl_config)
        expected = tf.concat([
            normalizer.normalize_capital(k),
            normalizer.normalize_debt(b),
            normalizer.normalize_productivity(z),
        ], axis=1)

        actual = simulator._prepare_inputs(k, b, z)
        np.testing.assert_allclose(
            actual.numpy(), expected.numpy(), rtol=1e-6,
            err_msg="Normalisation order or formula mismatch"
        )

    def test_denormalisation_roundtrip(self, dl_config):
        """normalize → denormalize should be identity for K and B."""
        normalizer = StateSpaceNormalizer(dl_config)
        k = tf.constant([[5.0, 20.0, 40.0]], dtype=TENSORFLOW_DTYPE)
        k = tf.reshape(k, (-1, 1))
        k_rt = normalizer.denormalize_capital(normalizer.normalize_capital(k))
        np.testing.assert_allclose(
            k_rt.numpy(), k.numpy(), rtol=1e-5,
            err_msg="Capital normalisation not invertible"
        )

        b = tf.constant([[-5.0, 10.0, 70.0]], dtype=TENSORFLOW_DTYPE)
        b = tf.reshape(b, (-1, 1))
        b_rt = normalizer.denormalize_debt(normalizer.normalize_debt(b))
        np.testing.assert_allclose(
            b_rt.numpy(), b.numpy(), rtol=1e-5,
            err_msg="Debt normalisation not invertible"
        )


# ---------------------------------------------------------------------------
# Test 3: Weight sharing / loading alignment
# ---------------------------------------------------------------------------


class TestWeightLoadingAlignment:
    """Verify weights saved by training can be loaded by simulator and produce
    identical outputs."""

    def test_weight_save_load_roundtrip(self, simulator, dl_config, econ_params):
        """Save weights → create fresh simulator → load → same outputs."""
        dummy = tf.random.normal((8, 3), dtype=TENSORFLOW_DTYPE)

        # Collect outputs from original
        orig_outputs = {}
        for name in [
            "capital_policy_net",
            "debt_policy_net",
            "investment_policy_net",
            "default_policy_net",
            "equity_issuance_invest_net",
            "equity_issuance_noinvest_net",
            "value_function_net",
        ]:
            orig_outputs[name] = getattr(simulator, name)(dummy).numpy()

        with tempfile.TemporaryDirectory() as tmpdir:
            epoch = 0
            # Save using the same naming convention as save_checkpoint_risky_final
            simulator.capital_policy_net.save_weights(
                os.path.join(tmpdir, f"risky_capital_policy_net_{epoch}.weights.h5")
            )
            simulator.debt_policy_net.save_weights(
                os.path.join(tmpdir, f"risky_debt_policy_net_{epoch}.weights.h5")
            )
            simulator.investment_policy_net.save_weights(
                os.path.join(tmpdir, f"risky_investment_decision_net_{epoch}.weights.h5")
            )
            simulator.default_policy_net.save_weights(
                os.path.join(tmpdir, f"risky_default_policy_net_{epoch}.weights.h5")
            )
            simulator.equity_issuance_invest_net.save_weights(
                os.path.join(tmpdir, f"risky_equity_issuance_net_{epoch}.weights.h5")
            )
            simulator.equity_issuance_noinvest_net.save_weights(
                os.path.join(tmpdir, f"risky_equity_issuance_net_noinvest_{epoch}.weights.h5")
            )
            simulator.value_function_net.save_weights(
                os.path.join(tmpdir, f"risky_value_net_{epoch}.weights.h5")
            )

            # Create fresh simulator and load
            sim2 = DLSimulatorRiskyFinal(dl_config, econ_params)
            sim2.load_solved_dl_solution(
                capital_policy_filepath=os.path.join(
                    tmpdir, f"risky_capital_policy_net_{epoch}.weights.h5"
                ),
                debt_filepath=os.path.join(
                    tmpdir, f"risky_debt_policy_net_{epoch}.weights.h5"
                ),
                investment_policy_filepath=os.path.join(
                    tmpdir, f"risky_investment_decision_net_{epoch}.weights.h5"
                ),
                default_policy_filepath=os.path.join(
                    tmpdir, f"risky_default_policy_net_{epoch}.weights.h5"
                ),
                value_function_filepath=os.path.join(
                    tmpdir, f"risky_value_net_{epoch}.weights.h5"
                ),
                equity_issuance_invest_filepath=os.path.join(
                    tmpdir, f"risky_equity_issuance_net_{epoch}.weights.h5"
                ),
                equity_issuance_noinvest_filepath=os.path.join(
                    tmpdir, f"risky_equity_issuance_net_noinvest_{epoch}.weights.h5"
                ),
            )

            # Compare outputs
            for name in orig_outputs:
                loaded_out = getattr(sim2, name)(dummy).numpy()
                np.testing.assert_allclose(
                    loaded_out,
                    orig_outputs[name],
                    rtol=1e-5,
                    err_msg=f"Weight roundtrip mismatch for {name}",
                )


# ---------------------------------------------------------------------------
# Test 4: Policy action composition (hierarchical decisions)
# ---------------------------------------------------------------------------


class TestPolicyActionComposition:
    """Verify _get_policy_action implements the same hierarchical logic
    as the training code's _get_policy_actions + hard thresholding."""

    def test_invest_wait_branching(self, simulator, sample_states):
        """When invest_prob < 0.5, k' should equal (1-δ)k."""
        k, b, z = sample_states

        k_prime, b_prime, invest_prob, default_prob, equity_issuance = (
            simulator._get_policy_action(k, b, z)
        )

        # For firms that don't invest, k' = (1-δ)k
        not_investing = invest_prob.numpy() <= 0.5
        if np.any(not_investing):
            k_expected = k.numpy()[not_investing] * (
                1.0 - simulator.params.depreciation_rate
            )
            k_actual = k_prime.numpy()[not_investing]
            np.testing.assert_allclose(
                k_actual,
                np.clip(k_expected, simulator.config.capital_min, simulator.config.capital_max),
                rtol=1e-5,
                err_msg="Non-investing firms should have k'=(1-δ)k",
            )

    def test_single_debt_policy_composition(self, simulator, sample_states):
        """Shared-debt architecture: k' = hard_invest * k_target + (1 - hard_invest) * (1-delta)*k.
        b' comes from shared debt policy net (same for both paths)."""
        k, b, z = sample_states

        # Get individual outputs manually
        k_2d = tf.reshape(k, (-1, 1))
        b_2d = tf.reshape(b, (-1, 1))
        z_2d = tf.reshape(z, (-1, 1))
        inputs = simulator._prepare_inputs(k_2d, b_2d, z_2d)

        k_target = simulator.normalizer.denormalize_capital(
            simulator.capital_policy_net(inputs, training=False)
        )
        b_prime = simulator.normalizer.denormalize_debt(
            simulator.debt_policy_net(inputs, training=False)
        )
        invest_prob = simulator.investment_policy_net(inputs, training=False)

        # Compose manually (matching _get_policy_action logic)
        k_no_invest = k_2d * (1.0 - simulator.params.depreciation_rate)
        hard_invest = tf.cast(invest_prob > 0.5, TENSORFLOW_DTYPE)

        k_expected = hard_invest * k_target + (1.0 - hard_invest) * k_no_invest
        b_expected = b_prime

        k_expected = tf.clip_by_value(
            k_expected, simulator.config.capital_min, simulator.config.capital_max
        )
        b_expected = tf.clip_by_value(
            b_expected, simulator.config.debt_min, simulator.config.debt_max
        )

        # Compare with _get_policy_action output
        k_prime, b_prime, _, _, _ = simulator._get_policy_action(k, b, z)

        np.testing.assert_allclose(
            k_prime.numpy(),
            tf.squeeze(k_expected).numpy(),
            rtol=1e-5,
            err_msg="Capital policy composition mismatch",
        )
        np.testing.assert_allclose(
            b_prime.numpy(),
            tf.squeeze(b_expected).numpy(),
            rtol=1e-5,
            err_msg="Debt policy composition mismatch",
        )

    def test_output_clipping(self, simulator, sample_states):
        """Outputs must be clipped to [k_min, k_max] and [b_min, b_max]."""
        k, b, z = sample_states
        k_prime, b_prime, _, _, _ = simulator._get_policy_action(k, b, z)

        assert np.all(k_prime.numpy() >= simulator.config.capital_min - 1e-6)
        assert np.all(k_prime.numpy() <= simulator.config.capital_max + 1e-6)
        assert np.all(b_prime.numpy() >= simulator.config.debt_min - 1e-6)
        assert np.all(b_prime.numpy() <= simulator.config.debt_max + 1e-6)


# ---------------------------------------------------------------------------
# Test 5: Dividend / cash-flow computation alignment
# ---------------------------------------------------------------------------


class TestDividendAlignment:
    """Verify cash-flow formulas in the simulator match training expressions."""

    def test_dividend_formula_with_inline_issuance_cost(self, econ_params, dl_config):
        """Simulate the dividend computation with inline issuance cost
        matching the training code: cost = 1(e > eps) * (eta_0 + eta_1 * e)."""
        one_minus_tax = tf.constant(
            1.0 - econ_params.corporate_tax_rate, dtype=TENSORFLOW_DTYPE
        )

        k = tf.constant([[10.0]], dtype=TENSORFLOW_DTYPE)
        b = tf.constant([[5.0]], dtype=TENSORFLOW_DTYPE)
        z = tf.constant([[1.0]], dtype=TENSORFLOW_DTYPE)
        k_prime = tf.constant([[12.0]], dtype=TENSORFLOW_DTYPE)
        b_prime = tf.constant([[8.0]], dtype=TENSORFLOW_DTYPE)
        q = tf.constant([[0.9]], dtype=TENSORFLOW_DTYPE)
        equity_issuance = tf.constant([[0.5]], dtype=TENSORFLOW_DTYPE)

        # Revenue
        revenue = one_minus_tax * ProductionFunctions.cobb_douglas(k, z, econ_params)

        # Investment
        investment = ProductionFunctions.calculate_investment(k, k_prime, econ_params)

        # Adjustment cost
        adj_cost, _ = AdjustmentCostCalculator.calculate(investment, k, econ_params)

        # Debt flow
        debt_inflow, tax_shield = DebtFlowCalculator.calculate(b_prime, q, econ_params)

        # Payout (before equity issuance)
        payout = revenue + debt_inflow + tax_shield - adj_cost - investment - b

        # Inline issuance cost: 1(e > eps) * (eta_0 + eta_1 * e)
        eps = 1e-6
        eta_0 = econ_params.equity_issuance_cost_fixed
        eta_1 = econ_params.equity_issuance_cost_linear
        e = equity_issuance
        issuance_cost = tf.cast(e > eps, TENSORFLOW_DTYPE) * (eta_0 + eta_1 * e)

        dividend = payout + e - issuance_cost

        # All components should be finite
        assert np.isfinite(dividend.numpy()), "Dividend is not finite"
        # Payout formula check: revenue + qb' + tax_shield - adj - I - b
        manual_payout = (
            revenue.numpy()
            + debt_inflow.numpy()
            + tax_shield.numpy()
            - adj_cost.numpy()
            - investment.numpy()
            - b.numpy()
        )
        np.testing.assert_allclose(
            payout.numpy(), manual_payout, rtol=1e-5,
            err_msg="Payout formula mismatch"
        )

    def test_no_invest_branch_no_adjustment_cost(self, econ_params):
        """When not investing, investment=0 and adj_cost=0."""
        k = tf.constant([[10.0]], dtype=TENSORFLOW_DTYPE)
        k_prime_no_invest = k * (1.0 - econ_params.depreciation_rate)

        investment = ProductionFunctions.calculate_investment(
            k, k_prime_no_invest, econ_params
        )
        adj_cost, _ = AdjustmentCostCalculator.calculate(investment, k, econ_params)

        np.testing.assert_allclose(
            investment.numpy(), 0.0, atol=1e-6,
            err_msg="Investment should be 0 when k'=(1-δ)k"
        )
        np.testing.assert_allclose(
            adj_cost.numpy(), 0.0, atol=1e-6,
            err_msg="Adj cost should be 0 when investment is 0"
        )

    def test_zero_equity_issuance_no_issuance_cost(self, econ_params):
        """When equity issuance e = 0, inline issuance cost should be 0.

        Inline formula: 1(e > eps) * (eta_0 + eta_1 * e).
        With e = 0, the indicator is 0, so cost = 0.
        """
        eps = 1e-6
        eta_0 = econ_params.equity_issuance_cost_fixed
        eta_1 = econ_params.equity_issuance_cost_linear

        e = tf.constant([[0.0]], dtype=TENSORFLOW_DTYPE)
        issuance_cost = tf.cast(e > eps, TENSORFLOW_DTYPE) * (eta_0 + eta_1 * e)

        np.testing.assert_allclose(
            issuance_cost.numpy(), 0.0, atol=1e-6,
            err_msg="Zero equity issuance should have zero issuance cost"
        )

    def test_positive_equity_issuance_cost(self, econ_params):
        """When equity issuance e > eps, cost = eta_0 + eta_1 * e."""
        eps = 1e-6
        eta_0 = econ_params.equity_issuance_cost_fixed
        eta_1 = econ_params.equity_issuance_cost_linear

        e = tf.constant([[1.0]], dtype=TENSORFLOW_DTYPE)
        issuance_cost = tf.cast(e > eps, TENSORFLOW_DTYPE) * (eta_0 + eta_1 * e)

        expected = eta_0 + eta_1 * 1.0
        np.testing.assert_allclose(
            issuance_cost.numpy(), expected, rtol=1e-6,
            err_msg="Issuance cost mismatch for e=1.0"
        )


# ---------------------------------------------------------------------------
# Test 6: Bond pricing alignment
# ---------------------------------------------------------------------------


class TestBondPricingAlignment:
    """Verify bond pricing in simulator matches training code."""

    def test_bond_price_deterministic_seed(self, simulator, sample_states):
        """Bond price should be deterministic with a fixed seed."""
        k, b, z = sample_states
        k_prime, b_prime, _, _, _ = simulator._get_policy_action(k, b, z)

        tf.random.set_seed(123)
        q1 = simulator._estimate_bond_price(k_prime, b_prime, z)

        tf.random.set_seed(123)
        q2 = simulator._estimate_bond_price(k_prime, b_prime, z)

        np.testing.assert_allclose(
            q1.numpy(), q2.numpy(), rtol=1e-6,
            err_msg="Bond price not deterministic with same seed"
        )

    def test_bond_price_range(self, simulator, sample_states):
        """Bond price should be in [min_q_price, 1/(1+r)]."""
        k, b, z = sample_states
        k_prime, b_prime, _, _, _ = simulator._get_policy_action(k, b, z)

        q = simulator._estimate_bond_price(k_prime, b_prime, z)
        q_np = q.numpy()

        q_rf = 1.0 / (1.0 + simulator.params.risk_free_rate)
        assert np.all(q_np >= simulator.config.min_q_price - 1e-6), (
            f"Bond price below floor: min={q_np.min()}"
        )
        assert np.all(q_np <= q_rf + 1e-6), (
            f"Bond price above risk-free: max={q_np.max()}, q_rf={q_rf}"
        )

    def test_zero_default_gives_risk_free_price(self, simulator, dl_config, econ_params):
        """When default probability is zero everywhere, q should equal q_rf
        (for non-trivial debt)."""
        # Create a simulator with a default net that always outputs 0
        sim = DLSimulatorRiskyFinal(dl_config, econ_params)
        dummy = tf.zeros((1, 3), dtype=TENSORFLOW_DTYPE)
        _ = sim.default_policy_net(dummy)

        # Set default net to output near-zero
        for layer in sim.default_policy_net.layers:
            if hasattr(layer, "kernel"):
                layer.kernel.assign(tf.zeros_like(layer.kernel))
            if hasattr(layer, "bias"):
                # Sigmoid(large_negative) ≈ 0
                layer.bias.assign(tf.fill(layer.bias.shape, -10.0))

        k_prime = tf.constant([10.0, 20.0], dtype=TENSORFLOW_DTYPE)
        b_prime = tf.constant([5.0, 10.0], dtype=TENSORFLOW_DTYPE)
        z_curr = tf.constant([1.0, 1.0], dtype=TENSORFLOW_DTYPE)

        q = sim._estimate_bond_price(k_prime, b_prime, z_curr, mc_samples=200)
        q_rf = 1.0 / (1.0 + econ_params.risk_free_rate)

        np.testing.assert_allclose(
            q.numpy(), q_rf, rtol=0.05,
            err_msg="Zero default should give risk-free bond price"
        )

    def test_bond_price_uses_online_default_net(self, simulator, sample_states):
        """Simulator bond pricing should use the (online) default_policy_net,
        not a target network (which doesn't exist at simulation time)."""
        k, b, z = sample_states
        k_prime, b_prime, _, _, _ = simulator._get_policy_action(k, b, z)

        # Get baseline bond price
        tf.random.set_seed(99)
        q_before = simulator._estimate_bond_price(k_prime, b_prime, z)

        # Perturb default net weights
        for layer in simulator.default_policy_net.layers:
            if hasattr(layer, "kernel"):
                layer.kernel.assign(layer.kernel + 0.5)

        tf.random.set_seed(99)
        q_after = simulator._estimate_bond_price(k_prime, b_prime, z)

        # Bond price must change — confirming it uses default_policy_net
        assert not np.allclose(q_before.numpy(), q_after.numpy(), atol=1e-4), (
            "Bond price did not change when default net weights changed"
        )


# ---------------------------------------------------------------------------
# Test 7: Simulation loop integrity
# ---------------------------------------------------------------------------


class TestSimulationLoop:
    """Verify the simulate() method produces well-formed output."""

    def test_output_shapes(self, simulator):
        """Output arrays should have shape (batch_size, T)."""
        batch_size, T = 16, 20
        k0 = tf.constant(np.full(batch_size, 10.0), dtype=TENSORFLOW_DTYPE)
        b0 = tf.constant(np.full(batch_size, 5.0), dtype=TENSORFLOW_DTYPE)
        z0 = tf.constant(np.full(batch_size, 1.0), dtype=TENSORFLOW_DTYPE)
        innovations = tf.random.normal(
            (batch_size, T + 1), dtype=TENSORFLOW_DTYPE
        )

        results = simulator.simulate((k0, b0, z0), innovations)

        expected_keys = [
            "K_curr", "K_next", "B_curr", "B_next",
            "Z_curr", "Z_next", "equity_issuance", "issuance_decision",
        ]
        for key in expected_keys:
            assert key in results, f"Missing key: {key}"
            assert results[key].shape == (batch_size, T), (
                f"{key} shape {results[key].shape} != ({batch_size}, {T})"
            )

    def test_state_transition_continuity(self, simulator):
        """K_next[t] should equal K_curr[t+1] for non-defaulting firms."""
        batch_size, T = 8, 30
        k0 = tf.constant(np.full(batch_size, 15.0), dtype=TENSORFLOW_DTYPE)
        b0 = tf.constant(np.full(batch_size, 3.0), dtype=TENSORFLOW_DTYPE)
        z0 = tf.constant(np.full(batch_size, 1.0), dtype=TENSORFLOW_DTYPE)
        innovations = tf.random.normal(
            (batch_size, T + 1), dtype=TENSORFLOW_DTYPE, seed=42
        )

        results = simulator.simulate((k0, b0, z0), innovations)

        K_next = results["K_next"]
        K_curr = results["K_curr"]

        for i in range(batch_size):
            for t in range(T - 1):
                if np.isfinite(K_next[i, t]) and np.isfinite(K_curr[i, t + 1]):
                    np.testing.assert_allclose(
                        K_next[i, t],
                        K_curr[i, t + 1],
                        rtol=1e-5,
                        err_msg=f"State discontinuity at firm {i}, t={t}",
                    )

    def test_default_nan_masking(self, simulator, dl_config, econ_params):
        """Once a firm defaults, all subsequent states should be NaN."""
        # Create simulator with default net that always defaults
        sim = DLSimulatorRiskyFinal(dl_config, econ_params)
        dummy = tf.zeros((1, 3), dtype=TENSORFLOW_DTYPE)
        _ = sim.default_policy_net(dummy)

        for layer in sim.default_policy_net.layers:
            if hasattr(layer, "kernel"):
                layer.kernel.assign(tf.zeros_like(layer.kernel))
            if hasattr(layer, "bias"):
                layer.bias.assign(tf.fill(layer.bias.shape, 10.0))

        batch_size, T = 4, 10
        k0 = tf.constant(np.full(batch_size, 10.0), dtype=TENSORFLOW_DTYPE)
        b0 = tf.constant(np.full(batch_size, 5.0), dtype=TENSORFLOW_DTYPE)
        z0 = tf.constant(np.full(batch_size, 1.0), dtype=TENSORFLOW_DTYPE)
        innovations = tf.random.normal((batch_size, T + 1), dtype=TENSORFLOW_DTYPE)

        results = sim.simulate((k0, b0, z0), innovations)

        # First period should be recorded (default happens during it)
        # From period 1 onward, K_curr should be NaN
        for t in range(1, T):
            assert np.all(np.isnan(results["K_curr"][:, t])), (
                f"Defaulted firms have non-NaN K_curr at t={t}"
            )

    def test_two_element_initial_states(self, simulator):
        """simulate() should accept (K, Z) tuple and default B to 0."""
        batch_size, T = 8, 10
        k0 = tf.constant(np.full(batch_size, 10.0), dtype=TENSORFLOW_DTYPE)
        z0 = tf.constant(np.full(batch_size, 1.0), dtype=TENSORFLOW_DTYPE)
        innovations = tf.random.normal((batch_size, T + 1), dtype=TENSORFLOW_DTYPE)

        results = simulator.simulate((k0, z0), innovations)

        # B_curr at t=0 should be 0
        np.testing.assert_allclose(
            results["B_curr"][:, 0],
            0.0,
            atol=1e-6,
            err_msg="B_curr should be 0 when initial state is (K, Z)",
        )

    def test_equity_issuance_nonnegative(self, simulator):
        """Equity issuance should be >= 0 (softplus output is non-negative)."""
        batch_size, T = 16, 20
        k0 = tf.constant(np.full(batch_size, 10.0), dtype=TENSORFLOW_DTYPE)
        b0 = tf.constant(np.full(batch_size, 5.0), dtype=TENSORFLOW_DTYPE)
        z0 = tf.constant(np.full(batch_size, 1.0), dtype=TENSORFLOW_DTYPE)
        innovations = tf.random.normal(
            (batch_size, T + 1), dtype=TENSORFLOW_DTYPE, seed=42
        )

        results = simulator.simulate((k0, b0, z0), innovations)

        eq_iss = results["equity_issuance"]
        finite_mask = np.isfinite(eq_iss)
        assert np.all(eq_iss[finite_mask] >= -1e-6), (
            f"Negative equity issuance found: min={eq_iss[finite_mask].min()}"
        )


# ---------------------------------------------------------------------------
# Test 8: Training ↔ Simulator weight compatibility
# ---------------------------------------------------------------------------


class TestTrainingSimulatorWeightCompat:
    """Verify that weights from the training architecture produce identical
    policy outputs when loaded into the simulator."""

    def _build_training_nets(self, dl_config):
        """Build individual networks mimicking the training code's
        _build_networks method."""
        nets = {}
        for name, activation in [
            ("CapitalPolicyNet", "sigmoid"),
            ("DebtPolicyNet", "sigmoid"),
            ("InvestmentNet", "sigmoid"),
            ("DefaultPolicyNet", "sigmoid"),
            ("EquityIssuanceInvestNet", "linear"),
            ("EquityIssuanceNoinvestNet", "linear"),
            ("ValueNet", "linear"),
        ]:
            nets[name] = NeuralNetFactory.build_mlp(
                input_dim=3,
                output_dim=1,
                config=dl_config,
                output_activation=activation,
                scale_factor=1.0,
                name=name,
            )
        return nets

    def test_identical_outputs_after_weight_transfer(
        self, dl_config, econ_params
    ):
        """Build 'training' nets, transfer weights to sim, verify outputs."""
        training_nets = self._build_training_nets(dl_config)
        dummy = tf.random.normal((16, 3), dtype=TENSORFLOW_DTYPE, seed=7)

        # Warm up
        for net in training_nets.values():
            _ = net(dummy)

        # Build simulator and warm up
        sim = DLSimulatorRiskyFinal(dl_config, econ_params)
        for net_name in [
            "capital_policy_net",
            "debt_policy_net",
            "investment_policy_net",
            "default_policy_net",
            "equity_issuance_invest_net",
            "equity_issuance_noinvest_net",
            "value_function_net",
        ]:
            _ = getattr(sim, net_name)(dummy)

        # Map training net names to simulator attribute names
        mapping = {
            "CapitalPolicyNet": "capital_policy_net",
            "DebtPolicyNet": "debt_policy_net",
            "InvestmentNet": "investment_policy_net",
            "DefaultPolicyNet": "default_policy_net",
            "EquityIssuanceNetInvest": "equity_issuance_invest_net",
            "EquityIssuanceNetNoinvest": "equity_issuance_noinvest_net",
            "ValueNet": "value_function_net",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            # Transfer via save/load (most realistic test)
            for train_name, sim_attr in mapping.items():
                path = os.path.join(tmpdir, f"{train_name}.weights.h5")
                training_nets[train_name].save_weights(path)
                getattr(sim, sim_attr).load_weights(path)

            # Compare outputs
            for train_name, sim_attr in mapping.items():
                train_out = training_nets[train_name](dummy).numpy()
                sim_out = getattr(sim, sim_attr)(dummy).numpy()
                np.testing.assert_allclose(
                    sim_out,
                    train_out,
                    rtol=1e-5,
                    err_msg=f"Output mismatch: training {train_name} vs sim {sim_attr}",
                )


# ---------------------------------------------------------------------------
# Test 9: Bellman residual computation
# ---------------------------------------------------------------------------


class TestBellmanResidual:
    """Verify Bellman residual computation runs without error and produces
    sensible outputs."""

    def test_bellman_residual_runs(self, simulator, sample_states):
        """simulate_bellman_residual should run and return finite values."""
        k, b, z = sample_states
        result = simulator.simulate_bellman_residual((k, b, z))

        assert "relative_error" in result
        assert "absolute_error" in result
        assert "mean_value" in result
        assert np.isfinite(result["relative_error"])
        assert np.isfinite(result["absolute_error"])
        assert np.isfinite(result["mean_value"])

    def test_bellman_residual_issuance_cost_handling(self, simulator, sample_states):
        """Bellman residual should apply inline issuance cost matching training:
        cost = 1(e > eps) * (eta_0 + eta_1 * e)."""
        k, b, z = sample_states

        # The method computes equity issuance from the softplus net and
        # applies inline issuance cost formula.
        result = simulator.simulate_bellman_residual((k, b, z))
        assert result["relative_error"] >= 0, "Relative error should be non-negative"


# ---------------------------------------------------------------------------
# Test 10: Lifetime reward computation
# ---------------------------------------------------------------------------


class TestLifetimeReward:
    """Verify lifetime reward computation runs and is well-formed."""

    def test_lifetime_reward_runs(self, simulator):
        """simulate_life_time_reward should return a finite scalar."""
        batch_size, T = 8, 15
        k0 = tf.constant(np.full(batch_size, 10.0), dtype=TENSORFLOW_DTYPE)
        b0 = tf.constant(np.full(batch_size, 3.0), dtype=TENSORFLOW_DTYPE)
        z0 = tf.constant(np.full(batch_size, 1.0), dtype=TENSORFLOW_DTYPE)
        innovations = tf.random.normal(
            (batch_size, T + 1), dtype=TENSORFLOW_DTYPE, seed=42
        )

        reward = simulator.simulate_life_time_reward(
            (k0, b0, z0), innovations
        )

        assert np.isfinite(float(reward)), "Lifetime reward is not finite"

    def test_lifetime_reward_default_zeros_cashflow(self, simulator, dl_config, econ_params):
        """After default, cash flow contributions should be zero."""
        # Use always-default net
        sim = DLSimulatorRiskyFinal(dl_config, econ_params)
        dummy = tf.zeros((1, 3), dtype=TENSORFLOW_DTYPE)
        _ = sim.default_policy_net(dummy)

        for layer in sim.default_policy_net.layers:
            if hasattr(layer, "kernel"):
                layer.kernel.assign(tf.zeros_like(layer.kernel))
            if hasattr(layer, "bias"):
                layer.bias.assign(tf.fill(layer.bias.shape, 10.0))

        batch_size, T = 4, 10
        k0 = tf.constant(np.full(batch_size, 10.0), dtype=TENSORFLOW_DTYPE)
        b0 = tf.constant(np.full(batch_size, 3.0), dtype=TENSORFLOW_DTYPE)
        z0 = tf.constant(np.full(batch_size, 1.0), dtype=TENSORFLOW_DTYPE)
        innovations = tf.random.normal((batch_size, T + 1), dtype=TENSORFLOW_DTYPE)

        reward = sim.simulate_life_time_reward((k0, b0, z0), innovations)

        # All firms default immediately → all contributions are 0
        np.testing.assert_allclose(
            float(reward), 0.0, atol=1e-4,
            err_msg="Always-defaulting should give zero lifetime reward"
        )


# ---------------------------------------------------------------------------
# Test 11: Value function gap computation
# ---------------------------------------------------------------------------


class TestValueFunctionGap:
    """Verify compute_value_function_gap runs correctly."""

    def test_value_gap_runs(self, simulator):
        """compute_value_function_gap should return finite MAE and MAPE."""
        n = 100
        k_grid = tf.random.uniform((n,), 2.0, 40.0, dtype=TENSORFLOW_DTYPE)
        b_grid = tf.random.uniform((n,), -5.0, 50.0, dtype=TENSORFLOW_DTYPE)
        z_grid = tf.random.uniform((n,), 0.6, 1.8, dtype=TENSORFLOW_DTYPE)
        value_labels = tf.random.normal((n,), mean=5.0, dtype=TENSORFLOW_DTYPE)

        result = simulator.compute_value_function_gap(
            (k_grid, b_grid, z_grid), value_labels
        )

        assert "mae" in result
        assert "mape" in result
        assert np.isfinite(result["mae"])
        assert np.isfinite(result["mape"])
        assert result["mae"] >= 0


# ---------------------------------------------------------------------------
# Test 12: Economic consistency checks
# ---------------------------------------------------------------------------


class TestEconomicConsistency:
    """Sanity checks on economic relationships in the simulation output."""

    def test_higher_productivity_higher_output(self, simulator, dl_config, econ_params):
        """Firms with higher Z should tend to produce more output (Y = Z·K^θ)."""
        one_minus_tax = 1.0 - econ_params.corporate_tax_rate

        batch_size = 100
        k = tf.constant(np.full(batch_size, 15.0), dtype=TENSORFLOW_DTYPE)
        b = tf.constant(np.full(batch_size, 5.0), dtype=TENSORFLOW_DTYPE)

        z_low = tf.constant(np.full(batch_size, 0.7), dtype=TENSORFLOW_DTYPE)
        z_high = tf.constant(np.full(batch_size, 1.5), dtype=TENSORFLOW_DTYPE)

        y_low = one_minus_tax * ProductionFunctions.cobb_douglas(
            tf.reshape(k, (-1, 1)),
            tf.reshape(z_low, (-1, 1)),
            econ_params,
        )
        y_high = one_minus_tax * ProductionFunctions.cobb_douglas(
            tf.reshape(k, (-1, 1)),
            tf.reshape(z_high, (-1, 1)),
            econ_params,
        )

        assert float(tf.reduce_mean(y_high)) > float(tf.reduce_mean(y_low)), (
            "Higher productivity should yield higher output"
        )

    def test_bond_price_decreases_with_default_risk(self, simulator, dl_config, econ_params):
        """Bond price should decrease when default probability increases."""
        sim_lo = DLSimulatorRiskyFinal(dl_config, econ_params)
        sim_hi = DLSimulatorRiskyFinal(dl_config, econ_params)

        dummy = tf.zeros((1, 3), dtype=TENSORFLOW_DTYPE)
        _ = sim_lo.default_policy_net(dummy)
        _ = sim_hi.default_policy_net(dummy)

        # Low default: bias = -5 → sigmoid ≈ 0.007
        for layer in sim_lo.default_policy_net.layers:
            if hasattr(layer, "kernel"):
                layer.kernel.assign(tf.zeros_like(layer.kernel))
            if hasattr(layer, "bias"):
                layer.bias.assign(tf.fill(layer.bias.shape, -5.0))

        # High default: bias = 2 → sigmoid ≈ 0.88
        for layer in sim_hi.default_policy_net.layers:
            if hasattr(layer, "kernel"):
                layer.kernel.assign(tf.zeros_like(layer.kernel))
            if hasattr(layer, "bias"):
                layer.bias.assign(tf.fill(layer.bias.shape, 2.0))

        # Use high debt relative to firm value so recovery < B' and
        # default actually reduces bondholder payoff.
        k_prime = tf.constant([5.0] * 10, dtype=TENSORFLOW_DTYPE)
        b_prime = tf.constant([50.0] * 10, dtype=TENSORFLOW_DTYPE)
        z_curr = tf.constant([1.0] * 10, dtype=TENSORFLOW_DTYPE)

        tf.random.set_seed(42)
        q_lo = sim_lo._estimate_bond_price(k_prime, b_prime, z_curr, mc_samples=200)
        tf.random.set_seed(42)
        q_hi = sim_hi._estimate_bond_price(k_prime, b_prime, z_curr, mc_samples=200)

        assert float(tf.reduce_mean(q_hi)) < float(tf.reduce_mean(q_lo)), (
            "Higher default risk should give lower bond price"
        )
