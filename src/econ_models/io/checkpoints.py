# econ_models/io/checkpoints.py
"""
Utilities for saving and loading neural network model weights.

This module provides checkpoint management for deep learning models,
enabling training resumption and model persistence.

Example:
    >>> from econ_models.io.checkpoints import save_checkpoint_basic
    >>> save_checkpoint_basic(value_net, policy_net, epoch=100)
"""

from pathlib import Path
from typing import Optional
import tensorflow as tf


def save_checkpoint_basic(
    value_net: tf.keras.Model,
    policy_net: tf.keras.Model,
    epoch: int,
    save_dir: str = "checkpoints_pretrain/basic"
) -> None:
    """
    Save checkpoints for basic model networks.

    Args:
        value_net: Value function neural network.
        policy_net: Policy function neural network.
        epoch: Current training epoch for naming.
        save_dir: Directory for checkpoint storage.
    """
    path = Path(save_dir)
    path.mkdir(parents=True, exist_ok=True)
    value_net.save_weights(path / f"basic_value_net_{epoch}.weights.h5")
    policy_net.save_weights(path / f"basic_policy_net_{epoch}.weights.h5")

def save_checkpoint_basic_dist(
    value_net: tf.keras.Model,
    policy_net: tf.keras.Model,
    epoch: int,
    save_dir: str = "checkpoints_pretrain_dist/basic"
) -> None:
    """
    Save checkpoints for basic model networks.

    Args:
        value_net: Value function neural network.
        policy_net: Policy function neural network.
        epoch: Current training epoch for naming.
        save_dir: Directory for checkpoint storage.
    """
    path = Path(save_dir)
    path.mkdir(parents=True, exist_ok=True)
    value_net.save_weights(path / f"basic_value_net_{epoch}.weights.h5")
    policy_net.save_weights(path / f"basic_policy_net_{epoch}.weights.h5")

def save_checkpoint_basic_final(
    value_net: tf.keras.Model,
    capital_policy_net: tf.keras.Model,
    investment_policy_net: tf.keras.Model,
    epoch: int,
    save_dir: str = "checkpoints_final/basic",
) -> None:
    """Save checkpoints for the basic-final model networks.

    Args:
        value_net: Value function neural network.
        capital_policy_net: Capital policy neural network.
        investment_policy_net: Investment policy neural network.
        epoch: Current training epoch for naming.
        save_dir: Directory for checkpoint storage.
    """
    path = Path(save_dir)
    path.mkdir(parents=True, exist_ok=True)
    value_net.save_weights(path / f"basic_value_net_{epoch}.weights.h5")
    capital_policy_net.save_weights(path / f"basic_capital_policy_net_{epoch}.weights.h5")
    investment_policy_net.save_weights(path / f"basic_investment_policy_net_{epoch}.weights.h5")


def save_checkpoint_basic_final_dist(
    value_net: tf.keras.Model,
    capital_policy_net: tf.keras.Model,
    investment_policy_net: tf.keras.Model,
    epoch: int,
    save_dir: str = "checkpoints_final_dist/basic"
) -> None:
    """
    Save checkpoints for basic model networks.

    Args:
        value_net: Value function neural network.
        policy_net: Policy function neural network.
        epoch: Current training epoch for naming.
        save_dir: Directory for checkpoint storage.
    """
    path = Path(save_dir)
    path.mkdir(parents=True, exist_ok=True)
    value_net.save_weights(path / f"basic_value_net_{epoch}.weights.h5")
    capital_policy_net.save_weights(path / f"basic_capital_policy_net_{epoch}.weights.h5")
    investment_policy_net.save_weights(path / f"basic_investment_policy_net_{epoch}.weights.h5")


def save_checkpoint_risky(
    value_net: tf.keras.Model,
    epoch: int,
    save_dir: str = "checkpoints/risky",
    suffix: str = ""
) -> None:
    """
    Save checkpoint for risky model value network.

    Args:
        value_net: Value function neural network.
        epoch: Current training epoch for naming.
        save_dir: Directory for checkpoint storage.
        suffix: Optional suffix for filename.
    """
    path = Path(save_dir)
    path.mkdir(parents=True, exist_ok=True)
    value_net.save_weights(path / f"risky_value_net_{epoch}{suffix}.weights.h5")


def save_checkpoint_risky_final_dist(
    value_net: tf.keras.Model,
    investment_decision_net: tf.keras.Model,
    capital_policy_net: tf.keras.Model,
    default_policy_net: tf.keras.Model,
    epoch: int,
    save_dir: str = "checkpoints_final_dist/risky",
    suffix: str = "",
    continuous_net: Optional[tf.keras.Model] = None,
    debt_net: Optional[tf.keras.Model] = None,
    debt_policy_net: Optional[tf.keras.Model] = None,
    equity_issuance_net: Optional[tf.keras.Model] = None,
    equity_issuance_net_noinvest: Optional[tf.keras.Model] = None,
) -> None:
    """
    Save checkpoint for risky dist model networks.

    Args:
        value_net: Value function neural network V(s, p).
        investment_decision_net: Investment decision network.
        capital_policy_net: Capital policy network πk(s, p).
        default_policy_net: Default policy network πd(s, p).
        epoch: Current training epoch for naming.
        save_dir: Directory for checkpoint storage.
        suffix: Optional suffix for filename.
        continuous_net: Optional continuous value network Vcont(s, p).
        debt_net: Shared debt policy network (preferred).
        debt_policy_net: Alias for debt_net (backward compat).
        equity_issuance_net: Equity issuance network (invest path).
        equity_issuance_net_noinvest: Equity issuance network (no-invest path).
    """
    # Accept either debt_net or debt_policy_net (backward compat)
    _debt = debt_net or debt_policy_net
    path = Path(save_dir)
    path.mkdir(parents=True, exist_ok=True)

    value_net.save_weights(path / f"risky_value_net_{epoch}{suffix}.weights.h5")
    investment_decision_net.save_weights(path / f"risky_investment_decision_net_{epoch}{suffix}.weights.h5")
    capital_policy_net.save_weights(path / f"risky_capital_policy_net_{epoch}{suffix}.weights.h5")
    if default_policy_net is not None:
        default_policy_net.save_weights(path / f"risky_default_policy_net_{epoch}{suffix}.weights.h5")
    if continuous_net is not None:
        continuous_net.save_weights(path / f"risky_continuous_net_{epoch}{suffix}.weights.h5")
    if _debt is not None:
        _debt.save_weights(path / f"risky_debt_policy_net_{epoch}{suffix}.weights.h5")
    if equity_issuance_net is not None:
        equity_issuance_net.save_weights(path / f"risky_equity_issuance_net_{epoch}{suffix}.weights.h5")
    if equity_issuance_net_noinvest is not None:
        equity_issuance_net_noinvest.save_weights(path / f"risky_equity_issuance_net_noinvest_{epoch}{suffix}.weights.h5")


# def save_checkpoint_risky_final(
#     value_net: tf.keras.Model,
#     investment_decision_net: tf.keras.Model,
#     capital_issue_net: tf.keras.Model,
#     capital_noissue_net: tf.keras.Model,
#     debt_invest_issue_net: tf.keras.Model,
#     debt_invest_noissue_net: tf.keras.Model,
#     debt_wait_issue_net: tf.keras.Model,
#     debt_wait_noissue_net: tf.keras.Model,
#     default_policy_net: tf.keras.Model,
#     epoch: int,
#     save_dir: str = "checkpoints/risky_final",
#     suffix: str = "",
#     continuous_net: Optional[tf.keras.Model] = None,
#     equity_issuance_net: Optional[tf.keras.Model] = None,
# ) -> None:
#     """
#     Save checkpoint for risky final model networks (4-branch architecture).

#     Args:
#         value_net: Value function neural network V(s).
#         investment_decision_net: Investment decision network.
#         capital_issue_net: Capital policy network for invest+issue branch.
#         capital_noissue_net: Capital policy network for invest+noissue branch.
#         debt_invest_issue_net: Debt policy network for invest+issue branch.
#         debt_invest_noissue_net: Debt policy network for invest+noissue branch.
#         debt_wait_issue_net: Debt policy network for wait+issue branch.
#         debt_wait_noissue_net: Debt policy network for wait+noissue branch.
#         default_policy_net: Default policy network πd(s).
#         epoch: Current training epoch for naming.
#         save_dir: Directory for checkpoint storage.
#         suffix: Optional suffix for filename.
#         continuous_net: Optional continuous value network Vcont(s).
#         equity_issuance_net: Optional equity issuance decision network.
#     """
#     path = Path(save_dir)
#     path.mkdir(parents=True, exist_ok=True)
    
#     # Save value network
#     value_net.save_weights(path / f"risky_value_net_{epoch}{suffix}.weights.h5")
    
#     # Save decision networks
#     investment_decision_net.save_weights(path / f"risky_investment_decision_net_{epoch}{suffix}.weights.h5")
#     if equity_issuance_net is not None:
#         equity_issuance_net.save_weights(path / f"risky_equity_issuance_net_{epoch}{suffix}.weights.h5")
    
#     # Save capital policy networks (2: issue vs no-issue)
#     capital_issue_net.save_weights(path / f"risky_capital_issue_net_{epoch}{suffix}.weights.h5")
#     capital_noissue_net.save_weights(path / f"risky_capital_noissue_net_{epoch}{suffix}.weights.h5")
    
#     # Save debt policy networks (4: invest/wait × issue/no-issue)
#     debt_invest_issue_net.save_weights(path / f"risky_debt_invest_issue_net_{epoch}{suffix}.weights.h5")
#     debt_invest_noissue_net.save_weights(path / f"risky_debt_invest_noissue_net_{epoch}{suffix}.weights.h5")
#     debt_wait_issue_net.save_weights(path / f"risky_debt_wait_issue_net_{epoch}{suffix}.weights.h5")
#     debt_wait_noissue_net.save_weights(path / f"risky_debt_wait_noissue_net_{epoch}{suffix}.weights.h5")
    
#     # Save default policy network
#     if default_policy_net is not None:
#         default_policy_net.save_weights(path / f"risky_default_policy_net_{epoch}{suffix}.weights.h5")
    
#     # Optionally save continuous network
#     if continuous_net is not None:
#         continuous_net.save_weights(path / f"risky_continuous_net_{epoch}{suffix}.weights.h5")


def save_checkpoint_risky_final(
    value_net: tf.keras.Model,
    investment_decision_net: tf.keras.Model,
    capital_policy_net: tf.keras.Model,
    default_policy_net: tf.keras.Model,
    epoch: int,
    save_dir: str = "checkpoints_final/risky",
    suffix: str = "",
    continuous_net: Optional[tf.keras.Model] = None,
    debt_net: Optional[tf.keras.Model] = None,
    equity_issuance_net: Optional[tf.keras.Model] = None,
    equity_issuance_net_noinvest: Optional[tf.keras.Model] = None,
) -> None:
    """
    Save checkpoint for risky final model networks.

    Args:
        value_net: Value function neural network V(s).
        investment_decision_net: Investment decision network.
        capital_policy_net: Capital policy network πk(s).
        default_policy_net: Default policy network πd(s).
        epoch: Current training epoch for naming.
        save_dir: Directory for checkpoint storage.
        suffix: Optional suffix for filename.
        continuous_net: Optional continuous value network Vcont(s).
        debt_net: Shared debt policy network.
        equity_issuance_net: Equity issuance network (invest path).
        equity_issuance_net_noinvest: Equity issuance network (no-invest path).
    """
    path = Path(save_dir)
    path.mkdir(parents=True, exist_ok=True)
    
    # Save value network
    value_net.save_weights(path / f"risky_value_net_{epoch}{suffix}.weights.h5")
    
    # Save policy networks
    investment_decision_net.save_weights(path / f"risky_investment_decision_net_{epoch}{suffix}.weights.h5")
    capital_policy_net.save_weights(path / f"risky_capital_policy_net_{epoch}{suffix}.weights.h5")
    default_policy_net.save_weights(path / f"risky_default_policy_net_{epoch}{suffix}.weights.h5")

    # Shared debt network
    if debt_net is not None:
        debt_net.save_weights(path / f"risky_debt_policy_net_{epoch}{suffix}.weights.h5")

    # Equity issuance networks (invest / no-invest paths)
    if equity_issuance_net is not None:
        equity_issuance_net.save_weights(path / f"risky_equity_issuance_net_{epoch}{suffix}.weights.h5")
    if equity_issuance_net_noinvest is not None:
        equity_issuance_net_noinvest.save_weights(path / f"risky_equity_issuance_net_noinvest_{epoch}{suffix}.weights.h5")

    # Optionally save continuous network
    if continuous_net is not None:
        continuous_net.save_weights(path / f"risky_continuous_net_{epoch}{suffix}.weights.h5")

def save_checkpoint_risk_free(
    value_net: tf.keras.Model,
    capital_policy_net: tf.keras.Model,
    debt_policy_net: tf.keras.Model,
    epoch: int,
    save_dir: str = "checkpoints_pretrain/risk_free",
    suffix: str = "",
    default_policy_net: Optional[tf.keras.Model] = None,
    continuous_net: Optional[tf.keras.Model] = None,
) -> None:
    """
    Save checkpoint for risky upgrade model networks.

    Args:
        value_net: Value function neural network V(s).
        capital_policy_net: Capital policy network πk(s).
        debt_policy_net: Debt policy network πb(s).
        epoch: Current training epoch for naming.
        save_dir: Directory for checkpoint storage.
        suffix: Optional suffix for filename.
        continuous_net: Optional continuous value network Vcont(s).
    """
    path = Path(save_dir)
    path.mkdir(parents=True, exist_ok=True)
    
    # Save value network
    value_net.save_weights(path / f"risk_free_value_net_{epoch}{suffix}.weights.h5")
    
    # Save policy networks
    capital_policy_net.save_weights(path / f"risk_free_capital_policy_net_{epoch}{suffix}.weights.h5")
    debt_policy_net.save_weights(path / f"risk_free_debt_policy_net_{epoch}{suffix}.weights.h5")
    if default_policy_net is not None:
        default_policy_net.save_weights(path / f"risk_free_default_policy_net_{epoch}{suffix}.weights.h5")
    # Optionally save continuous network
    if continuous_net is not None:
        continuous_net.save_weights(path / f"risk_free_continuous_net_{epoch}{suffix}.weights.h5")


def save_checkpoint_risk_free_dist(
    value_net: tf.keras.Model,
    capital_policy_net: tf.keras.Model,
    debt_policy_net: tf.keras.Model,
    epoch: int,
    save_dir: str = "checkpoints_pretrain_dist/risk_free",
    suffix: str = "",
    default_policy_net: Optional[tf.keras.Model] = None,
    continuous_net: Optional[tf.keras.Model] = None,
) -> None:
    """Save checkpoint for risk-free distributional model networks.

    Args:
        value_net: Value function neural network V(s, p).
        capital_policy_net: Capital policy network πk(s, p).
        debt_policy_net: Debt policy network πb(s, p).
        epoch: Current training epoch for naming.
        save_dir: Directory for checkpoint storage.
        suffix: Optional suffix for filename.
        default_policy_net: Optional default policy network πd(s, p).
        continuous_net: Optional continuous value network Vcont(s, p).
    """
    path = Path(save_dir)
    path.mkdir(parents=True, exist_ok=True)

    value_net.save_weights(path / f"risk_free_value_net_{epoch}{suffix}.weights.h5")
    capital_policy_net.save_weights(path / f"risk_free_capital_policy_net_{epoch}{suffix}.weights.h5")
    debt_policy_net.save_weights(path / f"risk_free_debt_policy_net_{epoch}{suffix}.weights.h5")
    if default_policy_net is not None:
        default_policy_net.save_weights(path / f"risk_free_default_policy_net_{epoch}{suffix}.weights.h5")
    if continuous_net is not None:
        continuous_net.save_weights(path / f"risk_free_continuous_net_{epoch}{suffix}.weights.h5")