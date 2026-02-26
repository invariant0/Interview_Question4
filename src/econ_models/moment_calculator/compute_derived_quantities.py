# src/econ_models/moment_calculator/compute_derived_quantities.py
"""Compute derived quantities from simulation data."""

import tensorflow as tf
from src.econ_models.core.types import TENSORFLOW_DTYPE
import numpy as np

def convert_to_numpy(tensor: tf.Tensor) -> np.ndarray:
    """Convert a TensorFlow tensor to a NumPy array."""
    if isinstance(tensor, tf.Tensor):
        return tensor.numpy()
    return tensor

def compute_all_derived_quantities(
    sim_dict: dict,
    delta: float,
    alpha: float,
    include_debt: bool = False,
) -> dict:
    """
    Compute all derived quantities from simulation data.
    
    Args:
        sim_dict: Dictionary containing:
            - K_curr: Current capital (batch_size, n_periods)
            - K_next: Next period capital (batch_size, n_periods)
            - Z_curr: Current productivity (batch_size, n_periods)
            - Z_next: Next period productivity (batch_size, n_periods)
            When include_debt=True, also requires:
            - B_curr: Current debt (batch_size, n_periods)
            - B_next: Next period debt (batch_size, n_periods)
            - equity_issuance: Equity issuance amount (batch_size, n_periods)
              (computed in the VFI simulator from the full payout equation
               using the bond-price schedule Q)
        delta: Depreciation rate
        alpha: Capital share in production
        include_debt: If True, compute debt-related derived quantities
            (leverage, equity issuance rate, issuance frequency binary,
             investment frequency binary). Defaults to False.
    
    Returns:
        Dictionary with derived quantities:
            - output: Production output
            - investment: Investment level
            - investment_rate: Investment / Capital
            - output_growth_rate: (Y_t+1 - Y_t) / Y_t
            - inaction_rate: Binary indicator for zero investment
            When include_debt=True, additionally:
            - leverage: B / K
            - equity_issuance: Raw equity issuance amount
            - equity_issuance_rate: equity_issuance / K
            - issuance_binary: 1 if equity_issuance > threshold, 0 otherwise
            - investment_binary: 1 if |investment_rate| > threshold, 0 otherwise
    """
    K_curr = sim_dict['K_curr']
    K_next = sim_dict['K_next']
    Z_curr = sim_dict['Z_curr']
    Z_next = sim_dict['Z_next']
    
    # Compute output using Cobb-Douglas production function
    output = Z_curr * tf.pow(K_curr, alpha)
    
    # Compute investment
    investment = K_next - (1.0 - delta) * K_curr
    
    # Compute investment rate
    # NUMERIC FIX: Safe division for firms with near-zero capital
    # If K < 1e-5, investment rate is likely meaningless or huge, cap denominator
    denom_K = tf.maximum(K_curr, 1e-5)
    investment_rate = investment / denom_K
    
    # output growth rate
    output_next = Z_next * tf.pow(K_next, alpha)
    
    # NUMERIC FIX: Safe division for growth rates
    # If output is tiny, growth rate explodes. 
    # We use a larger epsilon or mask. Here we use a safe denominator.
    denom_output = tf.maximum(output, 1e-4)
    output_growth_rate = (output_next - output) / denom_output
    
    # Inaction rate logic
    # Note: This checks strictly between -0.00 and 0.00. 
    # If floating point noise exists, this might be too strict, 
    # but compute_inaction_rate.py handles the thresholding logic better.
    # We keep this consistent with the original logic but ensure types match.
    # CRITICAL: Defaulted firms have NaN investment_rate. Comparison with NaN is False,
    # so they would fall into the 'active' (0.0) bucket if not handled.
    # We must explicitly propagate NaNs.
    is_inactive = (investment_rate >= -1e-6) & (investment_rate <= 1e-6)
    inaction_binary = tf.where(
        is_inactive,
        tf.ones_like(investment_rate),
        tf.zeros_like(investment_rate)
    )
    inaction_rate = tf.where(
        tf.math.is_nan(investment_rate),
        investment_rate,
        inaction_binary
    )
    
    # Investment frequency binary: 1 if firm is actively investing, 0 otherwise
    # (inverse of inaction — active means |investment_rate| > threshold)
    investment_binary = tf.where(
        tf.math.is_nan(investment_rate),
        investment_rate,   # propagate NaN
        tf.where(
            is_inactive,
            tf.zeros_like(investment_rate),
            tf.ones_like(investment_rate),
        ),
    )
    
    result = {
        'output': convert_to_numpy(output),
        'output_growth_rate': convert_to_numpy(output_growth_rate),
        'investment': convert_to_numpy(investment),
        'investment_rate': convert_to_numpy(investment_rate),
        'capital': convert_to_numpy(K_curr),
        'productivity': convert_to_numpy(Z_curr),
        'inaction_rate': convert_to_numpy(inaction_rate),
        'investment_binary': convert_to_numpy(investment_binary),
    }
    
    # ── Debt-related derived quantities ──────────────────────────────────
    if include_debt:
        B_curr = sim_dict['B_curr']
        
        # Leverage = B / K  (safe division)
        leverage = B_curr / denom_K
        
        # Equity issuance (pre-computed in VFI simulator using bond price Q)
        # = max(0, -payout);  positive when the firm raises external equity
        eq_iss = tf.constant(sim_dict['equity_issuance'], dtype=K_curr.dtype)
        
        # Equity issuance rate = equity_issuance / K
        equity_issuance_rate = eq_iss / denom_K
        
        # Issuance frequency binary: 1 if firm issues equity, 0 otherwise
        is_issuing = (eq_iss > 1e-6)
        issuance_binary = tf.where(
            is_issuing,
            tf.ones_like(eq_iss),
            tf.zeros_like(eq_iss),
        )
        # Propagate NaN (defaulted firms already have NaN equity_issuance)
        issuance_binary = tf.where(
            tf.math.is_nan(eq_iss),
            eq_iss,
            issuance_binary,
        )
        
        result.update({
            'leverage': convert_to_numpy(leverage),
            'equity_issuance': convert_to_numpy(eq_iss),
            'equity_issuance_rate': convert_to_numpy(equity_issuance_rate),
            'issuance_binary': convert_to_numpy(issuance_binary),
        })

        # Bond price passthrough (if computed by the simulator)
        if 'bond_price' in sim_dict:
            bp = sim_dict['bond_price']
            if not isinstance(bp, np.ndarray):
                bp = convert_to_numpy(bp)
            result['bond_price'] = bp
    
    return result