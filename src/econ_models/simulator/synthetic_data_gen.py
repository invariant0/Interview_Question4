import os 
import tensorflow as tf
import tensorflow_probability as tfp

from econ_models.config.economic_params import EconomicParams
from econ_models.core.sampling import StateSampler
from econ_models.config.bond_config import BondsConfig
from econ_models.core.types import TENSORFLOW_DTYPE
from econ_models.core.sampling.transitions import TransitionFunctions 
from econ_models.io.file_utils import load_json_file

tfd = tfp.distributions

class synthetic_data_generator():

    def __init__(
        self,
        econ_params_benchmark: EconomicParams,
        sample_bonds_config: dict,
        batch_size: int = 3000,
        T_periods: int = 500,
        include_debt: bool = False,
    ):
        """Initialize the DL simulator for the basic RBC model with default risk.

        Args:
            econ_params_benchmark: Economic parameters.
            sample_bonds_config: Bond configuration dictionary.
        """
        self.econ_params = econ_params_benchmark
        self.sample_bonds_config = sample_bonds_config
        self.batch_size = batch_size
        self.T_periods = T_periods
        self.state_sampler = StateSampler()
        self.shock_dist = tfd.Normal(
            loc=tf.cast(0.0, TENSORFLOW_DTYPE),
            scale=tf.cast(1.0, TENSORFLOW_DTYPE)
        )
        self.include_debt = include_debt    

    def gen(self) -> tuple[tf.Tensor, tf.Tensor]:
        """Generate synthetic data: initial states and shock sequence.

        Returns:
            initial_states: Tuple of tensors (K_0, Z_0) for initial states.
            shock_sequence: Tensor of shape (batch_size, T+1) with productivity path.
        """
        initial_states = self.state_sampler.sample_states(
        batch_size=self.batch_size,
        bonds_config= self.sample_bonds_config,
        include_debt=self.include_debt,
    ) 

        eps_sequence = self.shock_dist.sample(sample_shape=(self.batch_size, self.T_periods))

        return initial_states, eps_sequence