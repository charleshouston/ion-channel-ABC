from functools import wraps, reduce
import pandas as pd
import numpy as np
from pyabc.acceptor import SimpleFunctionAcceptor, accept_use_complete_history
from pyabc.transition.multivariatenormal import MultivariateNormalTransition

"""
This module contains utility classes/functions for use with pyabc.
"""

def log_transform(f):
    @wraps(f)
    def log_transformed(**log_kwargs):
        kwargs = dict([(key[4:], 10**value) if key.startswith("log")
                       else (key, value)
                       for key, value in log_kwargs.items()])
        return f(**kwargs)
    return log_transformed


def combine_sum_stats(*functions):
    def sum_stats_fn(x):
        sum_stats = []
        for i, flist in enumerate(functions):
            for f in flist:
                sum_stats = sum_stats+f(x[i])
        return sum_stats
    return lambda x: sum_stats_fn(x)


def ion_channel_sum_stats_calculator(model_output: pd.DataFrame) -> dict:
    """Converts myokit simulation wrapper output into ABC-readable output.

    Args:
        model_output (pd.DataFrame): Simulation measurements

    Returns:
        dict: Mapping of number for each measurement.
    """
    if not model_output.empty:
        keys = range(len(model_output))
        return dict(zip(keys, model_output.y))
    else:
        return {}


def theoretical_population_size(sampling_density: int,
                                n_parameters: int) -> int:
    """Calculate theoretical minimum particule population size.
    
    Determines theoretical particle population size required to
    sample hyperspace with sufficient fidelity.

    Args:
        sampling_density (int): Number of particles per dimension.
        n_parameters (int): Number of parameters (= number of 
            of the parameter hyperspace).

    Returns:
        Theoretical minimum particle population size.
    """
    return int((10**(np.log10(sampling_density)))**n_parameters)


class IonChannelAcceptor(SimpleFunctionAcceptor):
    """Identical to SimpleFunctionAcceptor other than uses complete history."""
    def __init__(self):
        fun = accept_use_complete_history
        super().__init__(fun)


class EfficientMultivariateNormalTransition(MultivariateNormalTransition):
    """Efficient implementation of multivariate normal for multiple samples.

    Only override the default `rvs` method.
    """
    def rvs(self, size=None):
        if size is None:
            return self.rvs_single()
        else:
            sample = (self.X.sample(n=size, replace=True, weights=self.w)
                      .iloc[:])
            perturbed = (sample +
                         np.random.multivariate_normal(
                             np.zeros(self.cov.shape[0]),
                             self.cov,
                             size=size))
            return pd.DataFrame(perturbed)