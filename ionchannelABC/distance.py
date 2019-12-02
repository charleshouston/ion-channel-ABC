from typing import Dict, List, Union, Callable
import numpy as np
import warnings

from pyabc.distance import PNormDistance, StochasticKernel
import scipy.stats as stats

import logging
abclogger = logging.getLogger('ABC')


class IonChannelDistance(PNormDistance):
    """Distance function for weighted P-norm distance.

    Weights applied to each point are:
        w_i = \sigma_i^(-1)w_(i|exp)
    where:
        w_(i|exp) = (n_tot-n_exp)/n_tot
    is a weighting to balance the varying number of data points in
    separate experiments and \sigma_i is the standard deviation of
    a data point.

    Calling is identical to pyabc.distance.PNormDistance, other than
    it checks whether the simulated input is empty, in which case it returns
    np.inf.

    Args:
        exp_id (List[int]): Number of experiment each data point belongs to.
        variance (List[float]): Optional mapping from data point index to
            reported variance.
        delta (float): A regularisation parameter to avoid divide by zero for
            zero (or zero reported) variance.
    """

    def __init__(self,
                 exp_id: List[int],
                 variance: List[float],
                 p: float=2,
                 delta: float=0.001):

        # Calculate weighting due to number of data points.
        w_iexp = np.asarray([1./(sum(1. for id_ in exp_id if id_ == id))
                             for id in exp_id])

        # Calculate weighting due to variance in experiment data points.
        w_ivar = np.asarray([max(delta, np.sqrt(var)) for var in variance])
        w_ivar = 1./w_ivar

        # Create dictionary of weights.
        ids = [str(i) for i in range(len(exp_id))]
        w = w_iexp * w_ivar
        weights = dict(zip(ids, w))

        # Balance weights.
        w_mean = np.mean(list(weights.values()))
        for k in weights.keys():
            weights[k] /= w_mean

        abclogger.debug('ion channel weights: {}'.format(weights))

        # now initialize PNormDistance
        super().__init__(p=p, weights={0: weights})

    def __call__(self,
                 x: Dict[str, float],
                 x_0: Dict[str, float],
                 t: int,
                 par: Dict[str, float]=None) -> float:
        """Calculate the error for measured model output.

        This wraps PNormDistance call to check for np.inf in model output.

        Args:
            x (Dict[str, float]): Simulated output measurements.
            x_0 (Dict[str, float]): Observed data.
            t (int): ABC iteration number.
            par (Dict[str, float]): Parameters which may be
                required by some distance functions (pyabc requirement).

        Returns:
            float: Error between x and x_0. If distance gives
                overflow warning will return inf.
        """
        # x is the simulated output
        if (len(x) is 0 or
            any(np.isinf(xi) for xi in x.values())):
            return np.inf

        # Catch possible runtime overflow warnings
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            try:
                distance = super().__call__(x, x_0, t, par)
                return distance
            except RuntimeWarning:
                return np.inf


class DiscrepancyKernel(StochasticKernel):
    """A kernel to infer model discrepancy variance with parameters.

    Expected to be used with pyABC stochastic kernel acceptance and
    epsilon functionality. The current form expects to receive a model
    variance parameter for each experiment, which is then added to
    *every* data point measurement variance in that experiment.

    Args:
        measure_var (Union[Callable, List[float], float]): Variances in
            experimental data.
        keys (List[str]): Keys of summary statistics (in order to be used).
        eps_keys (List[str]): Keys of model discrepancy variance parameters
            (in order to be used).
        exp_mask (List[int]): Experiment ID for each summary statistic.
            i.e. which experiment does this SS belong to?
        pdf_max: (float): Optional norm value for the returned probability
            density.
    """
    def __init__(
            self,
            measure_var: Union[Callable, List[float], float] = None,
            keys: List[str] = None,
            eps_keys: List[str] = None,
            exp_mask: List[int] = None,
            pdf_max: float = None):
        super().__init__(ret_scale="SCALE_LOG", keys=keys, pdf_max=pdf_max)

        self.keys = keys

        if measure_var is not None:
            # regularize by smallest value to avoid divide by zero errors
            measure_var = np.array(measure_var)
            min_measure_var = min(measure_var[measure_var > 0.])
            self.measure_var = np.array([max(vi, min_measure_var)
                                         for vi in measure_var])
        else:
            self.measure_var = measure_var
        self.eps_keys = eps_keys
        self.exp_mask = exp_mask
        self.pdf_max = pdf_max

    def initialize(
            self,
            t: int,
            get_sum_stats: Callable[[], List[dict]],
            x_0: dict = None):
        super().initialize(
            t=t,
            get_sum_stats=get_sum_stats,
            x_0=x_0)

        # dimensionality of summary statistics
        self.dim = sum(np.size(x_0[key]) for key in self.keys)

        # make sure this is in number format
        self.exp_mask = [int(e) for e in self.exp_mask]

        # pdf will be normalised by distance_function
        if self.pdf_max is None and self.measure_var is not None:
            self.pdf_max = np.sum(-0.5*(np.log(self.measure_var)+np.log(2)+np.log(np.pi)))
        else:
            self.pdf_max = 0.

    def __call__(
            self,
            x: dict,
            x_0: dict,
            t: int = None,
            par: dict = None) -> float:
        """Calculate log probability density."""

        # safety check
        if self.keys is None:
            self.initialize_keys(x_0)

        # check for inf values returned by model
        if np.any(np.isinf(list(x.values()))):
            # return a very small probability
            return -1e10 # TODO there has to be a better way?

        # array to hold variance for each summary statistic
        var = np.array([0.,]*self.dim)

        # add any model variance by experiment
        if par is not None:
            if self.eps_keys is not None:
                model_var = [par[k] for k in self.eps_keys]
                # arrange separately for per-experiment model_var
                if self.exp_mask is not None:
                    tmp_var = [model_var[i] for i in self.exp_mask]
                    var += tmp_var
                else:
                    # assuming that only one value was passed
                    try:
                        var += model_var
                    except:
                        raise Exception('error adding model variance')

        # add measurement variance
        if self.measure_var is not None:
            var += self.measure_var

        # difference from experimental data
        diff = _diff_arr(x, x_0, self.keys)

        # compute pdf (log-likelihood of multiple independent gaussians)
        log_2_pi = np.sum(np.log(2) + np.log(np.pi) + np.log(var))
        squares = np.sum((diff**2) / var)
        log_pd = - 0.5 * (log_2_pi + squares)

        return log_pd


def _diff_arr(x, x_0, keys):
    """Get difference array.

    See `pyabc/distance/kernel.py`.
    """
    diff = []
    for key in keys:
        d = x[key] - x_0[key]
        try:
            diff.extend(d)
        except Exception:
            diff.append(d)
    diff = np.array(diff)
    return diff
