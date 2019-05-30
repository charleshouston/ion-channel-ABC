from pyabc.distance import PNormDistance
from typing import Dict, List
import numpy as np
import warnings
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
                 delta: float=0.001):

        # Calculate weighting due to number of data points.
        w_iexp = np.asarray([len(exp_id)-sum(1. for id_ in exp_id if id_ == id)
                             for id in exp_id])
        w_iexp = w_iexp/len(exp_id)

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
        super().__init__(p=2, w={0: weights})

    def __call__(self,
                 x: Dict[str, float],
                 x_0: Dict[str, float],
                 t: int,
                 par: Dict[str, float]=None) -> float:
        """Calculate the error for measured model output.

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
        warnings.simplefilter('error', RuntimeWarning)
        try:
            distance = super().__call__(x, x_0, t, par)
            warnings.resetwarnings()
            return distance
        except RuntimeWarning:
            warnings.resetwarnings()
            return np.inf