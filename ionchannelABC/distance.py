from pyabc.distance import PNormDistance
from typing import Dict, List
import numpy as np
import warnings
import scipy.stats as stats

import logging
abclogger = logging.getLogger('ABC')


class IonChannelDistance(PNormDistance):
    """Distance function to automatically weights of simulation measurements.

    Reduces weighting between separate experimental data sets according
    to number of data points, scale of experiment y-axis, and optionally
    size of reported errors at data points.

    Calling is identical to pyabc.distance.PNormDistance, other than
    it checks whether the simulated input is empty, in which case it returns
    np.inf.

    Args:
        obs (Dict[int, float]): Mapping from data point index to experimental
            data point value.
        exp_map (Dict[int, int]): Mapping from data point index to experiment
            number.
        err_bars (Dict[int, float]): Optional mapping from data point index to
            reported experimental error.
        err_th (float): Upper threshold for error weighting as percentrage of
            the interquartile range of the experimental data points. This
            avoids overweighting points which have very low uncertainty.
            Defaults to 0.0, i.e. no threshold for errors.
    """

    def __init__(self,
                 obs: Dict[int, float],
                 exp_map: Dict[int, int],
                 err_bars: Dict[int, float]=None,
                 err_th: float=0.0):

        self.exp_map = exp_map
        data_by_exp = self._group_data_by_exp(obs)

        N_by_exp = {k: len(l) for k, l in data_by_exp.items()}
        iqr_by_exp = self._calculate_experiment_IQR(data_by_exp)

        # Optionally include error bars
        if err_bars is None:
            err_by_pt = [1.0] * len(obs)
        else:
            err_by_pt = self._calculate_err_by_pt(
                    err_bars, err_th, iqr_by_exp)

        # Create dictionary of weights
        w = {}
        for ss, exp_num in self.exp_map.items():
            w[ss] = 1. / (err_by_pt[ss] * np.sqrt(N_by_exp[exp_num] *
                                                  iqr_by_exp[exp_num]))

        mean_weight = np.mean(list(w.values()))
        for key in w.keys():
            w[key] /= mean_weight

        abclogger.debug('ion channel weights: {}'.format(w))

        # now initialize PNormDistance
        super().__init__(p=2, w={0: w})

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

        # On continuing a previous run using pyabc's load function, the database
        # stores our data index keys as strings. This doesn't play nice with
        # initializing the epsilon value.
        if not isinstance(list(x.keys())[0], int):
            x = {int(k): v for (k, v) in x.items()}

        # Catch possible runtime overflow warnings
        warnings.simplefilter('error', RuntimeWarning)
        try:
            distance = super().__call__(x, x_0, t, par)
            warnings.resetwarnings()
            return distance
        except RuntimeWarning:
            warnings.resetwarnings()
            return np.inf

    def _group_data_by_exp(
            self,
            obs: Dict[int, float]) -> Dict[int, List[float]]:
        """Computes a mapping experiment number to data points.

        Args:
            obs (Dict[int, float]): Observed data.

        Returns:
            Dict[int, List[float]]: Mapping from experiment number to
                data point indices.
        """
        data_by_exp = {}
        for index, value in obs.items():
            exp_num = self.exp_map[index]
            if exp_num not in data_by_exp:
                data_by_exp[exp_num] = []
            data_by_exp[exp_num].append(value)
        return data_by_exp

    def _calculate_experiment_IQR(
            self,
            data_by_exp: Dict[int, List[float]]) -> Dict[int, float]:
        """Calculate the interquartile range of each experimental dataset.

        Args:
            data_by_exp (Dict[int, List[float]]): Mapping from experiment
                number to data point indices.
            
        Returns:
            Dict[int, float]: Interquartile range for each experiment.
        """
        iqr_by_exp = {}
        for exp_num, l_obs in data_by_exp.items():
            if len(l_obs) > 1:
                weight = stats.iqr(l_obs)
                if weight == 0:
                    # each value is the same
                    weight = 1.0 # TODO: better way to handle this edge case?
            else:
                weight = abs(l_obs[0])
            iqr_by_exp[exp_num] = weight
        return iqr_by_exp

    def _calculate_err_by_pt(
            self,
            err_bars: Dict[int, float],
            err_th: float,
            iqr_by_exp: Dict[int, float]) -> Dict[int, float]:
        """Calculates weighting due to reported error in each point.

        Args:
            err_bars (Dict[int, float]): Mapping from data point index
                to size of reported error.
            err_th (float): Threshold for limiting how much weight can
                be applied.
            iqr_by_exp (Dict[int, float]): Mapping from experiment num
                to interquartile range for experiment.

        Returns:
            Dict[int, float]: Error weighting for each data point.
        """
        err_by_pt = {}
        err_by_exp = {}
        for index, err in err_bars.items():
            exp_num = self.exp_map[index]

            if exp_num not in err_by_exp:
                err_by_exp[exp_num] = []

            if np.isnan(err):
                # If no reported error value supplied
                weight = 1.0
            else:
                weight = err
                if err_th > 0.0:
                    iqr = iqr_by_exp[exp_num]
                    # Error threshold given as % of experiment IQR
                    th = err_th * iqr
                    weight /= th
                    # If weight is below the threshold, no downweighting
                    if weight < 1.0:
                        weight = 1.0
            err_by_pt[index] = weight
            err_by_exp[exp_num].append(weight)

        # Normalise between all points in a given experiment
        for index, weight in err_by_pt.items():
            exp_num = self.exp_map[index]
            if len(err_by_exp[exp_num]) > 1:
                err_by_pt[index] = (
                    (err_by_pt[index] / sum(err_by_exp[exp_num]) *
                     len(err_by_exp[exp_num])))
        return err_by_pt
