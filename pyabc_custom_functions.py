from pyabc.distance_functions import PNormDistance
from pyabc.acceptor import SimpleAcceptor, accept_use_complete_history
from pyabc.model import Model
from pyabc.transition.multivariatenormal import MultivariateNormalTransition

import scipy.stats as stats
import numpy as np
import pandas as pd
import subprocess
import io
from typing import List, Dict, Union
import logging

abclogger = logging.getLogger('ABC')


def ion_channel_sum_stats_calculator(model_output: pd.DataFrame):
    """
    Converts myokit simulation wrapper output into ABC-readable output.
    """
    if not model_output.empty:
        keys = range(len(model_output))
        return dict(zip(keys, model_output.y))
    else:
        return model_output


class EfficientMultivariateNormalTransition(MultivariateNormalTransition):
    """
    Efficient implementation of multivariate normal for multiple samples.

    Only overrides the default `rvs` method.
    """
    def rvs(self, size=None):
        if size is None:
            return self.rvs_single()
        else:
            sample = self.X.sample(n=size, replace=True, weights=self.w).iloc[:]
            perturbed = (sample +
                         np.random.multivariate_normal(
                             np.zeros(self.cov.shape[0]), self.cov, size=size))
            return pd.DataFrame(perturbed)



class IonChannelModel(Model):
    """
    Interface to run external simulations in myokit.

    Parameters
    ----------

    channel: str
        Shortcode name of channel to simulate.

    myokit_python_root: str
        Location of python 2 environment where myokit is installed.

    wrapper_script: str
        Location of wrapper script for running channel in python 2.

    external_par_samples: List[Dict[str, float]]
        List of dictionaries of parameters to randomly choose external to the
        ABC fitting process. These are set randomly prior to running each
        simulation.
    """

    def __init__(self,
                 channel: str,
                 myokit_python_root: str,
                 wrapper_script: str,
                 external_par_samples: List[Dict[str, float]]=None):
        self.external_par_samples = external_par_samples
        self.simulate_args = [myokit_python_root, wrapper_script, channel]
        super().__init__(name = channel+"_model")

    def sample(self,
               pars: Dict[str, float]) -> pd.DataFrame:
        if self.external_par_samples is not None:
            full_pars = dict(np.random.choice(self.external_par_samples),
                             **pars)
        else:
            full_pars = pars

        # Run simulation
        try:
            res = self._simulate(**full_pars)
        except:
            res = pd.DataFrame({})
        return res

    def _simulate(self,
                  n_x: int=None,
                  exp_num: int=None,
                  logvars: List[str]=None,
                  **pars) -> pd.DataFrame:
        """
        Wrapper to simulate a model in myokit using Python 2.

        Simulates in a subprocess python 2 by passing parameters as args
        to another wrapper script.

        Parameters
        ----------

        n_x: int
            Resolution of independent variable to simulate at.
        exp_num: int
            Number of specific experiment to run on its own.
        logvars: List[str]
            List of variables to log in simulations.
        pars: Dict[str, float]
            Parameters as kwargs.

        Returns
        -------
        
        Pandas dataframe with simulated output.

        Errors
        ------

        Throws a ValueError if the simulation fails.
        """
        args = self.simulate_args
        if n_x is not None:
            args.extend(['--n_x', str(n_x)])
        if exp_num is not None:
            args.extend(['--exp_num', str(exp_num)])
        if logvars is not None:
            args.append('--logvars')
            args.extend([var for var in logvars])
        for p, v in pars.items():
            args.extend(['-'+str(p), str(v)])

        # Run simulation in subprocess
        result = subprocess.run(args, stdout=subprocess.PIPE)
        if len(result.stdout) > 0:
            df = pd.read_table(io.BytesIO(result.stdout),
                               delim_whitespace=True,
                               header=0,
                               index_col=False)
            return df
        else:
            raise ValueError("Failed simulation.")
        

class IonChannelAcceptor(SimpleAcceptor):
    """
    Identical to SimpleAcceptor other than uses complete history.
    """
    def __init__(self):
        fun = accept_use_complete_history
        super().__init__(fun)


class IonChannelDistance(PNormDistance):
    """
    Distance function to automatically set weights for a given ion channel.
   
    Reduces weighting between separate experimental data sets according
    to number of data points, scale of experiment y-axis, and error bars
    in reported values.

    Calling is identical to PNormDistance, other than it checks whether
    the simulated input is empty, in which case it returns np.inf.

    Parameters
    ----------

    obs: dict
        Mapping from data point index to value.

    exp_map: dict
        Mapping from data point index to separate experiments.

    err_bars: dict
        Mapping from data point index to absolute error bars.

    err_th: float
        Upper threshold of errors as percentage of experiment IQR to include in 
        down-weighting, avoids overweighting of points with very low uncertainty.
        Defaults to 0.0, i.e. no threshold for the errors.
    """

    def __init__(self,
                 obs: dict,
                 exp_map: dict,
                 err_bars: dict=None,
                 err_th: float=0.0):

        data_by_exp = self._group_data_by_exp(obs, exp_map)
        
        N_by_exp = {k: len(l) for k, l in data_by_exp.items()}
        IQR_by_exp = self._calculate_experiment_IQR(data_by_exp)

        # Optionally include error bars
        if err_bars is None:
            err_by_pt = [1.0] * len(obs)
        else:
            err_by_pt = self._calculate_err_by_pt(
                    exp_map, err_bars, err_th, IQR_by_exp)

        # Create dictionary of weights
        w = {} 
        for ss, exp_num in exp_map.items():
            w[ss] = 1. / (err_by_pt[ss] * np.sqrt(N_by_exp[exp_num] *
                                                  IQR_by_exp[exp_num]))

        mean_weight = np.mean(list(w.values()))
        for key in w.keys():
            w[key] /= mean_weight
        
        abclogger.debug('ion channel weights: {}'.format(w))

        # now initialize PNormDistance
        super().__init__(p=2, w={0: w})

    def __call__(self,
                 t: int,
                 x: dict,
                 y: dict) -> float:
        # x is the simulated output
        if len(x) is 0:
            return np.inf
        return super().__call__(t, x, y)

    def _group_data_by_exp(
            self,
            obs: Dict[int, float],
            exp_map: Dict[int, int]) -> Dict[int, List[float]]:
        """
        Computes a dictionary mapping experiment number to data points.
        """
        data_by_exp = {}
        for index, value in obs.items():
            exp_num = exp_map[index]
            if exp_num not in data_by_exp:
                data_by_exp[exp_num] = []
            data_by_exp[exp_num].append(value)
        return data_by_exp

    def _calculate_experiment_IQR(
            self,
            data_by_exp: Dict[int, List[float]]) -> Dict[int, float]:
        """
        Calculate the interquartile range of each experimental dataset.
        """
        IQR_by_exp = {}
        for exp_num, l_obs in data_by_exp.items():
            weight = stats.iqr(l_obs)
            if weight == 0:
                # if list has only one entry
                weight = abs(l_obs[0])
            IQR_by_exp[exp_num] = weight
        return IQR_by_exp

    def _calculate_err_by_pt(
            self,
            exp_map: Dict[int, int],
            err_bars: Dict[int, float],
            err_th: float, 
            IQR_by_exp: Dict[int, float]) -> Dict[int, float]:
        """
        Calculates weighting due to larger error bars for each observed data point.
        """

        err_by_pt = {}
        err_by_exp = {}
        for index, err in err_bars.items():
            exp_num = exp_map[index]

            if exp_num not in err_by_exp:
                err_by_exp[exp_num] = []

            if np.isnan(err):
                # If no error bar value supplied
                weight = 1.0
            else:
                weight = err
                if err_th > 0.0:
                    iqr = IQR_by_exp[exp_num]
                    # Error threshold given as % of experiment data scale (IQR)
                    th = err_th * iqr
                    weight /= th
                    # If weight is below the threshold, set to no downweighting
                    if weight < 1.0:
                        weight = 1.0
            err_by_pt[index] = weight
            err_by_exp[exp_num].append(weight)

        # Normalise between all points in a given experiment
        for index, weight in err_by_pt.items():
            exp_num = exp_map[index]
            err_by_pt[index] = (err_by_pt[index] / sum(err_by_exp[exp_num]) *
                                len(err_by_exp[exp_num]))
        return err_by_pt
