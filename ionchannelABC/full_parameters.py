#from sklearn import linear_model
#from sklearn.metrics import r2_score
import scipy.optimize as so
import numpy as np
import pandas as pd
import copy
from typing import List, Dict, Tuple
import warnings

from .ion_channel_pyabc import (IonChannelModel,
                                ion_channel_sum_stats_calculator)
from .distance import IonChannelDistance

def min_fn(macro_vals, *args):
    (model, macro_parameters, abc_parameters,
     observations, w) = args
    result = ion_channel_sum_stats_calculator(
            model.sample({**dict(zip(macro_parameters, macro_vals)),
                          **abc_parameters}))
    x = result
    if len(x) == 0 | np.any(~np.isfinite(list(x.values()))):
        return np.inf
    y = observations
    # always going to be separate experiments in full model so no need to
    # do exp_map step
    d = [abs(w[key]*(x[key]-y[key]))
         if key in x and key in y else 0
         for key in w]
    return sum(d)


def generate_training_data(
        macro_parameters: List[str],
        abc_samples: List[Dict[str, float]],
        model: IonChannelModel,
        distance_fn: IonChannelDistance,
        limits: List[Tuple[float, float]],
        disp: bool=False,
        workers: int=1,
        optimise_args: dict=None):
    """
    Fit final parameters of full model by least squares regression.
    """
    # Send warning about experimental feature
    warnings.warn("experimental feature may produce unexpected results")

    # Get original values for perturbing later
    macro_original_vals = (np.asarray(list(
        model.get_parameter_vals(macro_parameters)
             .values()
             )))

    observations = ion_channel_sum_stats_calculator(
            model.get_experiment_data())
    _ = distance_fn(0, observations, observations)

    w = distance_fn.w[0]
   
    # Dependent variable for fitting - ABC parameter samples
    X = np.empty((len(abc_samples), len(abc_samples[0])))
    for i, sample in enumerate(abc_samples):
        X[i, :] = np.array(list(sample.values()))
    
    Y = np.empty((len(abc_samples), len(macro_parameters)))
    for i, abc_parameters in enumerate(abc_samples):
        if disp:
            print('=> Running ABC sample {}...'.format(i))

        # Fit macro parameters by differential evolution algorithm
        result = so.differential_evolution(min_fn,
                                 bounds=tuple(limits.values()),
                                 args=(copy.deepcopy(model),
                                       macro_parameters,
                                       abc_parameters,
                                       observations,
                                       w),
                                 disp=disp,
                                 workers=workers,
                                 **optimise_args)
#                                 updating='deferred',
#                                 tol=.05,
#                                 maxiter=200)

        if result.success:
            Y[i, :] = result.x
        else:
            print('differential_evolution failed with message: {}'
                  .format(result.message))
            # fill results vector with nan if optimisation fails
            Y[i, :] = [np.nan,]*len(macro_parameters)
            continue

    return (X, Y)

    # Now use estimates of best macro parameters for abc samples to
    # fit a linear model to choose macro parameters
#    if disp:
#        print('=> Fitting linear model...')
#    reg = linear_model.LinearRegression()
#    reg.fit(X, Y)
#    beta = reg.coef_
#    intercept = reg.intercept_
#
#    return (X, Y, beta, intercept)
