from sklearn import linear_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict

from .ion_channel_pyabc import (IonChannelModel,
                                IonChannelDistance,
                                ion_channel_sum_stats_calculator)


def plot_parameter_sensitivity(
        model: IonChannelModel,
        parameters: List[str],
        distance_fn: IonChannelDistance,
        sigma: float=0.1,
        n_samples: int=500,
        log_transform_x: bool=False,
        log_transform_y: bool=False) -> sns.FacetGrid:
    """
    Plots estimate of distance function to parameter variation.

    Based on work of:
    Sobie EA. Parameter sensitivity analysis in electrophysiological models
    using multivariable regression. Biophys J. 2009 Feb 18;96(4):1264-74.

    Parameters
    ----------

    model: IonChannelModel
        Model to interrogate.

    parameters: List[str]
        List of parameter names in model.

    distance_fn: IonChannelDistance
        Distance function to measure between simulation and observations.

    sigma: float
        Parameter lognormal distribution with moves parameters from original
        values.

    n_samples: int
        Number of parameter samples to make to fit model.

    log_transform_x: bool
        Whether to log transform the X variable (parameter values).

    log_transform_y: bool
        Whether to log transform the Y variable (distance results).

    Returns
    -------
    Seaborn catplot showing sensitivity of each parameter by experiment.
    """

    # Generate lognormal distribution for parameters
    scale_dist_ln = np.random.lognormal(mean=0.0, sigma=sigma,
                                        size=(n_samples, len(parameters)))

    # Get original parameter values
    original_values = np.asarray(model.get_parameter_vals(parameters))
    observations = ion_channel_sum_stats_calculator(
            model.get_experiment_data())

    m = len(model.experiments)
    X = np.empty((n_samples, len(original_values)))
    Y = np.empty((n_samples, m))

    # Initialize weights
    _ = distance_fn(0, observations, observations)

    p = distance_fn.p
    w = distance_fn.w[0]
    exp_map = distance_fn.exp_map
    def dist(x, y):
        d = [pow(abs(w[key]*(x[key]-y[key])), p)
             if key in x and key in y else 0
             for key in w]
        d_by_exp = {}
        for i in range(len(d)):
            if exp_map[i] not in d_by_exp.keys():
                d_by_exp[exp_map[i]] = []
            d_by_exp[exp_map[i]].append(d[i])

        d_out = []
        for _, val in d_by_exp.items():
            d_out.append(pow(sum(val), 1/p))
        return d_out

    for i in range(n_samples):
        X[i, :] = np.multiply(original_values, scale_dist_ln[i, :])
        results = ion_channel_sum_stats_calculator(
                model.sample(dict(zip(parameters, X[i, :]))))
        Y[i, :] = dist(results, observations)

    # Mean center and normalise
    if log_transform_x: X = np.log(X)
    if log_transform_y: Y = np.log(Y)
    X = np.divide(X - np.mean(X, axis=0), np.std(X, axis=0))
    Y = np.divide(Y - np.nanmean(Y, axis=0), np.nanstd(Y, axis=0))

    # Ordinary least squares regression
    reg = linear_model.LinearRegression()
    reg.fit(X, Y)
    beta = reg.coef_

    # Create dataframe for plotting
    parameter_names = [p.split('.')[-1] for p in parameters]
    exp = [0,]*len(parameter_names)
    for i in range(1, m):
        exp += [i,]*len(parameter_names)
    d = {'param': parameter_names*m,
         'beta': beta.reshape(-1),
         'exp': exp}
    fitted = pd.DataFrame(data=d)

    # Plot parameter sensitivity
    grid = (sns.catplot(x='param', y='beta',
                        row='exp',
                        data=fitted, kind='bar',
                        aspect=3,
                        sharey=False)
                        .despine(left=True))

    return grid
