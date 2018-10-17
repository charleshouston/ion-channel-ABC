from sklearn import linear_model
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple

from .ion_channel_pyabc import (IonChannelModel,
                                IonChannelDistance,
                                ion_channel_sum_stats_calculator)


def plot_parameter_sensitivity(
        model: IonChannelModel,
        parameters: List[str],
        distance_fn: IonChannelDistance,
        sigma: float=0.1,
        n_samples: int=500,
        plot_cutoff: float=0.0,
        log_transform_x: bool=False,
        log_transform_y: bool=False) -> Tuple[sns.FacetGrid, sns.FacetGrid]:
    """
    Plots estimate of sensitivity to parameter variation and regression fit.

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

    plot_cutoff: float
        Plot dotted line on graphs to indicate cutoff for non-sensitive
        parameters.

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
    original = model.get_parameter_vals(parameters)
    original_vals = np.asarray(list(original.values()))
    observations = ion_channel_sum_stats_calculator(
            model.get_experiment_data())

    # Initialize weights
    _ = distance_fn(0, observations, observations)

    p = distance_fn.p
    w = distance_fn.w[0]
    exp_map = distance_fn.exp_map

    m = max(exp_map.values())+1
    X = np.empty((n_samples, len(original_vals)))
    Y = np.empty((n_samples, m))
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
        X[i, :] = np.multiply(original_vals, scale_dist_ln[i, :])
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

    # Predicted values
    Ypred = np.dot(X, beta.transpose())

    # Create dataframe for plotting
    parameter_names = [p.split('.')[-1] for p in parameters]
    exp = [0,]*len(parameter_names)
    for i in range(1, m):
        exp += [i,]*len(parameter_names)
    d = {'param': parameter_names*m,
         'beta': beta.reshape(-1),
         'exp': exp}
    fitted = pd.DataFrame(data=d)

    # Setup seaborn
    pal = sns.cubehelix_palette(len(parameter_names), rot=-.25, light=.7)

    # Plot parameter sensitivity
    grid = (sns.catplot(x='param', y='beta',
                        row='exp',
                        data=fitted, kind='bar',
                        aspect=len(parameter_names)/m,
                        sharey=False,
                        palette=pal)
                        .despine(left=True, bottom=True))
    for i, ax in enumerate(grid.axes.flatten()):
        ax.axhline(y=plot_cutoff, linewidth=1, color='k', linestyle='--')
        ax.axhline(y=-1*plot_cutoff, linewidth=1, color='k', linestyle='--')

    # Plot regression fit
    r2 = []
    exp = []
    for i in range(m):
        r2.append(r2_score(Y[:, i], Ypred[:, i]))
        exp += [i,]*Y.shape[0]
    # Need to transpose Y and Ypred to correctly match up
    # with experiment map.
    d = {'Y': Y.transpose().reshape(-1),
         'Y_predicted': Ypred.transpose().reshape(-1),
         'exp': exp}
    regression_fit = pd.DataFrame(data=d)
    grid2 = (sns.relplot(x='Y', y='Y_predicted',
                         col='exp',
                         data=regression_fit,
                         palette='Purples',
                         facet_kws={'sharex': False,
                                    'sharey': False}))
    for i, ax in enumerate(grid2.axes.flatten()):
        lims = [np.min([ax.get_xlim(), ax.get_ylim()]),
                np.max([ax.get_xlim(), ax.get_ylim()])]
        ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
        ax.set_title('exp={exp}\nr_2 score={r2:.2f}'.format(exp=i, r2=r2[i]))

    # Reset model to original parameters
    model.set_parameters(**original)

    return (grid, grid2)
