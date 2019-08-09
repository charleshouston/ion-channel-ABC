from sklearn import linear_model
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Callable

from .distance import IonChannelDistance


def calculate_parameter_sensitivity(
        model: Callable,
        summary_statistics: Callable,
        exp_map: List[int],
        parameters: Dict[str, float],
        distance_fn: IonChannelDistance,
        sigma: float=0.1,
        n_samples: int=500) -> Tuple[pd.DataFrame, pd.DataFrame, List[float]]:
    """Estimate of sensitivity to parameter variation and regression fit.

    Based on work of:
    Sobie EA. Parameter sensitivity analysis in electrophysiological models
    using multivariable regression. Biophys J. 2009 Feb 18;96(4):1264-74.

    Args:
        model (Callable): Model to interrogate.
        summary_statistics (Callable): Summary statistics function
        exp_map (List[int]): List of experiment number for each data point.
        parameters (Dict[str, float]): Parameters and base values to perturb from.
        distance_fn (IonChannelDistance): ABC distance function.
        sigma (float): Standard deviation of normalised log-normal distribution.
        n_samples (int): Number of parameter samples for training data.

    Returns:
        Dataframes of sensitivty of each parameter and goodness of regression fit.
        List of r2 scores for regression fits.
        All outputs are used in subsequent functions to produce plots.
    """

    # Generate lognormal distribution for parameters
    scale_dist_ln = np.random.lognormal(mean=0.0, sigma=sigma,
                                        size=(n_samples, len(parameters.keys())))

    # Get original parameter values
    observations = summary_statistics(model(parameters))
    original_vals = np.asarray(list(parameters.values()))

    # Initialize weights
    _ = distance_fn(observations, observations, 0)

    p = distance_fn.p
    w = distance_fn.w[0]

    m = max(exp_map)+1
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

    # Loop through requested samples and eliminate NaN values
    i = 0
    while i < n_samples:
        X[i, :] = np.multiply(original_vals, scale_dist_ln[i, :])
        results = summary_statistics(
                model(dict(zip(parameters.keys(), X[i, :]))))
        if not np.all(np.isfinite(list(results.values()))):
            # Change scale multiplier for this sample
            scale_dist_ln[i, :] = np.random.lognormal(mean=0.0, sigma=sigma,
                                        size=(1, len(parameters.keys())))
            continue
        # Otherwise continue and add to results
        Y[i, :] = dist(results, observations)
        i = i + 1

    # Mean center and normalise
    X = np.divide(X - np.mean(X, axis=0), np.std(X, axis=0))
    Y = np.divide(Y - np.nanmean(Y, axis=0), np.nanstd(Y, axis=0))

    # Ordinary least squares regression
    reg = linear_model.LinearRegression()
    reg.fit(X, Y)
    beta = reg.coef_

    # Predicted values
    Ypred = np.dot(X, beta.transpose())

    # Create dataframe for plotting
    parameter_names = [p.split('.')[-1] for p in parameters.keys()]
    exp = [0,]*len(parameter_names)
    for i in range(1, m):
        exp += [i,]*len(parameter_names)
    d = {'param': parameter_names*m,
         'beta': beta.reshape(-1),
         'exp': exp}
    fitted = pd.DataFrame(data=d)

    # Error measure (r2)
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

    return (fitted, regression_fit, r2)


def plot_parameter_sensitivity(
        fitted: pd.DataFrame,
        plot_cutoff: float=0.0) -> sns.FacetGrid:
    """Plot sensitivity of each model measure to parameters."""

    parameter_names = fitted.param.unique()
    m = len(fitted[fitted.param==parameter_names[0]])

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
    for ax in grid.axes.flatten():
        ax.axhline(y=plot_cutoff, linewidth=1, color='k', linestyle='--')
        ax.axhline(y=-1*plot_cutoff, linewidth=1, color='k', linestyle='--')

    return grid


def plot_regression_fit(
        regression_fit: pd.DataFrame,
        r2: List[float]) -> sns.FacetGrid:
    """Plot goodness of fit of regression for sensitivity study."""

    grid = (sns.relplot(x='Y', y='Y_predicted',
                         col='exp',
                         data=regression_fit,
                         palette='Purples',
                         facet_kws={'sharex': False,
                                    'sharey': False}))
    for i, ax in enumerate(grid.axes.flatten()):
        lims = [np.min([ax.get_xlim(), ax.get_ylim()]),
                np.max([ax.get_xlim(), ax.get_ylim()])]
        ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
        ax.set_title('exp={exp}\nr_2 score={r2:.2f}'.format(exp=i, r2=r2[i]))

    return grid
