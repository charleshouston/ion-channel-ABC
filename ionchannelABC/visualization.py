from .ion_channel_pyabc import (IonChannelModel,
                                IonChannelDistance,
                                ion_channel_sum_stats_calculator)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_sim_results(df: pd.DataFrame,
                     w: np.ndarray,
                     model: IonChannelModel,
                     n_samples: int=None,
                     obs: pd.DataFrame=None,
                     n_x: int=None):
    """
    Plot model summary statistics output from posterior parameters.

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe of posterior output from pyabc.History data store.

    w: np.ndarray
        Corresponding weight array for posterior output.

    model: IonChannelModel
        Model to produce output from parameter samples.

    n_samples: int
        Number of samples taken to approximate the distribution. Defaults to
        length of `df`.

    obs: pd.DataFrame
        Measurements dataframe to also plot fitting data.

    n_x: int
        Custom x resolution on plots.

    Returns
    -------
    Seaborn relplot of each experiment separately showing mean and standard
    deviation, optionally with fitting data points.
    """
    samples = pd.DataFrame()

    if n_samples is None:
        n_samples = len(w)

    # Get posterior samples
    posterior_theta = (df.sample(n=n_samples,
                                 weights=w,
                                 replace=True)
                       .to_dict(orient='records'))

    for i, theta in enumerate(posterior_theta):
        output = model.sample(theta, n_x)
        output['distribution'] = 'post'
        samples = samples.append(output, ignore_index=True)

    # Plotting measurements
    def measured_plot(**kwargs):
        measurements = kwargs.pop('measurements')
        ax = plt.gca()
        data = kwargs.pop('data')
        exp = data['exp'].unique()[0]
        plt.errorbar(measurements.loc[measurements['exp']==exp]['x'],
                     measurements.loc[measurements['exp']==exp]['y'],
                     yerr=measurements.loc[measurements['exp']==exp]['errs'],
                     label='obs',
                     ls='None', marker='x', c='k')

    with sns.color_palette("gray"):
        grid = sns.relplot(x='x', y='y',
                           col='exp', kind='line',
                           data=samples,
                           facet_kws={'sharex': 'col',
                                      'sharey': 'col'})

    # Format lines in all plots
    for ax in grid.axes.flatten():
        for l in ax.lines:
            l.set_linestyle('--')

    if obs is not None:
        grid = (grid.map_dataframe(measured_plot, measurements=obs)
                .add_legend())
    else:
        grid = grid.add_legend()
    return grid


def plot_distance_weights(
        model: IonChannelModel,
        distance_fn: IonChannelDistance) -> sns.FacetGrid:
    """
    Plots weighting of each sampling statistic by distance function.
    """
    m = len(model.experiments)
    observations = ion_channel_sum_stats_calculator(
            model.get_experiment_data())

    # Initialize weights
    _ = distance_fn(0, observations, observations)

    w = distance_fn.w[0]
    exp = distance_fn.exp_map

    df = pd.DataFrame({'data_point': list(w.keys()),
                       'weights': list(w.values())})
    grid = (sns.catplot(x='data_point', y='weights',
                        data=df, aspect=m,
                        kind='bar')
                        .set(xticklabels=[], 
                             xticks=[]))
    for ax in grid.axes.flatten():
        ax.axhline(y=1, color='k', linestyle='--')
    return grid
