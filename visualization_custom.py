from pyabc.transition import MultivariateNormalTransition
import pandas as pd
from pyabc_custom import simulate
import matplotlib.pyplot as plt
import seaborn as sns


def plot_sim_results(df, w, channel, n_samples=10, obs=None):
    """Plot the simulation results using parameter distributions.

    Args:
        df (DataFrame): Distribution results from pyabc output.
        w: Weights for each parameter.
        channel (str): Name of channel to simulate.
        n_samples (int): Number of samples to compute statistics on plot.
        obs (DataFrame): Experimental observations to overlay on simulation.

    Returns:
        Seaborn plot of each experiment separately showing mean and standard
        deviation for simulation results.
    """
    kde = MultivariateNormalTransition()
    kde.fit(df, w)
    samples = pd.DataFrame()
    for i in range(n_samples):
        output = None
        while output is None:
            theta_s = kde.rvs_single()
            try:
                output = simulate(channel, continuous=True, **theta_s)
            except:
                output = None
        output['sample'] = i
        samples = samples.append(output, ignore_index=True)
    def measured_plot(**kwargs):
        measurements = kwargs.pop('measurements')
        ax = plt.gca()
        data = kwargs.pop('data')
        exp = data['exp'].unique()[0]
        plt.plot(measurements.loc[measurements['exp']==exp]['x'],
                 measurements.loc[measurements['exp']==exp]['y'],
                 ls='None', marker='o', c=sns.color_palette()[1])
    grid = sns.FacetGrid(samples, col="exp",
                         sharex=False, sharey=False)
    grid = grid.map_dataframe(sns.tsplot, time="x", value="y",
                              unit="sample")
    if obs is not None:
        grid = grid.map_dataframe(measured_plot, measurements=measurements)
    return grid
