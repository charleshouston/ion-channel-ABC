from pyabc import History, Distribution
from pyabc.transition import MultivariateNormalTransition
from pyabc.visualization import plot_kde_matrix, plot_kde_2d, plot_kde_1d
from pyabc.visualization import kde_1d, kde_2d
import pandas as pd
import numpy as np
from pyabc_custom import simulate
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
import seaborn as sns


class SeabornFig2Grid():
    """Class to allow seaborn plots on matplotlib subplots.

    https://stackoverflow.com/a/47664533
    """

    def __init__(self, seaborngrid, fig,  subplot_spec):
        self.fig = fig
        self.sg = seaborngrid
        self.subplot = subplot_spec
        if isinstance(self.sg, sns.axisgrid.FacetGrid) or \
            isinstance(self.sg, sns.axisgrid.PairGrid):
            self._movegrid()
        elif isinstance(self.sg, sns.axisgrid.JointGrid):
            self._movejointgrid()
        self._finalize()

    def _movegrid(self):
        """ Move PairGrid or Facetgrid """
        self._resize()
        n = self.sg.axes.shape[0]
        m = self.sg.axes.shape[1]
        self.subgrid = (gridspec.GridSpecFromSubplotSpec(
            n, m, subplot_spec=self.subplot))
        for i in range(n):
            for j in range(m):
                self._moveaxes(self.sg.axes[i,j], self.subgrid[i,j])

    def _movejointgrid(self):
        """ Move Jointgrid """
        h= self.sg.ax_joint.get_position().height
        h2= self.sg.ax_marg_x.get_position().height
        r = int(np.round(h/h2))
        self._resize()
        self.subgrid = (gridspec.GridSpecFromSubplotSpec(
            r+1,r+1, subplot_spec=self.subplot))

        self._moveaxes(self.sg.ax_joint, self.subgrid[1:, :-1])
        self._moveaxes(self.sg.ax_marg_x, self.subgrid[0, :-1])
        self._moveaxes(self.sg.ax_marg_y, self.subgrid[1:, -1])

    def _moveaxes(self, ax, gs):
        #https://stackoverflow.com/a/46906599/4124317
        ax.remove()
        ax.figure=self.fig
        self.fig.axes.append(ax)
        self.fig.add_axes(ax)
        ax._subplotspec = gs
        ax.set_position(gs.get_position(self.fig))
        ax.set_subplotspec(gs)

    def _finalize(self):
        plt.close(self.sg.fig)
        self.fig.canvas.mpl_connect("resize_event", self._resize)
        self.fig.canvas.draw()

    def _resize(self, evt=None):
        self.sg.fig.set_size_inches(self.fig.get_size_inches())


def plot_sim_results(history, channel, n_samples=10, obs=None,
                     original=True):
    """Plot the simulation results using parameter distributions.

    Args:
        history (History): pyabc history object storing ABC results.
        channel (str): Name of channel to simulate.
        n_samples (int): Number of samples to compute statistics on plot.
        obs (DataFrame): Experimental observations to overlay on simulation.
        original (bool): Whether to also plot original settings.

    Returns:
        Seaborn plot of each experiment separately showing mean and standard
        deviation for simulation results.
    """
    #sns.set_context('paper')
    #sns.set_style('white')

    samples = pd.DataFrame()

    # Get prior samples
    # prior_df, prior_w = history.get_distribution(t=0, m=0)
    # prior_th = (prior_df.sample(n=n_samples,
    #                             weights=prior_w,
    #                             replace=True)
    #             .to_dict(orient='records'))
    # for i, th in enumerate(prior_th):
    #     output = simulate(channel, continuous=True, **th)
    #     output['sample'] = i
    #     output['distribution'] = 'prior'
    #     samples = samples.append(output, ignore_index=True)

    # Get posterior samples
    post_df, post_w = history.get_distribution(m=0)
    post_th = (post_df.sample(n=n_samples,
                              weights=post_w,
                              replace=True)
               .to_dict(orient='records'))
    for i, th in enumerate(post_th):
        output = simulate(channel, continuous=True, **th)
        output['sample'] = i
        output['distribution'] = 'post'
        samples = samples.append(output, ignore_index=True)

    # Plotting measurements
    def measured_plot(**kwargs):
        measurements = kwargs.pop('measurements')
        ax = plt.gca()
        data = kwargs.pop('data')
        exp = data['exp'].unique()[0]
        plt.plot(measurements.loc[measurements['exp']==exp]['x'],
                 measurements.loc[measurements['exp']==exp]['y'],
                 label='obs',
                 ls='None', marker='x', c='k')

    # Plotting original settings
    if original:
        output = simulate(channel, {})
        output['sample'] = 0
        output['distribution'] = 'original'
        samples = samples.append(output, ignore_index=True)

    grid = sns.FacetGrid(samples,
                         col="exp", sharex='col', sharey='col',
                         legend_out=True)
    grid = grid.map_dataframe(sns.tsplot, time="x", value="y",
                              unit="sample", condition="distribution",
                              estimator=np.median,
                              err_style="ci_band", ci=95,
                              color='black')

    # Fix markers for different observations
    #markers = ["o", "s"]
    #if original:
    #    markers.append("v")
    for ax in grid.axes.flatten():
        #for l, m in zip(ax.lines, markers):
        for l in ax.lines:
            #l.set_marker(m)
            l.set_linestyle('--')

    if obs is not None:
        grid = (grid.map_dataframe(measured_plot, measurements=obs)
                .add_legend())
    else:
        grid = grid.add_legend()
    return grid


def plot_kde_2d_custom(history, x, y, n_samples=100,
                       times=None, limits=None):
    """Custom 2D KDE plot at selection of time points in history.

    Args:
        history (History): pyabc History object.
        x (str): Variable for x axis.
        y (str): Variable for y axis.
        n_samples (int): number of samples for kde approximation
        times (List[int]): time points to plot.
        limits (dict): Dictionary containing plotting limits for each
            parameter.

    Returns:
        Joint plots for each time point.
    """
    if times is None:
        times = [0, history.max_t]

    xlim, ylim = None, None
    if limits is not None:
        xlim = limits[x]
        ylim = limits[y]

    grids = []
    sns.set(style="white", color_codes=True)
    for t in times:
        df, w = history.get_distribution(m=0, t=t)
        samples = df.sample(n=n_samples,
                            weights=w,
                            replace=True)
        g = sns.jointplot(x=x, y=y, xlim=xlim, ylim=ylim,
                          data=samples,
                          kind="kde", space=0, color="b",
                          stat_func=None,
                          joint_kws=dict(shade_lowest=False))
        grids.append(g)

    fig = plt.figure(figsize=(len(times)*3, 3))
    gs = gridspec.GridSpec(1, len(times))

    for i, g in enumerate(grids):
        _ = SeabornFig2Grid(g, fig, gs[i])
    gs.tight_layout(fig)
    return fig


def animate_kde_matrix(history, n_frames=None, limits=None, colorbar=True):
    """Generate animation of pyabc's `plot_kde_matrix` function.

    Args:
        history (History): pyabc History object.
        n_frames (int): Number of ABCSMC timesteps to plot. If `None` plots
            all by querying history.
        limits (dict): Dictionary containing plotting limits for each
            parameter.
        colorbar (bool): Whether to plot the colorbar on applicable
            subplots.

    Returns:
        Matplotlib FuncAnimation object.
    """

    df, w = history.get_distribution(m=0, t=0)
    if limits is None:
        limits = {}

    if n_frames is None:
        n_frames = history.max_t

    default = (None, None)
    g = plot_kde_matrix(df, w, limits=limits, colorbar=False)

    def animate(i):
        g.data, w = history.get_distribution(m=0, t=i)

        def prep_axes(g):
            g.hue_vals = pd.Series(["_nolegend_"] * len(g.data),
                                   index=g.data.index)
            for ax in g.axes.flatten():
                ax.clear()
            for ax in g.diag_axes.flatten():
                ax.clear()

        def off_diagonal(x, y, **kwargs):
            df = pd.concat((x, y), axis=1)
            plot_kde_2d(df, w,
                        x.name, y.name,
                        xmin=limits.get(x.name, default)[0],
                        xmax=limits.get(x.name, default)[1],
                        ymin=limits.get(y.name, default)[0],
                        ymax=limits.get(y.name, default)[1],
                        ax=plt.gca(), title=False, colorbar=colorbar)

        def scatter(x, y, **kwargs):
            alpha = w / w.max()
            colors = np.zeros((alpha.size, 4))
            colors[:, 3] = alpha
            plt.gca().scatter(x, y, color="k")
            plt.gca().set_xlim(*limits.get(x.name, default))
            plt.gca().set_ylim(*limits.get(y.name, default))

        def diagonal(x, **kwargs):
            df = pd.concat((x,), axis=1)
            plot_kde_1d(df, w, x.name,
                        xmin=limits.get(x.name, default)[0],
                        xmax=limits.get(x.name, default)[1],
                        ax=plt.gca())

        prep_axes(g)
        g.map_diag(diagonal)
        g.map_upper(scatter)
        g.map_lower(off_diagonal)

    frames = np.arange(0, n_frames)
    anim = FuncAnimation(g.fig, animate, frames=frames, repeat=True)
    return anim
