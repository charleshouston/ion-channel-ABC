from .distance import IonChannelDistance
from .experiment import (Experiment,
                         setup,
                         get_observations_df)
import numpy as np
import myokit
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Callable, List, Union, Tuple
import pymc3.stats

from pyabc import Distribution
from pyabc.visualization.kde import kde_1d, plot_kde_matrix

def normalise(df, limits=None):
    result = df.copy()
    for feature_name in df.columns:
        if limits is None:
            max_value = df[feature_name].max()
            min_value = df[feature_name].min()
        else:
            max_value = limits[feature_name][1]
            min_value = limits[feature_name][0]
        result[feature_name] = ((df[feature_name] - min_value) /
                                (max_value - min_value))
    return result


def plot_sim_results(modelfiles: Union[str,List[str]],
                     *experiments: Experiment,
                     temp_match_model: int=0,
                     masks: List[List[Union[int,Tuple[int]]]]=None,
                     pacevar: str='membrane.V',
                     tvar: str='phys.T',
                     prev_runs: List[str]=[],
                     additional_pars: Distribution=None,
                     df: Union[pd.DataFrame,List[pd.DataFrame]]=None,
                     w: Union[np.ndarray,List[np.ndarray]]=None,
                     n_samples: int=100,
                     #credible_interval: float=0.89,
                     alpha: float=0.2,
                     exclude_infs: bool=False) -> sns.FacetGrid:
    """Plot output of ABC against experimental and/or original output.

    Note that excluding infinite values assumes that previous runs or
    additional parameters have been passed which gives a stochastic model
    on each parameter sample (i.e. the parameter is not re-sampled on each
    attempt).

    Args:
        modelfile (str): Path to Myokit MMT file.
        *experiments (Experiment): Experiments to plot.
        temp_match_model (int): index of modelfile that all simulation
            output will be temperature adjusted to.
        masks (list): Optional masking list for the case when comparing
            multiple models and you only want some models to plot on
            a subset of the experiments. Format is a list of lists with
            experiments to use having a value not None. A tuple value should
            be used in the case that an experiment produces more than
            one plot.
        pacevar (str): Name of pacing variable in model.
        tvar (str): Name of temperature variable in model.
        prev_runs (list): Previous pyabc.History objects to randomly
            sample from when producing plots.
        df (pd.DataFrame): Dataframe of parameters (see pyabc.History).
            If `None` runs model with current parameter settings.
        w (np.ndarray): The corresponding weights (see pyabc.History).
        n_samples (int): Number of ABC posterior samples used to
            generate summary output.

    Returns
        sns.FacetGrid: Plots of measured output.
    """
    model_samples = pd.DataFrame({})

    # wrap inputs if necessary and only one
    if not isinstance(modelfiles, list):
        modelfiles = [modelfiles]
    if not isinstance(df, list):
        df = [df]
    if not isinstance(w, list):
        w = [w]

    all_observations = []
    model_names = []
    for i, modelfile in enumerate(modelfiles):
        if masks is not None and masks[i] is not None:
            experiment_list = [experiments[j] for j,val in enumerate(masks[i])
                                              if val is not None]
        else:
            experiment_list = experiments
        observations, model, summary_statistics = setup(modelfile,
                                                        *experiment_list,
                                                        pacevar=pacevar,
                                                        tvar=tvar,
                                                        prev_runs=prev_runs,
                                                        additional_pars=additional_pars,
                                                        normalise=False)

        m = myokit.load_model(modelfile)
        name = m.name()
        model_names.append(name)

        # Create list of exp_ids to map to
        exp_map = []
        if masks is not None and masks[i] is not None:
            for j in range(len(masks[i])):
                if masks[i][j] is not None:
                    if isinstance(masks[i][j], tuple):
                        for k in range(len(masks[i][j])):
                            exp_map.append(str(masks[i][j][k]))
                    else:
                        exp_map.append(str(masks[i][j]))

        # Generate model samples from ABC approximate posterior or create default
        # samples if posterior was not provided as input.
        if df is not None and df[i] is not None:
            posterior_samples = (df[i].sample(n=n_samples, weights=w[i], replace=True)
                                      .to_dict(orient='records'))
        else:
            posterior_samples = [{}]

        for j, th in enumerate(posterior_samples):
            results = {'0': float('inf')}
            if exclude_infs:
                try_limit = 100
                try_cnt = 0
                while not np.all(np.isfinite(list(results.values()))) and try_cnt < try_limit:
                    results = summary_statistics(model(th))
                    try_cnt += 1
                if try_cnt == try_limit:
                    raise Exception('exclude_infs failed in 100 iterations')
            else:
                results = summary_statistics(model(th))
            output = pd.DataFrame({'x': observations.x,
                                   'y': list(results.values()),
                                   'exp_id': observations.exp_id})
            output['sample'] = j
            output['model'] = name
            if masks is not None and masks[i] is not None:
                output.exp_id = [exp_map[int(exp_id)] for exp_id in output.exp_id]
            model_samples = model_samples.append(output, ignore_index=True)

        if masks is not None and masks[i] is not None:
            observations.exp_id = [exp_map[int(exp_id)] for exp_id in observations.exp_id]
        all_observations.append(observations)

    # Temperature adjust to model temperature specified in index
    # i.e. scale those model variables not at the same temperature
    # This is done by checking the observation y values (which are scaled
    # to model temperature in `setup`
    plot_obs = all_observations[temp_match_model]
    for i, model_obs in enumerate(all_observations):
        name = model_names[i]
        for exp_id in model_obs['exp_id'].unique():
            po_temp = plot_obs[plot_obs['exp_id']==exp_id]
            mo_temp = model_obs[model_obs['exp_id']==exp_id]
            temp_adjust_factor = np.nanmean(po_temp['y'].values/mo_temp['y'].values)
            model_samples.loc[(model_samples['model']==name) &
                              (model_samples['exp_id']==exp_id),'y'] *= temp_adjust_factor


    # Function for mapping observations onto plot later
    def measured_plot(**kwargs):
        measurements = kwargs.pop('measurements')
        data = kwargs.pop('data')
        exp = data['exp_id'].unique()[0]
        plt.errorbar(measurements.loc[measurements['exp_id']==exp]['x'],
                     measurements.loc[measurements['exp_id']==exp]['y'],
                     yerr=np.sqrt(measurements.loc[measurements['exp_id']==exp]['variance']),
                     ls='None', marker='x', c='k')

    # Actually make the plot
    grid = sns.relplot(x='x', y='y',
                       col='exp_id', kind='line',
                       hue='model', style='model',
                       data=model_samples,
                       estimator=np.median,
                       ci=None,
                       markers=True,
                       facet_kws={'sharex': 'col',
                                  'sharey': 'col'})

    current_palette = sns.color_palette()

    # calculate HPDI for each sample
    for j, exp_id in enumerate(model_samples['exp_id'].unique()):
        ax = grid.axes.flatten()[j]
        for mi, model in enumerate(model_samples['model'].unique()):
            data = model_samples[(model_samples['exp_id']==exp_id) & \
                                 (model_samples['model']==model)]
            hpd67 = np.zeros((len(data['x'].unique()), 2))
            hpd89 = np.zeros((len(data['x'].unique()), 2))
            hpd97 = np.zeros((len(data['x'].unique()), 2))
            for i, xi in enumerate(data['x'].unique()):
                data_xi = data[data['x']==xi]
                hpd67[i,:] = pymc3.stats.hpd(data_xi['y'],
                                             credible_interval=0.67)
                hpd89[i,:] = pymc3.stats.hpd(data_xi['y'],
                                             credible_interval=0.89)
                hpd97[i,:] = pymc3.stats.hpd(data_xi['y'],
                                             credible_interval=0.97)


            # plot on graph
            ax.fill_between(data['x'].unique(), hpd67[:,0], hpd67[:,1],
                            alpha=alpha, color=current_palette[mi])
            ax.fill_between(data['x'].unique(), hpd89[:,0], hpd89[:,1],
                            alpha=alpha, color=current_palette[mi])
            ax.fill_between(data['x'].unique(), hpd97[:,0], hpd97[:,1],
                            alpha=alpha, color=current_palette[mi])

    # Format lines in all plots
    for ax in grid.axes.flatten():
        for l in ax.lines:
            l.set_linestyle('-')

    grid = grid.map_dataframe(measured_plot, measurements=observations)

    return grid


def plot_experiment_traces(modelfile: str,
                           recordvars: List[str],
                           split_data_fns: List[Callable],
                           *experiments: Experiment,
                           df: pd.DataFrame=None,
                           w: np.ndarray=None,
                           prev_runs: List[str]=[],
                           additional_pars: Distribution=None,
                           #credible_interval: float=0.89,
                           alpha: float=0.2,
                           pacevar: str='membrane.V',
                           timevar: str='engine.time',
                           log_interval: float=None,
                           n_samples: int=100,
                           timeout: int=None,
                           exclude_fails: bool=False,
                           try_limit: int=100
                           ) -> sns.FacetGrid:

    if timevar not in recordvars:
        recordvars = [timevar,]+recordvars
    _, model, _ = setup(modelfile,
                        *experiments,
                        log_interval=log_interval,
                        logvars=[pacevar,]+recordvars,
                        pacevar=pacevar,
                        prev_runs=prev_runs,
                        timeout=timeout,
                        additional_pars=additional_pars,
                        normalise=False)

    model_samples = pd.DataFrame({})
    if df is not None:
        posterior_samples = (df.sample(n=n_samples, weights=w, replace=True)
                               .to_dict(orient='records'))
    else:
        posterior_samples = [{}]

    for i, th in enumerate(posterior_samples):
        data = None
        if exclude_fails:
            try_limit = try_limit
            try_cnt = 0
            while data is None and try_cnt<try_limit:
                data = model(th)
                try_cnt += 1
            if try_cnt == try_limit:
                raise Exception('failed to simulate after 100 attempts')
        else:
            data = model(th)

        output = pd.DataFrame({})
        for j, d in enumerate(data):
            split_f = split_data_fns[j]
            data_exp = split_f(data[j])
            for k, step in enumerate(data_exp):
                output_exp = pd.DataFrame({'time': step[timevar],
                                           'y': step[pacevar],
                                           'measure': 'pace',
                                           'step': k,
                                           'exp_id': j})
                output = output.append(output_exp, ignore_index=True)
                for rvar in recordvars:
                    if rvar == timevar:
                        continue
                    output_exp = pd.DataFrame({'time': step[timevar],
                                               'y': step[rvar],
                                               'measure': rvar,
                                               'step': k,
                                               'exp_id': j})
                    output = output.append(output_exp, ignore_index=True)
        output['sample'] = i
        model_samples = model_samples.append(output, ignore_index=True)

    grid = sns.relplot(x='time', y='y',
                       hue='step',
                       col='exp_id', row='measure',
                       palette='viridis',
                       legend=False,
                       data=model_samples,
                       kind='line',
                       estimator=np.median,
                       ci=None,
                       facet_kws={'sharex': 'col',
                                  'sharey': 'row'})

    current_palette = sns.color_palette()

    # calculate HPDI for each sample
    for j, exp_id in enumerate(model_samples['exp_id'].unique()):
        for k, measure in enumerate(model_samples['measure'].unique()):
            ax = grid.axes.flatten()[j]
            for si, step_id in enumerate(model_samples['step'].unique()):
                data = model_samples[(model_samples['exp_id']==exp_id)   & \
                                     (model_samples['measure']==measure) & \
                                     (model_samples['step']==step_id)]
                hpd67 = np.zeros((len(data['time'].unique()), 2))
                hpd89 = np.zeros((len(data['time'].unique()), 2))
                hpd97 = np.zeros((len(data['time'].unique()), 2))
                for i, ti in enumerate(data['time'].unique()):
                    data_ti = data[data['time']==ti]
                    hpd67[i,:] = pymc3.stats.hpd(data_ti['y'],
                                                 credible_interval=0.67)
                    hpd89[i,:] = pymc3.stats.hpd(data_ti['y'],
                                                 credible_interval=0.89)
                    hpd97[i,:] = pymc3.stats.hpd(data_ti['y'],
                                                 credible_interval=0.97)

                # plot on graph
                ax.fill_between(data['time'].unique(), hpd67[:,0], hpd67[:,1],
                                alpha=alpha, color=current_palette[si])
                ax.fill_between(data['time'].unique(), hpd89[:,0], hpd89[:,1],
                                alpha=alpha, color=current_palette[si])
                ax.fill_between(data['time'].unique(), hpd97[:,0], hpd97[:,1],
                                alpha=alpha, color=current_palette[si])

    return grid


def plot_distance_weights(
        observations: pd.DataFrame,
        distance_fn: IonChannelDistance,
        sum_stats_fn: Callable) -> sns.FacetGrid:
    """Plots weights of each sampling statistic in distance function.

    Args:
        observations (pd.DataFrame): Observation results.
        distance_fn (IonChannelDistance): ABC distance function.
        sum_stats_fn (Callable): ABC summary statistics calculator.

    Returns:
        sns.FacetGrid: Bar graph showing relative weights for each
            data point in distance function.
    """
    # Initialize weights
    _ = distance_fn(observations, observations, 0)

    w = distance_fn.w[0]
    exp = distance_fn.exp_map
    m = np.max(exp)

    df = pd.DataFrame({'data_point': list(w.keys()),
                       'weights': list(w.values())})

    pal = sns.cubehelix_palette(len(w), rot=-.25, light=.7)
    grid = (sns.catplot(x='data_point', y='weights',
                        data=df, aspect=m,
                        kind='bar',
                        palette=pal)
                        .set(xticklabels=[],
                             xticks=[]))
    for ax in grid.axes.flatten():
        ax.axhline(y=1, color='k', linestyle='--')
    return grid


def plot_variables(v: np.ndarray,
                   variables: Union[dict,List[dict]],
                   modelfiles: Union[str,List[str]],
                   par_samples: Union[dict,List[dict]]=None,
                   original: Union[bool,List[bool]]=False,
                   #credible_interval: float=0.89,
                   alpha: float=0.2,
                   figshape: Tuple[int]=None):
    """Plot model variables over voltage range.

    v (np.ndarray): Voltage range to plot over.
    variables (dict): Dictionary mapping common names to variables in the
        model to be plotted.
    modelfiles (str): Location of modelfiles to plot.
    par_samples (dict): Samples of parameter values to set in model to
        plot summarised results.
    original (bool): Whether to plot original parameter values as
        dashed lines.
    figshape (Tuple[int]): Shape of figure. Defaults to single row
    """
    if original:
        assert par_samples is not None, 'Require parameter value samples to plot original values.'

    if not isinstance(variables, list):
        variables = [variables]
    if not isinstance(modelfiles, list):
        modelfiles = [modelfiles]
    if not isinstance(par_samples, list):
        par_samples = [par_samples,]*len(variables)
    if not isinstance(original, list):
        original = [original,]*len(variables)

    if figshape is None:
        ncols = len(variables[0])
        nrows = 1
    else:
        ncols = figshape[0]
        nrows = figshape[1]

    fig, ax = plt.subplots(ncols=ncols, nrows=nrows,
                           figsize=(ncols*5, nrows*5),
                           sharex=True)

    samples = pd.DataFrame({})
    for i, modelfile in enumerate(modelfiles):
        m = myokit.load_model(modelfile)

        original_vals = {}
        if original[i]:
            for p in par_samples[i][0].keys():
                name = p
                if p.startswith("log"):
                    name = p[4:]
                original_vals[name] = m.value(name)

        if par_samples[i] is not None:
            for j, pars in enumerate(par_samples[i]):
                for p, val in pars.items():
                    if p.startswith("log"):
                        val = 10**val
                        p = p[4:]
                    m.set_value(p, val)

                output = pd.DataFrame({})
                for key, var in variables[i].items():
                    output[key] = m.get(var).pyfunc()(v)
                output['V'] = v
                output['samples'] = j
                output['model'] = m.name()
                output['data'] = 'recalibrated'
                samples = samples.append(output, ignore_index=True)

        # Plot original values
        if par_samples[i] is None or original[i]:
            output = pd.DataFrame({})
            if original:
                for p, val in original_vals.items():
                    m.set_value(p, val)
            for key, var in variables[i].items():
                try:
                    output[key] = m.get(var).pyfunc()(v)
                except:
                    raise Exception('Could not find variable '+key+' in modelfile '+modelfile)
                output['V'] = v
            output['model'] = m.name()
            output['data'] = 'original'
            samples = samples.append(output)

    current_palette = sns.color_palette()

    # redorder axes for plotting
    for i, key in enumerate(variables[0].keys()):
        sns.lineplot(x='V', y=key, hue='model', style='data',
                     data=samples,
                     estimator=np.median,
                     ci=None,
                     ax=ax.flatten()[i],
                     legend=False)
        sns.despine(ax=ax.flatten()[i])
        for mi, model in enumerate(samples['model'].unique()):
            data = samples[(samples['model']==model)]
            hpd67 = np.zeros((len(data['V'].unique()), 2))
            hpd89 = np.zeros((len(data['V'].unique()), 2))
            hpd97 = np.zeros((len(data['V'].unique()), 2))
            for k, vk in enumerate(data['V'].unique()):
                data_vk = data[data['V']==vk]
                hpd67[k,:] = pymc3.stats.hpd(data_vk[key],
                                             credible_interval=0.67)
                hpd89[k,:] = pymc3.stats.hpd(data_vk[key],
                                             credible_interval=0.89)
                hpd97[k,:] = pymc3.stats.hpd(data_vk[key],
                                             credible_interval=0.97)

            # plot on graph
            ax.flatten()[i].fill_between(data['V'].unique(), hpd67[:,0], hpd67[:,1],
                                         alpha=alpha, color=current_palette[mi])
            ax.flatten()[i].fill_between(data['V'].unique(), hpd89[:,0], hpd89[:,1],
                                         alpha=alpha, color=current_palette[mi])
            ax.flatten()[i].fill_between(data['V'].unique(), hpd97[:,0], hpd97[:,1],
                                         alpha=alpha, color=current_palette[mi])

    plt.tight_layout()

    return fig, ax


def plot_parameters_kde(df, w, limits, aspect=None, height=None):
    """Plot grid of parameter KDE density estimates.

    EXPERIMENTAL: probably better off using functions from `pyabc`
    library to plot KDEs.
    """

    if aspect is None:
        aspect=5
    if height is None:
        height=.5
    sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    pal = sns.cubehelix_palette(len(limits), rot=-.25, light=.7)

    df_melt = pd.melt(normalise(df, limits))
    g = sns.FacetGrid(df_melt, row="name", hue="name", aspect=aspect,
                      height=height, palette=pal, sharex=False)

    def custom_kde(x, shade=False, **kwargs):
        df = pd.concat((x,), axis=1)
        x_vals, pdf = kde_1d(df, w, x.name, xmin=0.0, xmax=1.0, numx=1000)
        pdf = (pdf-pdf.min())/(pdf.max()-pdf.min())
        facecolor = kwargs.pop("facecolor", None)
        ax = plt.gca()
        line, = ax.plot(x_vals, pdf, **kwargs)
        color = line.get_color()
        line.remove()
        kwargs.pop("color", None)
        facecolor = color if facecolor is None else facecolor
        ax.plot(x_vals, pdf, color=color, **kwargs)
        shade_kws = dict(
                facecolor=facecolor,
                alpha=kwargs.get("alpha", 0.25),
                clip_on=kwargs.get("clip_on", True),
                zorder=kwargs.get("zorder", 1)
                )
        if shade:
            ax.fill_between(x_vals, 0, pdf, **shade_kws)
        ax.set_ylim(0, auto=None)
        return ax

    g.map(custom_kde, "value", alpha=1, lw=1, shade=True)
    g.map(custom_kde, "value", color="w", lw=1)
    g.map(plt.axhline, y=0, lw=2, clip_on=False)

    def label(x, color, label):
        ax = plt.gca()
        ax.text(0, .2, label, fontweight="bold", color=color,
                ha="left", va="center", transform=ax.transAxes)
    g.map(label, "name")

    def xlims(x, color, label):
        ax = plt.gca()
        ax.set(xticks=[0, 1])
        ax.set(xticklabels=[limits[label][0], limits[label][1]])
    g.map(xlims, "name")

    # Set subplots to overlap
    g.fig.subplots_adjust(hspace=-.25)

    # Update axes details
    g.set_xlabels("posterior")
    g.set_titles("")
    g.set(yticks=[])
    g.despine(bottom=True, left=True)

    return g

def plot_kde_matrix_custom(df, w, limits=None, refval=None):
    """Wrapper around pyabc.visualization.plot_kde_matrix."""

    arr_ax = plot_kde_matrix(df, w, limits=limits, refval=refval)

    # Remove titles
    n_par = df.shape[1]
    for i in range(0, n_par):
        for j in range(0, i):
            # lower axis is 2D density
            ax = arr_ax[i, j]
            ax.set_title(None)

    plt.set_cmap('viridis')

    return arr_ax


def plot_data_adjustment(modelfiles: List[str],
                         *experiments: Experiment,
                         tvar: str='phys.T') -> sns.FacetGrid:
    """Plots raw data before and after adjustment to model temperature.

    Args:
        modelfiles (List[str]): Models with temperature to adjust.
        experiments (Experiment): Experimental data to plot.
        tvar (str): Name of temperature variable in model file.

    Returns:
        sns.FacetGrid: Plots of experimental data.
    """
    if not isinstance(modelfiles, list):
        modelfiles = [modelfiles]

    tvars = [tvar,]*len(modelfiles)

    temperatures = []
    for i, model in enumerate(modelfiles):
        m = myokit.load_model(model)
        temperatures.append(m.get(tvars[i]).value())

    observations = pd.DataFrame({})
    # get adjusted for each model
    for temp in temperatures:
        observations_temp = get_observations_df(
                list(experiments),
                normalise=False,
                temp_adjust=True,
                model_temperature=temp)
        observations_temp['T'] = str(temp)+'K'
        observations = observations.append(observations_temp)

    # get unadjusted raw data
    observations_unadjusted = get_observations_df(list(experiments),
                                       normalise=False,
                                       temp_adjust=False,
                                       model_temperature=None)
    observations_unadjusted['T'] = 'unadjusted'

    grid = sns.relplot(x='x', y='y',
                       col='exp_id',
                       kind='scatter',
                       hue='T', style='T',
                       data=observations,
                       markers=True,
                       facet_kws={'sharex': 'col',
                                  'sharey': 'col'})

    def error_bars(**kwargs):
        measurements = kwargs.pop('measurements')
        data = kwargs.pop('data')
        # add error bars to adjusted
        for temp in data['T'].unique():
            plt.errorbar(data.loc[data['T']==temp]['x'],
                         data.loc[data['T']==temp]['y'],
                         yerr=np.sqrt(data.loc[data['T']==temp]['variance']),
                         ls='None', marker=None)
        # add unadjusted data and error bars
        for exp_id in data['exp_id'].unique():
            plt.errorbar(measurements.loc[measurements['exp_id']==exp_id]['x'],
                         measurements.loc[measurements['exp_id']==exp_id]['y'],
                         yerr=np.sqrt(measurements.loc[measurements['exp_id']==exp_id]['variance']),
                         ls='None', marker='x', color='k')

    grid = grid.map_dataframe(error_bars, measurements=observations_unadjusted)

    return grid
