### Class to build channel for ABCSolver.

import os
import myokit
import distributions as dist
import matplotlib.pyplot as plt
if os.environ.get('DISPLAY') is None:
    plt.switch_backend('agg')
import numpy as np
import dill as pickle
import plotting_helpers as ph
import logging


class Channel(object):
    def __init__(self, modelfile, abc_params, vvar='membrane.V',
                 logvars=myokit.LOG_ALL):
        """Initialisation.

        Args:
            modelfile (str): Path to model for myokit to load.
            abc_params (Dict[str, Tuple[float]]): Dict mapping list of
                parameter name string to upper and lower limit of prior
                for ABC algorithm.
            vvar (str): String name of dependent variable in simulations,
                usually the voltage.
            logvars (Union[str, List[str]]): Model variables to log during
                simulation runs.
        """
        self.modelfile = modelfile
        self.vvar = vvar
        self.logvars = logvars

        self.param_names = []
        self.param_priors = []
        self.param_ranges = []
        self.kernel = dist.Normal(0.0, 1.0)
        for param, val in abc_params.iteritems():
            self.param_names.append(param)
            self.param_priors.append((val[0], val[1]))
            self.param_ranges.append(val[1] - val[0])

        self.experiments = []
        self._sim = None

        self.abc_plotting_results = None

    def _generate_sim(self):
        """Creates class instance of Model and Simulation."""
        m, _, _ = myokit.load(self.modelfile)
        try:
            v = m.get(self.vvar)
        except:
            print('Model does not have vvar: ' + self.vvar)

        if v.is_state():
            v.demote()
        v.set_rhs(0)
        v.set_binding(None)

        # Check model has all parameters listed.
        for param_name in self.param_names:
            assert m.has_variable(param_name), (
                    'The parameter ' + param_name + ' does not exist.')

        self._sim = myokit.Simulation(m)

    def get_original_param_vals(self):
        """Return original values of parameters for ABC."""
        m, _, _ = myokit.load(self.modelfile)
        original_vals = []
        for param in self.param_names:
            try:
                original_vals.append(m.get(param).value())
            except:
                raise ValueError("Could not access parameter: " + param)
        return original_vals

    def run_experiments(self, step_override=-1):
        """Run channel with defined experiments.

        Args:
            step_override (int): Steps between min and max of experiment,
                defaults to -1 which uses default steps from ExperimentData.

        Returns:
            Simulation output data points.
        """
        assert len(self.experiments) > 0, 'Need to add at least one experiment!'
        if self._sim is None:
            self._generate_sim()
        sim_results = []
        for exp in self.experiments:
            sim_results.append(exp.run(self._sim, self.vvar, self.logvars,
                                       step_override))
        return sim_results

    def eval_error(self, error_fn):
        """Evaluates error with experiment data over all experiments.

        Args:
            error_fn (Callable): Function to calculate error that accepts
                first argument as simulation results and second argument
                as ExperimentData object.

        Returns:
            Loss value as float.
        """
        if self._sim is None:
            self._generate_sim()
        errs = []
        for exp in self.experiments:
            errs.append(exp.eval_err(error_fn, self._sim, self.vvar,
                                     self.logvars))
        return errs

    def set_abc_params(self, new_params):
        """Set model ABC parameters to new values.

        Args:
            new_params (List[float]): List of new parameters corresponding
                to `param_names` attribute order.
        """
        if self._sim is None:
            self._generate_sim()
        else:
            self._sim.reset()
        for param_name, new_val in zip(self.param_names, new_params):
            try:
                self._sim.set_constant(param_name, new_val)
            except:
                print("Could not set parameter " + param_name
                      + " to value: " + str(new_val))

    def add_experiment(self, experiment):
        """Adds experiment to channel for ABC algorithm.

        Args:
            experiment (Experiment): Experiment class containing data and
                definition of stimulus protocol.
        """
        self.experiments.append(experiment)

    def reset(self):
        """Erases previously created simulations and experiment logs."""
        self._sim = None

    def plot_results(self, abc_distr, step=-1):
        """Plot results from ABC solver.

        Args:
            abc_distr (dist.Arbitrary): Posterior distribution estimate from
                ABC algorithm.

        Returns:
            Handle to figure with summary plots for each experiment.
        """
        pool = abc_distr.pool
        weights = abc_distr.weights
        self.reset()

        if self.abc_plotting_results is None:
            # Updated values for each ABC posterior particle.
            results_abc = []
            for i, params in enumerate(pool):
                self.set_abc_params(params)
                results_abc.append(self.run_experiments(step))
            results_abc = np.array(results_abc).swapaxes(0, 1)
            self.abc_plotting_results = results_abc
        else:
            results_abc = self.abc_plotting_results

        results_abc_mean = []
        results_abc_sd = []
        for exper in results_abc:
            d = dist.Arbitrary(exper[:, 1], weights)
            results_abc_mean.append(d.getmean())
            results_abc_sd.append(np.sqrt(d.getvar()))

        # Generate plots.
        plt.style.use('seaborn-colorblind')
        ncols = len(results_abc)
        x = [results_abc[i][0][0] for i in range(ncols)]
        fig, ax = plt.subplots(nrows=1, ncols=ncols, figsize=(3*ncols, 2.8))
        for i in range(ncols):
            if ncols > 1:
                axi = ax[i]
            else:
                axi = ax
            axi.plot(x[i], results_abc_mean[i], '-', label='ABC posterior')
            axi.fill_between(x[i], results_abc_mean[i]-results_abc_sd[i],
                             results_abc_mean[i]+results_abc_sd[i], alpha=0.25,
                             lw=0)
            axi.errorbar(x=self.experiments[i].data.x,
                         y=self.experiments[i].data.y,
                         yerr=self.experiments[i].data.errs,
                         fmt='o')

            if i == ncols-1:
                handles, labels = axi.get_legend_handles_labels()
                lgd = fig.legend(handles, labels, loc='lower center', ncol=3)
                bb = lgd.get_bbox_to_anchor().inverse_transformed(
                        fig.transFigure)
                bb.y0 -= 0.1
                lgd.set_bbox_to_anchor(bb, transform=fig.transFigure)

        # Mark letters on subfigures.
        if ncols > 1:
            ax = ph.add_subfig_letters(ax)
        plt.tight_layout()
        return fig

    def plot_final_params(self, abc_distr):
        """Plot posterior distributions for each parameter.

        Args:
            abc_distr (dist.Arbitrary): Posterior distribution estimate from
                ABC algorithm.

        Returns:
            Figure handle with violin plots for each parameter posterior.
        """
        pool = abc_distr.pool
        wts = abc_distr.weights
        pool = np.asarray(pool)
        pool = pool.swapaxes(0, 1)

        # Scale between min and max of prior.
        pool_norm = np.empty_like(pool)
        for i, param in enumerate(pool):
            pool_norm[i] = ((param - self.param_priors[i][0])
                            / self.param_ranges[i])
        pool_norm = pool_norm.swapaxes(0, 1)

        vpstats = ph.custom_violin_stats(pool_norm, wts)
        pos = range(pool_norm.shape[1])
        fig, ax = plt.subplots()
        vplot = ax.violin(vpstats, pos, vert=False,
                          showmeans=True, showextrema=True, showmedians=True)
        vplot['cmeans'].set_color('r')
        ax.set_yticks(pos)
        ax.set_yticklabels(self.param_names)
        return fig

    def save(self, filename):
        """Save channel and underlying experiments to disk."""
        self._sim = None
        pickle.dump(self, open(filename, 'wb'))
