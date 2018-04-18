### Classes to store experimental and simulation output data.

import numpy as np
import myokit


class ExperimentData(object):
    """Organises raw digitised data extracted from publications."""

    def __init__(self, x, y, N=None, errs=None, err_type=None):
        """Initialisation.

        Args:
            x (List[float]): X-axis data from extracted source.
            y (List[float]): Y-axis data from extracted source.
            N (int): N number (if reported).
            errs (List[float]): Corresponding error bars (if reported).
            err_type (string): One of `SEM` or `STD` for error type.

        Raises:
            ValueError: When err_type is not a valid string.
        """
        self.x = x
        self.y = y
        self.N = N
        if errs is not None:
            self.errs = np.abs(np.asarray(errs) - np.asarray(y))
        else:
            self.errs = None
        self.err_type = err_type

        if self.err_type not in (None, 'SEM', 'STD'):
            raise ValueError('`err_type` must be either `SEM` or `STD`\n'
                             'Passed value: ' + err_type)


class ExperimentStimProtocol(object):
    """Stimulation times and measurement points for simulations."""

    def __init__(self, stim_times, stim_levels, measure_index, measure_fn,
                 post_fn=None, time_independent=False):
        """Initialisation.

        `stim_times` and `stim_levels` may contain nested lists for multiple
        experimental runs. These nested lists must be the same size.

        Args:
            stim_times (List[Union[float, List[float]]): List of stimulation
                times from sim start.
            stim_levels (List[Union[float, List[float]]): List of stimulation
                levels corresponding to `stim_times`.
            measure_index (Union[int, Tuple[int]]): Indices of stim_times of
                interest for measurement function.
            measure_fn (Callable): Function that accepts datalogs at times
                of measure index and computes summary statistic for experiment.
            post_fn (Callable): Function that accepts results after all
                simulations have run and carries out any post processing.
            time_independent (bool): Whether the simulation needs to run or
                variables can be extracted from the model.

        Raises:
            ValueError: When `stim_times` and `stim_levels` are not of equal
                length.
        """
        if len(stim_times) != len(stim_levels):
            raise ValueError('`stim_times` and `stim_levels` must be same size')

        self.n_runs = None
        for time, level in zip(stim_times, stim_levels):
            if isinstance(time, list):
                if self.n_runs is None:
                    # Take first nested list as number of runs.
                    self.n_runs = len(time)
                    self.ind_var = time
                else:
                    assert len(time) == self.n_runs, (
                            'Inconsistent number of experiment runs.')
            if isinstance(level, list):
                if self.n_runs is None:
                    self.n_runs = len(level)
                    self.ind_var = level
                else:
                    assert len(level) == self.n_runs, (
                            'Inconsistent number of experiment runs.')
        self.stim_times = stim_times
        self.stim_levels = stim_levels

        if isinstance(measure_index, int):
            self.measure_index = (measure_index,)
        else:
            self.measure_index = measure_index
        self.measure_fn = measure_fn
        self.post_fn = post_fn
        self.time_independent = time_independent

    def __call__(self, sim, vvar, logvars, step_override=-1):
        """Runs the protocol in Myokit using the passed simulation model.

        Args:
            sim (Simulation): Myokit simulation object.
            vvar (str): Name of voltage variable in Simulation.
            logvars (str): Name of variables to log when running.
            step_override (int): Override defined stimulation step, default
                to -1 for no override.

        Returns:
            Independent (changing) variable and results
            from measured experiment values.
        """
        if step_override != -1:
            times = []
            levels = []
            for time, level in zip(self.stim_times,
                                   self.stim_levels):
                if isinstance(time, list):
                    times.append(range(min(time), max(time) + step_override,
                                       step_override))
                    n_runs = len(times[-1])
                    ind_var = times[-1]
                else:
                    times.append(time)

                if isinstance(level, list):
                    levels.append(range(min(level), max(level) + step_override,
                                        step_override))
                    n_runs = len(levels[-1])
                    ind_var = levels[-1]
                else:
                    levels.append(level)
        else:
            n_runs = self.n_runs
            times = self.stim_times
            levels = self.stim_levels
            ind_var = self.ind_var

        res_sim = []
        try:
            for run in range(n_runs):
                data = []
                sim.reset()
                for i, (time, level) in enumerate(zip(times, levels)):
                    if isinstance(time, list):
                        t = time[run]
                    else:
                        t = time
                    if isinstance(level, list):
                        l = level[run]
                    else:
                        l = level
                    sim.set_constant(vvar, l)
                    if i in self.measure_index:
                        if self.time_independent:
                            d = myokit.DataLog()
                            for logi in logvars:
                                d[logi] = sim._model.get(logi).value()
                            data.append(d)
                        else:
                            data.append(sim.run(t, log=logvars))
                    else:
                        if not self.time_independent:
                            d = sim.run(t)
                result = self.measure_fn(data)
                res_sim.append(result)
            if self.post_fn is not None:
                res_sim = self.post_fn(res_sim)
        except:
            res_sim = None
        return ind_var, res_sim


class Experiment(object):
    """Organises protocol and data related to a single experiment instance."""

    def __init__(self, protocol, data):
        """Initialisation.

        Args:
            protocol (ExperimentStimProtocol): Time and voltage steps for
                experiment to replicate experimental results.
            data (ExperimentData): Experimental data to compare with.
        """
        self.protocol = protocol
        self.data = data

    def run(self, sim, vvar, logvars, step_override=-1):
        """Wrapper to run simulation."""
        return self.protocol(sim, vvar, logvars, step_override)

    def eval_err(self, error_fn, sim=None, vvar=None, logvars=None):
        """Evaluate difference between experimental and simulation output.

        Args:
            error_fn (Callable): Error function to use.
            sim (Simulation): Simulation object.
            vvar (string): Name of voltage variable in myokit model.
            logvars (List[string]): List of variables in model to log.

        Returns:
            Loss value as float.
        """
        res = self.run(sim, vvar, logvars)
        # Results y values will be None if simulation failed.
        if res[1] is None:
            return float("inf")
        return error_fn(res[1], self.data)
