### Classes to store experimental and simulation output data.

import numpy as np
import myokit
import pandas as pd


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

        if self.errs is None:
            df_errs = [1.]*len(x)
        else:
            df_errs = self.errs
        self.df = pd.DataFrame({'x': x, 'y': y, 'errs': df_errs})


class ExperimentStimProtocol(object):
    """Stimulation times and measurement points for simulations."""

    def __init__(self, stim_times, stim_levels, measure_index=None, 
                 measure_fn=None, post_fn=None, time_independent=False, 
                 ind_var=None):
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
            ind_var (list): Custom independent variable (x-axis) can be
                passed explicitly.

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
        # Defaults to one run TODO: not sure this is the best default
        if self.n_runs is None: 
            self.n_runs = 1

        if ind_var is not None:
            self.ind_var = ind_var


    def __call__(self, sim, vvar, logvars, n_x=None):
        """Runs the protocol in Myokit using the passed simulation model.

        Args:
            sim (Simulation): Myokit simulation object.
            vvar (str): Name of voltage variable in Simulation.
            logvars (list(str)): Name of variables to log when running.
            n_x (int): Override defined x resolution.
            ind_var (list): Custom independent variable (x-axis) can be
                passed explicitly.

        Returns:
            Independent (changing) variable and results
            from measured experiment values.
        """

        # Setup if x resolution is being overridden.
        if n_x is not None:
            n_runs = None
            times = []
            levels = []
            for time, level in zip(self.stim_times,
                                   self.stim_levels):
                if isinstance(time, list):
                    times.append(
                         [float(min(time)) +
                          x*(float(max(time))-float(min(time))) /
                          (n_x-1) for x in range(n_x)]
                        )
                    n_runs = len(times[-1])
                    ind_var = times[-1]
                else:
                    times.append(time)

                if isinstance(level, list):
                    levels.append(
                         [float(min(level)) +
                          x*(float(max(level))-float(min(level))) / (n_x-1)
                          for x in range(n_x)]
                        )
                    n_runs = len(levels[-1])
                    ind_var = levels[-1]
                else:
                    levels.append(level)
            if n_runs is None:
                n_runs = 1
                ind_var = [min(self.ind_var) +
                           x*(max(self.ind_var)-min(self.ind_var)) / (n_x-1)
                           for x in range(n_x)]
        else:
            n_runs = self.n_runs
            times = self.stim_times
            levels = self.stim_levels
            ind_var = self.ind_var

        # If no measure_index specified, record all
        if self.measure_index is None:
            measure_index = range(len(zip(times, levels)))
        else:
            measure_index = self.measure_index

        # Run simulations
        full_results = []
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

                # Store data values if it is a measurment region of interest
                # or if we are simply spitting out the raw sim output.
                if i in measure_index:
                    # Query values if simulation does not depend
                    # on time (and thus running would error).
                    if self.time_independent:
                        d = myokit.DataLog()
                        for logi in logvars:
                            d[logi] = sim._model.get(logi).value()
                        data.append(d)
                    # Otherwise run simulation and store output.
                    else:
                        data.append(sim.run(t, log=logvars))
                else:
                    if not self.time_independent:
                        d = sim.run(t)

            if self.measure_fn is not None:
                data = self.measure_fn(data)
            else:
                # Combine by default
                d0 = data[0]
                for d in data[1:]:
                    d0 = d0.extend(d)
                data = d0
                data['run'] = run
            full_results.append(data)

        # Apply any post-processing function
        if self.post_fn is not None:
            full_results = self.post_fn(full_results, ind_var)
        return pd.DataFrame({'x': ind_var, 'y': full_results})
        #out = pd.DataFrame({})
        #if len(full_results) == 1:
        #    for stage in full_results:
        #        out = out.append(pd.DataFrame(stage), ignore_index=True)
        #else:
        #    out = out.append(pd.DataFrame(full_results), ignore_index=True)

class Experiment(object):
    """Organises protocol and data related to a single experiment instance."""

    def __init__(self, protocol, data, conditions):
        """Initialisation.

        Args:
            protocol (ExperimentStimProtocol): Time and voltage steps for
                experiment to replicate experimental results.
            data (ExperimentData): Experimental data to compare with.
            conditions (dict[str: float]): Experimental conditions, usually
                ion concentrations and temperature.
        """
        self.protocol = protocol
        self.data = data
        self.conditions = conditions

    def run(self, sim, vvar, logvars, n_x=None):
        """Wrapper to run simulation."""
        for c_name, c_val in self.conditions.items():
            try:
                sim.set_constant('membrane.'+c_name, c_val)
            except:
                print("Could not set condition " + c_name
                      + " to value: " + str(c_val))
        return self.protocol(sim, vvar, logvars, n_x)
#
#    def eval_err(self, error_fn, sim=None, vvar=None, logvars=None):
#        """Evaluate difference between experimental and simulation output.
#
#        Args:
#            error_fn (Callable): Error function to use.
#            sim (Simulation): Simulation object.
#            vvar (string): Name of voltage variable in myokit model.
#            logvars (List[string]): List of variables in model to log.
#
#        Returns:
#            Loss value as float.
#        """
#        res = self.run(sim, vvar, logvars)
#        # Results y values will be None if simulation failed.
#        if res[1] is None:
#            return float("inf")
#        return error_fn(res[1], self.data)
