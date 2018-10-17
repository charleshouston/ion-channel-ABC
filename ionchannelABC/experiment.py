import numpy as np
import myokit
import pandas as pd
from typing import List, Union, Tuple, Callable, Dict


class ExperimentData(object):
    """
    Organises raw digitised data extracted from publications.

    Parameters
    ----------

    x: List[float]
        X-axis data from extracted source.

    y: List[float]
        Y-axis data from extracted source.

    N: int
        Number of experimental repetitions.

    errs: List[float]
        Error bars points. NOT VALUE OF ERROR BARS RELATIVE TO Y POINTS - THIS
        IS CALCULATED AT INITIALISATION.

    err_type: str
        Type of error given by error bars, e.g. `std` or `sem`.
    """

    def __init__(self,
                 x: List[float],
                 y: List[float],
                 N: int=None,
                 errs: List[float]=None,
                 err_type: str=None):

        if errs is not None:
            errs = np.abs(np.asarray(errs) - np.asarray(y))
        else:
            errs = [np.nan]*len(x)
        self.df = pd.DataFrame({'x': x, 'y': y, 'errs': errs})
        self.N = N
        self.err_type=err_type


class ExperimentStimProtocol(object):
    """
    Stimulation times and measurement points for simulations.

    Parameters
    ----------

    stim_times: List[Union[float, List[float]]]
        List of stimulation protocol time points from sim start. A nested
        list indicates that the experiment is repeated for different
        time intervals.

    stim_levels: List[Union[float, List[float]]]
        List of stimulation levels corresponding to `stim_times`. A nested
        list indicates that the experiment is repeated for different
        stimulation levels.

    measure_index: int
        Indices of stim_times of interest which data will be passed to
        measure function.

    measure_fn: Callable
        Function that accepts Datalogs at times of measure index and
        computes the experiment output from raw simulation data.

    post_fn: Callable
        Function that accepts results after all simulations have run
        for any additional processing (e.g. normalising). Inputs to function
        are list of results for each measure index and the independent variable.
        Outputs are the processed data and a boolean flag for whether the keys
        of the results should be used to fill up the results dataframe.

    time_independent: bool
        Whether the simulation needs to be solved or variables can be
        extracted directly from the model at each time point (i.e. no
        differential equations to solve).

    ind_var: List[float]
        Custom independent variable (x-axis) can be passed explicitly.
    """

    def __init__(self,
                 stim_times: List[Union[float, List[float]]],
                 stim_levels: List[Union[float, List[float]]],
                 measure_index: Union[int, Tuple[int]]=None,
                 measure_fn: Callable=None,
                 post_fn: Callable=None,
                 time_independent: bool=False,
                 ind_var: List[float]=None):

        if len(stim_times) != len(stim_levels):
            raise ValueError('`stim_times` and `stim_levels` must',
                             'be same size')

        self.n_runs = None
        self.ind_var = None
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
        # TODO: check this is an appropriate default
        if self.n_runs is None: 
            self.n_runs = 1

        if self.ind_var is None:
            self.ind_var = ind_var

    def __call__(self,
                 sim: myokit.Simulation,
                 vvar: str,
                 logvars: List[str],
                 n_x: int=None) -> pd.DataFrame:
        """
        Runs the protocol in Myokit using the passed simulation model.

        Parameters
        ----------
        sim: myokit.Simulation
            Myokit simulation object to run.

        vvar: str
            Name of voltage variable in simulation.

        logvars: List[str]
            Name of variables to log in simulation.

        n_x: int
            Override for defined x resolution.

        Returns
        -------
        Pandas dataframe containing changing variable and results.
        """
        # Setup if x resolution is being overridden.
        if n_x is not None:
            (n_runs, times, levels, ind_var) = self._calculate_custom_res(n_x)
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
                        # Record everything is unspecified
                        if logvars is None or logvars is myokit.LOG_ALL:
                            logvars = sim._model.variables()
                        for logi in logvars:
                            d[logi] = sim._model.get(logi).value()
                        data.append(d)
                    # Otherwise run simulation and store output.
                    else:
                        data.append(sim.run(t, log=logvars))
                else:
                    # Run unless doing so would error
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
        output = pd.DataFrame({})
        use_result_keys = False
        if self.post_fn is not None:
            full_results, use_result_keys = self.post_fn(full_results, ind_var)
        if not use_result_keys:
            output = output.append(pd.DataFrame({'x': ind_var, 'y': full_results}),
                                   ignore_index=True)
        else:
            output = output.append(pd.DataFrame({'x': list(full_results.keys()),
                                                 'y': list(full_results.values())}),
                                   ignore_index=True)
        return output

    def _calculate_custom_res(self,
                              n_x: int) -> Tuple[int,
                                                 List[Union[float,
                                                      List[float]]],
                                                 List[Union[float,
                                                      List[float]]],
                                                 List[float]]:
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
                               x*(float(max(level))-float(min(level))) /
                               (n_x-1) for x in range(n_x)]
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

        return (n_runs, times, levels, ind_var)


class Experiment(object):
    """
    Protocol and observed data for a single experiment.

    Parameters
    ----------

    protocol: ExperimentStimProtocol
        Defined protocol to run simulation.

    data: ExperimentData
        Observed data to compare to.

    conditions: Dict[str, float]
        Experimental conditions, usually ion concentrations and
        temperature.
    """

    def __init__(self,
                 protocol: ExperimentStimProtocol,
                 data: ExperimentData,
                 conditions: Dict[str, float]):
        self.protocol = protocol
        self.data = data
        self.conditions = conditions

    def run(self,
            sim: myokit.Simulation,
            vvar: str,
            logvars: List[str],
            n_x: int=None):
        for c_name, c_val in self.conditions.items():
            try:
                sim.set_constant('membrane.'+c_name, c_val)
            except:
                raise ValueError('Could not set condition {1} to {2}'
                                 .format(c_name, c_val))
        return self.protocol(sim, vvar, logvars, n_x)
