### Classes to store experimental and simulation output data.

import numpy as np


class ExperimentData():
    '''Organises experimental data extracted from publications.'''

    def __init__(self, x, y, N=None, errs=None, err_type=None):
        '''Initialisation.

        Args:
            x (List[float]): X-axis data from extracted source.
            y (List[float]): Y-axis data from extracted source.
            N (int): N number (if reported).
            errs (List[float]): Corresponding error bars (if reported).
            err_type (string): One of `SEM` or `STD` for error type.

        Raises:
            ValueError: When err_type is not a valid string.
        '''
        self.x = x
        self.y = y
        self.N = N
        if errs is not None:
            self.errs = np.abs(np.asarray(errs) - np.asarray(y))
        else:
            self.errs = errs
        self.err_type = err_type

        if self.err_type not in (None, 'SEM', 'STD'):
            raise ValueError('`err_type` must be either `SEM` or `STD`\n'
                             'Passed value: ' + err_type)


class ExperimentStimProtocol():
    '''Stimulation times and measurement points for simulations.'''

    def __init__(self, stim_times, stim_levels, measure_index, measure_fn):
        '''Initialisation.

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

        Raises:
            ValueError: When `stim_times` and `stim_levels` are not of equal
                length.
        '''
        if len(stim_times) != len(stim_levels):
            raise ValueError('`stim_times` and `stim_levels` must be same size')

        self.n_runs = None
        for time in stim_times:
            if isinstance(stim, list):
                if self.n_runs is None:
                    # Take first nested list as number of runs.
                    self.n_runs = len(stim)
                else:
                    assert len(stim) == self.n_runs, (
                            'Inconsistent number of experiment runs.')
        self.stim_times = stim_times
        for level in stim_levels:
            if isinstance(level, list):
                assert len(level) == self.n_runs, (
                        'Inconsistent number of experiment runs.')
        self.stim_levels = stim_levels
        if isinstance(measure_index, int):
            self.measure_index = (measure_index,)
        else
            self.measure_index = measure_index
        self.measure_fn = measure_fn

    def __call__(self, sim, vvar, logvars):
        '''Runs the protocol in Myokit using the passed simulation model.

        Args:
            sim (Simulation): Myokit simulation object.
            vvar (str): Name of voltage variable in Simulation.
            logvars (str): Name of variables to log when running.

        Returns:
            Results from measured experiment values.
        '''
        res_sim = []
        for run in n_runs:
            data = []
            sim.reset()
            sim.set_constant(self.vvar, self.stim_levels[0])
            sim.run(self.stim_times[0], log=None)
            for i, t in enumerate(self.stim_times[1:]):
                sim.set_constant(self.vvar, self.stim_levels[i+1])
                if (i+1) in self.measure_index:
                    data.append(sim.run(t, log=logvars))
            result = self.measure_fn(data)
            res_sim.append(result)
        return res_sim


class Experiment():
    '''Organises protocol and data related to a single experiment instance.'''

    def __init__(self, name):
        self.name = name
