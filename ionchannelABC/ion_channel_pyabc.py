from pyabc.distance import PNormDistance
from pyabc.acceptor import SimpleFunctionAcceptor, accept_use_complete_history
from pyabc.model import Model
from pyabc.transition.multivariatenormal import MultivariateNormalTransition

from .experiment import Experiment
import myokit

import numpy as np
import warnings
import pandas as pd
import subprocess
import io
from typing import List, Dict, Union
import logging
import signal # for function timeouts
from contextlib import contextmanager # for function timeouts
import weakref # for cleaning up model

abclogger = logging.getLogger('ABC')


class TimeoutException(Exception): pass


@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

### Deprecated
class IonChannelModel(Model):
    """Pyabc model to run Myokit simulations for ion channels.

    Parameters:
        channel (str): Shortcode reference name of channel to simulate.
        modelfile (str): Path to Myokit mmt file defining model formulation.
        vvar (str): Name of voltage variable in Myokit model file.
        logvars (List[str]): Variables logged during simulation runs.
        external_par_samples (List[Dict[str, float]]): List of dictionaries of 
            parameters to randomly choose external to the ABC fitting process. 
            These are set randomly prior to running each simulation.
    """

    def __init__(self,
                 channel: str,
                 modelfile: str,
                 vvar: str,
                 logvars: List[str]=myokit.LOG_ALL,
                 external_par_samples: List[Dict[str, float]]=None):

        warnings.warn(
           "The IonChannelModel class is deprecated and will be removed."
           ) 
        self.channel = channel
        self.external_par_samples = external_par_samples
        self.modelfile = modelfile

        self.vvar = vvar
        self.logvars = logvars

        self._sim = None
        self.experiments = []

        # Attempt to force cleanup during parallel subprocesses
        self._finalizer = weakref.finalize(self, self._remove_simulation)

        super().__init__(name = channel+"_model")

    def sample(self,
               pars: Dict[str, float],
               n_x: int=None,
               exp_num: int=None,
               logvars: List[str]=None) -> pd.DataFrame:
        """Sample from the ion channel model.

        Args:
            pars (Dict[str, float]): Dictionary mapping the name of 
                individual parameters to their values for this sample.
            n_x (int): Override the default independent variable for
                simulations. Defaults to None.
            exp_num (int): Only run a specific experiment. Default to
                run all.
            logvars (List[str]): Variables to log during simulation.
        
        Returns:
            pd.DataFrame: Output from experiment measurements.
        """
        if self.external_par_samples is not None:
            full_pars = dict(np.random.choice(self.external_par_samples),
                             **pars)
        else:
            full_pars = pars

        # Run myokit simulation and return empty dataframe if failure.
        try:
            results = self._simulate(n_x, exp_num, logvars, **full_pars)
        except:
            results = pd.DataFrame({})
        return results

    def add_external_par_samples(self,
                                 samples: List[Dict[str, float]]=None):
        """Add defined sample values for removing stochasticity.

        Useful for testing purposes or for sequentially fitting
        larger models.

        Args:
            samples (List[Dict[str, float]]): Each element in the
                list is a sample which could be randomly chosen in
                the simulation. Passing a simple value will ensure it
                is always chosen.
        """
        self.external_par_samples = samples

    def _simulate(self,
                  n_x: int=None,
                  exp_num: int=None,
                  logvars: List[str]=None,
                  **pars: float) -> pd.DataFrame:
        """Simulate model in Myokit and measure using defined experiments.

        Args:
            n_x (int): Override resolution of independent variable. If None will
                use the default resolution defined by the experiment.
            exp_num (int): Number of specific experiment to run on its own.
            logvars (List[str]): List of variables to log from simulation.
            pars (float): Values of named parameters passed as kwargs.

        Returns:
            pd.DataFrame: Summary measurements from simulated output.

        Raises:
            RuntimeError: Myokit fails to run simulation.
            ValueError: `exp_num` is outside of the range of experiments.
        """
        if logvars is None:
            logvars = self.logvars

        # Sanity check for experiment number
        if (exp_num is not None and
                (exp_num < 0 or exp_num > len(self.experiments)-1)):
            raise ValueError("Experiment number is outside range.")

        # Get original parameter values to reset later
        original_vals = self.get_parameter_vals(list(pars.keys()))

        self.set_parameters(**pars)
        all_results = pd.DataFrame({})

        for i, e in enumerate(self.experiments):
            if exp_num is not None and i is not exp_num:
                continue
            try:
                with time_limit(5):
                    results = e.run(sim=self._sim,
                                    vvar=self.vvar,
                                    logvars=logvars,
                                    n_x=n_x)
            except:
                self.set_parameters(**original_vals)
                raise RuntimeError("Failed simulation.")

            results['exp'] = i

            if i is 0:
                all_results = results
            else:
                all_results = all_results.append(results)

        # Reset to original values
        self.set_parameters(**original_vals)

        return all_results

    def set_parameters(self, **pars: float):
        """Set parameters of the Myokit model.

        Args:
            pars (float): Values of parameter values passed as kwargs.

        Raises:
            ValueError: Fails to set a parameter value.
        """

        # Make sure a myokit simulation exists
        if self._sim is None:
            self._build_simulation()
        else:
            self._sim.reset()

        # Set parameters
        for name, value in pars.items():
            try:
                if value is not None:
                    if '.' not in name:
                        self._sim.set_constant(
                                self.channel+'.'+name, value)
                    else:
                        self._sim.set_constant(name, value)
            except:
                raise ValueError('Could not set parameter {0} to {1}'
                                 .format(name, value))

    def get_parameter_vals(self, parameters: List[str]) -> Dict[str, float]:
        """Get the value of current model parameters.

        Args:
            parameters (List[str]): Names of parameter values to retrieve.
        """
        m, _, _ = myokit.load(self.modelfile)
        parameter_values = {}
        for p in parameters:
            if m.has_variable(p):
                parameter_values[p] = m.get(p).value()
            else:
                parameter_values[p] = np.nan
        return parameter_values

    def _build_simulation(self):
        """Creates a class instance of myokit.Model and myokit.Simulation.

        Raises:
            ValueError: Cannot find voltage parameter in model.
        """
        m, _, _ = myokit.load(self.modelfile)
        try:
            v = m.get(self.vvar)
        except:
            raise ValueError('Model does not have vvar: {}'
                             .format(self.vvar))
        if v.is_state():
            v.demote()
        v.set_rhs(0)
        v.set_binding(None)
        self._sim = myokit.Simulation(m)
    
    def _remove_simulation(self):
        """Removes reference to simulation object.
        
        This is part attempt for the object to be cleanup up
        in parallel ABC runs.
        """
        self._sim = None

    def add_experiments(self,
                        experiments: Union[Experiment, List[Experiment]]):
        """Add Experiment measurement for model output.

        Args:
            experiments (Union[Experiment, List[Experiment]]): Either a
                single or multiple instances of ionchannelABC.Experiment.
        """
        self.experiments.extend(experiments)

    def get_experiment_data(self) -> pd.DataFrame:
        """Returns experimental data from Experiments.

        Returns:
            pd.DataFrame: Experimental data in columns exp, x, y, err.
        """
        measurements = pd.DataFrame(columns=['exp', 'x', 'y', 'errs'])
        for i, exp in enumerate(self.experiments):
            data = exp.data.df
            data['exp'] = i
            measurements = measurements.append(data, ignore_index=True,
                                               sort=True)
        return measurements
