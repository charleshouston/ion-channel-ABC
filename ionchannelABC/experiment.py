from functools import wraps, reduce
import numpy as np
import myokit
import pandas as pd
from typing import List, Callable, Dict, Union, Tuple
import warnings


def log_transform(f):
    @wraps(f)
    def log_transformed(**log_kwargs):
        kwargs = dict([(key[4:], 10**value) if key.startswith("log")
                       else (key, value)
                       for key, value in log_kwargs.items()])
        return f(**kwargs)
    return log_transformed


def combine_sum_stats(*functions):
    def sum_stats_fn(x):
        sum_stats = []
        for i, flist in enumerate(functions):
            for f in flist:
                sum_stats = sum_stats+f(x[i])
        return sum_stats
    return lambda x: sum_stats_fn(x)


class Experiment:
    """Contains related information from patch clamp experiment."""
    def __init__(self,
                 dataset: Union[np.ndarray, List[np.ndarray]],
                 protocol: myokit.Protocol,
                 conditions: Dict[str, float],
                 sum_stats: Union[Callable, List[Callable]],
                 description: str=""):
        """Initialisation.

        Args:
            dataset (Union[np.ndarray, List[np.ndarray]]):
                Experimental data in format (x, y, variance). More than one
                dataset can be supplied in a list and will be assigned
                separate exp_id. Care must be taken in this case that sum stats
                function produces appropriate output.
            protocol (myokit.Protocol): Voltage step protocol from experiment.
            conditions (Dict[str, float]): Reported experimental conditions.
            sum_stats (Union[Callable, List[Callable]]): Summary statistics
                function(s) which may be list of functions as more than one
                measurement could be made from one protocol.
            description (str): Optional descriptor.
        """
        if isinstance(dataset, list):
            self._dataset = dataset
        else:
            self._dataset = [dataset]
        self._protocol = protocol
        self._conditions = conditions
        if isinstance(sum_stats, list):
            self._sum_stats = sum_stats
        else:
            self._sum_stats = [sum_stats]
        self._description = description

    def __call__(self) -> None:
        """Print descriptor"""
        print(self._description)

    @property
    def dataset(self) -> np.ndarray:
        return self._dataset

    @property
    def protocol(self) -> myokit.Protocol:
        return self._protocol

    @property
    def conditions(self) -> Dict:
        return self._conditions

    @property
    def sum_stats(self) -> Callable:
        return self._sum_stats


def setup(modelfile: str,
          *experiments: Experiment,
          vvar: str='membrane.V',
          logvars: List[str]=myokit.LOG_ALL
          ) -> Tuple[pd.DataFrame, Callable, Callable]:
    """Combine chosen experiments into inputs for ABC.
    
    Args:
        modelfile (str): Path to Myokit MMT file.
        *experiments (Experiment): Any number of experiments to run in ABC.
        vvar (str): Optionally specify name of membrane voltage in modelfile.
        logvars (str): Optionally specify variables to log in simulations.

    Returns:
        Tuple[pd.DataFrame, Callable, Callable]:
            Observations combined from experiments.
            Model function to run combined protocols from experiments.
            Summary statistics function to convert 'raw' simulation output.
    """

    # Create Myokit model instance
    m = myokit.load_model(modelfile)
    v = m.get(vvar)
    v.demote()
    v.set_rhs(0)
    v.set_binding('pace')

    # Initialise combined variables
    cols = ['x', 'y', 'variance', 'exp_id']
    observations = pd.DataFrame(columns=cols)
    simulations, times = [], []

    cnt = 0
    for exp in list(experiments):
        # Combine datasets
        for d in exp.dataset:
            dataset = d.T.tolist()
            dataset = [d_+[str(cnt),] for d_ in dataset]
            observations = observations.append(
                pd.DataFrame(dataset, columns=cols),
                ignore_index=True
            )
            cnt += 1

        # Combine protocols into Myokit simulations
        s = myokit.Simulation(m, exp.protocol)
        for ci, vi in exp.conditions.items():
            s.set_constant(ci, vi)
        simulations.append(s)
        times.append(exp.protocol.characteristic_time())
        
    # Create model function
    def simulate_model(**pars):
        sim_output = []
        for sim, time in zip(simulations, times):
            for p, v in pars.items():
                try:
                    sim.set_constant(p, v)
                except:
                    warnings.warn("Could not set value of {}"
                                  .format(p))
                    return None
            sim.reset()
            try:
                sim_output.append(sim.run(time, log=logvars))
            except:
                del(sim_output)
                return None
        return sim_output
    def model(x):
        return log_transform(simulate_model)(**x)

    # Combine summary statistic functions
    sum_stats_combined = combine_sum_stats(
        *[e.sum_stats for e in list(experiments)]
    )
    def summary_statistics(data):
        if data is None:
            return {}
        return {str(i): val 
                for i, val in enumerate(sum_stats_combined(data))}

    return observations, model, summary_statistics