import pandas as pd
import subprocess
import numpy as np
from io import BytesIO
from pyabc import Model, ModelResult
from typing import List, Dict, Callable


def simulate(channel, n_x=None, exp_num=None,
             logvars=None, **pars):
    """Wrapper to simulate the myokit model.

    Simulates in a subprocess running python2 by passing
    parameters as arguments to (another) wrapper script.

    Args:
        channel (str): Name of channel.
        n_x (int): Resolution of independent variable.
        exp_num (int): Specific experiment number to run and
            return raw output rather than measured results.
        logvars (list(str)): List of variables to log in sim.
        pars (Dict[string, float]): Parameters as kwargs.

    Returns:
        Dataframe with simulated output or empty if
        the simulation failed.
    """
    myokit_python = ("/scratch/cph211/miniconda3/envs" +
                     "/ion_channel_ABC/bin/python")
    #myokit_python = ("/Users/charles/miniconda3/envs" +
    #                 "/ion_channel_ABC/bin/python")
    script = "run_channel.py"
    args = [myokit_python, script]
    args.append(channel)
    if n_x is not None:
        args.append('--n_x')
        args.append(str(n_x))
    if exp_num is not None:
        args.append('--exp_num')
        args.append(str(exp_num))
    if logvars is not None:
        args.append('--logvars')
        for var in logvars:
            args.append(var)
    for p in pars:
        try:
            args.append("-" + str(p))
            args.append(str(pars[p]))
        except:
            print("Error: " +
                  "args is " + str(args))
    re = subprocess.run(args, stdout=subprocess.PIPE)
    if len(re.stdout) > 0:
        d = pd.read_table(BytesIO(re.stdout),
                          delim_whitespace=True,
                          header=0, index_col=False)
        return d
    else:
        raise ValueError("Failed simulation.")
        return None


def voltage_dependence(channel, variables, **pars):
    """Returns underlying model variables at different voltages.

    Args:
        channel (str): name of channel.
        variables (list(str)): variables to log at voltages.
        pars (dict(str -> float)): Dictionary mapping parameter
            names to new parameter values.

    Returns:
        Dataframe containing voltage column and recorded variables.
    """
    myokit_python = ("/tmp/chouston/miniconda3/envs" +
                     "/ion_channel_ABC/bin/python")
    script = "run_channel.py"
    args = [myokit_python, script]
    args.append(channel)

    args.append('--vdep')
    for var in variables:
        args.append(var)
    for p in pars:
        try:
            args.append("-" + str(p))
            args.append(str(pars[p]))
        except:
            print("Error: " +
                  "args is " + str(args))

    re = subprocess.run(args, stdout=subprocess.PIPE)
    if len(re.stdout) > 0:
        d = pd.read_table(BytesIO(re.stdout),
                          delim_whitespace=True,
                          header=0, index_col=False)
        return d
    else:
        raise ValueError("Failed simulation.")
        return None


class MyokitSimulation(Model):

    def __init__(self, channel: str, par_samples: List[Dict[str, float]] =None,
                 measure_fn: Callable =lambda x: x):
        super().__init__()
        self.channel = channel
        # List of samples to randomly chosen external from ABC sampling.
        self.par_samples = par_samples
        # How to convert results from `simulate` into model output.
        # Default is identity function.
        self.measure_fn = measure_fn

    def sample(self, pars):
        # Add any parameters varying external to ABC fitting
        if self.par_samples is not None:
            full_pars = dict(np.random.choice(self.par_samples), **pars)
        else:
            full_pars = pars

        # Run simulation
        try:
            res = self.measure_fn(simulate(self.channel, **full_pars))
        except:
            res = {}
        return res

    def accept(self, pars, sum_stats_calculator, distance_calculator, eps) \
            -> ModelResult:

        # Run simulation and catch failed sims
        result = self.summary_statistics(pars, sum_stats_calculator)
        if len(result.sum_stats) == 0:
            result.accepted = False
            result.distance = float('inf')
            return result

        distances = distance_calculator(result.sum_stats)

        # Check result satisfies all previous iterations of distance
        # and epsilon history
        for d_i, eps_i in zip(distances, eps):
            if d_i > eps_i:
                result.accepted = False
                result.distance = distances[-1]
                return result

        result.accepted = True
        result.distance = distances[-1]
        return result
