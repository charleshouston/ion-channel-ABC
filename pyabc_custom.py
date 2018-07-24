import pandas as pd
import subprocess
from io import BytesIO
from pyabc import Model, ModelResult


def simulate(channel, continuous=False, exp_num=None,
             logvars=None, **pars):
    """Wrapper to simulate the myokit model.

    Simulates in a subprocess running python2 by passing
    parameters as arguments to (another) wrapper script.

    Args:
        channel (str): Name of channel.
        continuous (bool): Whether to run only at experimental
            data points or finer x resolution.
        exp_num (int): Specific experiment number to run and
            return raw output rather than measured results.
        logvars (list(str)): List of variables to log in sim.
        pars (Dict[string, float]): Parameters as kwargs.

    Returns:
        Dataframe with simulated output or empty if
        the simulation failed.
    """
    #myokit_python = ("/storage/hhecm/cellrotor/chouston/miniconda3/envs" +
    #                 "/ion_channel_ABC/bin/python")
    myokit_python = ("/Users/charles/miniconda3/envs" +
                     "/ion_channel_ABC/bin/python")
    script = "run_channel.py"
    args = [myokit_python, script]
    args.append(channel)
    if continuous:
        args.append('--continuous')
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

    def __init__(self, channel: str):
        super().__init__()
        self.channel = channel

    def sample(self, pars):
        try:
            res = simulate(self.channel, **pars).to_dict()['y']
        except:
            res = {}
        return res

    def accept(self, pars, sum_stats_calculator, distance_calculator, eps) \
            -> ModelResult:

        # Run simulation and catch failed sims
        result = self.summary_statistics(pars, sum_stats_calculator)
        if not result.sum_stats:
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
