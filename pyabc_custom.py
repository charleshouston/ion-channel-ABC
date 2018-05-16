import pandas as pd
import subprocess
from io import BytesIO


def simulate(channel, continuous=False, experiment=None,
             logvars=None, **pars):
    """Wrapper to simulate the myokit model.

    Simulates in a subprocess running python2 by passing
    parameters as arguments to (another) wrapper script.

    Args:
        channel (str): Name of channel.
        continuous (bool): Whether to run only at experimental
            data points or finer x resolution.
        experiment (int): Specific experiment number to run and
            return raw output rather than measured results.
        logvars (list(str)): List of variables to log in sim.
        pars (Dict[string, float]): Parameters as kwargs.

    Returns:
        Dataframe with simulated output or empty if
        the simulation failed.
    """
    myokit_python = ("/tmp/chouston/miniconda3/envs" +
                     "/ion_channel_ABC/bin/python")
    script = "run_channel.py"
    args = [myokit_python, script]
    args.append(channel)
    if continuous:
        args.append('--continuous')
    if experiment is not None:
        args.append('--experiment')
        args.append(str(experiment))
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
