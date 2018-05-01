import pandas as pd
import subprocess


def simulate(channel, continuous=False, **pars):
    """Wrapper to simulate the myokit model.

    Simulates in a subprocess running python2 by passing
    parameters as arguments to (another) wrapper script.

    Args:
        pars (Dict[string, float]): Parameters as kwargs.
        continuous (bool): Whether to run only at experimental
            data points or finer x resolution.

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
