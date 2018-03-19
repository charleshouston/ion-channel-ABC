### Error functions for ABC parameter estimation.

# All error functions should accept two arguments:
# Arg 1 is a list or array of the simulation results.
# Arg 2 is an ExperimentData object corresponding the the results.

import warnings
import numpy as np


def normaliseby(a, b):
    """Normalise `a` between -1 and 1 using `b` as normalisation data."""
    temp = np.copy(a)
    temp /= np.max(np.abs(b), axis=0)
    return temp


def cvchisq(sim_results, experiment_data):
    """Coefficient of variation of weighted residuals.

    Args:
        sim_results (List[float]): Output from simulation run.
        experiment_data (ExperimentData): Corresponding experimental data.

    Returns:
        Error value as float.

    Raises:
        ValueError: If `experiment_data` has no `err_type` reported.
    """
    warnings.filterwarnings('error')

    s = np.array(sim_results)
    e = np.array(experiment_data.y)
    N = np.array(experiment_data.N)
    err_bars = np.array(experiment_data.errs)

    if experiment_data.err_type == 'SEM':
        sd = err_bars * np.sqrt(N)
    elif experiment_data.err_type == 'STD':
        sd = err_bars
    else:
        raise ValueError('Need `err_type` to compute cvchisq loss!')

    # Set SD values to minimum rather than 0 to avoid divide by zero.
    for i, sdi in enumerate(sd):
        if sdi == 0:
            sd[i] = np.min(sd[sd!=0])

    # Normalise weights for WSS loss so that limit as SD -> 0
    # is equivalent to sum of squares loss.
    w = [1 / np.square(sdi) for sdi in sd]
    w = w / np.max(w)

    try:
        err = np.sum(w * np.square(e - s))
    except Warning:
        return float("inf")
    except:
        return float("inf")

    err = pow(err / len(s), 0.5)
    err = err / np.ptp(e)
    warnings.resetwarnings()
    return err


def cvrmsd(sim_results, experiment_data):
    """Coefficient of variation of the real mean sq distance."""
    warnings.filterwarnings('error')
    s = np.array(sim_results)
    e = np.array(experiment_data.y)

    try:
        err = np.sum(np.square(s - e))
    except Warning:
        return float("inf")
    except:
        return float("inf")

    err = pow(err / len(s), 0.5)
    err = err / np.ptp(e)
    warnings.resetwarnings()
    return err
