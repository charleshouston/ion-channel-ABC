### Error functions for ABC parameter estimation.


import warnings
import numpy as np


def normaliseby(a, b):
    '''Normalise `a` between -1 and 1 using `b` as normalisation data.'''
    temp = a
    temp /= np.max(np.abs(b), axis=0)
    return temp


def cvchisq(res_sim, res_exper):
    '''Coefficient of variation of weighted residuals.

    Args:
        res_sim (List[List[float]]): Results for each simulation run.
        res_exper (List[List[float]]): Corresponding experimental data.

    Returns:
        Error value as float.
    '''
    warnings.filterwarnings('error')
    tot_err = 0

    for i, m in enumerate(model):
        m = np.array(m)
        e = np.array(exper[i][1]) # experimental mean
        err_bars = np.array(exper[i][2]) # experimental standard deviation
        N = np.array(exper[i][3]) # number of experimental measurements taken for this data mean

        # normalise all data points to between -1 and 1 by experimental value
        m = normaliseby(m, e)
        e = normaliseby(e, e)
        err_bars = normaliseby(err_bars, e)
        sd = err_bars * np.sqrt(N) # calculate standard deviations from normalised values of SEM

        # if any sd value is zero, set to minimum of other recorded values
        for i,sdi in enumerate(sd):
            if sdi == 0:
                sd[i] = np.min(sd[sd!=0])

        # normalise weights for weighted sum of squares loss so that in the limit as sd -> 0
        # is equivalent to sum of squares loss
        w = [1 / np.square(sdi) for sdi in sd]
        w = w / np.max(w)

        try:
            err = np.sum(w * np.square(e - m))
        except Warning:
            return float("inf")
        except:
            return float("inf")

        # root error
        err = pow(err / len(m), 0.5)
        # err = err / np.ptp(e) # normalised earlier so shouldn't need to normalise again here
        tot_err += err

    warnings.resetwarnings()

    return tot_err

# Calculate coefficient of variation of the real mean squared distance
def cvrmsd(model, exper):

    warnings.filterwarnings('error')

    tot_err = 0

    for i, m in enumerate(model):
        m = np.array(m)
        e = np.array(exper[i][1])

        # normalise between -1 and 1 by experimental values
        m = normaliseby(m, e)
        e = normaliseby(e, e)

        try:
            err = np.sum(np.square(m - e))
        except Warning:
            return float("inf")
        except:
            return float("inf")

        # root and normalise error by range
        err = pow(err / len(m), 0.5)
        # err = err / np.ptp(e)
        tot_err += err

    warnings.resetwarnings()

    return tot_err


