import matplotlib.pyplot as plt
from matplotlib.cbook import violin_stats
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.weightstats import DescrStatsW
import numpy as np


def vdensity_with_weights(weights):

    def vdensity(data, coords):
        """Custom matplotlib weighted violin stats function."""
        weighted_cost = sm.nonparametric.KDEUnivariate(data)
        weighted_cost.fit(fft=False, weights=np.asarray(weights))
        return weighted_cost.evaluate(coords)
    return vdensity

def custom_violin_stats(data, weights):
    median = np.asarray(DescrStatsW(data, weights).quantile(0.5))[0]
    mean, sumw = np.ma.average(data, weights=list(weights), axis=0,
                               returned=True)

    results = violin_stats(data, vdensity_with_weights(weights))
    for i in range(len(results)):
        results[i][u"mean"] = mean[i]
        results[i][u"median"] = median[i]

    return results
