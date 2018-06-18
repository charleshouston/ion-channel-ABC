from pyabc.distance_functions import DistanceFunction
from scipy.stats import iqr
import numpy as np


class ExperimentNormalisedDistance(DistanceFunction):
    def __init__(self, exp: dict):
        """Initialisation function.

        Args
            exp: Dictionary mapping summary statistics to
                separate experiments for normalisation.
        """
        self.exp = exp
        self.exp_N = {}
        self.exp_IQR = {}

    def __call__(self, x: dict, y: dict):
        if not self.exp_N:
            self._count_experiments()

        if not self.exp_IQR:
            self._calculate_IQR(y)

        distances = {}
        for key in x.keys():
            if self.exp[key] not in distances:
                distances[self.exp[key]] = 0
            distances[self.exp[key]] += pow(x[key] - y[key], 2)

        result = 0.
        for k, v in distances.items():
            result += np.sqrt(v / self.exp_N[k]) / self.exp_IQR[k]
        return result

    def _count_experiments(self):
        for i in self.exp.values():
            if i not in self.exp_N.keys():
                self.exp_N[i] = 0
            self.exp_N[i] += 1

    def _calculate_IQR(self, obs: dict):
        exp_lists = {}
        for k, v in obs.items():
            if self.exp[k] not in exp_lists:
                exp_lists[self.exp[k]] = []
            exp_lists[self.exp[k]].append(v)

        for k, l in exp_lists.items():
            self.exp_IQR[k] = iqr(l)
