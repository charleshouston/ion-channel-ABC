from pyabc.distance_functions import DistanceFunction
from pyabc.distance_functions import median_absolute_deviation
from pyabc.sampler import Sampler
from pyabc.epsilon import Epsilon
from pyabc.storage import History
from pyabc.populationstrategy import AdaptivePopulationSize
from pyabc.population import Population
from pyabc.transition import Transition
from pyabc.weighted_statistics import weighted_quantile
import numpy as np
from scipy.stats import iqr
from typing import List
import statistics
import logging
import warnings
df_logger = logging.getLogger("DistanceFunction")
eps_logger = logging.getLogger("Epsilon")


class PranglePopulationSize(AdaptivePopulationSize):
    def __init__(self, start_nr_particles, alpha, adapt=False,
                 **kwargs):
        super().__init__(start_nr_particles, **kwargs)
        self.alpha = alpha
        self.adapt = adapt
        # Adapt population size to accept larger proportion
        self.nr_particles = int(self.nr_particles / self.alpha)

    def get_config(self):
        return {"name": self.__class__.__name__,
                "max_population_size": self.max_population_size,
                "mean_cv": self.mean_cv,
                "alpha": self.alpha,
                "adapt": self.adapt}

    def adapt_population_size(self, transitions: List[Transition],
                              model_weights: np.ndarray):
        if self.adapt:
            # Adapt population based on original setting
            self.nr_particles = int(self.nr_particles * self.alpha)
            super().adapt_population_size(transitions, model_weights)
            self.nr_particles = int(self.nr_particles / self.alpha)


class PrangleEpsilon(Epsilon):
    def __init__(self,
                 initial_epsilon: float =float('inf'),
                 alpha: float =0.5):
        eps_logger.debug(
            "init prangle_epsilon initial_epsilon={}, alpha={}"
            .format(initial_epsilon, alpha))
        super().__init__()
        self.alpha = alpha
        self._initial_epsilon = initial_epsilon

        self._look_up = {}

        if self.alpha > 1 or self.alpha <= 0:
            raise ValueError("It must be 0 < alpha <= 1")

    def get_config(self):
        config = super().get_config()
        config.update({"initial_epsilon": self._initial_epsilon,
                       "alpha": self.alpha})
        return config

    def initialize(self,
                   sample_from_prior, distance_to_ground_truth_function):
        super().initialize(sample_from_prior,
                           distance_to_ground_truth_function)

        self._look_up = {0: self._initial_epsilon}

        # logging
        eps_logger.info("initial epsilon is {}".format(self._look_up[0]))

    def __call__(self, t: int,
                 history: History,
                 latest_population: Population =None) -> float:
        try:
            eps = self._look_up[t]
        except KeyError:
            # this will be the usual case after the first iteration
            self._update(t, history, latest_population)
            eps = self._look_up[t]

        # We need to pass the full history of epsilon values
        eps = self._look_up
        return eps

    def _update(self, t: int,
                history: History,
                latest_population: Population =None):
        # If latest_population is None, e.g. after smc.load(), read latest
        # weighted distances from history, else use the passed latest
        # population.
        if latest_population is None:
            df_weighted = history.get_weighted_distances(None)
        else:
            df_weighted = latest_population.get_weighted_distances()

        distances = df_weighted.distance.as_matrix()
        len_distances = len(distances)
        weights = df_weighted.w.as_matrix()
        weights /= weights.sum()

        quantile = weighted_quantile(
                points=distances, weights=weights, alpha=self.alpha)

        self._look_up[t] = quantile

        # logger
        eps_logger.debug("new eps, t={}, eps={}".format(t, self._look_up[t]))


class PrangleDistance(DistanceFunction):
    def __init__(self, exp_map: dict, err_bars: dict =None, 
                 adapt: bool =False, alpha: float =0.5, delta: float =0.5):
        """Initialisation.

        Args:
            exp_map: mapping from summary statistics to separate
                experimental data sources.
            err_bars: error bars for observed data points used to
                weight between points within an individual experiment.
            alpha: tuning parameter for distance algorithm.
            delta: parameter to ensure bounded eccentricity of
                algorithm.
            adapt: whether to use the prangle weighting algorithm.
        """
        super().__init__()
        self.exp = exp_map
        self.errs = err_bars
        self.adapt = adapt
        self.alpha = alpha
        self.delta = delta
        # need to store previous distances
        self.w = [] # adaptive distance weight based on MAD of ss
        self.v = {} # constant weights

    def __call__(self, x: dict, y: dict):
        """Return distance.

        Args:
            x: mapping from arbitrary index to output from
                simulation.
            y: mapping from arbitrary index to observed
                output.
        """
        # make sure weights are initialized
        if not self.w:
            self._initialize_adaptive_weights(x.keys())
        if not self.v:
            self._initialize_constant_weights(y, self.errs)

        # calculate all distances up to this iteration
        distances = []
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                for w in self.w:
                    distances.append(pow(
                        sum(pow(abs(w[key] * self.v[key]
                                    * (x[key] - y[key])), 2)
                            for key in w.keys()),
                        1 / 2))
            except RuntimeWarning:
                return [float('inf')] * len(self.w)
        return distances

    def configure_sampler(self, sampler: Sampler):
        super().configure_sampler(sampler)
        sampler.require_all_sum_stats()

    def initialize(self, sample_from_prior: List[dict],
                   x_0: dict =None):
        # retrieve keys, and init weights with 1
        if self.adapt:
            self._initialize_adaptive_weights(
                    sample_from_prior[0].keys())
        # initialize constant weights
        if x_0:
            self._initialize_constant_weights(x_0, self.errs)

    def _initialize_adaptive_weights(self, summary_statistics_keys):
        # init prangle weights with 1, and retrieve keys
        self.w.append({k: 1 for k in summary_statistics_keys})

    def _initialize_constant_weights(self, x_0: dict, errs: dict):
        """
        Initialise the constant weights based from obs and err bars.

        Args:
            x_0: mapping from data points index to observed output.
            errs: mapping from data points index to error bar in
                publication. Note that this only changes the relative
                weight for data point within an experiment.
        """
        exp_N = self._count_experiments()
        exp_IQR = self._calculate_experiment_IQR(x_0)
        exp_sdi = self._calculate_experiment_sdi(errs)
        for ss, exp_i in self.exp.items():
            self.v[ss] = 1. / (exp_sdi[ss] * np.sqrt(exp_N[exp_i] * exp_IQR[exp_i]))
        mean_weight = statistics.mean(list(self.v.values()))
        for key in self.v.keys():
            self.v[key] /= mean_weight

    def _count_experiments(self):
        exp_N = {}
        for i in self.exp.values():
            if i not in exp_N.keys():
                exp_N[i] = 0
            exp_N[i] += 1
        return exp_N

    def _calculate_experiment_IQR(self, x_0: dict):
        exp_lists = {}
        for k, v in x_0.items():
            if self.exp[k] not in exp_lists:
                exp_lists[self.exp[k]] = []
            exp_lists[self.exp[k]].append(v)

        exp_IQR = {}
        for k, l in exp_lists.items():
            weight = iqr(l)
            if weight == 0:
                # if `l` is a single value
                weight = abs(l[0])
            exp_IQR[k] = weight
        return exp_IQR

    def _calculate_experiment_sdi(self, errs: dict):
        exp_lists = {}
        exp_sdi = {}
        for k, e in errs.items():
            if self.exp[k] not in exp_lists:
                exp_lists[self.exp[k]] = []
            if np.isnan(e):
                weight = 1.0
            else:
                weight = e
            exp_lists[self.exp[k]].append(weight)
            exp_sdi[k] = weight

        # Normalise between experiments
        for k, weight in exp_sdi.items():
            exp_sdi[k] = (exp_sdi[k] / sum(exp_lists[self.exp[k]]) *
                          len(exp_lists[self.exp[k]]))
        return exp_sdi

    def update(self, all_summary_statistics_list: List[dict],
               x_0: dict =None):
        # make sure weights are initialized
        if not self.w:
            self._initialize_adaptive_weights(
                    all_summary_statistics_list[0].keys())
        if not self.v and x_0:
            self._initialize_constant_weights(x_0)
        
        if not self.adapt:
            self._initialize_adaptive_weights(
                    all_summary_statistics_list[0].keys())
            return True

        m = len(all_summary_statistics_list)
        df_logger.debug('number of summary_statistics: {}'.format(m))

        new_w = {}
        for key in self.w[0].keys():
            # prepare list for key
            current_list = []
            for j in range(m):
                # if a failed sim add inf as the output
                ss = all_summary_statistics_list[j]
                if not ss:
                    current_list.append(float('inf'))
                else:
                    current_list.append(self.v[key] * ss[key])

            # compute weighting
            val = median_absolute_deviation(current_list)
            if val == 0:
                new_w[key] = 1
            else:
                new_w[key] = 1 / val

        # Ensuring bounded eccentricity.
        max_w = max(list(new_w.values()))
        for key in new_w.keys():
            new_w[key] += self.delta * max_w

        self.w.append(new_w)

        # normalize weights to have mean 1. This has just the effect that the
        # epsilon will decrease more smoothly, but is not important otherwise.
        mean_weight = statistics.mean(list(self.w[-1].values()))
        for key in self.w[-1].keys():
            self.w[-1][key] /= mean_weight

        # logging
        df_logger.debug("update distance weights = {}".format(self.w[-1]))
        return True
