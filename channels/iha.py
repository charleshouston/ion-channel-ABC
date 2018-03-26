from experiment import Experiment
from experiment import ExperimentData
from experiment import ExperimentStimProtocol
from channel import Channel
import data.iha.data_iha as data
import numpy as np


modelfile = 'models/Majumder2016_iha.mmt'

iha_params = {'iha.k_yss1': (0, 100),
              'iha.k_yss2': (1, 10),
              'iha.k_ytau1': (0, 10),
              'iha.k_ytau2': (0, 1.0),
              'iha.k_ytau3': (0, 100),
              'iha.k_ytau4': (1, 100),
              'iha.k_ytau5': (0, 1.0),
              'iha.k_ytau6': (0, 100),
              'iha.k_ytau7': (0, 100),
              'iha.k_i_haNa': (0, 1.0),
              'iha.g_ha': (0, 0.1)}

iha = Channel(modelfile, iha_params,
              vvar='membrane.V', logvars=['environment.time', 'iha.i_ha',
                                          'iha.G_ha'])

### Exp 1 - IV curve.
iv_vsteps, iv_curr, iv_errs, iv_N = data.IV_Sartiani()
iv_data = ExperimentData(x=iv_vsteps, y=iv_curr, N=iv_N,
                         errs=iv_errs, err_type='SEM')
stim_times = [500, 1500, 500]
stim_levels = [-40, -120, iv_vsteps]
def max_curr(data):
    import numpy as np
    return max(data[0]['iha.i_ha'], key=abs)
iv_prot = ExperimentStimProtocol(stim_times, stim_levels,
                                 measure_index=2, measure_fn=max_curr)
iv_exp = Experiment(iv_prot, iv_data)
iha.add_experiment(iv_exp)

### Exp 2, 3 - Activation.
act_vsteps, act_cond, act_errs, act_N = data.Act_Sartiani()
_, act_tau, act_tau_errs, _ = data.ActTau_Sartiani()
act_data = ExperimentData(x=act_vsteps, y=act_cond, N=act_N,
                          errs=act_errs, err_type='SEM')
act_tau_data = ExperimentData(x=act_vsteps, y=act_tau, N=act_N,
                              errs=act_tau_errs, err_type='SEM')
stim_times = [500, 1500, 500]
stim_levels = [-40, act_vsteps, 40]
def fit_mono_exp(data):
    import numpy as np
    import scipy.optimize as so
    import warnings
    old_settings = np.seterr(all="ignore")
    with warnings.catch_warnings():
        try:
            t = data[0]['environment.time']
            t = [(ti - min(t))/1000 for ti in t] # Units in seconds.
            I = data[0]['iha.i_ha']
            def single_exp(t, I_max, I_diff, tau):
                return I_max * np.exp(-t / tau) - I_diff
            [I_max, I_diff, tau], _ = so.curve_fit(single_exp, t, I,
                                                   bounds=([0, -np.inf, 0],
                                                           np.inf))
            I_amplitude = abs(I_diff - I[0])
            np.seterr(**old_settings)
            return [I_amplitude, tau]
        except:
            np.seterr(**old_settings)
            return float("inf")
def takesecond(data): return [d[1] for d in data]
def normalise_positives_firsts(data):
    sim_results = [d[0] for d in data]
    m = np.max(sim_results)
    if m > 0:
        sim_results = sim_results / m
    return sim_results

act_prot = ExperimentStimProtocol(stim_times, stim_levels,
                                  measure_index=1,
                                  measure_fn=fit_mono_exp,
                                  post_fn=normalise_positives_firsts)
act_tau_prot = ExperimentStimProtocol(stim_times, stim_levels,
                                      measure_index=1,
                                      measure_fn=fit_mono_exp,
                                      post_fn=takesecond)
act_exp = Experiment(act_prot, act_data)
act_tau_exp = Experiment(act_tau_prot, act_tau_data)
iha.add_experiment(act_exp)
iha.add_experiment(act_tau_exp)
