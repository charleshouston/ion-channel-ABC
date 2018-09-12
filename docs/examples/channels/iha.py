from ionchannelABC import (Experiment,
                           ExperimentData,
                           ExperimentStimProtocol,
                           IonChannelModel)
import data.iha.data_iha as data
import numpy as np


modelfile = 'models/Korhonen2009_iha.mmt'

iha = IonChannelModel('iha',
                      modelfile,
                      vvar='membrane.V',
                      logvars=['environment.time',
                               'iha.i_ha',
                               'iha.G_ha'])

### Exp 1 - IV curve.
iv_vsteps, iv_curr, iv_errs, iv_N = data.IV_Sartiani()
iv_data = ExperimentData(x=iv_vsteps, y=iv_curr, N=iv_N,
                         errs=iv_errs, err_type='SEM')
stim_times = [500, 1500, 500]
stim_levels = [-40, -120, iv_vsteps]
def max_curr(data):
    return max(data[0]['iha.i_ha'], key=abs)
iv_prot = ExperimentStimProtocol(stim_times, stim_levels,
                                 measure_index=2,
                                 measure_fn=max_curr)
sartiani_conditions = dict(T=293,
                           Ko=25000,
                           Ki=120000,
                           Nao=140000,
                           Nai=10800)
iv_exp = Experiment(iv_prot, iv_data, sartiani_conditions)

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
    curr = data[0]['iha.i_ha']
    time = data[0]['environment.time']
    t0 = time[0]
    time = [(t - t0)/1000 for t in time] # Units in seconds.

    with warnings.catch_warnings():
        warnings.simplefilter('error')
        try:
            def single_exp(t, I_max, I_diff, tau):
                return I_max * np.exp(-t / tau) + I_diff
            [I_max, _, tau], _ = so.curve_fit(
                    single_exp, time, curr, bounds=([-np.inf, -np.inf, 0],
                                                    np.inf))
            return [I_max, tau]
        except:
            return [float("inf"), float("inf")]

def takesecond(data, ind_var):
    return [d[1] for d in data]
def normalise(data, ind_var):
    sim_results = [d[0] for d in data]
    m = max(sim_results, key=abs)
    sim_results = [result / m for result in sim_results]
    return sim_results

act_prot = ExperimentStimProtocol(stim_times, stim_levels,
                                  measure_index=1,
                                  measure_fn=fit_mono_exp,
                                  post_fn=normalise)
act_tau_prot = ExperimentStimProtocol(stim_times, stim_levels,
                                      measure_index=1,
                                      measure_fn=fit_mono_exp,
                                      post_fn=takesecond)
act_exp = Experiment(act_prot, act_data, sartiani_conditions)
act_tau_exp = Experiment(act_tau_prot, act_tau_data, sartiani_conditions)

iha.add_experiments([iv_exp, act_exp, act_tau_exp])
