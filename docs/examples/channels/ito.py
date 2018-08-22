from ionchannelABC import (Experiment,
                           ExperimentData,
                           ExperimentStimProtocol,
                           IonChannelModel)
import data.ito.data_ito as data
import numpy as np


modelfile = 'models/Takeuchi2013_ito.mmt'

#ito_params = {'g_to': (0, 1),
#              'k_xss1': (0, 10),
#              'k_xss2': (0, 100),
#              'k_xtau1': (0, 10),
#              'k_xtau2': (0, 10),
#              'k_xtau3': (0, 100),
#              'k_yss1': (0, 100),
#              'k_yss2': (0, 100),
#              'k_ytau1': (0, 100),
#              'k_ytau2': (0, 100),
#              'k_ytau3': (0, 100),
#              'k_ytau4': (0, 100)}

ito = IonChannelModel('ito',
                      modelfile,
                      vvar='membrane.V',
                      logvars=['environment.time',
                               'ito.i_to',
                               'ito.G_to'])

### Exp 1 - IV curve
iv_vsteps, iv_curr, iv_errs, _ = data.IV_Lu()
iv_data = ExperimentData(x=iv_vsteps, y=iv_curr,
                         errs=iv_errs)
stim_times = [500, 30, 300, 500]
stim_levels = [-80, -40, iv_vsteps, -80]
def peak_curr(data):
    return max(data[0]['ito.i_to'], key=abs)
iv_prot = ExperimentStimProtocol(stim_times, stim_levels,
                                 measure_index=2, measure_fn=peak_curr)
lu_conditions = dict(T=309,
                     Ko=4000,
                     Ki=130000)
iv_exp = Experiment(iv_prot, iv_data, lu_conditions)

### Exp 2 - Activation time constant
act_vsteps, act_tau, act_tau_errs, _ = data.ActTau_Xu()
act_kin_data = ExperimentData(x=act_vsteps, y=act_tau,
                              errs=act_tau_errs)
stim_times = [500, 20, 4500, 500]
stim_levels = [-70, -20, act_vsteps, -70]
def fit_exp_rising_phase(data):
    import numpy as np
    import scipy.optimize as so
    import warnings
    # Only want rising phase of current
    curr = data[0]['ito.i_to']
    curr_diff = np.diff(curr)
    index = 0
    if curr_diff[0] > 0:
        index = np.argwhere(curr_diff < 0)[0][0]
    else:
        index = np.argwhere(curr_diff > 0)[0][0]
    curr = curr[:index+1]
    i0 = curr[0]
    curr = [i - i0 for i in curr]
    # Corresponding time variable
    time = data[0]['environment.time'][:index+1]
    t0 = time[0]
    time = [t - t0 for t in time]
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        try:
            def single_exp(t, I_max, tau):
                return I_max * (1 - np.exp(-t / tau))
            [_, tau], _ = so.curve_fit(single_exp, time, curr)
            return tau
        except:
            return float("inf")
act_kin_prot = ExperimentStimProtocol(stim_times, stim_levels,
                                      measure_index=2,
                                      measure_fn=fit_exp_rising_phase)
xu_conditions = dict(T=296,
                     Ko=4000,
                     Ki=135000)
act_kin_exp = Experiment(act_kin_prot, act_kin_data, xu_conditions)

### Exp 3 - Inactivation
inact_vsteps, inact_cond, inact_errs, inact_N = data.Inact_Xu()
inact_data = ExperimentData(x=inact_vsteps, y=inact_cond, N=inact_N,
                            errs=inact_errs, err_type='SEM')
stim_times = [5000, 5000, 500]
stim_levels = [inact_vsteps, 50, -70]
def max_gto(data):
    return max(data[0]['ito.G_to'])
def normalise(sim_results, ind_var):
    m = max(sim_results, key=abs)
    sim_results = [result / m for result in sim_results]
    return sim_results
inact_prot = ExperimentStimProtocol(stim_times, stim_levels,
                                    measure_index=1, measure_fn=max_gto,
                                    post_fn=normalise)
inact_exp = Experiment(inact_prot, inact_data, xu_conditions)

ito.add_experiments([iv_exp, act_kin_exp, inact_exp])
