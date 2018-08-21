from experiment import Experiment
from experiment import ExperimentData
from experiment import ExperimentStimProtocol
from channel import Channel
import data.ikur.data_ikur as data
import numpy as np


modelfile = 'models/Bondarenko2004_iKur.mmt'

ikur_params = {'g_Kur': (0, 1),
               'k_ass1': (0, 100),
               'k_ass2': (0, 100),
               'k_atau1': (0, 100),
               'k_atau2': (0, 100),
               'k_atau3': (0, 10),
               'k_iss1': (0, 100),
               'k_iss2': (0, 100),
               'k_itau1': (0, 10),
               'k_itau2': (0, 100)}

ikur = Channel('ikur', modelfile, ikur_params,
               vvar='membrane.V', logvars=['environment.time',
                                           'ikur.i_Kur',
                                           'ikur.G_Kur'])

### Exp 1 - IV curve
iv_vsteps, iv_curr, iv_errs, _ = data.IV_Maharani()
iv_data = ExperimentData(x=iv_vsteps, y=iv_curr, errs=iv_errs,
                         err_type='STD')
stim_times = [5000, 300]
stim_levels = [-60, iv_vsteps]
def max_ikur(data):
    return max(data[0]['ikur.i_Kur'])
iv_prot = ExperimentStimProtocol(stim_times, stim_levels,
                                 measure_index=1, measure_fn=max_ikur)
maharani_conditions = dict(T=310,
                           Ko=4000,
                           Ki=120000)
iv_exp = Experiment(iv_prot, iv_data, maharani_conditions)
ikur.add_experiment(iv_exp)

### Exp 2 - Activation time constants
act_vsteps, act_tau, act_errs, _ = data.ActTau_Xu()
act_data = ExperimentData(x=act_vsteps, y=act_tau, errs=act_errs,
                          err_type='STD')
stim_times = [15000, 0.3, 4500]
stim_levels = [-70, act_vsteps, act_vsteps]
def rising_exponential_fit(data):
    import numpy as np
    import scipy.optimize as so
    import warnings
    # First subset so only rising phase of current
    curr = data[0]['ikur.i_Kur']
    curr_diff = np.diff(curr)
    index = 0
    if curr_diff[0] > 0:
        index = np.argwhere(curr_diff < 0)[0][0]
    else:
        index = np.argwhere(curr_diff > 0)[0][0]
    curr = curr[:index+1]
    i0 = curr[0]
    curr = [i - i0 for i in curr]
    # Get time and move to zero
    time = data[0]['environment.time'][:index+1]
    t0 = time[0]
    time = [t - t0 for t in time]

    old_settings = np.seterr(all="ignore")
    with warnings.catch_warnings():
        try:
            def single_exp(t, I_max, tau):
                return I_max * (1 - np.exp(-t / tau))
            [_, tau], _ = so.curve_fit(single_exp, time, curr)
            np.seterr(**old_settings)
            return tau
        except:
            np.seterr(**old_settings)
            return float("inf")

act_prot = ExperimentStimProtocol(stim_times, stim_levels,
                                  measure_index=2,
                                  measure_fn=rising_exponential_fit)
xu_conditions = dict(T=296,
                     Ko=4000,
                     Ki=135000)
act_exp = Experiment(act_prot, act_data, xu_conditions)
ikur.add_experiment(act_exp)

### Exp 3 - Inactivation curve
inact_vsteps, inact_cond, inact_errs, inact_N = data.Inact_Brouillette()
inact_data = ExperimentData(x=inact_vsteps, y=inact_cond, N=inact_N,
                            errs=inact_errs, err_type='SEM')
stim_times = [5000, 100, 2500]
stim_levels = [inact_vsteps, -40, 30]
def max_ikur(data):
    return max(data[0]['ikur.G_Kur'])
def normalise(sim_results, ind_var):
    cond_max = max(sim_results, key=abs)
    sim_results = [result / cond_max for result in sim_results]
    return sim_results
inact_prot = ExperimentStimProtocol(stim_times, stim_levels,
                                    measure_index=2, measure_fn=max_ikur,
                                    post_fn=normalise)
brouillette_conditions = dict(T=294,
                              Ko=5400,
                              Ki=28000)
inact_exp = Experiment(inact_prot, inact_data, brouillette_conditions)
ikur.add_experiment(inact_exp)

### Exp 4 - Recovery curve
rec_steps, rec_react, rec_errs, rec_N = data.Rec_Brouillette()
rec_data = ExperimentData(x=rec_steps, y=rec_react, N=rec_N,
                          errs=rec_errs, err_type='SEM')
stim_times = [5000, 100, 1500, rec_steps, 100, 500]
stim_levels = [-80, -40, 30, -80, -40, 30]
def ratio_ikur(data):
    return max(data[1]['ikur.G_Kur'])/max(data[0]['ikur.G_Kur'])
rec_prot = ExperimentStimProtocol(stim_times, stim_levels,
                                   measure_index = [2, 5],
                                   measure_fn=ratio_ikur,
                                   post_fn=normalise)
rec_exp = Experiment(rec_prot, rec_data, brouillette_conditions)
ikur.add_experiment(rec_exp)
