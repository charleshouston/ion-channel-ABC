from experiment import Experiment
from experiment import ExperimentData
from experiment import ExperimentStimProtocol
from channel import Channel
import data.ikur.data_ikur as data
import numpy as np


modelfile = 'models/Bondarenko2004_iKur.mmt'

ikur_params = {'ikur.g_Kur': (0, 1),
               'ikur.k_ass1': (0, 100),
               'ikur.k_ass2': (1, 10),
               'ikur.k_atau1': (0, 1),
               'ikur.k_atau2': (0, 10),
               'ikur.k_atau3': (0, 10),
               'ikur.k_iss1': (0, 100),
               'ikur.k_iss2': (1, 10),
               'ikur.k_itau1': (0, 10),
               'ikur.k_itau2': (0, 10)}

ikur = Channel(modelfile, ikur_params,
               vvar='membrane.V', logvars=['environment.time', 'ikur.i_Kur',
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
iv_exp = Experiment(iv_prot, iv_data)
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
    old_settings = np.seterr(all="warn")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        try:
            tmax_i = data[0]['ikur.G_Kur'].index(max(data[0]['ikur.G_Kur']))
            tmax = data[0]['environment.time'][tmax_i]
            t = [time for time in data[0]['environment.time']
                 if time <= tmax]
            t = [ti - min(t) for ti in t]
            I = [curr for (time, curr) in zip(data[0]['environment.time'],
                                              data[0]['ikur.G_Kur'])
                 if time <= tmax]

            if len(t) == 0 or len(I) == 0:
                np.seterr(**old_settings)
                return float("inf")

            def single_exp(t, I_max, tau):
                return I_max * (1 - np.exp(-t / tau))

            [_, tau], _ = so.curve_fit(single_exp, t, I)
            np.seterr(**old_settings)
            return tau
        except:
            np.seterr(**old_settings)
            return float("inf")

act_prot = ExperimentStimProtocol(stim_times, stim_levels,
                                  measure_index=2,
                                  measure_fn=rising_exponential_fit)
act_exp = Experiment(act_prot, act_data)
ikur.add_experiment(act_exp)

### Exp 3 - Inactivation curve
inact_vsteps, inact_cond, inact_errs, inact_N = data.Inact_Brouillette()
inact_data = ExperimentData(x=inact_vsteps, y=inact_cond, N=inact_N,
                            errs=inact_errs, err_type='SEM')
stim_times = [5000, 100, 2500]
stim_levels = [inact_vsteps, -40, 30]
def max_ikur(data):
    return max(data[0]['ikur.G_Kur'])
def normalise_positives(sim_results):
    m = np.max(sim_results)
    if m > 0:
        sim_results = sim_results / m
    return sim_results
inact_prot = ExperimentStimProtocol(stim_times, stim_levels,
                                    measure_index=2, measure_fn=max_ikur,
                                    post_fn=normalise_positives)
inact_exp = Experiment(inact_prot, inact_data)
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
                                   post_fn=normalise_positives)
rec_exp = Experiment(rec_prot, rec_data)
ikur.add_experiment(rec_exp)
