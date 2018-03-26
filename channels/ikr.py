from experiment import Experiment
from experiment import ExperimentData
from experiment import ExperimentStimProtocol
from channel import Channel
import data.ikr.data_ikr as data
import numpy as np
from functools import partial


modelfile = 'models/Korhonen2009_iKr.mmt'

ikr_params = {'ikr.g_Kr': (0, 1),
              'ikr.p1': (0, 0.1),
              'ikr.p2': (0, 0.1),
              'ikr.p3': (0, 0.1),
              'ikr.p4': (0, 0.1),
              'ikr.p5': (-0.1, 0.1),
              'ikr.p6': (0, 0.1),
              'ikr.q1': (0, 0.1),
              'ikr.q2': (-0.1, 0),
              'ikr.q3': (0, 0.0001),
              'ikr.q4': (-0.1, 0),
              'ikr.q5': (0, 0.01),
              'ikr.q6': (-0.1, 0),
              'ikr.k_f': (0, 0.5),
              'ikr.k_b': (0, 0.1)}

ikr = Channel(modelfile, ikr_params,
              vvar='membrane.V', logvars=['environment.time',
                                          'ikr.i_Kr', 'ikr.G_Kr'])

### Exp 1 - IV curve.
iv_vsteps, iv_curr, iv_errs, iv_N = data.IV_Toyoda()
iv_data = ExperimentData(x=iv_vsteps, y=iv_curr, N=iv_N, errs=iv_errs,
                         err_type='SEM')
stim_times = [1000, 1000, 100]
stim_levels = [-50, iv_vsteps, -50]
def tail_curr(data):
    return data[0]['ikr.i_Kr'][-1]
iv_prot = ExperimentStimProtocol(stim_times, stim_levels,
                                 measure_index=1, measure_fn=tail_curr)
iv_exp = Experiment(iv_prot, iv_data)
ikr.add_experiment(iv_exp)

### Exp 2 - Activation curve.
act_vsteps, act_cond, act_errs, act_N = data.Act_Toyoda()
act_data = ExperimentData(x=act_vsteps, y=act_cond, N=act_N, errs=act_errs,
                          err_type='SEM')
stim_times = [1000, 1000, 500]
stim_levels = [-50, act_vsteps, -50]
def max_gkr(data):
    return max(data[0]['ikr.G_Kr'])
def normalise_positives(sim_results):
    m = np.max(sim_results)
    if m > 0:
        sim_results = sim_results / m
    return sim_results
act_prot = ExperimentStimProtocol(stim_times, stim_levels,
                                  measure_index=2, measure_fn=max_gkr,
                                  post_fn=normalise_positives)
act_exp = Experiment(act_prot, act_data)
ikr.add_experiment(act_exp)

### Exp 3 - Activation kinetics.
akin_vsteps, akin_tau, akin_errs, akin_N = data.ActKin_Toyoda()
akin_data = ExperimentData(x=akin_vsteps, y=akin_tau, N=akin_N, errs=akin_errs,
                           err_type='SEM')
intervals = np.arange(25, 975+50, 50)
stim_times = []
stim_levels = []
measure_index = []
for i, interval in enumerate(intervals):
    stim_times = stim_times + [1000, interval, 1000]
    stim_levels = stim_levels + [-50, akin_vsteps, -50]
    measure_index = measure_index + [3*i + 2,]
def measure_maxes(data):
    maxes = []
    for d in data:
        maxes.append(max(d['ikr.G_Kr']))
    return maxes
def fit_single_exp(data, xvar=intervals):
    import numpy as np
    import scipy.optimize as so
    import warnings
    old_settings = np.seterr(all="ignore")
    with warnings.catch_warnings():
        try:
            def single_exp(t, I_max, tau):
                return I_max * (1 - np.exp(-t / tau))
            [_, tau], _ = so.curve_fit(single_exp, xvar, data)
            np.seterr(**old_settings)
            return tau
        except:
            np.seterr(**old_settings)
            return float("inf")
akin_prot = ExperimentStimProtocol(stim_times, stim_levels,
                                   measure_index=measure_index,
                                   measure_fn=measure_maxes,
                                   post_fn=partial(map, fit_single_exp))
akin_exp = Experiment(akin_prot, akin_data)
ikr.add_experiment(akin_exp)

### Exp 4, 5, 6 - Deactivation kinetics (fast and slow).
deact_vsteps, deact_tauf, deact_f_errs, deact_N = data.DeactKinFast_Toyoda()
_, deact_taus, deact_s_errs, _ = data.DeactKinSlow_Toyoda()
_, deact_amp, deact_amp_errs, _ = data.DeactKinRelAmp_Toyoda()
deact_f_data = ExperimentData(x=deact_vsteps, y=deact_tauf, N=deact_N,
                              errs=deact_f_errs, err_type='SEM')
deact_s_data = ExperimentData(x=deact_vsteps, y=deact_taus, N=deact_N,
                              errs=deact_s_errs, err_type='SEM')
deact_amp_data = ExperimentData(x=deact_vsteps, y=deact_amp, N=deact_N,
                                errs=deact_amp_errs, err_type='SEM')
stim_times = [1000, 1000]
stim_levels = [20, deact_vsteps]
def double_exp_decay_fit(data):
    import numpy as np
    import scipy.optimize as so
    import warnings
    old_settings = np.seterr(all="warn")
    with warnings.catch_warnings():
        try:
            tmax_i = data[0]['ikr.G_Kr'].index(max(np.abs(data[0]['ikr.G_Kr'])))
            tmax = data[0]['environment.time'][tmax_i]
            t = [time for time in data[0]['environment.time']
                 if time >= tmax]
            t = [ti - min(t) for ti in t]
            G = [cond for (time, cond) in zip(data[0]['environment.time'],
                                              data[0]['ikr.G_Kr'])
                 if time >= tmax]

            if len(t) == 0 or len(G) == 0:
                np.seterr(**old_settings)
                return float("inf")

            def double_exp(t, G_max1, G_max2, tauf, taus):
                return G_max1 * np.exp(-t / tauf) + G_max2 * np.exp(-t / taus)
            popt, _ = so.curve_fit(double_exp, t, G, bounds=(0, np.inf))
            G_max = (popt[0], popt[1])
            tau = (popt[2], popt[3])
            tauf = min(tau)
            taus = max(tau)
            tauf_i = tau.index(min(tau))
            taus_i = tau.index(max(tau))
            G_maxf = G_max[tauf_i]
            G_maxs = G_max[taus_i]
            A_relative = G_maxf / (G_maxf + G_maxs)

            np.seterr(**old_settings)
            return (tauf, taus, A_relative)
        except:
            np.seterr(**old_settings)
            return float("inf")
def takefirst(data): return [d[0] for d in data]
def takesecond(data): return [d[1] for d in data]
def takethird(data): return [d[2] for d in data]

deact_f_prot = ExperimentStimProtocol(stim_times, stim_levels,
                                      measure_index=1,
                                      measure_fn=double_exp_decay_fit,
                                      post_fn=takefirst)
deact_s_prot = ExperimentStimProtocol(stim_times, stim_levels,
                                      measure_index=1,
                                      measure_fn=double_exp_decay_fit,
                                      post_fn=takesecond)
deact_amp_prot = ExperimentStimProtocol(stim_times, stim_levels,
                                        measure_index=1,
                                        measure_fn=double_exp_decay_fit,
                                        post_fn=takethird)
deact_f_exp = Experiment(deact_f_prot, deact_f_data)
deact_s_exp = Experiment(deact_s_prot, deact_s_data)
deact_amp_exp = Experiment(deact_amp_prot, deact_amp_data)
ikr.add_experiment(deact_f_exp)
ikr.add_experiment(deact_s_exp)
ikr.add_experiment(deact_amp_exp)

### Exp 7 - Inactivation.
inact_vsteps, inact_cond, _, _ = data.Inact_Toyoda()
inact_data = ExperimentData(x=inact_vsteps, y=inact_cond)
stim_times = [1000, 1000, 10, 1000]
stim_levels = [-50, 20, inact_vsteps, 20]
inact_prot = ExperimentStimProtocol(stim_times, stim_levels,
                                    measure_index=3,
                                    measure_fn=max_gkr,
                                    post_fn=normalise_positives)
inact_exp = Experiment(inact_prot, inact_data)
ikr.add_experiment(inact_exp)
