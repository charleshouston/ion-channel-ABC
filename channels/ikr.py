from experiment import Experiment
from experiment import ExperimentData
from experiment import ExperimentStimProtocol
from channel import Channel
import data.ikr.data_ikr as data
import numpy as np
from functools import partial


modelfile = 'models/Majumder2016_iKr.mmt'

ikr_params = {'g_Kr': (0, 0.001),
              'p1': (0, 100),
              'p2': (0, 100),
              'p3': (0, 0.1),
              'p4': (0, 100),
              'p5': (-1.0, 0),
              'p6': (0, 0.001),
              'p7': (0, 100),
              'p8': (0, 1.0),
              'q1': (0, 100),
              'q2': (0, 100)}

ikr = Channel('ikr', modelfile,
              ikr_params,
              vvar='membrane.V',
              logvars=['environment.time',
                       'ikr.i_Kr',
                       'ikr.G_Kr'
                       ])

### Exp 1 - IV curve.
iv_vsteps, iv_curr, iv_errs, iv_N = data.IV_Toyoda()
iv_data = ExperimentData(x=iv_vsteps, y=iv_curr,
                         errs=iv_errs)
stim_times = [1000, 1000, 100]
stim_levels = [-50, iv_vsteps, -50]
def tail_curr(data):
    return data[0]['ikr.i_Kr'][-1]
iv_prot = ExperimentStimProtocol(stim_times, stim_levels,
                                 measure_index=1,
                                 measure_fn=tail_curr)
toyoda_conditions1 = dict(K_o=5400,
                          K_i=130000,
                          Na_o=140330,
                          Na_i=6000,
                          T=308)
iv_exp = Experiment(iv_prot, iv_data, toyoda_conditions1)
ikr.add_experiment(iv_exp)

### Exp 2 - Activation curve.
act_vsteps, act_cond, act_errs, act_N = data.Act_Toyoda()
act_data = ExperimentData(x=act_vsteps, y=act_cond,
                          errs=act_errs)
stim_times = [1000, 1000, 500]
stim_levels = [-50, act_vsteps, -50]
def max_gkr(data):
    return max(data[0]['ikr.G_Kr'], key=abs)
def normalise(sim_results, ind_var):
    max_cond = abs(max(sim_results, key=abs))
    sim_results = [result / max_cond for result in sim_results]
    return sim_results
act_prot = ExperimentStimProtocol(stim_times, stim_levels,
                                  measure_index=2, measure_fn=max_gkr,
                                  post_fn=normalise)
act_exp = Experiment(act_prot, act_data, toyoda_conditions1)
ikr.add_experiment(act_exp)

### Exp 3 - Activation kinetics.
akin_vsteps, akin_tau, akin_errs, akin_N = data.ActKin_Toyoda()
akin_data = ExperimentData(x=akin_vsteps, y=akin_tau,
                           errs=akin_errs)
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
def fit_single_exp(data, ind_var, xvar=intervals):
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
akin_exp = Experiment(akin_prot, akin_data, toyoda_conditions1)
#ikr.add_experiment(akin_exp)

### Exp 4, 5, 6 - Deactivation kinetics (fast and slow).
deact_vsteps, deact_tauf, deactfast_errs, _ = data.DeactKinFast_Toyoda()
_, deact_taus, deactslow_errs, _ = data.DeactKinSlow_Toyoda()
_, deact_amp, deactamp_errs, _ = data.DeactKinRelAmp_Toyoda()
deact_f_data = ExperimentData(x=deact_vsteps, y=deact_tauf,
                              errs=deactfast_errs)
deact_s_data = ExperimentData(x=deact_vsteps, y=deact_taus,
                              errs=deactslow_errs)
deact_amp_data = ExperimentData(x=deact_vsteps, y=deact_amp,
                                errs=deactamp_errs)
stim_times = [1000, 1000, 1000]
stim_levels = [-50, 20, deact_vsteps]
def double_exp_decay_fit(data):
    import numpy as np
    import scipy.optimize as so
    import warnings
    old_settings = np.seterr(all="warn")
    with warnings.catch_warnings():
        try:
            curr = data[0]['ikr.i_Kr']
            curr_diff = np.diff(curr)
            if curr_diff[0] > 0:
                index = np.argwhere(curr_diff < 0)[0][0]
            else:
                index = np.argwhere(curr_diff > 0)[0][0]
            curr = curr[index:]
            i0 = curr[0]
            curr = [i - i0 for i in curr]
            # Get time and move to zero
            time = data[0]['environment.time'][index:]
            t0 = time[0]
            time = [t - t0 for t in time]
            if len(time) == 0 or len(curr) == 0:
                np.seterr(**old_settings)
                return float("inf")

            def double_exp(t, I_maxf, I_maxs, tauf, taus):
                return (I_maxf * (1 - np.exp(-t / tauf)) +
                        I_maxs * (1 - np.exp(-t / taus)))
            popt, _ = so.curve_fit(double_exp, time, curr,
                                   bounds=([-np.inf, -np.inf, 10, 0],
                                           [np.inf, np.inf, 1000, 100])
                                  )
            I_max = (popt[0], popt[1])
            tau = (popt[2], popt[3])
            tauf = max(tau)
            taus = min(tau)
            tauf_i = tau.index(max(tau))
            taus_i = tau.index(min(tau))
            I_maxf = abs(I_max[tauf_i])
            I_maxs = abs(I_max[taus_i])
            A_rel = I_maxf / (I_maxf + I_maxs)

            np.seterr(**old_settings)
            return (tauf, taus, A_rel)
        except:
            np.seterr(**old_settings)
            return (float("inf"), float("inf"), float("inf"))
def takefirst(data, ind_var): return [d[0] for d in data]
def takesecond(data, ind_var): return [d[1] for d in data]
def takethird(data, ind_var): return [d[2] for d in data]

deact_f_prot = ExperimentStimProtocol(stim_times, stim_levels,
                                      measure_index=2,
                                      measure_fn=double_exp_decay_fit,
                                      post_fn=takefirst)
deact_s_prot = ExperimentStimProtocol(stim_times, stim_levels,
                                      measure_index=2,
                                      measure_fn=double_exp_decay_fit,
                                      post_fn=takesecond)
deact_amp_prot = ExperimentStimProtocol(stim_times, stim_levels,
                                        measure_index=2,
                                        measure_fn=double_exp_decay_fit,
                                        post_fn=takethird)
deact_f_exp = Experiment(deact_f_prot, deact_f_data, toyoda_conditions1)
deact_s_exp = Experiment(deact_s_prot, deact_s_data, toyoda_conditions1)
deact_amp_exp = Experiment(deact_amp_prot, deact_amp_data,
                           toyoda_conditions1)
#ikr.add_experiment(deact_f_exp)
#ikr.add_experiment(deact_s_exp)
#ikr.add_experiment(deact_amp_exp)

### Exp 7 - Kinetic properties of recovery from inactivation
inact_vsteps, inact_tau, inactkin_errs, _, = data.InactKin_Toyoda()
inact_kin_data = ExperimentData(x=inact_vsteps, y=inact_tau,
                                errs=inactkin_errs)
stim_times = [1000, 1000, 1000]
stim_levels = [-50, 20, inact_vsteps]
def fit_exp_rising_phase(data):
    import numpy as np
    import scipy.optimize as so
    import warnings
    # First subset so only rising phase of current
    curr = data[0]['ikr.i_Kr']
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
inact_kin_prot = ExperimentStimProtocol(stim_times, stim_levels,
                                        measure_index=2,
                                        measure_fn=fit_exp_rising_phase)
toyoda_conditions2 = dict(K_o=5400,
                          K_i=130000,
                          Na_o=140330,
                          Na_i=6000,
                          T=298)
inact_kin_exp = Experiment(inact_kin_prot, inact_kin_data,
                           toyoda_conditions2)
#ikr.add_experiment(inact_kin_exp)

### Exp 8 - Voltage dependence of steady-state inactivation.
inact_vsteps, inact_cond, _, _ = data.Inact_Toyoda()
inact_data = ExperimentData(x=inact_vsteps, y=inact_cond)
stim_times = [1000, 1000, 10, 1000]
stim_levels = [-50, 20, inact_vsteps, 20]
inact_prot = ExperimentStimProtocol(stim_times, stim_levels,
                                    measure_index=3,
                                    measure_fn=max_gkr,
                                    post_fn=normalise)
inact_exp = Experiment(inact_prot, inact_data, toyoda_conditions2)
ikr.add_experiment(inact_exp)
