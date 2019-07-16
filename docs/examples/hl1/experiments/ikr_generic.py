from ionchannelABC import (Experiment,
                           ExperimentData,
                           ExperimentStimProtocol,
                           IonChannelModel)
import data.ikr.data_ikr as data
import numpy as np
from functools import partial
from scipy.optimize import OptimizeWarning
import myokit


modelfile = 'models/Generic_iKr.mmt'

ikr = IonChannelModel('ikr',
                      modelfile,
                      vvar='membrane.V',
                      logvars=myokit.LOG_ALL)#['environment.time',
                               #'ikr.i_Kr',
                               #'ikr.G_Kr'])

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
    return sim_results, False
act_prot = ExperimentStimProtocol(stim_times, stim_levels,
                                  measure_index=2, measure_fn=max_gkr,
                                  post_fn=normalise)
act_exp = Experiment(act_prot, act_data, toyoda_conditions1)

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
        maxes.append(max(d['ikr.i_Kr'], key=abs))
    return maxes
def fit_single_exp(data, xvar=intervals):
    import numpy as np
    import scipy.optimize as so
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('error', OptimizeWarning)
        warnings.simplefilter('error', RuntimeWarning)
        try:
            def single_exp(t, Ass, A, tau):
                return Ass + A * np.exp(-t / tau)
            [_, _, tau], _ = so.curve_fit(single_exp, xvar, data,
                                       bounds=([-50, -50, 0],
                                               [50, 50, 5000]))

            if np.isclose(tau, 5000):
                raise Exception('Optimisation hit bounds')

            return tau
        except (Exception, OptimizeWarning, RuntimeWarning):
            return float("inf")
def map_return(func, iterable, ind_var=None):
    out = []
    for i in iterable:
        out.append(func(i))
    return out, False
akin_prot = ExperimentStimProtocol(stim_times, stim_levels,
                                   measure_index=measure_index,
                                   measure_fn=measure_maxes,
                                   post_fn=partial(map_return, fit_single_exp))
akin_exp = Experiment(akin_prot, akin_data, toyoda_conditions1)

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
    with warnings.catch_warnings():
        warnings.simplefilter('error',OptimizeWarning)
        warnings.simplefilter('error',RuntimeWarning)
        try:
            curr = data[0]['ikr.i_Kr']
            time = data[0]['environment.time']

            # Get peak current
            index = np.argwhere(np.isclose(curr, max(curr, key=abs)))
       
            # Set time zero to peak current
            index = index[0][0]
            curr = curr[index:]

            # Zero time from peak current time
            time = time[index:]
            t0 = time[0]
            time = [t - t0 for t in time]
            if len(time) == 0 or len(curr) == 0:
                raise Exception('Could not find peak current')

            def sum_of_exp(t, Ass, Af, tauf, As, taus):
                return (Ass + Af * np.exp(-t / tauf) + As * np.exp(-t / taus))
            popt, _ = so.curve_fit(sum_of_exp, time, curr,
                                   p0=[1, 1, 5, 0.2, 70],
                                   bounds=([-50, -50, 0, -50, 50], 
                                           [50, 50, 100, 50, 2000]))

            tauf = popt[2]
            taus = popt[4]
            Af = abs(popt[1])
            As = abs(popt[3])
            A_rel = Af / (Af + As)
            
            return (tauf, taus, A_rel)
        except (Exception, RuntimeWarning, OptimizeWarning, RuntimeError):
            return (float("inf"), float("inf"), float("inf"))
def takefirst(data, ind_var): return [d[0] for d in data], False
def takesecond(data, ind_var): return [d[1] for d in data], False
def takethird(data, ind_var): return [d[2] for d in data], False

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
    with warnings.catch_warnings():
        warnings.simplefilter('error',OptimizeWarning)
        warnings.simplefilter('error',RuntimeWarning)
        try:
            curr = data[0]['ikr.i_Kr']
            time = data[0]['environment.time']

            # Get peak current
            index = np.argwhere(np.isclose(curr, max(curr, key=abs)))

            # Take subset up to peak current
            index = index[0][0]
            curr = curr[:index+1]

            # Zero time
            time = time[:index+1]
            t0 = time[0]
            time = [t - t0 for t in time]
            if len(time) == 0 or len(curr) == 0:
                raise Exception('Could not find a peak current')

            def single_exp(t, Ass, A, tau):
                return Ass + A * np.exp(-t / tau)
            [_, _, tau], _ = so.curve_fit(single_exp, time, curr,
                                       p0=[1, -1, 5],
                                       bounds=([-50, -50, 0],
                                               [50, 50, 100]))
            if np.isclose(tau, 100):
                raise Exception('Optimisation hit bounds')

            return tau
        except (Exception, OptimizeWarning, RuntimeWarning):
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

ikr.add_experiments([iv_exp, act_exp, akin_exp, 
                     deact_f_exp, deact_s_exp, deact_amp_exp,
                     inact_kin_exp, inact_exp])
