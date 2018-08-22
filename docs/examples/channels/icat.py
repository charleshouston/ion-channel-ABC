from ionchannelABC import (Experiment,
                           ExperimentData,
                           ExperimentStimProtocol,
                           IonChannelModel)
import data.icat.data_icat as data
import numpy as np


modelfile = 'models/Korhonen2009_iCaT.mmt'

icat = IonChannelModel('icat',
                       modelfile,
                       vvar='membrane.V',
                       logvars=['environment.time', 
                                'icat.i_CaT',
                                'icat.G_CaT'])

### Exp 1 - IV curve
iv_vsteps, iv_curr, iv_errs, iv_N = data.IV_Nguyen()
iv_data = ExperimentData(x=iv_vsteps, y=iv_curr, N=iv_N, errs=iv_errs,
                         err_type='SEM')
stim_times = [5000, 300, 500]
stim_levels = [-75, iv_vsteps, -75]
def max_icat(data):
    return max(data[0]['icat.i_CaT'],
               key=lambda x: abs(x))
iv_prot = ExperimentStimProtocol(stim_times, stim_levels,
                                 measure_index=1,
                                 measure_fn=max_icat)
nguyen_conditions = dict(Ca_o=5000,
                         T=295)
iv_exp = Experiment(iv_prot, iv_data, nguyen_conditions)

### Exp 2 - Activation curve
act_vsteps, act_cond, act_errs, act_N = data.Act_Nguyen()
act_data = ExperimentData(x=act_vsteps, y=act_cond, N=act_N, errs=act_errs,
                          err_type='SEM')
stim_times = [5000, 300, 500]
stim_levels = [-75, act_vsteps, -75]
def max_gcat(data):
    return max(data[0]['icat.G_CaT'])
def normalise(sim_results, ind_var):
    cond_max = max(sim_results, key=abs)
    sim_results = [result / cond_max for result in sim_results]
    return sim_results
act_prot = ExperimentStimProtocol(stim_times, stim_levels,
                                  measure_index=1, measure_fn=max_gcat,
                                  post_fn=normalise)
act_exp = Experiment(act_prot, act_data, nguyen_conditions)

### Exp 3 - Inactivation curve
inact_vsteps, inact_cond, inact_errs, inact_N = data.Inact_Nguyen()
inact_data = ExperimentData(x=inact_vsteps, y=inact_cond, N=inact_N,
                            errs=inact_errs, err_type='SEM')
stim_times = [1000, 200]
stim_levels = [inact_vsteps, -10]
inact_prot = ExperimentStimProtocol(stim_times, stim_levels,
                                    measure_index=1, measure_fn=max_gcat,
                                    post_fn=normalise)
inact_exp = Experiment(inact_prot, inact_data, nguyen_conditions)

### Exp 4 - Recovery curve
rec_intervals, rec_cond, rec_errs, rec_N = data.Rec_Deng()
rec_data = ExperimentData(x=rec_intervals, y=rec_cond, N=rec_N, errs=rec_errs,
                          err_type='SEM')
stim_times = [5000, 300, rec_intervals, 300]
stim_levels = [-80, -20, -80, -20]
def ratio_cond(data):
    return max(data[1]['icat.G_CaT'])/max(data[0]['icat.G_CaT'])
rec_prot = ExperimentStimProtocol(stim_times, stim_levels,
                                  measure_index=[1, 3], measure_fn=ratio_cond,
                                  post_fn=normalise)
deng_conditions = dict(Ca_o=5000,
                       T=298)
rec_exp = Experiment(rec_prot, rec_data, deng_conditions)

### Exp 5 - Current trace
time, curr, _, _ = data.CurrTrace_Deng()
peak_curr = abs(max(curr, key=abs))
curr = [c / peak_curr for c in curr]
trace_data = ExperimentData(x=time, y=curr)
stim_times = [5000, 300]
stim_levels = [-80, -20]
def interpolate_align(data, time):
    import numpy as np
    simtime = data[0]['environment.time']
    simtime_min = min(simtime)
    simtime = [t - simtime_min for t in simtime]
    curr = data[0]['icat.i_CaT']
    max_curr = abs(max(curr, key=abs))
    curr = [c / max_curr for c in curr]
    return np.interp(time, simtime, curr)
trace_prot = ExperimentStimProtocol(stim_times, stim_levels,
                                    measure_index=1,
                                    post_fn=interpolate_align,
                                    ind_var=time)
trace_exp = Experiment(trace_prot, trace_data, deng_conditions)

icat.add_experiments([iv_exp, act_exp, inact_exp, rec_exp, trace_exp])
