from ionchannelABC import (Experiment,
                           ExperimentData,
                           ExperimentStimProtocol,
                           IonChannelModel)
import data.ina.data_ina as data
import numpy as np


modelfile = 'models/Korhonen2009_iNa_all.mmt'

#ina_params = dict(g_Na=(0, 100),
#                  p1=(0, 100),
#                  p2=(-10, 0),
#                  p3=(0, 1),
#                  p4=(0, 100),
#                  p5=(-1, 0),
#                  p6=(0, 1),
#                  p7=(0, 100),
#                  q1=(0, 100),
#                  q2=(0, 10),
#                  q4=(0, 100),
#                  q5=(-100, 0),
#                  q6=(0, 10),
#                  q7=(0, 10),
#                  q8=(0, 100),
#                  q9=(-100, 0),
#                  q10=(0, 10),
#                  q11=(0, 1),
#                  q12=(0, 10),
#                  q13=(0, 1),
#                  r2=(-1, 0),
#                  r3=(0, 100),
#                  r4=(-10, 0),
#                  r5=(0, 10),
#                  r6=(0, 100),
#                  r7=(0, 1),
#                  r8=(0, 100),
#                  r9=(-10, 0),
#                  r10=(0, 1),
#                  r11=(0, 10),
#                  r12=(-1, 1),
#                  r13=(0, 1),
#                  r14=(-1, 1),
#                  r15=(-1, 1),
#                  r16=(0, 100))

ina = IonChannelModel('ina',
                      modelfile,
                      vvar='membrane.V',
                      logvars=['environment.time',
                               'ina.i_Na',
                               'ina.G_Na'])

### Exp 1 - IV curve
iv_vsteps, iv_curr, iv_errs, iv_N = data.IV_Nakajima()
iv_data = ExperimentData(x=iv_vsteps, y=iv_curr, N=iv_N, errs=iv_errs,
                         err_type='SEM')
stim_times = [1000, 20]
stim_levels = [-120, iv_vsteps]
def peak_curr(data):
    return max(data[0]['ina.i_Na'], key=abs)
iv_prot = ExperimentStimProtocol(stim_times, stim_levels,
                                 measure_index=1, measure_fn=peak_curr)
nakajima_conditions = dict(Nao=145000,
                           Nai=10000,
                           T=296)
iv_exp = Experiment(iv_prot, iv_data, nakajima_conditions)

### Exp 2 - Activation
act_vsteps, act_cond, act_errs, act_N = data.Act_Nakajima()
act_data = ExperimentData(x=act_vsteps, y=act_cond, N=act_N, errs=act_errs,
                          err_type='SEM')
stim_times = [1000, 20, 100]
stim_levels = [-120, act_vsteps, -120]
def max_gna(data):
    return max(data[0]['ina.G_Na'])
def normalise(sim_results, ind_var):
    m = max(sim_results, key=abs)
    sim_results = [result / m for result in sim_results]
    return sim_results
act_prot = ExperimentStimProtocol(stim_times, stim_levels,
                                  measure_index=1, measure_fn=max_gna,
                                  post_fn=normalise)
act_exp = Experiment(act_prot, act_data, nakajima_conditions)

### Exp 3 - Inactivation
inact_vsteps, inact_cond, inact_errs, inact_N = data.Inact_Nakajima()
inact_data = ExperimentData(x=inact_vsteps, y=inact_cond, N=inact_N,
                            errs=inact_errs, err_type='SEM')
stim_times = [500, 500, 20, 500]
stim_levels = [-120, inact_vsteps, 20, -120]
inact_prot = ExperimentStimProtocol(stim_times, stim_levels,
                                    measure_index=2, measure_fn=max_gna,
                                    post_fn=normalise)
inact_exp = Experiment(inact_prot, inact_data, nakajima_conditions)

### Exp 4 - Recovery
rec_intervals, rec_cond, rec_errs, rec_N = data.Recovery_Zhang()
rec_data = ExperimentData(x=rec_intervals, y=rec_cond, N=rec_N,
                          errs=rec_errs, err_type='SEM')
stim_times = [3000, 20, rec_intervals, 20]
stim_levels = [-120, -40, -120, -40]
def ratio_cond(data):
    return max(data[1]['ina.G_Na'])/max(data[0]['ina.G_Na'])
rec_prot = ExperimentStimProtocol(stim_times, stim_levels,
                                  measure_index=[1, 3],
                                  measure_fn=ratio_cond)
zhang_conditions = dict(Nao=136330,
                        Nai=15000,
                        T=293)
rec_exp = Experiment(rec_prot, rec_data, zhang_conditions)

### Exp 5 - Current Trace
time, curr, _, _ = data.Trace_Nakajima()
peak_curr = abs(max(curr, key=abs))
curr = [c / peak_curr for c in curr]
trace_data = ExperimentData(x=time, y=curr)
stim_times = [1000, 20]
stim_levels = [-120, -20]
def interpolate_align(data, time):
    import numpy as np
    simtime = data[0]['environment.time']
    simtime_min = min(simtime)
    simtime = [t - simtime_min for t in simtime]
    curr = data[0]['ina.i_Na']
    max_curr = abs(max(curr, key=abs))
    curr = [c / max_curr for c in curr]
    return np.interp(time, simtime, curr)
trace_prot = ExperimentStimProtocol(stim_times, stim_levels,
                                    measure_index=1,
                                    post_fn=interpolate_align,
                                    ind_var=time)
trace_exp = Experiment(trace_prot, trace_data, nakajima_conditions)

ina.add_experiments([iv_exp, act_exp, inact_exp, rec_exp, trace_exp])
