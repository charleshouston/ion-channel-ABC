from ionchannelABC import (Experiment,
                           ExperimentData,
                           ExperimentStimProtocol,
                           IonChannelModel)
import data.ina.data_ina as data
import numpy as np


modelfile = 'models/Korhonen2009_iNa.mmt'

#parameters = dict(g_Na=(0, 100),
#                  p1=(0, 100),
#                  p2=(-100, 0),
#                  p3=(0, 1),
#                  p4=(0, 100),
#                  p5=(-100, 0), # Expanded from (-10, 0) after test run
#                  p6=(0, 1),
#                  p7=(0, 1000), # Expanded from (0, 500) after test run
#                  q1=(0, 100),
#                  q2=(0, 50),
#                  q3=(0, 10),
#                  q4=(-50, 50),
#                  q5=(-50, 50),
#                  q6=(0, 1),
#                  q7=(0, 100),
#                  q8=(-50, 50),
#                  q9=(0, 10),
#                  q10=(0, 1),
#                  q11=(0, 10),
#                  r1=(0, 100),
#                  r2=(-1, 0),
#                  r3=(-50, 50),
#                  r4=(-10, 0),
#                  r5=(0, 10),
#                  r6=(0, 100),
#                  r7=(0, 1),
#                  r8=(0, 100),
#                  r9=(-10, 0),
#                  r10=(0, 1),
#                  r11=(0, 10),
#                  r12=(-0.1, 0),
#                  r13=(0, 1),
#                  r14=(-1, 0),
#                  r15=(-1, 0),
#                  r16=(0, 100))

ina = IonChannelModel('ina',
                      modelfile,
                      vvar='membrane.V',
                      logvars=['environment.time',
                               'ina.i_Na',
                               'ina.G_Na'])

### Exp 1 - IV curve
iv_vsteps, iv_curr, _, _ = data.IV_Lee()
iv_data = ExperimentData(x=iv_vsteps, y=iv_curr)
stim_times = [5000, 20]
stim_levels = [-80, iv_vsteps]
def peak_curr(data):
    return max(data[0]['ina.i_Na'], key=abs)
#def normalise_by_peak(sim_results):
#    peak = 445.1308 # peak current determined from unmodified sim
#    sim_results = [r / peak for r in sim_results]
#    return sim_results
def normalise(sim_results):
    m = abs(max(sim_results, key=abs))
    sim_results = [r / m for r in sim_results]
    return sim_results, False
iv_prot = ExperimentStimProtocol(stim_times, stim_levels,
                                 measure_index=1, measure_fn=peak_curr,
                                 post_fn=normalise)
conditions = dict(Nao=20000, Nai=10000, T=295)
iv_exp = Experiment(iv_prot, iv_data, conditions)

### Exp 2 - Activation curve
act_vsteps, act_cond, _, _ = data.Act_Lee()
act_data = ExperimentData(x=act_vsteps, y=act_cond)
stim_times = [5000, 20]
stim_levels = [-80, act_vsteps]
def max_gna(data):
    return max(data[0]['ina.G_Na'])
act_prot = ExperimentStimProtocol(stim_times, stim_levels,
                                  measure_index=1, measure_fn=max_gna,
                                  post_fn=normalise)
act_exp = Experiment(act_prot, act_data, conditions)

### Exp 3 - Inactivation
inact_vsteps, inact_cond, _, _ = data.Inact_Lee()
inact_data = ExperimentData(x=inact_vsteps, y=inact_cond)
stim_times = [500, 20]
stim_levels = [inact_vsteps, -20]
inact_prot = ExperimentStimProtocol(stim_times, stim_levels,
                                    measure_index=1, measure_fn=max_gna,
                                    post_fn=normalise)
inact_exp = Experiment(inact_prot, inact_data, conditions)

### Exp 4 - Recovery (pulse train)
rec_intervals, rec_curr, _, _ = data.Rec_Lee1()
rec_data1 = ExperimentData(x=rec_intervals, y=rec_curr)
stim_times = [180, 20] * 20
cumul_times = np.cumsum(stim_times).tolist()
cumul_times = [0] + cumul_times
ind_var = [cumul_times[i] for i in range(0, 40, 2)]
stim_levels = [-80, -30] * 20
def normalise_curr(data):
    sim_results = []
    for record in data:
        sim_results.append(max(record['ina.i_Na'], key=abs))
    m = max(sim_results, key=abs)
    sim_results = [r / m for r in sim_results]
    return sim_results
rec_prot1 = ExperimentStimProtocol(stim_times, stim_levels,
                                   measure_index=range(1, 40, 2),
                                   measure_fn=normalise_curr,
                                   ind_var=ind_var)
rec_exp1 = Experiment(rec_prot1, rec_data1, conditions)

### Exp 5 - Recovery (two pulse protocol)
rec_intervals, rec_curr, _, _ = data.Rec_Lee2()
rec_data2 = ExperimentData(x=rec_intervals, y=rec_curr)
stim_times = [500, 500, rec_intervals, 20]
stim_levels = [-80, 0, -80, -30]
def ratio_cond(data):
    return max(data[1]['ina.G_Na']) / max(data[0]['ina.G_Na'])
rec_prot2 = ExperimentStimProtocol(stim_times, stim_levels,
                                  measure_index=[1, 3], measure_fn=ratio_cond,
                                  post_fn=normalise)
rec_exp2 = Experiment(rec_prot2, rec_data2, conditions)

### Exp 6 - Current trace
time, curr, _, _ = data.Trace_Lee()
peak_curr = abs(max(curr, key=abs))
curr = [c / peak_curr for c in curr]
trace_data = ExperimentData(x=time, y=curr)
stim_times = [500, 20]
stim_levels = [-80, -30]
def interpolate_align(data):
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
                                    measure_fn=interpolate_align,
                                    ind_var=time)
trace_exp = Experiment(trace_prot, trace_data, conditions)

ina.add_experiments([iv_exp, act_exp, inact_exp,
                     rec_exp1, rec_exp2, trace_exp])
