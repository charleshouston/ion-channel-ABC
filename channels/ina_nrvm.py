from experiment import Experiment
from experiment import ExperimentData
from experiment import ExperimentStimProtocol
from channel import Channel
import data.ina.data_ina as data
import numpy as np


modelfile = 'models/BeelerReuter1976_iNa.mmt'

parameters = dict(k1=(0, 100),
                  k2=(-1, 0),
                  k3=(0, 100),
                  k4=(0, 100),
                  k5=(-1, 0),
                  k6=(0, 100),
                  k7=(0, 1),
                  k8=(-1, 0),
                  k9=(0, 100),
                  k10=(0, 10),
                  k11=(-1, 0),
                  k12=(0, 100),
                  k13=(0, 1),
                  k14=(-1, 0),
                  k15=(0, 100),
                  k16=(-1, 0),
                  k17=(0, 1),
                  k18=(-1, 0),
                  k19=(0, 100))

#parameters = dict(p1=(0, 100),
#                  p2=(-10, 0),
#                  p3=(0, 1),
#                  p4=(0, 100),
#                  p5=(-1, 0),
#                  p6=(0, 1),
#                  p7=(0, 100),
#                  q1=(0, 100),
#                  q2=(0, 10))

ina = Channel('ina', modelfile, parameters,
              vvar='membrane.V', logvars=['ina.i_Na', 'ina.G_Na'])

### Exp 1 - IV curve
iv_vsteps, iv_curr, _, _ = data.IV_Lee()
iv_data = ExperimentData(x=iv_vsteps, y=iv_curr)
stim_times = [5000, 20]
stim_levels = [-80, iv_vsteps]
def peak_curr(data):
    return max(data[0]['ina.i_Na'], key=abs)
def normalise(sim_results):
    m = abs(max(sim_results, key=abs))
    sim_results = [r / m for r in sim_results]
    return sim_results
iv_prot = ExperimentStimProtocol(stim_times, stim_levels,
                                 measure_index=1, measure_fn=peak_curr,
                                 post_fn=normalise)
conditions = dict(Nao=20000, Nai=10000, T=295)
iv_exp = Experiment(iv_prot, iv_data, conditions)
ina.add_experiment(iv_exp)

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
ina.add_experiment(act_exp)

### Exp 3 - Inactivation
inact_vsteps, inact_cond, _, _ = data.Inact_Lee()
inact_data = ExperimentData(x=inact_vsteps, y=inact_cond)
stim_times = [500, 20]
stim_levels = [inact_vsteps, -20]
inact_prot = ExperimentStimProtocol(stim_times, stim_levels,
                                    measure_index=1, measure_fn=max_gna,
                                    post_fn=normalise)
inact_exp = Experiment(inact_prot, inact_data, conditions)
ina.add_experiment(inact_exp)

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
    print(sim_results)
    return sim_results
rec_prot1 = ExperimentStimProtocol(stim_times, stim_levels,
                                   measure_index=range(1, 40, 2),
                                   measure_fn=normalise_curr,
                                   ind_var=ind_var)
rec_exp1 = Experiment(rec_prot1, rec_data1, conditions)
ina.add_experiment(rec_exp1)

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
ina.add_experiment(rec_exp2)
