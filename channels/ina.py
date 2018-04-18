from experiment import Experiment
from experiment import ExperimentData
from experiment import ExperimentStimProtocol
from channel import Channel
import data.ina.data_ina as data
import numpy as np


modelfile = 'models/Korhonen2009_iNa.mmt'

ina_params = {'ina.g_Na': (0, 100),
              'ina.E_Na': (0, 100),
              'ina.p1': (0, 100),
              'ina.p2': (-10, 0),
              'ina.p3': (0, 1),
              'ina.p4': (0, 100),
              'ina.p5': (-1, 0),
              'ina.p6': (0, 1),
              'ina.p7': (0, 100),
              'ina.q1': (0, 100),
              'ina.q2': (0, 10)}

ina = Channel(modelfile, ina_params,
              vvar='membrane.V', logvars=['ina.i_Na', 'ina.G_Na'])

### Exp 1 - IV curve
iv_vsteps, iv_curr, iv_errs, iv_N = data.IV_Nakajima()
iv_data = ExperimentData(x=iv_vsteps, y=iv_curr, N=iv_N, errs=iv_errs,
                         err_type='SEM')
stim_times = [1000, 20]
stim_levels = [-120, iv_vsteps]
def peak_curr(data):
    import numpy as np
    return max(data[0]['ina.i_Na'], key=abs)
iv_prot = ExperimentStimProtocol(stim_times, stim_levels,
                                 measure_index=1, measure_fn=peak_curr)
iv_exp = Experiment(iv_prot, iv_data)
ina.add_experiment(iv_exp)

### Exp 2 - Activation
act_vsteps, act_cond, act_errs, act_N = data.Act_Nakajima()
act_data = ExperimentData(x=act_vsteps, y=act_cond, N=act_N, errs=act_errs,
                          err_type='SEM')
stim_times = [1000, 20, 100]
stim_levels = [-120, act_vsteps, -120]
def max_gna(data):
    return max(data[0]['ina.G_Na'])
def normalise_positives(sim_results):
    import numpy as np
    m = np.max(sim_results)
    if m > 0:
        sim_results = sim_results / m
    return sim_results
act_prot = ExperimentStimProtocol(stim_times, stim_levels,
                                  measure_index=1, measure_fn=max_gna,
                                  post_fn=normalise_positives)
act_exp = Experiment(act_prot, act_data)
ina.add_experiment(act_exp)

### Exp 3 - Inactivation
inact_vsteps, inact_cond, inact_errs, inact_N = data.Inact_Nakajima()
inact_data = ExperimentData(x=inact_vsteps, y=inact_cond, N=inact_N,
                            errs=inact_errs, err_type='SEM')
stim_times = [500, 500, 20, 500]
stim_levels = [-120, inact_vsteps, 20, -120]
inact_prot = ExperimentStimProtocol(stim_times, stim_levels,
                                    measure_index=2, measure_fn=max_gna,
                                    post_fn=normalise_positives)
inact_exp = Experiment(inact_prot, inact_data)
ina.add_experiment(inact_exp)

### Exp 4 - Recovery
rec_intervals, rec_cond, rec_errs, rec_N = data.Recovery_Zhang()
rec_data = ExperimentData(x=rec_intervals, y=rec_cond, N=rec_N,
                          errs=rec_errs, err_type='SEM')
stim_times = [3000, 20, rec_intervals, 20]
stim_levels = [-120, -40, -120, -40]
def ratio_cond(data):
    return max(data[1]['ina.G_Na'])/max(data[0]['ina.G_Na'])
rec_prot = ExperimentStimProtocol(stim_times, stim_levels,
                                  measure_index=[1, 3], measure_fn=ratio_cond,
                                  post_fn=normalise_positives)
rec_exp = Experiment(rec_prot, rec_data)
ina.add_experiment(rec_exp)
