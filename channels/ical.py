from experiment import Experiment
from experiment import ExperimentData
from experiment import ExperimentStimProtocol
from channel import Channel
import data.ical.data_ical as data
import numpy as np


modelfile = 'models/Korhonen2009_iCaL.mmt'

ical_params = {'ical.g_CaL': (0, 0.001),
               'ical.p1': (-50, 50),
               'ical.p2': (0, 50),
               'ical.p3': (-100, 50),
               'ical.p4': (-50, 50),
               'ical.p5': (-50, 50),
               'ical.p6': (-50, 50),
               'ical.p7': (0, 200),
               'ical.p8': (0, 200),
               'ical.q1': (0, 100),
               'ical.q2': (0, 50),
               'ical.q3': (0, 10000),
               'ical.q4': (0, 100),
               'ical.q5': (0, 1000),
               'ical.q6': (0, 1000),
               'ical.q7': (0, 100),
               'ical.q8': (0, 100),
               'ical.q9': (-500, 500)}

ical = Channel(modelfile, ical_params,
               vvar='membrane.V', logvars=['ical.i_CaL', 'ical.G_CaL',
                                           'environment.time'])

### Exp 1 - IV curve
iv_vsteps, iv_curr, iv_errs, iv_N = data.IV_Rao()
iv_data = ExperimentData(x=iv_vsteps, y=iv_curr, N=iv_N,
                         errs=iv_errs, err_type='SEM')
stim_times = [1000, 100, 500]
stim_levels = [-80, iv_vsteps, -80]
def peak_curr(data):
    return max(data[0]['ical.i_CaL'], key=abs)
iv_prot = ExperimentStimProtocol(stim_times, stim_levels,
                                 measure_index=1, measure_fn=peak_curr)
iv_exp = Experiment(iv_prot, iv_data)
ical.add_experiment(iv_exp)

### Exp 2 - activation
act_vsteps, act_cond, act_errs, act_N = data.Act_Rao()
act_data = ExperimentData(x=act_vsteps, y=act_cond, N=act_N,
                          errs=act_errs, err_type='SEM')
stim_times = [1000, 100, 500]
stim_levels = [-80, act_vsteps, -80]
def max_gcal(data):
    return max(data[0]['ical.G_CaL'])
def normalise_positives(sim_results):
    import numpy as np
    m = np.max(sim_results)
    if m > 0:
        sim_results = sim_results / m
    return sim_results
act_prot = ExperimentStimProtocol(stim_times, stim_levels,
                                  measure_index=1, measure_fn=max_gcal,
                                  post_fn=normalise_positives)
act_exp = Experiment(act_prot, act_data)
ical.add_experiment(act_exp)

### Exp 3 - inactivation
inact_vsteps, inact_cond, inact_errs, inact_N = data.Inact_Rao()
inact_data = ExperimentData(x=inact_vsteps, y=inact_cond, N=inact_N,
                            errs=inact_errs, err_type='SEM')
stim_times = [1000, 1000, 400]
stim_levels = [-80, inact_vsteps, -20]
def gcal_inact_max(data):
    d = data[0]['ical.G_CaL']
    if d[1] - d[0] > 0:
        # Increasing -> find max
        return max(d)
    else:
        # Decreasing -> find min
        return min(d)
inact_prot = ExperimentStimProtocol(stim_times, stim_levels,
                                    measure_index=2, measure_fn=gcal_inact_max,
                                    post_fn=normalise_positives)
inact_exp = Experiment(inact_prot, inact_data)
ical.add_experiment(inact_exp)

### Exp 4 - recovery
rec_intervals, rec_cond, rec_errs, rec_N = data.Rec_Rao()
rec_data = ExperimentData(x=rec_intervals, y=rec_cond, N=rec_N,
                          errs=rec_errs, err_type='SEM')
stim_times = [1000, 400, rec_intervals, 400]
stim_levels = [-80, -20, -80, -20]
def ratio_cond(data):
    return max(data[1]['ical.G_CaL'])/max(data[0]['ical.G_CaL'])
rec_prot = ExperimentStimProtocol(stim_times, stim_levels,
                                  measure_index=[1, 3], measure_fn=ratio_cond,
                                  post_fn=normalise_positives)
rec_exp = Experiment(rec_prot, rec_data)
ical.add_experiment(rec_exp)