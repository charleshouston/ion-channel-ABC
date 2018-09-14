from ionchannelABC import (Experiment,
                           ExperimentData,
                           ExperimentStimProtocol,
                           IonChannelModel)
import data.ical.data_ical as data
import numpy as np


modelfile = 'models/Korhonen2009_iCaL.mmt'

#ical_params = {'g_CaL': (0, 0.001),
#               'p1': (-50, 50),
#               'p2': (0, 50),
#               'p3': (-100, 50),
#               'p4': (-50, 50),
#               'p5': (-50, 50),
#               'p6': (-50, 50),
#               'p7': (0, 200),
#               'p8': (0, 200),
#               'q1': (0, 100),
#               'q2': (0, 50),
#               'q3': (0, 10000),
#               'q4': (0, 100),
#               'q5': (0, 1000),
#               'q6': (0, 1000),
#               'q7': (0, 100),
#               'q8': (0, 100),
#               'q9': (-500, 500)}

ical = IonChannelModel('ical',
                       modelfile,
                       vvar='membrane.V',
                       logvars=['ical.i_CaL',
                                'ical.G_CaL',
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
                                 measure_index=1,
                                 measure_fn=peak_curr)
rao_conditions = dict(T=298,
                      Ca_o=5000)
iv_exp = Experiment(iv_prot, iv_data, rao_conditions)

### Exp 2 - activation
act_vsteps, act_cond, act_errs, act_N = data.Act_Rao()
act_data = ExperimentData(x=act_vsteps, y=act_cond, N=act_N,
                          errs=act_errs, err_type='SEM')
stim_times = [1000, 100, 500]
stim_levels = [-80, act_vsteps, -80]
def max_gcal(data):
    return max(data[0]['ical.G_CaL'])
def normalise(sim_results, ind_var):
    m = max(sim_results, key=abs)
    sim_results = [result / m for result in sim_results]
    return sim_results
act_prot = ExperimentStimProtocol(stim_times, stim_levels,
                                  measure_index=1, measure_fn=max_gcal,
                                  post_fn=normalise)
act_exp = Experiment(act_prot, act_data, rao_conditions)

### Exp 3 - Inactivation
inact_vsteps, inact_cond, inact_errs, inact_N = data.Inact_Rao()
inact_data = ExperimentData(x=inact_vsteps, y=inact_cond, N=inact_N,
                            errs=inact_errs, err_type='SEM')
stim_times = [1000, 1000, 400]
stim_levels = [-80, inact_vsteps, -20]
def gcal_inact_max(data):
    return max(data[0]['ical.G_CaL'], key=abs)
inact_prot = ExperimentStimProtocol(stim_times, stim_levels,
                                    measure_index=2,
                                    measure_fn=gcal_inact_max,
                                    post_fn=normalise)
inact_exp = Experiment(inact_prot, inact_data, rao_conditions)

### Exp 4 - recovery
rec_intervals, rec_cond, rec_errs, rec_N = data.Rec_Rao()
rec_data = ExperimentData(x=rec_intervals, y=rec_cond, N=rec_N,
                          errs=rec_errs, err_type='SEM')
stim_times = [1000, 400, rec_intervals, 400]
stim_levels = [-80, -20, -80, -20]
def ratio_cond(data):
    return max(data[1]['ical.G_CaL']) / max(data[0]['ical.G_CaL'])
rec_prot = ExperimentStimProtocol(stim_times, stim_levels,
                                  measure_index=[1, 3], 
                                  measure_fn=ratio_cond)
rec_exp = Experiment(rec_prot, rec_data, rao_conditions)

ical.add_experiments([iv_exp, act_exp, inact_exp, rec_exp])
