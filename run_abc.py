import ion_channel_ABC as abc
from experiment import Experiment
from experiment import ExperimentData
from experiment import ExperimentStimProtocol
from channel import Channel
from error_functions import cvrmsd, cvchisq
import data.icat.data_icat as data_icat

import matplotlib.pyplot as plt
import myokit
import numpy as np


def pretty_results_print(channel, distr):
    lmeans = distr.getmean()
    lvars = distr.getvar()
    print "Results\n======="
    for i, param in enumerate(channel.param_names):
        print param + ": Mean " + str(lmeans[i]) + " Var " + str(lvars[i])


modelfile = 'models/Korhonen2009_iCaT.mmt'

icat_params = {'icat.g_CaT': (0, 2),
               'icat.E_CaT': (0, 50),
               'icat.p1': (0, 100),
               'icat.p2': (1, 10),
               'icat.p3': (0, 1),
               'icat.p4': (0, 10),
               'icat.p5': (0, 0.1),
               'icat.p6': (0, 200),
               'icat.q1': (0, 100),
               'icat.q2': (1, 10),
               'icat.q3': (0, 10),
               'icat.q4': (0, 100),
               'icat.q5': (0, 0.1),
               'icat.q6': (0, 100)}

icat = Channel(modelfile, icat_params,
               vvar='membrane.V', logvars=['icat.i_CaT', 'icat.G_CaT'])

### Exp 1 - IV curve
iv_vsteps, iv_curr, iv_errs, iv_N = data_icat.IV_Nguyen()
iv_data = ExperimentData(x=iv_vsteps, y=iv_curr, N=iv_N, errs=iv_errs,
                         err_type='SEM')
stim_times = [5000, 300, 500]
stim_levels = [-75, iv_vsteps, -75]
def min_icat(data):
    return min(data[0]['icat.i_CaT'])
iv_prot = ExperimentStimProtocol(stim_times, stim_levels,
                                 measure_index=1, measure_fn=min_icat)
iv_exp = Experiment(iv_prot, iv_data)
icat.add_experiment(iv_exp)

### Exp 2 - Activation curve
act_vsteps, act_cond, act_errs, act_N = data_icat.Act_Nguyen()
act_data = ExperimentData(x=act_vsteps, y=act_cond, N=act_N, errs=act_errs,
                          err_type='SEM')
stim_times = [5000, 300, 500]
stim_levels = [-75, act_vsteps, -75]
def max_gcat(data):
    return max(data[0]['icat.G_CaT'])
def normalise_positives(sim_results):
    m = np.max(sim_results)
    if m > 0:
        sim_results = sim_results / m
    return sim_results
act_prot = ExperimentStimProtocol(stim_times, stim_levels,
                                  measure_index=1, measure_fn=max_gcat,
                                  post_fn=normalise_positives)
act_exp = Experiment(act_prot, act_data)
icat.add_experiment(act_exp)

### Exp 3 - Inactivation curve
inact_vsteps, inact_cond, inact_errs, inact_N = data_icat.Inact_Nguyen()
inact_data = ExperimentData(x=inact_vsteps, y=inact_cond, N=inact_N,
                            errs=inact_errs, err_type='SEM')
stim_times = [1000, 200]
stim_levels = [inact_vsteps, -10]
inact_prot = ExperimentStimProtocol(stim_times, stim_levels,
                                    measure_index=1, measure_fn=max_gcat,
                                    post_fn=normalise_positives)
inact_exp = Experiment(inact_prot, inact_data)
icat.add_experiment(inact_exp)

### Exp 4 - Recovery curve
rec_intervals, rec_cond, rec_errs, rec_N = data_icat.Rec_Deng()
rec_data = ExperimentData(x=rec_intervals, y=rec_cond, N=rec_N, errs=rec_errs,
                          err_type='SEM')
stim_times = [5000, 300, rec_intervals, 300]
stim_levels = [-80, -20, -80, -20]
def ratio_cond(data):
    return max(data[1]['icat.G_CaT'])/max(data[0]['icat.G_CaT'])
rec_prot = ExperimentStimProtocol(stim_times, stim_levels,
                                  measure_index=[1, 3], measure_fn=ratio_cond,
                                  post_fn=normalise_positives)
rec_exp = Experiment(rec_prot, rec_data)
icat.add_experiment(rec_exp)

abc_solver = abc.ABCSolver(error_fn=cvchisq, post_size=20, maxiter=500)
final_distr = abc_solver(icat, logfile='logs/icat_cvchisq.log')

pretty_results_print(icat, final_distr)
