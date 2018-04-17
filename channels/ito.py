from experiment import Experiment
from experiment import ExperimentData
from experiment import ExperimentStimProtocol
from channel import Channel
import data.ito.data_ito as data
import numpy as np


modelfile = 'models/Takeuchi2013_ito.mmt'

ito_params = {'ito.g_to': (0, 1),
              'ito.k_xss1': (0, 10),
              'ito.k_xss2': (0, 100),
              'ito.k_xtau1': (0, 10),
              'ito.k_xtau2': (0, 10),
              'ito.k_xtau3': (0, 100),
              'ito.k_yss1': (0, 100),
              'ito.k_yss2': (0, 100),
              'ito.k_ytau1': (0, 100),
              'ito.k_ytau2': (0, 100),
              'ito.k_ytau3': (0, 100),
              'ito.k_ytau4': (0, 100)}

ito = Channel(modelfile, ito_params,
              vvar='membrane.V', logvars=['ito.i_to', 'ito.G_to'])

### Exp 1 - IV curve
iv_vsteps, iv_curr, iv_errs, iv_N = data.IV_Kao()
iv_data = ExperimentData(x=iv_vsteps, y=iv_curr, N=iv_N, errs=iv_errs,
                         err_type='SEM')
stim_times = [500, 30, 300, 500]
stim_levels = [-80, -40, iv_vsteps, -80]
def peak_curr(data):
    import numpy as np
    return max(data[0]['ito.i_to'], key=abs)
iv_prot = ExperimentStimProtocol(stim_times, stim_levels,
                                 measure_index=2, measure_fn=peak_curr)
iv_exp = Experiment(iv_prot, iv_data)
ito.add_experiment(iv_exp)

### Exp 2 - Inactivation
inact_vsteps, inact_cond, inact_errs, inact_N = data.Inact_Xu()
inact_data = ExperimentData(x=inact_vsteps, y=inact_cond, N=inact_N,
                            errs=inact_errs, err_type='SEM')
stim_times = [5000, 5000, 500]
stim_levels = [inact_vsteps, 50, -70]
def max_gto(data):
    return max(data[0]['ito.G_to'])
def normalise_positives(sim_results):
    import numpy as np
    m = np.max(sim_results)
    if m > 0:
        sim_results = sim_results / m
    return sim_results
inact_prot = ExperimentStimProtocol(stim_times, stim_levels,
                                    measure_index=1, measure_fn=max_gto,
                                    post_fn=normalise_positives)
inact_exp = Experiment(inact_prot, inact_data)
ito.add_experiment(inact_exp)
