from experiment import Experiment
from experiment import ExperimentData
from experiment import ExperimentStimProtocol
from channel import Channel
import data.ik1.data_ik1 as data
import numpy as np


modelfile = 'models/Korhonen2009_iK1.mmt'

ik1_params = dict(g_K1=(0, 0.2),
                  k_1=(-500, 500),
                  k_2=(0, 50),
                  k_3=(0, 1),
                  k_4=(0, 0.1))

ik1 = Channel('ik1', modelfile, ik1_params,
              vvar='membrane.V', logvars=['ik1.i_K1'])

### Exp 1 - IV Curve
iv_vsteps, iv_curr, iv_errs, iv_N = data.IV_Goldoni()
iv_data = ExperimentData(x=iv_vsteps, y=iv_curr, N=iv_N,
                         errs=iv_errs, err_type='STD')
stim_times = [500, 100, 500]
stim_levels = [-50, iv_vsteps, -50]
def ss_curr(data):
    return data[0]['ik1.i_K1']
iv_prot = ExperimentStimProtocol(stim_times, stim_levels,
                                 measure_index=1, measure_fn=ss_curr,
                                 time_independent=True)

conditions = dict(Ko=120000,
                  Ki=140000,
                  T=310)
iv_exp = Experiment(iv_prot, iv_data, conditions)
ik1.add_experiment(iv_exp)
