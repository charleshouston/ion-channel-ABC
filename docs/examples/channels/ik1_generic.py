from ionchannelABC import (Experiment,
                           ExperimentData,
                           ExperimentStimProtocol,
                           IonChannelModel)
import data.ik1.data_ik1 as data
import numpy as np


modelfile = 'models/Generic_iK1.mmt'

ik1 = IonChannelModel('ik1',
                      modelfile,
                      vvar='membrane.V',
                      logvars=['ik1.i_K1'])

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
goldoni_conditions = dict(Ko=120000,
                          Ki=140000,
                          T=310)
iv_exp = Experiment(iv_prot, iv_data, goldoni_conditions)

ik1.add_experiments([iv_exp])
