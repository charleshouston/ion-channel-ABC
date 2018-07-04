from experiment import (Experiment, ExperimentData,
                        ExperimentStimProtocol)
from channel import Channel
import numpy as np


modelfile = 'models/Houston2017.mmt'

# TODO: this is a bit of a hack of the Channel class...
hl1 = Channel('hl1', modelfile, {}, vvar='membrane.i_stim')
stim_times = [10000]
stim_levels = [0.0]
# See resting state behaviour....
def get_V(data):
    return data[0]['membrane.V']
resting_prot = ExperimentStimProtocol(stim_times, stim_levels,
                                      measure_index=0, measure_fn=get_V)
dias_conditions = dict(T=305,
                       Ca_o=1800,
                       Na_o=1.4e5,
                       K_o=4e3)
resting_exp = Experiment(resting_prot, None, dias_conditions)
hl1.add_experiment(resting_exp)
