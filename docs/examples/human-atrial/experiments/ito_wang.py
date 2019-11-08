"""
Experiments from [Wang1993]
Charles Houston 2019-10-18
"""
from ionchannelABC.experiment import Experiment
from ionchannelABC.protocol import recovery, availability
import data.ito.Wang1993.data_Wang1993 as data
import numpy as np
import myokit

import warnings
from scipy.optimize import OptimizeWarning
import scipy.optimize as so


Q10_tau = 2.2 # [Wang1993]


fit_threshold = 0.9


#
# Activation [Wang1993]
#
wang_act_desc = """
    Steady-state activation curve for ito in human
    atrial myocytes [Wang1993] cf Fig 2c.

    Voltage dependence of activation as determined from peak current
    during 1000 ms depolarising pulses from a holding potential of
    -80 mV to various test potentials.
    """
vsteps_act, act, sd_act = data.Act_Wang()
variances_act = [sd_**2 for sd_ in sd_act]
wang_act_dataset = np.asarray([vsteps_act, act, variances_act])

#wang_act_protocol = myokit.Protocol()
#for v in vsteps_act:
#    wang_act_protocol.add_step(-80, 20000) # tpre at vhold
#    wang_act_protocol.add_step(v, 5)      # test pulse
#    wang_act_protocol.add_step(-40, 100)   # measure pulse
wang_act_protocol = myokit.pacing.steptrain(
    vsteps_act, -80, 10000, 1000)

wang_conditions = {'phys.T': 295.15,  # K
                   'k_conc.K_i': 130, # mM
                   'k_conc.K_o': 5.4
                  }

def wang_act_sum_stats(data):
    output = []
    for d in data.split_periodic(11000, adjust=True, closed_intervals=False):
        d = d.trim_left(10000, adjust=True)
        act_gate = d['ito.g']
        output = output + [max(act_gate, key=abs)]
    norm = max(output)
    try:
        for i in range(len(output)):
            output[i] /= norm
    except:
        output = [float('inf'),]*len(output)
    return output

wang_act = Experiment(
    dataset=wang_act_dataset,
    protocol=wang_act_protocol,
    conditions=wang_conditions,
    sum_stats=wang_act_sum_stats,
    description=wang_act_desc,
    Q10=None,
    Q10_factor=0.)


#
# Inactivation [Wang1993]
#
wang_inact_desc = """
    Steady-state inactivation curve for ito in human
    atrial myocytes [Wang1993] cf Fig 2c.

    Voltage-dependent inactivation was assessed with the use of a
    two-pulse protocol with a 1000 ms prepulse to voltages between -80
    and +40 mV, followed by a 1000 ms test pulse to +60 mV.
    """
vsteps_inact, inact, sd_inact = data.Inact_Wang()
variances_inact = [sd_**2 for sd_ in sd_inact]
wang_inact_dataset = np.asarray([vsteps_inact, inact, variances_inact])

wang_inact_protocol = availability(
    vsteps_inact, -80, 60, 20000, 1000, 0, 1000)

def wang_inact_sum_stats(data):
    output = []
    for d in data.split_periodic(22000, adjust=True, closed_intervals=False):
        d = d.trim_left(21000, adjust=True)
        inact_gate = d['ito.g']
        output = output + [max(inact_gate, key=abs)]
    norm = max(output)
    try:
        for i in range(len(output)):
            output[i] /= norm
    except:
        output = [float('inf'),]*len(output)
    return output

wang_inact = Experiment(
    dataset=wang_inact_dataset,
    protocol=wang_inact_protocol,
    conditions=wang_conditions,
    sum_stats=wang_inact_sum_stats,
    description=wang_inact_desc,
    Q10=None,
    Q10_factor=0.)
