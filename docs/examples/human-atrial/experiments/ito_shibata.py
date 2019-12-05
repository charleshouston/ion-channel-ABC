"""
Experiments from [Shibata1989]
Charles Houston 2019-10-18
"""
from ionchannelABC.experiment import Experiment
import data.ito.Shibata1989.data_Shibata1989 as data
import numpy as np
import myokit


Q10_tau = 2.2 # [Wang1993]


#
# Activation [Shibata1989]
#
shibata_act_desc = """
    Steady-state activation curve for ito in human
    atrial myocytes [Shibata1989] cf Fig 3a.

    The activation curve was obtained by holding cells at
    a potential of -60 mV and applying 15 ms depolarising prepulses
    at 0.05 Hz to potentials ranging from -30 to +80 mV. The second
    pulse was to -40 mV for 100 ms. The activation curve was constructed by
    normalizing the amplitude of the tail current at each voltage to
    the largest tail current.
    """
vsteps_act, act, sd_act = data.Act_Shibata()
variances_act = [sd_**2 for sd_ in sd_act]
shibata_act_dataset = np.asarray([vsteps_act, act, variances_act])

shibata_act_protocol = myokit.Protocol()
for v in vsteps_act:
    shibata_act_protocol.add_step(-60, 20000) # tpre at vhold
    shibata_act_protocol.add_step(v, 15)      # test pulse
    shibata_act_protocol.add_step(-40, 100)   # measure pulse

shibata_conditions = {'phys.T': 295.15,  # K
                      'k_conc.K_i': 150, # mM
                      'k_conc.K_o': 4.5
                     }

def shibata_act_sum_stats(data):
    output = []
    for d in data.split_periodic(20115, adjust=True, closed_intervals=False):
        d = d.trim_left(20015, adjust=True)
        act_gate = d['ito.g']
        output = output + [max(act_gate, key=abs)]
    norm = max(output)
    for i in range(len(output)):
        output[i] /= norm
    return output

shibata_act = Experiment(
    dataset=shibata_act_dataset,
    protocol=shibata_act_protocol,
    conditions=shibata_conditions,
    sum_stats=shibata_act_sum_stats,
    description=shibata_act_desc,
    Q10=None,
    Q10_factor=0.)
