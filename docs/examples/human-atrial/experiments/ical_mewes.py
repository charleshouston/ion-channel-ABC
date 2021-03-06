"""
Experiments from [Mewes1994]
Charles Houston 2019-10-17
"""
from ionchannelABC.experiment import Experiment
import data.ical.Mewes1994.data_Mewes1994 as data
import numpy as np
import myokit


Q10_cond = 1.6      # [Li1997]
Q10_tau_act = 1.7   # [Li1997]
Q10_tau_inact = 1.3 # [Li1997]


#
# Activation [Mewes1994]
#
mewes_act_desc = """
    Steady-state activation curve for ical in human
    atrial myocytes [Mewes1994] cf. Fig5c.

    ICa was measured as the inward peak current with reference to the
    current level at the end of the test pulse.

    Current voltage relationships were determined by clamping for 450 ms
    from the holding potential (-40 mV) to different test potentials
    between -30 and +40 mV. Activation plots were generated by dividing
    peak ICa measured at a particular potential by the ion-driving force.
    """
vsteps_act, act, sd_act = data.Act_Mewes()
variances_act = [sd_**2 for sd_ in sd_act]
mewes_act_dataset = np.asarray([vsteps_act, act, variances_act])

mewes_act_protocol = myokit.pacing.steptrain(
    vsteps_act, -40, 10000, 450)

mewes_conditions = {'phys.T': 295.15,    # K
                    'ca_conc.Ca_o': 1.8 # mM
                   }

def mewes_act_sum_stats(data):
    output = []
    for d in data.split_periodic(10450, adjust=True,
            closed_intervals=False):
        d = d.trim_left(10000, adjust=True)
        act_gate = d['ical.g']
        output = output + [max(act_gate, key=abs)]
    norm = max(output)
    if norm > 0:
        for i in range(len(output)):
            output[i] /= norm
    else:
        for i in range(len(output)):
            output[i] = norm
    return output

mewes_act = Experiment(
    dataset=mewes_act_dataset,
    protocol=mewes_act_protocol,
    conditions=mewes_conditions,
    sum_stats=mewes_act_sum_stats,
    description=mewes_act_desc,
    Q10=None,
    Q10_factor=0.)
