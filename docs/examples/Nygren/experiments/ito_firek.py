"""
Experiments from [Firek1995]
Charles Houston 2019-10-18
"""
from ionchannelABC.experiment import Experiment
from ionchannelABC.protocol import availability
import data.ito.Firek1995.data_Firek1995 as data
import numpy as np
import myokit


Q10_tau = 2.2 # [Wang1993]


#
# Inactivation [Firek1995]
#
firek_inact_desc = """
    Steady-state inactivation curve for ito in human
    atrial myocytes [Firek1995] cf Fig 3c.

    A two-pulse protocol was applied to obtain the steady-state
    inactivation relationships. P1 depolarized the cell to selected
    voltages from the holding potential (-80 mV). P2 was of fixed
    amplitude and duration; it depolarized the cell to 0 mV to activate
    the outward currents.
    """
vsteps_inact, inact, sd_inact = data.Inact_Firek()
variances_inact = [sd_**2 for sd_ in sd_inact]
firek_inact_dataset = np.asarray([vsteps_inact, inact, variances_inact])

firek_inact_protocol = availability(
    vsteps_inact, -80, 0, 20000, 400, 0, 400)

firek_conditions = {'phys.T': 306.15,  # K
                    'k_conc.K_i': 140, # mM
                    'k_conc.K_o': 5.4
                   }

def firek_inact_sum_stats(data):
    output = []
    for d in data.split_periodic(20800, adjust=True, closed_intervals=False):
        d = d.trim_left(20400, adjust=True)
        inact_gate = d['ito.g']
        output = output + [max(inact_gate, key=abs)]
    norm = max(output)
    for i in range(len(output)):
        output[i] /= norm
    return output

firek_inact = Experiment(
    dataset=firek_inact_dataset,
    protocol=firek_inact_protocol,
    conditions=firek_conditions,
    sum_stats=firek_inact_sum_stats,
    description=firek_inact_desc,
    Q10=None,
    Q10_factor=0.)
