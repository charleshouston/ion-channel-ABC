import numpy as np
import pandas as pd
import myokit
from ionchannelABC.experiment import Experiment


ap_desc = """Action potential and calcium transient characteristics
from paced whole cell simulation. 80pA/pF for 0.5ms at 1Hz for 100s.
"""

ap_protocol = myokit.pacing.blocktrain(
    period=1000, duration=0.5, limit=101
)

ap_conditions = {'extra.Ca_o': 1.8e3,
                 'extra.K_o' : 4.0e3,
                 'extra.Na_o': 130e3,
                 'phys.T'    : 295}

# Ion channel pyABC runs
pre_optimised = ['ina/hl1-ina.db',
                 'ical/hl1-ical.db',
                 'icat/hl1-icat.db',
                 'iha/hl1-iha.db',
                 'ik1/hl1-ik1.db',
                 'ikr/hl1-ikr.db',
                 'ito/hl1-ito.db',
                 'ikss/hl1-ikss.db']
pre_optimised = ['sqlite:////storage/hhecm/cellrotor/chouston/'+path
                 for path in pre_optimised]

def ap_sum_stats(data):
    output = []
    d = data.trim_left(1000*100, adjust=True)
    v = d['membrane.V']
    CaT = d['calcium.Ca_i']

ap = Experiment(
    dataset=ap_dataset,
    protocol=ap_protocol,
    conditions=ap_conditions,
    sum_stats=ap_sum_stats,
    description=ap_desc
)
