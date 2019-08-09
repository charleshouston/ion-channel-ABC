import data.ik1.data_ik1 as data
import numpy as np
import scipy.optimize as so
import warnings
import myokit
from ionchannelABC.experiment import Experiment


room_temp = 296
Q10_cond = 1.5 # [Kiyosue1993]


#
# IV curve [Goldoni2010]
#
goldoni_iv_desc = """IV curve for ik1 in HL-1 cells from Goldoni 2010.
Experiments carried out at room temperature.
"""

vsteps_iv, peaks, _ = data.IV_Goldoni()
goldoni_iv_dataset = np.asarray([vsteps_iv, peaks, [0.]*len(peaks)])

goldoni_iv_protocol = myokit.pacing.steptrain_linear(
    -150, 40, 10, -50, 5000., 100
)

goldoni_conditions = {'extra.K_o': 120e3,
                      'potassium.K_i': 140e3,
                      'phys.T': room_temp}

def goldoni_iv_sum_stats(data):
    out_peak = []
    for d in data.split_periodic(5100, adjust=True):
        d = d.trim(5000, 5100, adjust=True)
        out_peak = out_peak + [max(d['ik1.i_k1'], key=abs)]
    return out_peak

goldoni_iv = Experiment(
    dataset=goldoni_iv_dataset,
    protocol=goldoni_iv_protocol,
    conditions=goldoni_conditions,
    sum_stats=goldoni_iv_sum_stats,
    description=goldoni_iv_desc,
    Q10=Q10_cond,
    Q10_factor=1
)
