import data.ik1.data_ik1 as data
import numpy as np
import scipy.optimize as so
import warnings
import myokit
from ionchannelABC.experiment import Experiment


room_temp = 295
Q10_cond = 1.5 # Correa 1991
Q10_tau = 2.79 # ten Tusscher 2004

def temperature_adjust_cond(R0, T0, T1, Q10):
    return R0*Q10**((T1-T0)/10)

def temperature_adjust_tau(R0, T0, T1, Q10):
    return R0*Q10**((T0-T1)/10)


### IV curve Goldoni 2010
goldoni_iv_desc = """IV curve for ik1 in HL-1 cells from Goldoni 2010.
Experiments carried out at room temperature.
"""

vsteps_iv, peaks, _ = data.IV_Goldoni()
max_iv_peak = np.max(np.abs(peaks))
peaks = [p/max_iv_peak for p in peaks]
goldoni_iv_dataset = np.asarray([vsteps_iv, peaks, [0.]*len(peaks)])

goldoni_iv_protocol = myokit.pacing.steptrain_linear(
    -150, 40, 10, -50, 5000., 100
)

goldoni_conditions = {'membrane.K_o': 120e3,
                      'membrane.K_i': 140e3,
                      'membrane.T': room_temp}

def goldoni_iv_sum_stats(data):
    out_peak = []
    for d in data.split_periodic(5100, adjust=True):
        d = d.trim(5000, 5100, adjust=True)
        out_peak = out_peak + [max(d['ik1.i_k1'], key=abs)/max_iv_peak]
    return out_peak

goldoni_iv = Experiment(
    dataset=goldoni_iv_dataset,
    protocol=goldoni_iv_protocol,
    conditions=goldoni_conditions,
    sum_stats=goldoni_iv_sum_stats,
    description=goldoni_iv_desc
)
