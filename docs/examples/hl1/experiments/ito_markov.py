import data.ito.data_ito as data
import numpy as np
from ionchannelABC.experiment import Experiment
import myokit


room_temp = 295
Q10_cond = 1.5
Q10_tau = 2.79

def temperature_adjust_cond(R0, T0, T1, Q10):
    return R0*Q10**((T1-T0)/10)

def temperature_adjust_tau(R0, T0, T1, Q10):
    return R0*Q10**((T0-T1)/10)


### IV curve Yang 2005
yang_iv_desc = """Peak and steady-state current density at voltage steps for HL-1 from Yang 2005 Figure 8B.
Measurements taken at room temperature.
"""

vsteps_iv, peaks, sd_pk = data.IV_Yang()
max_peak = np.max(peaks)
peaks = [pk/max_peak for pk in peaks]
variances_pk = [(sd_/max_peak)**2 for sd_ in sd_pk]

_, ss, sd_ss = data.SS_Yang()
max_ss = np.max(ss)
ss = [ss_/max_ss for ss_ in ss]
variances_ss = [(sd_/max_ss)**2 for sd_ in sd_ss]

yang_iv_dataset = [np.asarray([vsteps_iv, peaks, variances_pk]),
                   np.asarray([vsteps_iv, ss, variances_ss])]

yang_iv_protocol = myokit.pacing.steptrain(
    vsteps_iv, -70, 5000., 450.
)

yang_conditions = {'membrane.K_o': 4000,
                   'membrane.K_i': 145000,
                   'membrane.T': room_temp}

def yang_iv_sum_stats(data):
    out_pk = []
    out_ss = []
    for d in data.split_periodic(5450, adjust=True):
        d = d.trim(5000, 5450, adjust=True)
        peak_curr = np.max(d['ito.i_to'])
        ss_curr = d['ito.i_to'][-1]
        out_pk = out_pk + [peak_curr/max_peak]
        out_ss = out_ss + [ss_curr/max_ss]
    return out_pk + out_ss

yang_iv = Experiment(
    dataset=yang_iv_dataset,
    protocol=yang_iv_protocol,
    conditions=yang_conditions,
    sum_stats=yang_iv_sum_stats,
    description=yang_iv_desc
)
