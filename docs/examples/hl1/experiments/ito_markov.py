import data.ito.data_ito as data
import numpy as np
from ionchannelABC.experiment import Experiment
import myokit
import warnings
import scipy.optimize as so


room_temp = 295
Q10_cond = 1.5
Q10_tau = 2.79

def temperature_adjust_cond(R0, T0, T1, Q10):
    return R0*Q10**((T1-T0)/10)

def temperature_adjust_tau(R0, T0, T1, Q10):
    return R0*Q10**((T0-T1)/10)


### Yang 2005
yang_iv_desc = """Peak, steady-state current density and inactivation
time constants at voltage steps for HL-1 from Yang 2005 Figure 8B.
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

vsteps_taui, taui, _ = data.TauInact_Yang()
max_taui = np.max(taui)
taui = [ti/max_taui for ti in taui]

yang_iv_dataset = [np.asarray([vsteps_iv, peaks, variances_pk]),
                   np.asarray([vsteps_iv, ss, variances_ss]),
                   np.asarray([vsteps_taui, taui, [0.]*len(taui)])]

yang_iv_protocol = myokit.pacing.steptrain(
    vsteps_iv, -70, 5000., 450.
)

yang_conditions = {'membrane.K_o': 4000,
                   'membrane.K_i': 145000,
                   'membrane.T': room_temp}

def yang_iv_sum_stats(data):
    out_pk = []
    out_ss = []
    out_taui = []
    def single_exp(t, tau, A):
        return A*np.exp(-t/tau)
    for i, d in enumerate(data.split_periodic(5450, adjust=True)):
        d = d.trim(5000, 5450, adjust=True)
        peak_curr = np.max(d['ito.i_to'])
        ss_curr = d['ito.i_to'][-1]
        out_pk = out_pk + [peak_curr/max_peak]
        out_ss = out_ss + [ss_curr/max_ss]

        # Only time constants for higher voltage steps
        if i >= len(vsteps_iv)-5:
            time = d['environment.time']
            curr = d['ito.i_to']

            # Set time zero to peak current
            index = np.argmax(np.abs(curr))
            curr = curr[index:]
            time = time[index:]
            t0 = time[0]
            time = [t-t0 for t in time]

            with warnings.catch_warnings():
                warnings.simplefilter('error', so.OptimizeWarning)
                warnings.simplefilter('error', RuntimeWarning)
                try:
                    imax = max(curr, key=abs)
                    curr = [c_/imax for c_ in curr]
                    if len(time)<=1 or len(curr)<=1:
                        raise Exception('failed simulation')
                    popt, _ = so.curve_fit(single_exp, time, curr,
                                           p0=[10, 1],
                                           bounds=([0., -np.inf],
                                                   [100., np.inf]))
                    taui = popt[0]
                    out_taui = out_taui + [taui/max_taui]
                except:
                    out_taui = out_taui + [float('inf')]
    return out_pk + out_ss + out_taui

yang_iv = Experiment(
    dataset=yang_iv_dataset,
    protocol=yang_iv_protocol,
    conditions=yang_conditions,
    sum_stats=yang_iv_sum_stats,
    description=yang_iv_desc
)
