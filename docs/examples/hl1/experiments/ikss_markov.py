import data.ikss.data_ikss as data
import numpy as np
from ionchannelABC.experiment import Experiment
from ionchannelABC.protocol import availability
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


# Lu 2016 IV curve
lu_iv_desc = """Current density of i_Kss in HL-1 from Lu 2016
Figure 3A. Measurements taken at 35C (308K) and adjusted for
room temperature.
"""

vsteps_iv, peaks, sd_iv = data.IV_Lu()
peaks = [temperature_adjust_cond(pk, 308, room_temp, Q10_cond)
         for pk in peaks]
max_peak = np.max(peaks)
peaks = [pk/max_peak for pk in peaks]
variances_iv = [(sd_/max_peak)**2 for sd_ in sd_iv]
lu_iv_dataset = np.asarray([vsteps_iv, peaks, variances_iv])

lu_iv_protocol = myokit.Protocol()
time_ = 0.
vhold = -80.
thold = 5000.
vpre = -40.
tpre = 30.
tstep = 300.
for v in vsteps_iv:
    lu_iv_protocol.schedule(vhold, time_, thold)
    time_ += thold
    lu_iv_protocol.schedule(vpre, time_, tpre)
    time_ += tpre
    lu_iv_protocol.schedule(v, time_, tstep)
    time_ += tstep

lu_conditions = {'membrane.K_o': 2.68e3, # normal Tyrode's
                 'membrane.K_i': 130e3,
                 'membrane.T': room_temp}

def lu_iv_sum_stats(data):
    output = []
    for d in data.split_periodic(5330, adjust=True):
        d = d.trim(5030, 5330, adjust=True)
        peak_curr = max(d['ikss.i_Kss'], key=abs)
        output = output + [peak_curr/max_peak]
    return output

lu_iv = Experiment(
    dataset=lu_iv_dataset,
    protocol=lu_iv_protocol,
    conditions=lu_conditions,
    sum_stats=lu_iv_sum_stats,
    description=lu_iv_desc
)

### Xu 1999 activation time constants
xu_ta_desc = """Activation time constants recorded in mouse
ventricular myocytes. Used in the absence of reliable data for
HL-1 as ito in mouse atrial myocytes is similar to ito_f in
mouse ventricular myocytes [Xu1999] and HL-1 are derived from
mouse atrial myocytes.

Experiments recorded at room temperature.
"""

vsteps_ta, tau_a, sd_ta = data.ActTau_Xu()
max_taua = np.max(tau_a)
tau_a = [ta/max_taua for ta in tau_a]
variances_ta = [(sd_/max_taua)**2 for sd_ in sd_ta]
xu_ta_dataset = np.asarray([vsteps_ta, tau_a, variances_ta])

xu_ta_protocol = myokit.pacing.steptrain(
    vsteps_ta, -70, 15000., 4500.
)

xu_conditions = {'membrane.K_o': 4e3,
                 'membrane.K_i': 135e3,
                 'membrane.T': room_temp}

def xu_ta_sum_stats(data):
    output = []
    def single_exp(t, tau, A):
        return A*(1-np.exp(-t/tau))
    for d in data.split_periodic(19500, adjust=True):
        d = d.trim(15000, 19500, adjust=True)
        curr = d['ikss.i_Kss']
        time = d['environment.time']

        # Get rising phase of current
        index = np.argmax(np.abs(curr))
        curr = curr[:index+1]
        time = time[:index+1]
        with warnings.catch_warnings():
            warnings.simplefilter('error', so.OptimizeWarning)
            warnings.simplefilter('error', RuntimeWarning)
            try:
                if len(time) <= 1 or len(curr) <= 1:
                    raise Exception('failed simulation')
                imax = max(curr, key=abs)
                curr = [c_/imax for c_ in curr]
                popt, _ = so.curve_fit(single_exp, time, curr,
                                       p0=[50., 1.],
                                       bounds=([0., -np.inf],
                                               [10000., np.inf]))
                taua = popt[0]
                output = output + [taua/max_taua]
            except:
                output = output + [float('inf')]
    return output

xu_taua = Experiment(
    dataset=xu_ta_dataset,
    protocol=xu_ta_protocol,
    conditions=xu_conditions,
    sum_stats=xu_ta_sum_stats,
    description=xu_ta_desc
)
