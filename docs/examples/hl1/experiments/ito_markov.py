import data.ito.data_ito as data
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


### Lu 2016 IV curve
lu_iv_desc = """Current density of ito in HL-1 from Lu 2016 Figure 3A.
Measurements taken at 35C (308K) and adjusted for room temperature.
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
        peak_curr = max(d['ito.i_to'], key=abs)
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
        curr = d['ito.i_to']
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
                                       p0=[10., 1.],
                                       bounds=([0., -np.inf],
                                               [1000., np.inf]))
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


# Xu 1999 steady-state inactivation
xu_inact_desc = """Steady-state inactivation for ito_f in mouse
ventricular cells from Xu 1999 Figure 9.

Used in the absence of reliable data for
HL-1 as ito in mouse atrial myocytes is similar to ito_f in
mouse ventricular myocytes [Xu1999] and HL-1 are derived from
mouse atrial myocytes.

Experiments recorded at room temperature.
"""

vsteps_inact, inact, sd_inact = data.Inact_Xu()
variances_inact = [sd_**2 for sd_ in sd_inact]
xu_inact_dataset = np.asarray([vsteps_inact, inact, variances_inact])

xu_inact_protocol = availability(
    vsteps_inact, -70., 50., 15000., 5000., 0., 5000.
)

def xu_inact_sum_stats(data):
    output = []
    for d in data.split_periodic(25000., adjust=True):
        d = d.trim(20000., 25000., adjust=True)
        current = d['ito.i_to']
        output = output + [max(current, key=abs)]
    for i in range(1, len(output)):
        output[i] = output[i]/output[0]
    output[0] = 1.
    return output

xu_inact = Experiment(
    dataset=xu_inact_dataset,
    protocol=xu_inact_protocol,
    conditions=xu_conditions,
    sum_stats=xu_inact_sum_stats,
    description=xu_inact_desc
)


### Yang 2005
yang_iv_desc = """Peak, steady-state current density and inactivation
time constants at voltage steps for HL-1 from Yang 2005 Figure 8B.
Measurements taken at room temperature.
"""

vsteps_iv, peaks, sd_pk = data.Peak_Yang()
#max_peak = np.max(peaks)
#peaks = [pk/max_peak for pk in peaks]
#variances_pk = [(sd_/max_peak)**2 for sd_ in sd_pk]

_, ss, sd_ss = data.SS_Yang()
#max_ss = np.max(ss)
#ss = [ss_/max_ss for ss_ in ss]
#variances_ss = [(sd_/max_ss)**2 for sd_ in sd_ss]

peaks = [pk-ss for (pk,ss) in zip(peaks,ss)]
sd_iv = [np.sqrt(sd_pk_i**2+sd_ss_i**2) for (sd_pk_i,sd_ss_i) in zip(sd_pk,sd_ss)]
max_peak = np.max(peaks)
peaks = [pk/max_peak for pk in peaks]
variances_iv = [(sd_/max_peak)**2 for sd_ in sd_iv]

vsteps_taui, taui, _ = data.TauInact_Yang()
max_taui = np.max(taui)
taui = [ti/max_taui for ti in taui]

yang_iv_dataset = [np.asarray([vsteps_iv, peaks, variances_iv]),
                   #np.asarray([vsteps_iv, ss, variances_ss]),
                   np.asarray([vsteps_taui, taui, [0.]*len(taui)])]

yang_iv_protocol = myokit.pacing.steptrain(
    vsteps_iv, -70, 5000., 450.
)

yang_conditions = {'membrane.K_o': 4000,
                   'membrane.K_i': 145000,
                   'membrane.T': room_temp}

def yang_iv_sum_stats(data):
    out_pk = []
    #out_ss = []
    out_taui = []
    def single_exp(t, tau, A):
        return A*np.exp(-t/tau)
    for i, d in enumerate(data.split_periodic(5450, adjust=True)):
        d = d.trim(5000, 5450, adjust=True)
        peak_curr = np.max(d['ito.i_to'])
        #ss_curr = d['ito.i_to'][-1]
        out_pk = out_pk + [peak_curr/max_peak]
        #out_ss = out_ss + [ss_curr/max_ss]

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
    return out_pk + out_taui
    #return out_pk + out_ss + out_taui

yang_iv = Experiment(
    dataset=yang_iv_dataset,
    protocol=yang_iv_protocol,
    conditions=yang_conditions,
    sum_stats=yang_iv_sum_stats,
    description=yang_iv_desc
)
