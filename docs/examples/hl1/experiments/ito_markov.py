import data.ito.data_ito as data
import numpy as np
from ionchannelABC.experiment import Experiment
from ionchannelABC.protocol import availability
import myokit
import warnings
import scipy.optimize as so


# Q10 [Kiyosue1993] from iK currents (may not be VALID!)
Q10_cond = 4.4
Q10_tau = 4.1
room_temp = 296


#
# IV curve [Yang2005]
#
yang_iv_desc = """Peak, steady-state current density and inactivation
time constants at voltage steps for HL-1 from Yang 2005 Figure 8B.

Assuming peak-steadystate current is the ito (steady state is ikss).

Measurements taken at room temperature.
"""

vsteps_iv, peaks, sd_pk = data.Peak_Yang()
_, ss, sd_ss = data.SS_Yang()
peaks = [pk-ss for (pk,ss) in zip(peaks,ss)]
sd_iv = [np.sqrt(sd_pk_i**2+sd_ss_i**2) for (sd_pk_i,sd_ss_i) in zip(sd_pk,sd_ss)]
variances_iv = [sd_**2 for sd_ in sd_iv]

yang_iv_dataset = np.asarray([vsteps_iv, peaks, variances_iv])

yang_iv_protocol = myokit.pacing.steptrain(
    vsteps_iv, -70, 5000., 450.
)

yang_conditions = {'extra.K_o': 4e3,
                   'potassium.K_i': 145e3,
                   'phys.T': room_temp}

def yang_iv_sum_stats(data):
    out_pk = []
    for i, d in enumerate(data.split_periodic(5450, adjust=True)):
        d = d.trim(5000, 5450, adjust=True)
        peak_curr = np.max(d['ito.i_to'])
        out_pk = out_pk + [peak_curr]
    return out_pk

yang_iv = Experiment(
    dataset=yang_iv_dataset,
    protocol=yang_iv_protocol,
    conditions=yang_conditions,
    sum_stats=yang_iv_sum_stats,
    description=yang_iv_desc,
    Q10=Q10_cond,
    Q10_factor=1
)


#
# IV curve [Lu2016]
#
lu_iv_desc = """Current density of ito in HL-1 from Lu 2016 Figure 3A.
Measurements taken at 35C (308K).
"""

vsteps_iv, peaks, sd_iv = data.IV_Lu()
variances_iv = [sd_**2 for sd_ in sd_iv]
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

lu_conditions = {'extra.K_o': 4e3, # normal Tyrode's
                 'potassium.K_i': 130e3,
                 'phys.T': 308}

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
    description=lu_iv_desc,
    Q10=Q10_cond,
    Q10_factor=1
)


#
# Activation kinetics [Xu1999]
#
xu_ta_desc = """Activation time constants recorded in mouse
ventricular myocytes. Used in the absence of reliable data for
HL-1 as ito in mouse atrial myocytes is similar to ito_f in
mouse ventricular myocytes [Xu1999] and HL-1 are derived from
mouse atrial myocytes.

Experiments recorded at room temperature.
"""

vsteps_ta, tau_a, sd_ta = data.ActTau_Xu()
variances_ta = [sd_**2 for sd_ in sd_ta]
xu_ta_dataset = np.asarray([vsteps_ta, tau_a, variances_ta])

xu_ta_protocol = myokit.pacing.steptrain(
    vsteps_ta, -70, 15000., 4500.
)

xu_conditions = {'extra.K_o': 4e3,
                 'potassium.K_i': 135e3,
                 'phys.T': room_temp}

def xu_ta_sum_stats(data):
    output = []
    def single_exp(t, tau, A):
        return A*(1-np.exp(-t/tau))
    for d in data.split_periodic(19500, adjust=True):
        d = d.trim(15000, 19500, adjust=True)
        curr = d['ito.i_to']
        time = d['engine.time']

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
                                               [np.inf, np.inf]))
                taua = popt[0]
                output = output + [taua]#/max_taua]
            except:
                output = output + [float('inf')]
    return output

xu_taua = Experiment(
    dataset=xu_ta_dataset,
    protocol=xu_ta_protocol,
    conditions=xu_conditions,
    sum_stats=xu_ta_sum_stats,
    description=xu_ta_desc,
    Q10=Q10_tau,
    Q10_factor=-1
)


#
# Steady-state inactivation [Xu1999]
#
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
    description=xu_inact_desc,
    Q10=None,
    Q10_factor=0
)
