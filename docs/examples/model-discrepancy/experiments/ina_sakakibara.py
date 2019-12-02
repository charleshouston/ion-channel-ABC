import numpy as np
import warnings
from scipy.optimize import OptimizeWarning
import scipy.optimize as so
import myokit

from ionchannelABC.experiment import Experiment
import data.ina.Sakakibara1992.data_Sakakibara1992 as data
from ionchannelABC.protocol import availability_linear, recovery


# Q10 temperature adjustment factors
Q10_tau = 2.79 # [tenTusscher2004]
Q10_cond = 1.5 # [Correa1991]

# Threshold for goodness-of-fit of exponential curves
fit_threshold = 0.9


#
# Inactivation [Sakakibara1992]
#
sakakibara_inact_desc = """
    describes the protocol used to measure the activation curve in the Sakakibara Paper (figure 2)

    page 7 of the paper :
    The voltage dependence of h, was studied using a double-pulse protocol consisting of a
    1-second conditioning pulse from holding a potential of-140 mV 0.1 Hz (inset at lower left).
    Current amplitude elicited during the test pulse was normalized to that in absence of a conditioning pulse.

    The protocol is a double pulse protocol at the frequency of 0.1Hz
    """
vsteps_inact, inact, sd_inact = data.Inact_Sakakibara()
variances_inact = [(sd_)**2 for sd_ in sd_inact]
sakakibara_inact_dataset = np.asarray([vsteps_inact, inact, variances_inact])
nygren_inact_shift = 32.2 # mV
vsteps_inact_shifted = [v+nygren_inact_shift for v in vsteps_inact]
sakakibara_inact_shifted_dataset = np.asarray([vsteps_inact_shifted, inact, variances_inact])

tpre = 10000 # ms
tstep = 1000
twait = 0
ttest = 30

Vhold = -140 # mV
Vtest = -20
Vlower = -140
dV = 10
Vupper = -30

sakakibara_inact_protocol = availability_linear(
    Vlower, Vupper, dV, Vhold, Vtest, tpre, tstep, twait, ttest)
sakakibara_inact_shifted_protocol = availability_linear(
    Vlower+nygren_inact_shift,
    Vupper+nygren_inact_shift, dV,
    Vhold+nygren_inact_shift,
    Vtest+nygren_inact_shift, tpre, tstep, twait, ttest)

sakakibara_conditions = {'na_conc.Na_o': 5, # mM
                         'na_conc.Na_i': 5, # mM
                         'phys.T': 290.15}  # K

def sakakibara_inact_sum_stats(data):
    output = []
    for d in data.split_periodic(11030, adjust=True):
        d = d.trim_left(11000, adjust = True)
        inact_gate = d['ina.g']
        output = output+[max(inact_gate, key=abs)]
    norm = max(output)
    try:
        for i in range(len(output)):
            output[i] /= norm
    except:
        for i in range(len(output)):
            output[i] = float('inf')
    return output

sakakibara_inact = Experiment(
    dataset=sakakibara_inact_dataset,
    protocol=sakakibara_inact_protocol,
    conditions=sakakibara_conditions,
    sum_stats=sakakibara_inact_sum_stats,
    description=sakakibara_inact_desc,
    Q10=None,
    Q10_factor=0)

sakakibara_inact_shifted = Experiment(
    dataset=sakakibara_inact_shifted_dataset,
    protocol=sakakibara_inact_shifted_protocol,
    conditions=sakakibara_conditions,
    sum_stats=sakakibara_inact_sum_stats,
    description=sakakibara_inact_desc,
    Q10=None,
    Q10_factor=0)


#
# Inactivation kinetics [Sakakibara1992]
#
sakakibara_inact_kin_desc =   """
    describes the protocol used to measure the inactivation kinetics (tau_f and tau_s) in the Sakakibara Paper (figure 5B)

    the Voltage goes from -50mV to -20mV for this function with a dV = 10 mV.

    page 5 of the paper :
    Figure 5A shows INa elicited at holding potentials of -140 to -40 mV (top)
    and -20 mV (bottom).

    single test pulse at a frequency of 1Hz (since the step is a 100 msec test pulse)
    """
# Fast inactivation kinetics
vsteps_th1, th1, sd_th1 = data.TauF_Inactivation_Sakakibara()
variances_th1 = [(sd_)**2 for sd_ in sd_th1]
sakakibara_inact_kin_fast_dataset = np.asarray([vsteps_th1, th1, variances_th1])
# Slow inactivation kinetics
vsteps_th2, th2, sd_th2 = data.TauS_Inactivation_Sakakibara()
variances_th2 = [(sd_)**2 for sd_ in sd_th2]
sakakibara_inact_kin_slow_dataset = np.asarray([vsteps_th2, th2, variances_th2])

sakakibara_inact_kin_dataset = [sakakibara_inact_kin_fast_dataset,
                                sakakibara_inact_kin_slow_dataset]

tstep = 100 # ms
tpre = 10000 # before the first pulse occurs
Vhold = -140 # mV
Vlower = -50
dV = 10
Vupper = -20+dV
sakakibara_inact_kin_protocol = myokit.pacing.steptrain_linear(
    Vlower, Vupper, dV, Vhold, tpre, tstep)

def sakakibara_inact_kin_sum_stats(data, fast=True, slow=True):
    def double_exp(t, tauh, taus, Ah, As, A0):
        return Ah*np.exp(-t/tauh) + As*np.exp(-t/taus) + A0

    output_fast = []
    output_slow =  []
    for d in data.split_periodic(10100, adjust=True):
        d = d.trim_left(10000, adjust=True)

        current = d['ina.i_Na'][:-1]
        time = d['engine.time'][:-1]
        index = np.argmax(np.abs(current))

        # Set time zero to peak current
        current = current[index:]
        time = time[index:]
        t0 = time[0]
        time = [t-t0 for t in time]

        with warnings.catch_warnings():
            warnings.simplefilter('error', OptimizeWarning)
            warnings.simplefilter('error', RuntimeWarning)
            try:
                current = [c/current[0] for c in current]
                if len(time)<=1 or len(current)<=1:
                    raise Exception('Failed simulation')
                popt, _ = so.curve_fit(double_exp,
                                       time,
                                       current,
                                       p0=[2,20,0.9,0.1,0],
                                       bounds=(0.,
                                               [np.inf, np.inf, 1.0, 1.0, 1.0]),
                                       max_nfev=1000)
                fit = [double_exp(t,popt[0],popt[1],popt[2],popt[3],popt[4]) for t in time]
                # Calculate r2
                ss_res = np.sum((np.array(current)-np.array(fit))**2)
                ss_tot = np.sum((np.array(current)-np.mean(np.array(current)))**2)
                r2 = 1 - (ss_res / ss_tot)

                tauh = min(popt[0],popt[1])
                taus = max(popt[0],popt[1])

                if r2 > fit_threshold:
                    if fast:
                        output_fast = output_fast+[tauh]
                    if slow:
                        output_slow = output_slow+[taus]
                else:
                    raise RuntimeWarning('scipy.optimize.curve_fit found a poor fit')
            except:
                if fast:
                    output_fast = output_slow+[float('inf')]
                if slow:
                    output_slow = output_slow+[float('inf')]
    output = output_fast+output_slow
    return output

def sakakibara_inact_kin_fast_sum_stats(data):
    return sakakibara_inact_kin_sum_stats(data, fast=True, slow=False)

def sakakibara_inact_kin_slow_sum_stats(data):
    return sakakibara_inact_kin_sum_stats(data, fast=False, slow=True)

sakakibara_inact_kin = Experiment(
    dataset=sakakibara_inact_kin_dataset,
    protocol=sakakibara_inact_kin_protocol,
    conditions=sakakibara_conditions,
    sum_stats=sakakibara_inact_kin_sum_stats,
    description=sakakibara_inact_kin_desc,
    Q10=Q10_tau,
    Q10_factor=-1)
sakakibara_inact_kin_fast = Experiment(
    dataset=sakakibara_inact_kin_fast_dataset,
    protocol=sakakibara_inact_kin_protocol,
    conditions=sakakibara_conditions,
    sum_stats=sakakibara_inact_kin_fast_sum_stats,
    description=sakakibara_inact_kin_desc,
    Q10=Q10_tau,
    Q10_factor=-1)
sakakibara_inact_kin_slow = Experiment(
    dataset=sakakibara_inact_kin_slow_dataset,
    protocol=sakakibara_inact_kin_protocol,
    conditions=sakakibara_conditions,
    sum_stats=sakakibara_inact_kin_slow_sum_stats,
    description=sakakibara_inact_kin_desc,
    Q10=Q10_tau,
    Q10_factor=-1)


#
# Recovery [Sakakibara1992]
#
sakakibara_rec_desc =    """
    describes the protocol used to measure the Recovery of I_na in the Sakakibara Paper (figure 8A)

    page 8 of the paper :
    The double-pulseprotocol shown in the inset was applied at various recovery potentials at a frequency of 0.1 Hz. The magnitude of the fast
    Na+ current during the test pulse was normalized to that
    induced by the conditioning pulse.


    The protocol is a double pulse protocol at the frequency of 0.1Hz
    with differing wait potentials.
"""
prepulse_rec, rec_tauf, sd_rec_tauf = data.TauF_Recovery()
variances_rec_tauf = [sd_**2 for sd_ in sd_rec_tauf]
sakakibara_rec_tauf_dataset = np.array(
    [prepulse_rec, rec_tauf, variances_rec_tauf])

prepulse_rec, rec_taus, sd_rec_taus = data.TauS_Recovery()
variances_rec_taus = [sd_**2 for sd_ in sd_rec_taus]
sakakibara_rec_taus_dataset = np.array(
    [prepulse_rec, rec_taus, variances_rec_taus])

tpre = 10000 # ms
tstep1 = 1000
twaits_rec = [2**i for i in range(1,11)]
tstep2 = 1000
vstep1 = -20
vstep2 = -20
vhold = -140

tmp_protocols = []
for v in prepulse_rec:
    tmp_protocols.append(
        recovery(twaits_rec,vhold,vstep1,vstep2,tpre,tstep1,tstep2,v)
    )
sakakibara_rec_protocol = tmp_protocols[0]
tsplit_rec = tmp_protocols[0].characteristic_time()
for p in tmp_protocols[1:]:
    for e in p.events():
        sakakibara_rec_protocol.add_step(e.level(), e.duration())

tsplits_rec = [t+tstep1+tstep2+tpre for t in twaits_rec]
for i in range(len(tsplits_rec)-1):
    tsplits_rec[i+1] += tsplits_rec[i]

def sakakibara_rec_sum_stats(data, fast=True, slow=True):
    def double_exp(t, tau_r1, tau_r2, A0, A1, A2):
        return A0-A1*np.exp(-t/tau_r1)-A2*np.exp(-t/tau_r2)
    output1 = []
    output2 = []
    timename = 'engine.time'
    for i, d in enumerate(data.split_periodic(tsplit_rec, adjust=True, closed_intervals=False)):
        recov = []
        for t in tsplits_rec:
            d_, d = d.split(t)
            step1 = d_.trim(d_[timename][0]+10000,
                            d_[timename][0]+10000+1000,
                            adjust=True)
            step2 = d_.trim_left(t-1000, adjust=True)
            try:
                max1 = max(step1['ina.i_Na'], key=abs)
                max2 = max(step2['ina.i_Na'], key=abs)
                recov = recov + [max2/max1]
            except:
                recov = recov + [float('inf')]

        # Now fit output to double exponential
        with warnings.catch_warnings():
            warnings.simplefilter('error', so.OptimizeWarning)
            warnings.simplefilter('error', RuntimeWarning)
            try:
                popt, _ = so.curve_fit(double_exp,
                                       twaits_rec,
                                       recov,
                                       p0=[1.,10.,0.9,0.1,0.],
                                       bounds=(0.,
                                               [100,1000,1.0,1.0,1.0]),
                                       max_nfev=1000)

                fit = [double_exp(t,popt[0],popt[1],popt[2],popt[3],popt[4])
                       for t in twaits_rec]

                # Calculate r2
                ss_res = np.sum((np.array(recov)-np.array(fit))**2)
                ss_tot = np.sum((np.array(recov)-np.mean(np.array(recov)))**2)
                r2 = 1 - (ss_res / ss_tot)

                tauf = min(popt[0],popt[1])
                taus = max(popt[0],popt[1])
                if r2 > fit_threshold:
                    if fast:
                        output1 = output1+[tauf]
                    if slow:
                        output2 = output2+[taus]
                else:
                    raise RuntimeWarning('scipy.optimize.curve_fit found a poor fit')
            except:
                if fast:
                    output1 = output1+[float('inf')]
                if slow:
                    output2 = output2+[float('inf')]
    output = output1+output2
    return output

def sakakibara_rec_fast_sum_stats(data):
    return sakakibara_rec_sum_stats(data, fast=True, slow=False)
def sakakibara_rec_slow_sum_stats(data):
    return sakakibara_rec_sum_stats(data, fast=False, slow=True)

sakakibara_rec = Experiment(
    dataset=[sakakibara_rec_tauf_dataset,
             sakakibara_rec_taus_dataset],
    protocol=sakakibara_rec_protocol,
    conditions=sakakibara_conditions,
    sum_stats=sakakibara_rec_sum_stats,
    description=sakakibara_rec_desc,
    Q10=Q10_tau,
    Q10_factor=-1)
sakakibara_rec_fast = Experiment(
    dataset=sakakibara_rec_tauf_dataset,
    protocol=sakakibara_rec_protocol,
    conditions=sakakibara_conditions,
    sum_stats=sakakibara_rec_fast_sum_stats,
    description=sakakibara_rec_desc,
    Q10=Q10_tau,
    Q10_factor=-1)
sakakibara_rec_slow = Experiment(
    dataset=sakakibara_rec_taus_dataset,
    protocol=sakakibara_rec_protocol,
    conditions=sakakibara_conditions,
    sum_stats=sakakibara_rec_slow_sum_stats,
    description=sakakibara_rec_desc,
    Q10=Q10_tau,
    Q10_factor=-1)
