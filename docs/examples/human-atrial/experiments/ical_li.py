#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 17:42:47 2019

@author: brm18
Updated: Charles Houston 2019-10-02
"""

from ionchannelABC.experiment import Experiment
import data.ical.Li1997.data_Li1997 as data
from ionchannelABC.protocol import availability_linear, recovery
import numpy as np
import myokit
import warnings
from scipy.optimize import OptimizeWarning
import scipy.optimize as so


Q10_cond = 1.6      # [Li1997]
Q10_tau_act = 1.7   # [Li1997]
Q10_tau_inact = 1.3 # [Li1997]

fit_threshold = 0.9


#
# IV curve [Li1997]
#
li_iv_desc ="""
    describes the protocol used to measure the IV peak-current curve in the Li Paper figure 1C

    page 2 of the paper : I-V relations of Ica were determined using 300-ms depolarizing steps every 10s from HP of -80,-60, and -40 mV
    The magnitude was measured as the difference between the peak inward current and the steady state current at the end of the depolarizing step

    protocol used in figure 1C: single test pulse at a frequency of 0.1Hz : every 10s, the voltage step occurs.
    """
li_iv_80_desc = li_iv_desc + "\n HP = -80mV"
li_iv_60_desc = li_iv_desc + "\n HP = -60mV"
li_iv_40_desc = li_iv_desc + "\n HP = -40mV"

vsteps_iv_80, peaks_iv_80, sd_iv_80 = data.IV_Li_all(-80)
variances_iv_80 =  [(sd_)**2 for sd_ in sd_iv_80]
li_iv_80_dataset = np.asarray([vsteps_iv_80, peaks_iv_80, variances_iv_80])
vsteps_iv_60, peaks_iv_60, sd_iv_60 = data.IV_Li_all(-60)
variances_iv_60 =  [(sd_)**2 for sd_ in sd_iv_60]
li_iv_60_dataset = np.asarray([vsteps_iv_60, peaks_iv_60, variances_iv_60])
vsteps_iv_40, peaks_iv_40, sd_iv_40 = data.IV_Li_all(-40)
variances_iv_40 =  [(sd_)**2 for sd_ in sd_iv_40]
li_iv_40_dataset = np.asarray([vsteps_iv_40, peaks_iv_40, variances_iv_40])

maxpeak = max(peaks_iv_80, key=abs)
peaks_iv_80_norm = [p/abs(maxpeak) for p in peaks_iv_80]
sd_iv_80_norm = [sd_/abs(maxpeak) for sd_ in sd_iv_80]
variances_iv_80_norm =  [(sd_)**2 for sd_ in sd_iv_80_norm]
li_iv_80_norm_dataset = np.asarray([vsteps_iv_80, peaks_iv_80_norm, variances_iv_80_norm])

tpre = 10000 # ms
tstep = 300
dv = 10

li_iv_80_protocol = myokit.pacing.steptrain_linear(
    min(vsteps_iv_80), max(vsteps_iv_80)+dv, dv, -80, tpre, tstep)
li_iv_60_protocol = myokit.pacing.steptrain_linear(
    min(vsteps_iv_60), max(vsteps_iv_60)+dv, dv, -60, tpre, tstep)
li_iv_40_protocol = myokit.pacing.steptrain_linear(
    min(vsteps_iv_40), max(vsteps_iv_40)+dv, dv, -40, tpre, tstep)

li_conditions = {'phys.T': 309.15, # K
                 'ca_conc.Ca_o': 2.0}   # mM

def li_iv_sum_stats(data):
    output = []
    for d in data.split_periodic(10300, adjust=True, closed_intervals=False):
        d = d.trim_left(10000, adjust=True)
        current = d['ical.i_CaL'][:-1] # the last value for some models can be NaN
        # magnitude measured as difference between peak and steady-state at end of step
        output = output + [max(current, key=abs)-current[-1]]
    return output
def li_iv_norm_sum_stats(data):
    output = li_iv_sum_stats(data)
    norm = max(output, key=abs)
    for i in range(len(output)):
        output[i] /= abs(norm)
    return output


li_iv_80 = Experiment(
    dataset=li_iv_80_dataset,
    protocol=li_iv_80_protocol,
    conditions=li_conditions,
    sum_stats=li_iv_sum_stats,
    description=li_iv_80_desc,
    Q10=Q10_cond,
    Q10_factor=1)
li_iv_80_norm = Experiment(
    dataset=li_iv_80_norm_dataset,
    protocol=li_iv_80_protocol,
    conditions=li_conditions,
    sum_stats=li_iv_norm_sum_stats,
    description=li_iv_80_desc,
    Q10=None,
    Q10_factor=0)
li_iv_60 = Experiment(
    dataset=li_iv_60_dataset,
    protocol=li_iv_60_protocol,
    conditions=li_conditions,
    sum_stats=li_iv_sum_stats,
    description=li_iv_60_desc,
    Q10=Q10_cond,
    Q10_factor=1)
li_iv_40 = Experiment(
    dataset=li_iv_40_dataset,
    protocol=li_iv_40_protocol,
    conditions=li_conditions,
    sum_stats=li_iv_sum_stats,
    description=li_iv_40_desc,
    Q10=Q10_cond,
    Q10_factor=1)


#
# Activation [Li1997]
#
li_act_desc = """
    the protocol used to measure the activation curve in the Li Paper (figure 2B) is not decribed

    the protocol is assumed to be the same than for the IV curves

    single 300ms test pulse at a frequency of 0.1Hz
    """
vsteps_act, act, sd_act = data.Act_Li()
variances_act = [(sd_)**2 for sd_ in sd_act]
li_act_dataset = np.asarray([vsteps_act, act, variances_act])

vstep_tau_a, tau_a, sd_tau_a = data.Act_Tau_Li()
variances_tau_a = [(sd_)**2 for sd_ in sd_tau_a]
li_act_tau_dataset = np.asarray([vstep_tau_a, tau_a, variances_tau_a])

tpre  = 10000 # ms
tstep = 300
vhold = -80 # mV
vlower = -80
dv = 10
vupper = 20+dv

li_act_protocol = myokit.pacing.steptrain(
    vsteps_act, vhold, tpre, tstep)

def li_act_sum_stats(data, ss=True, act_tau_10=False):
    def single_exp(t, tau, A):
        return 1-A*np.exp(-t/tau)
    output = []
    output_tau = []

    for d in data.split_periodic(10300, adjust=True, closed_intervals=False):
        d = d.trim_left(10000, adjust=True)
        act_gate = d['ical.g']
        if ss:
            output = output + [max(act_gate, key=abs)]

        if act_tau_10 and np.isclose(d['membrane.V'][0], 10.0):
            time = d['engine.time']
            index = np.argmax(act_gate)

            # trim anything after peak
            act_gate = act_gate[:index]
            time = time[:index]

            with warnings.catch_warnings():
                warnings.simplefilter('error', OptimizeWarning)
                warnings.simplefilter('error', RuntimeWarning)
                try:
                    norm = max(act_gate, key=abs)
                    act_gate = [a/norm for a in act_gate]
                    if len(time)<=1 or len(act_gate)<=1:
                        raise Exception('Failed simulation')
                    popt, _ = so.curve_fit(single_exp,
                                           time,
                                           act_gate,
                                           p0=[0.5, 1],
                                           bounds=(0., np.inf),
                                           max_nfev=1000)
                    fit = [single_exp(t,popt[0],popt[1]) for t in time]
                    # Calculate r2
                    ss_res = np.sum((np.array(act_gate)-np.array(fit))**2)
                    ss_tot = np.sum((np.array(act_gate)-np.mean(np.array(act_gate)))**2)
                    r2 = 1 - (ss_res / ss_tot)

                    tau = popt[0]

                    if r2 > fit_threshold:
                        output_tau = [tau]
                    else:
                        raise RuntimeWarning('scipy.optimize.curve_fit found a poor fit')
                except:
                    output_tau = [float('inf')]
    if ss:
        norm = max(output)
        for i in range(len(output)):
            output[i] /= norm
    return output+output_tau

def li_act_and_tau_sum_stats(data):
    return li_act_sum_stats(data, act_tau_10=True)

def li_act_tau_sum_stats(data):
    return li_act_sum_stats(data, ss=False, act_tau_10=True)

li_act = Experiment(
    dataset=li_act_dataset,
    protocol=li_act_protocol,
    conditions=li_conditions,
    sum_stats=li_act_sum_stats,
    description=li_act_desc,
    Q10=None,
    Q10_factor=0)

li_act_tau = Experiment(
    dataset=[li_act_tau_dataset],
    protocol=li_act_protocol,
    conditions=li_conditions,
    sum_stats=li_act_tau_sum_stats,
    description=li_act_desc,
    Q10=Q10_tau_act,
    Q10_factor=-1)

li_act_and_tau = Experiment(
    dataset=[li_act_dataset,
             li_act_tau_dataset],
    protocol=li_act_protocol,
    conditions=li_conditions,
    sum_stats=li_act_and_tau_sum_stats,
    description=li_act_desc,
    Q10=Q10_tau_act,
    Q10_factor=[0,-1])


#
# Inactivation [Li1997]
#
li_inact_desc = """
    describes the protocol used to measure the activation curve in the Li Paper (figure 2B)

    page 3 of the paper :
        Prepulses of varying duration (150, 300, or 1,000 ms)
        were applied to conditioning potentials between -70
        and +50 mV, and then i_ca, was recorded during a
        300-ms test pulse to + 10 mV. The inactivation variable
        ( f ) was determined as i_ca, at a given prepulse potential
        divided by the maximum i_ca, in the absence of a
        prepulse.

    The protocol is a double pulse protocol at the frequency of 0.1Hz
    """
li_inact_1000_desc = li_inact_desc + "\n prepulse=1000ms"
li_inact_300_desc = li_inact_desc + "\n prepulse=300ms"
li_inact_150_desc = li_inact_desc + "\n prepulse=150ms"

vsteps_inact, inact_1000, sd_inact_1000 = data.inact_Li_all(1000)
variances_inact_1000 = [(sd_)**2 for sd_ in sd_inact_1000]
li_inact_1000_dataset = np.asarray([vsteps_inact, inact_1000, variances_inact_1000])
vsteps_inact, inact_300, sd_inact_300 = data.inact_Li_all(300)
variances_inact_300 = [(sd_)**2 for sd_ in sd_inact_300]
li_inact_300_dataset = np.asarray([vsteps_inact, inact_300, variances_inact_300])
vsteps_inact, inact_150, sd_inact_150 = data.inact_Li_all(150)
variances_inact_150 = [(sd_)**2 for sd_ in sd_inact_150]
li_inact_150_dataset = np.asarray([vsteps_inact, inact_150, variances_inact_150])

tpre = 10000
twait = 0
ttest = 300
vhold = -80 # mV
vtest = 10
vlower = -80
dv = 10
vupper = 50+dv

li_inact_1000_protocol = availability_linear(
    vlower, vupper, dv, vhold, vtest, tpre, 1000, twait, ttest)
li_inact_300_protocol = availability_linear(
    vlower, vupper, dv, vhold, vtest, tpre, 300, twait, ttest)
li_inact_150_protocol = availability_linear(
    vlower, vupper, dv, vhold, vtest, tpre, 150, twait, ttest)

def li_inact_sum_stats(data, tstep):
    output = []
    for d in data.split_periodic(10300+tstep, adjust=True, closed_intervals=False):
        d = d.trim_left(10000+tstep, adjust=True)
        inact_gate = d['ical.g']
        output = output + [max(inact_gate, key=abs)]
    norm = output[0] # absence of prepulse
    for i in range(len(output)):
        output[i] /= norm
    # account for incomplete voltage-dependent inactivation
    # (only if fitting to Boltzmann function
    #fmin = min(output)
    #for i in range(len(output)):
    #    output[i] = (output[i]-fmin)/(1-fmin)
    return output

def li_inact_1000_sum_stats(data):
    return li_inact_sum_stats(data,1000)
def li_inact_300_sum_stats(data):
    return li_inact_sum_stats(data,300)
def li_inact_150_sum_stats(data):
    return li_inact_sum_stats(data,150)

li_inact_1000 = Experiment(
    dataset=li_inact_1000_dataset,
    protocol=li_inact_1000_protocol,
    conditions=li_conditions,
    sum_stats=li_inact_1000_sum_stats,
    description=li_inact_1000_desc,
    Q10=None,
    Q10_factor=0)
li_inact_300 = Experiment(
    dataset=li_inact_300_dataset,
    protocol=li_inact_300_protocol,
    conditions=li_conditions,
    sum_stats=li_inact_300_sum_stats,
    description=li_inact_300_desc,
    Q10=None,
    Q10_factor=0)
li_inact_150 = Experiment(
    dataset=li_inact_150_dataset,
    protocol=li_inact_150_protocol,
    conditions=li_conditions,
    sum_stats=li_inact_150_sum_stats,
    description=li_inact_150_desc,
    Q10=None,
    Q10_factor=0)


#
# Inactivation kinetics [Li1997]
#
li_inact_kin_desc = """
    describes the protocol used to measure the inactivation kinetics (tau_f and tau_s) in the Li Paper (figure 3B)

    the Voltage goes from -10mV to 30mV for this function with a dV = 10 mV.

    page 3 of the paper :
        The development of Ica inactivation was studied with the use of
        depolarizing pulses from various HP. Figure 3A shows a
        typical recording obtained during a 3OO-ms voltage step
        from -80 to + 10 mV. The raw data were well fitted by a
        biexponential relation with the time constants shown.
        At all voltages, inactivation was well fitted by a biexponential
        relation and poorly fitted by a monoexponential
        function.
    """
li_inact_kin_80_desc = li_inact_kin_desc + "\nHP=-80mV"
li_inact_kin_60_desc = li_inact_kin_desc + "\nHP=-60mV"
li_inact_kin_40_desc = li_inact_kin_desc + "\nHP=-40mV"


vsteps_th1, th1, sd_th1 = data.Tau1_Li_all(-80)
variances_th1 = [(sd_)**2 for sd_ in sd_th1]
li_inact_kin_80_tauf_dataset = np.asarray([vsteps_th1, th1, variances_th1])
vsteps_th2, th2, sd_th2 = data.Tau2_Li_all(-80)
variances_th2 = [(sd_)**2 for sd_ in sd_th2]
li_inact_kin_80_taus_dataset = np.asarray([vsteps_th2, th2, variances_th2])

vsteps_th1, th1, sd_th1 = data.Tau1_Li_all(-60)
variances_th1 = [(sd_)**2 for sd_ in sd_th1]
dataset1 = np.asarray([vsteps_th1, th1, variances_th1])
vsteps_th2, th2, sd_th2 = data.Tau2_Li_all(-60)
variances_th2 = [(sd_)**2 for sd_ in sd_th2]
dataset2 = np.asarray([vsteps_th2, th2, variances_th2])
li_inact_kin_60_dataset = [dataset1,dataset2]

vsteps_th1, th1, sd_th1 = data.Tau1_Li_all(-40)
variances_th1 = [(sd_)**2 for sd_ in sd_th1]
dataset1 = np.asarray([vsteps_th1, th1, variances_th1])
vsteps_th2, th2, sd_th2 = data.Tau2_Li_all(-40)
variances_th2 = [(sd_)**2 for sd_ in sd_th2]
dataset2 = np.asarray([vsteps_th2, th2, variances_th2])
li_inact_kin_40_dataset = [dataset1,dataset2]

tpre = 10000 # ms
tstep = 300
vlower = -10
dv = 10
vupper = 30+dv

li_inact_kin_80_protocol = myokit.pacing.steptrain_linear(
    vlower, vupper, dv, -80, tpre, tstep)
li_inact_kin_60_protocol = myokit.pacing.steptrain_linear(
    vlower, vupper, dv, -60, tpre, tstep)
li_inact_kin_40_protocol = myokit.pacing.steptrain_linear(
    vlower, vupper, dv, -40, tpre, tstep)

def li_inact_kin_sum_stats(data, fast=True, slow=True):
    def double_exp(t, tauh, taus, Ah, As, A0):
        return A0 + Ah*np.exp(-t/tauh) + As*np.exp(-t/taus)
    output_fast = []
    output_slow = []
    for d in data.split_periodic(10300, adjust=True, closed_intervals=False):
        d = d.trim_left(10000, adjust=True)

        current = d['ical.i_CaL'][:-1]
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
                                       p0=[10,80,0.5,0.5,0],
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
                    output_fast = output_fast+[tauh]
                    output_slow = output_slow+[taus]
                else:
                    raise RuntimeWarning('scipy.optimize.curve_fit found a poor fit')
            except:
                output_fast = output_slow+[float('inf')]
                output_slow = output_slow+[float('inf')]
    output = []
    if fast:
        output += output_fast
    if slow:
        output += output_slow
    return output

def li_inact_kin_tauf_sum_stats(data):
    return li_inact_kin_sum_stats(data, fast=True, slow=False)

def li_inact_kin_taus_sum_stats(data):
    return li_inact_kin_sum_stats(data, fast=False, slow=True)

li_inact_kin_80 = Experiment(
    dataset=[li_inact_kin_80_tauf_dataset,
             li_inact_kin_80_taus_dataset],
    protocol=li_inact_kin_80_protocol,
    conditions=li_conditions,
    sum_stats=li_inact_kin_sum_stats,
    description=li_inact_kin_80_desc,
    Q10=Q10_tau_inact,
    Q10_factor=-1)

li_inact_kin_taus_80 = Experiment(
    dataset=li_inact_kin_80_taus_dataset,
    protocol=li_inact_kin_80_protocol,
    conditions=li_conditions,
    sum_stats=li_inact_kin_taus_sum_stats,
    description=li_inact_kin_80_desc,
    Q10=Q10_tau_inact,
    Q10_factor=-1)
li_inact_kin_tauf_80 = Experiment(
    dataset=li_inact_kin_80_tauf_dataset,
    protocol=li_inact_kin_80_protocol,
    conditions=li_conditions,
    sum_stats=li_inact_kin_tauf_sum_stats,
    description=li_inact_kin_80_desc,
    Q10=Q10_tau_inact,
    Q10_factor=-1)

li_inact_kin_60 = Experiment(
    dataset=li_inact_kin_60_dataset,
    protocol=li_inact_kin_60_protocol,
    conditions=li_conditions,
    sum_stats=li_inact_kin_sum_stats,
    description=li_inact_kin_60_desc,
    Q10=Q10_tau_inact,
    Q10_factor=-1)
li_inact_kin_40 = Experiment(
    dataset=li_inact_kin_40_dataset,
    protocol=li_inact_kin_40_protocol,
    conditions=li_conditions,
    sum_stats=li_inact_kin_sum_stats,
    description=li_inact_kin_40_desc,
    Q10=Q10_tau_inact,
    Q10_factor=-1)


#
# Recovery [Li1997]
#
li_recov_desc =   """
    describes the protocol used to measure the Recovery of I_ca_L in the Li Paper (figure 4B)

    page 4 of the paper :
        The time dependence of recovery from Ic, inactivation was studied with
        the paired-pulse protocol illustrated in Fig. 4. Identical 300-ms pulses
        (P1 and P2) from the HP (-80, -60, or -40 mV) to +1O mV were
        delivered every 10 s, with varying PI-p2 intervals. The
        current during p2 (i2) relative to the current during P1
        (i1) was determined as a function of the PI-P2 recovery
        interval.

    Data is fitted to mono- (-80mV) or bi-exponential (-60,-40) function

    The protocol is a double pulse protocol at the frequency of 0.1Hz
    """
prepulses_recov_tauf, recov_tauf, sd_recov_tauf = data.TauF_Recov_Li()
variances_recov_tauf = [sd_**2 for sd_ in sd_recov_tauf]
li_recov_tauf_dataset = np.asarray([prepulses_recov_tauf, recov_tauf, variances_recov_tauf])

prepulses_recov_taus, recov_taus, sd_recov_taus = data.TauS_Recov_Li()
variances_recov_taus = [sd_**2 for sd_ in sd_recov_taus]
li_recov_taus_dataset = np.asarray([prepulses_recov_taus, recov_taus, variances_recov_taus])

tpre = 10000 # ms
tstep1 = 300
twaits_recov = [2**i for i in range(1,12)]
tstep2 = 300
vstep1 = 10
vstep2 = 10

tmp_protocols = []
for v in prepulses_recov_taus:
    tmp_protocols.append(
        recovery(twaits_recov, v, vstep1, vstep2, tpre, tstep1, tstep2)
    )
li_recov_protocol = tmp_protocols[0]
tsplit_recov = tmp_protocols[0].characteristic_time()
for p in tmp_protocols[1:]:
    for e in p.events():
        li_recov_protocol.add_step(e.level(), e.duration())

tsplits_recov = [t+tstep1+tstep2+tpre for t in twaits_recov]
for i in range(len(tsplits_recov)-1):
    tsplits_recov[i+1] += tsplits_recov[i]

def li_recov_sum_stats(data, fast=True, slow=True):
    def double_exp(t, tau_f, tau_s, Af, As, A0):
        return A0-Af*np.exp(-t/tau_f)-As*np.exp(-t/tau_s)
    def single_exp(t, tau_s, As, A0):
        return A0-As*np.exp(-t/tau_s)

    output1 = []
    output2 = []
    timename = 'engine.time'
    for i,d in enumerate(data.split_periodic(tsplit_recov, adjust=True, closed_intervals=False)):
        recov = []
        for t in tsplits_recov:
            d_, d = d.split(t)
            step1 = d_.trim(d_[timename][0]+10000,
                            d_[timename][0]+10300,
                            adjust=True)
            step2 = d_.trim_left(t-300, adjust=True)
            try:
                max1 = max(step1['ical.i_CaL'], key=abs)
                max2 = max(step2['ical.i_CaL'], key=abs)
                recov = recov + [max2/max1]
            except:
                recov = recov + [float('inf')]
        # Now fit output to single/double exponential
        with warnings.catch_warnings():
            warnings.simplefilter('error', so.OptimizeWarning)
            warnings.simplefilter('error', RuntimeWarning)
            if i > 0:
                try:
                    popt, _ = so.curve_fit(double_exp,
                                           twaits_recov,
                                           recov,
                                           p0=[50.,500.,0.5,0.5,0.],
                                           bounds=(0.,
                                                   [np.inf,np.inf,1.0,1.0,1.0]),
                                           max_nfev=1000)

                    fit = [double_exp(t,popt[0],popt[1],popt[2],popt[3],popt[4])
                           for t in twaits_recov]
                    # Calculate r2
                    ss_res = np.sum((np.array(recov)-np.array(fit))**2)
                    ss_tot = np.sum((np.array(recov)-np.mean(np.array(recov)))**2)
                    r2 = 1 - (ss_res / ss_tot)

                    tauf = min(popt[0],popt[1])
                    taus = max(popt[0],popt[1])
                    if r2 > fit_threshold:
                        output1 = output1+[tauf]
                        output2 = output2+[taus]
                    else:
                        raise RuntimeWarning('scipy.optimize.curve_fit found a poor fit')

                except:
                    output1 = output1+[float('inf')]
                    output2 = output2+[float('inf')]
            else:
                try:
                    popt, _ = so.curve_fit(single_exp,
                                           twaits_recov,
                                           recov,
                                           p0=[50.,0.5,0.],
                                           bounds=(0.,
                                                   [np.inf,1.0,1.0]),
                                           max_nfev=1000)

                    fit = [single_exp(t,popt[0],popt[1],popt[2])
                           for t in twaits_recov]
                    # Calculate r2
                    ss_res = np.sum((np.array(recov)-np.array(fit))**2)
                    ss_tot = np.sum((np.array(recov)-np.mean(np.array(recov)))**2)
                    r2 = 1 - (ss_res / ss_tot)

                    taus = popt[0]
                    if r2 > fit_threshold:
                        output2 = output2+[taus]
                    else:
                        raise RuntimeWarning('scipy.optimize.curve_fit found a poor fit')
                except:
                    output2 = output2+[float('inf')]
    output = []
    if fast:
        output += output1
    if slow:
        output += output2
    return output

def li_recov_tauf_sum_stats(data):
    return li_recov_sum_stats(data, fast=True, slow=False)

def li_recov_taus_sum_stats(data):
    return li_recov_sum_stats(data, fast=False, slow=True)

li_recov = Experiment(
    dataset=[li_recov_tauf_dataset,
             li_recov_taus_dataset],
    protocol=li_recov_protocol,
    conditions=li_conditions,
    sum_stats=li_recov_sum_stats,
    description=li_recov_desc,
    Q10=Q10_tau_inact,
    Q10_factor=-1)
li_recov_taus = Experiment(
    dataset=li_recov_taus_dataset,
    protocol=li_recov_protocol,
    conditions=li_conditions,
    sum_stats=li_recov_taus_sum_stats,
    description=li_recov_desc,
    Q10=Q10_tau_inact,
    Q10_factor=-1)
li_recov_tauf = Experiment(
    dataset=li_recov_tauf_dataset,
    protocol=li_recov_protocol,
    conditions=li_conditions,
    sum_stats=li_recov_tauf_sum_stats,
    description=li_recov_desc,
    Q10=Q10_tau_inact,
    Q10_factor=-1)


#
# Use-dependent inactivation [Li1997]
#
li_use_inact_desc =   """
    Use-dependent inactivation of I_Ca current from [Li1997] (Fig 5).

    Protocol is a pulse train from various holding potentials using
    300ms pulses to +10mV.

    Holding potentials were -80,-60,-40mV.
    """
pulse_freq_80, pulse_const_80, sd_pulse_const_80 = data.Use_Inact_Li_80()
variances_pulse_const_80 = [sd_**2 for sd_ in sd_pulse_const_80]
li_use_inact_80_dataset = np.array([pulse_freq_80, pulse_const_80, variances_pulse_const_80])
_, ss_decay_80, sd_ss_decay_80 = data.Use_Inact_Li_SS_80()
variances_ss_decay_80 = [sd_**2 for sd_ in sd_ss_decay_80]
li_use_inact_ss_80_dataset = np.array([pulse_freq_80, ss_decay_80, variances_ss_decay_80])

pulse_freq_60, pulse_const_60, sd_pulse_const_60 = data.Use_Inact_Li_60()
variances_pulse_const_60 = [sd_**2 for sd_ in sd_pulse_const_60]
li_use_inact_60_dataset = np.array([pulse_freq_60, pulse_const_60, variances_pulse_const_60])
_, ss_decay_60, sd_ss_decay_60 = data.Use_Inact_Li_SS_60()
variances_ss_decay_60 = [sd_**2 for sd_ in sd_ss_decay_60]
li_use_inact_ss_60_dataset = np.array([pulse_freq_60, ss_decay_60, variances_ss_decay_60])

pulse_freq_40, pulse_const_40, sd_pulse_const_40 = data.Use_Inact_Li_40()
variances_pulse_const_40 = [sd_**2 for sd_ in sd_pulse_const_40]
li_use_inact_40_dataset = np.array([pulse_freq_40, pulse_const_40, variances_pulse_const_40])
_, ss_decay_40, sd_ss_decay_40 = data.Use_Inact_Li_SS_40()
variances_ss_decay_40 = [sd_**2 for sd_ in sd_ss_decay_40]
li_use_inact_ss_40_dataset = np.array([pulse_freq_40, ss_decay_40, variances_ss_decay_40])

freq = pulse_freq_80
vpulse = 10 # mV
tpulse = 300 # ms
tpre = 60000
npulses = 15

tmp_protocols = []
for vhold in [-80,-60,-40]:
    for f in freq:
        p = myokit.Protocol()
        period = 1000./f # ms
        p.add_step(vhold, tpre)
        for n in range(npulses):
            p.add_step(vpulse, tpulse)
            p.add_step(vhold, period-tpulse)
        tmp_protocols.append(p)
li_use_inact_protocol = tmp_protocols[0]
tsplits = [tmp_protocols[0].characteristic_time(),
           tmp_protocols[1].characteristic_time(),
           tmp_protocols[2].characteristic_time()]

for p in tmp_protocols[1:]:
    for e in p.events():
        li_use_inact_protocol.add_step(e.level(), e.duration())

def li_use_inact_sum_stats(data):
    def single_exp(n, k, D):
        return (1-D)+D*np.exp(-k*n)

    output1 = []
    output2 = []

    # split separate vholds
    for dvhold in data.split_periodic(sum(tsplits), adjust=True, closed_intervals=False):
        for i, f in enumerate(freq):
            pulses = [] # to hold the fitting data

            period = 1000./f

            dtrain, dvhold = dvhold.split(sum(tsplits[:i+1]))
            t0 = dtrain['engine.time'][0]
            dtrain['engine.time'] = [t-t0 for t in dtrain['engine.time']]
            dtrain = dtrain.trim_left(tpre, adjust=True)
            for d in dtrain.split_periodic(period, adjust=True, closed_intervals=False):
                d = d.trim_right(period-tpulse)
                current = d['ical.i_CaL']
                pulses = pulses + [max(current, key=abs)]

            # fit to exponential equation
            with warnings.catch_warnings():
                warnings.simplefilter('error', so.OptimizeWarning)
                warnings.simplefilter('error', RuntimeWarning)
                try:
                    # normalise to first pulse
                    norm = pulses[0]
                    for i in range(len(pulses)):
                        pulses[i] /= norm
                    popt, _ = so.curve_fit(single_exp,
                                           list(range(npulses)),
                                           pulses,
                                           p0 = [0.2, 0.2],
                                           bounds = (0.,
                                                     [np.inf, 1.0]),
                                           max_nfev=1000)

                    fit = [single_exp(n, popt[0], popt[1]) for n in range(npulses)]
                    # Calculate r2
                    ss_res = np.sum((np.array(pulses)-np.array(fit))**2)
                    ss_tot = np.sum((np.array(pulses)-np.mean(np.array(pulses)))**2)
                    r2 = 1 - (ss_res / ss_tot)
                    pulse_const = 1./popt[0]
                    ss_decay = 1.-popt[1]
                    if r2 > fit_threshold:
                        output1 = output1+[pulse_const]
                        output2 = output2+[ss_decay]
                    else:
                        raise RuntimeWarning('scipy.optimize.curve_fit found a poor fit')
                except:
                    output1 = output1 + [float('inf')]
                    output2 = output2 + [float('inf')]
    return output1+output2

li_use_inact = Experiment(
    dataset=[li_use_inact_80_dataset,
             li_use_inact_60_dataset,
             li_use_inact_40_dataset,
             li_use_inact_ss_80_dataset,
             li_use_inact_ss_60_dataset,
             li_use_inact_ss_40_dataset],
    protocol=li_use_inact_protocol,
    conditions=li_conditions,
    sum_stats=li_use_inact_sum_stats,
    description=li_use_inact_desc,
    Q10=None,
    Q10_factor=0.)
