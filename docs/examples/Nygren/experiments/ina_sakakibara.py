#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 15:38:28 2019

@author: brm18 (Benjamin Marchand)
edited: Charles Houston 2019-09-19
"""

from ionchannelABC.experiment import Experiment

import data.ina.Sakakibara1992.data_Sakakibara1992 as data
from ionchannelABC.protocol import availability_linear, availability, recovery
from custom_protocols import varying_test_duration_double_pulse
import numpy as np
import myokit
import warnings
from scipy.optimize import OptimizeWarning
import scipy.optimize as so

Q10_tau = 2.79 # [tenTusscher2004]
Q10_cond = 1.5 # [Correa1991]

fit_threshold = 0.9


#
# IV curve [Sakakibara1992]
#
sakakibara_iv_desc ="""
    describes the protocol used to measure the IV peak-current curve in the Sakakibara Paper figure 1B

    page 3 of the paper : The holding potential (Vh) was -140 mV. Depolarizing pulses in 10-mVsteps were applied from
     -70 to +20mV. Testpulses were applied at 0.1 Hz.

    protocol used in figure 1B: single test pulse at a frequency of 0.1Hz : every 10s, the voltage step occurs.
    """
vsteps, peaks, _ = data.IV_Sakakibara()
sakakibara_iv_dataset = np.asarray([vsteps, peaks, [0.,]*len(vsteps)])

tpre = 10000
tstep = 1000
Vhold = -140 # mV
Vlower = -100+1e-5 # modified to not go through V = 0 that makes the nygren model crash
dV = 10
Vupper = 20+1e-5
sakakibara_iv_protocol = myokit.pacing.steptrain_linear(
    Vlower, Vupper+dV, dV, Vhold, tpre, tstep)

sakakibara_conditions = {'na_conc.Na_o': 5, # mM
                         'na_conc.Na_i': 5, # mM
                         'phys.T': 290.15}  # K

def sakakibara_iv_sum_stats(data):
    output = []
    for d in data.split_periodic(11000, adjust=True):
        d = d.trim_left(10000, adjust=True)
        current = d['ina.i_Na']
        output = output+[max(current, key=abs)]
    return output

sakakibara_iv = Experiment(
    dataset=sakakibara_iv_dataset,
    protocol=sakakibara_iv_protocol,
    conditions=sakakibara_conditions,
    sum_stats=sakakibara_iv_sum_stats,
    description=sakakibara_iv_desc,
    Q10=Q10_cond,
    Q10_factor=1)

#
# IV curves for varying extracellular Sodium [Sakakibara1992]
#
sakakibara_iv_Nao2_desc ="""
    describes the protocol used to measure the differents IV-curves in the Sakakibara Paper (figure 3A)
    this protocol is for measuring the curve with Nao = 2mM

    page 5 of the paper :
    Test pulses were applied at 0.1 Hz
    """
Na_o = 2
vsteps, peaks, _ = data.IV_Sakakibara_fig3A_all(Na_o)
sakakibara_iv_Nao2_dataset = np.asarray([vsteps, peaks, [0.,]*len(vsteps)])

tpre = 10000 # ms
tstep = 1000 # ms (not precised in the paper, 1s seems enough to mesure the peak current)

Vhold = -140 # mV
Vlower = -100+1e-5 # modified to not go through V = 0 that makes the nygren model crash ( expression of I_na)
dV = 10
Vupper = 40+1e-5
sakakibara_iv_Nao2_protocol = myokit.pacing.steptrain_linear(
    Vlower, Vupper+dV, dV, Vhold, tpre, tstep)

sakakibara_iv_Nao2_conditions = {'na_conc.Na_o': Na_o,
                                 'na_conc.Na_i': 5,
                                 'phys.T': 290.15}
sakakibara_iv_Nao2 = Experiment(
    dataset=sakakibara_iv_Nao2_dataset,
    protocol=sakakibara_iv_Nao2_protocol,
    conditions=sakakibara_iv_Nao2_conditions,
    sum_stats=sakakibara_iv_sum_stats,
    description=sakakibara_iv_Nao2_desc,
    Q10=Q10_cond,
    Q10_factor=1)


sakakibara_iv_Nao5_desc ="""
    describes the protocol used to measure the differents IV-curves in the Sakakibara Paper (figure 3A)
    this protocol is for measuring the curve with Nao = 5mM

    page 5 of the paper :
    Test pulses were applied at 0.1 Hz
    """
Na_o = 5
vsteps, peaks, _ = data.IV_Sakakibara_fig3A_all(Na_o)
sakakibara_iv_Nao5_dataset = np.asarray([vsteps, peaks, [0.,]*len(vsteps)])
sakakibara_iv_Nao5_conditions = {'na_conc.Na_o': Na_o,
                                 'na_conc.Na_i': 5,
                                 'phys.T': 290.15}
sakakibara_iv_Nao5 = Experiment(
    dataset=sakakibara_iv_Nao5_dataset,
    protocol=sakakibara_iv_Nao2_protocol,
    conditions=sakakibara_iv_Nao5_conditions,
    sum_stats=sakakibara_iv_sum_stats,
    description=sakakibara_iv_Nao5_desc,
    Q10=Q10_cond,
    Q10_factor=1)


sakakibara_iv_Nao20_desc ="""
    describes the protocol used to measure the differents IV-curves in the Sakakibara Paper (figure 3A)
    this protocol is for measuring the curve with Nao = 20mM

    page 5 of the paper :
    Test pulses were applied at 0.1 Hz
    """
Na_o = 20
vsteps, peaks, _ = data.IV_Sakakibara_fig3A_all(Na_o)
sakakibara_iv_Nao20_dataset = np.asarray([vsteps, peaks, [0.,]*len(vsteps)])
sakakibara_iv_Nao20_conditions = {'na_conc.Na_o': Na_o,
                                  'na_conc.Na_i': 5,
                                  'phys.T': 290.15}
sakakibara_iv_Nao20 = Experiment(
    dataset=sakakibara_iv_Nao20_dataset,
    protocol=sakakibara_iv_Nao2_protocol,
    conditions=sakakibara_iv_Nao20_conditions,
    sum_stats=sakakibara_iv_sum_stats,
    description=sakakibara_iv_Nao20_desc,
    Q10=Q10_cond,
    Q10_factor=1)


#
# Activation [Sakakibara1992]
#
sakakibara_act_desc = """
    describes the protocol used to measure the activation curve in the Sakakibara Paper (figure 2)

    The protocol is not described !
    By hypothesis, this will be the same as the protocol used in figure 1B :
    single test pulse at a frequency of 0.1Hz
    """
vsteps_act, act, sd_act = data.Act_Sakakibara()
variances_act = [(sd_)**2 for sd_ in sd_act]
sakakibara_act_dataset = np.asarray([vsteps_act, act, variances_act])

def sakakibara_act_sum_stats(data):
    output = []
    for d in data.split_periodic(11000, adjust=True):
        d = d.trim_left(10000, adjust=True)
        act_gate = d['ina.g']
        output = output+[max(act_gate, key=abs)]
    norm = max(output)
    for i in range(len(output)):
        output[i] /= norm
    return output

sakakibara_act = Experiment(
    dataset=sakakibara_act_dataset,
    protocol=sakakibara_iv_protocol,
    conditions=sakakibara_conditions,
    sum_stats=sakakibara_act_sum_stats,
    description=sakakibara_act_desc,
    Q10=None,
    Q10_factor=0)


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
# Inactivation kinetics [Sakakibara1992]
#
sakakibara_inact_kin_2_desc = """
    describes the protocol used to measure the inactivation kinetics in the Sakakibara Paper (figure 6)

    page 7 of the paper :
    Inactivation was induced by increasing
    the conditioning pulse duration. Conditioning
    pulses were applied to -80 and -100 mV Pulse
    frequency was 0.1 Hz. Current magnitude during
    the test pulse was normalized to levels obtained in
    the absence of a conditioning pulse.

    The protocol is a double pulse protocol at the frequency of 0.1Hz
    """
sakakibara_inact_kin_80_desc = sakakibara_inact_kin_2_desc + "\n V=-80mV"
sakakibara_inact_kin_100_desc = sakakibara_inact_kin_2_desc + "\n V=-100mV"

timecourse80, inact_timecourse_80, _ = data.Time_course_Inactivation_Sakakibara_all(-80)
timecourse100, inact_timecourse_100, _ = data.Time_course_Inactivation_Sakakibara_all(-100)
sakakibara_inact_kin_80_dataset = np.array(
    [timecourse80, inact_timecourse_80, [0.,]*len(timecourse80)])
sakakibara_inact_kin_100_dataset = np.array(
    [timecourse100, inact_timecourse_100, [0.,]*len(timecourse100)])

tpre = 10000 # ms
twait = 0
ttest = 30
vhold = -140 # mV
vtest = -20
tsteps80  = [0.,] + timecourse80  # add 0. for 'absence of conditioning pulse'
tsteps100 = [0.,] + timecourse100 # add 0. for 'absence of conditioning pulse'
sakakibara_inact_kin_80_protocol = varying_test_duration_double_pulse(
    -80, vhold, vtest, tpre, tsteps80, twait, ttest)
sakakibara_inact_kin_100_protocol = varying_test_duration_double_pulse(
    -100, vhold, vtest, tpre, tsteps100, twait, ttest)

tsplits80_kin = [t+ttest+twait+tpre for t in tsteps80]
for i in range(len(tsplits80_kin)-1):
    tsplits80_kin[i+1] += tsplits80_kin[i]
tsplits100_kin = [t+ttest+twait+tpre for t in tsteps100]
for i in range(len(tsplits100_kin)-1):
    tsplits100_kin[i+1] += tsplits100_kin[i]

def sakakibara_inact_kin_sum_stats_wrapper(data, tsplits):
    output = []
    for t in tsplits:
        d, data = data.split(t)
        d = d.trim_left(t-ttest, adjust=True)
        current = d['ina.i_Na']
        output = output + [max(current, key=abs)]
    norm = output[0]
    for i in range(len(output)):
        output[i] /= norm
    return output[1:] # first element is 'absence of conditioning pulse'

def sakakibara_inact_kin_80_sum_stats(data):
    return sakakibara_inact_kin_sum_stats_wrapper(data, tsplits80_kin)
def sakakibara_inact_kin_100_sum_stats(data):
    return sakakibara_inact_kin_sum_stats_wrapper(data, tsplits100_kin)

sakakibara_inact_kin_80 = Experiment(
    dataset=sakakibara_inact_kin_80_dataset,
    protocol=sakakibara_inact_kin_80_protocol,
    conditions=sakakibara_conditions,
    sum_stats=sakakibara_inact_kin_80_sum_stats,
    description=sakakibara_inact_kin_80_desc,
    Q10=None,
    Q10_factor=0
)
sakakibara_inact_kin_100 = Experiment(
    dataset=sakakibara_inact_kin_100_dataset,
    protocol=sakakibara_inact_kin_100_protocol,
    conditions=sakakibara_conditions,
    sum_stats=sakakibara_inact_kin_100_sum_stats,
    description=sakakibara_inact_kin_100_desc,
    Q10=None,
    Q10_factor=0
)


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
    """
sakakibara_rec140_desc = sakakibara_rec_desc + "\n Vhold = -140mV"
sakakibara_rec120_desc = sakakibara_rec_desc + "\n Vhold = -120mV"
sakakibara_rec100_desc = sakakibara_rec_desc + "\n Vhold = -100mV"

twaits140, rec140, _ = data.Recovery_Sakakibara_all(-140)
sakakibara_rec140_dataset = np.array(
    [twaits140, rec140, [0.,]*len(twaits140)])
twaits120, rec120, _ = data.Recovery_Sakakibara_all(-120)
sakakibara_rec120_dataset = np.array(
    [twaits120, rec120, [0.,]*len(twaits120)])
twaits100, rec100, _ = data.Recovery_Sakakibara_all(-100)
sakakibara_rec100_dataset = np.array(
    [twaits100, rec100, [0.,]*len(twaits100)])

tpre = 10000 # ms
tstep1 = 1000
tstep2 = 1000
vstep1 = -20
vstep2 = -20

sakakibara_rec140_protocol = recovery(
    twaits140, -140, vstep1, vstep2, tpre, tstep1, tstep2)
sakakibara_rec120_protocol = recovery(
    twaits120, -120, vstep1, vstep2, tpre, tstep1, tstep2)
sakakibara_rec100_protocol = recovery(
    twaits100, -100, vstep1, vstep2, tpre, tstep1, tstep2)

tsplits140 = [t+tpre+tstep1+tstep2 for t in twaits140]
for i in range(len(tsplits140)-1):
    tsplits140[i+1] += tsplits140[i]
tsplits120 = [t+tpre+tstep1+tstep2 for t in twaits120]
for i in range(len(tsplits120)-1):
    tsplits120[i+1] += tsplits120[i]
tsplits100 = [t+tpre+tstep1+tstep2 for t in twaits100]
for i in range(len(tsplits100)-1):
    tsplits100[i+1] += tsplits100[i]

def sakakibara_rec_sum_stats_wrapper(data, tsplits):
    output = []
    timename = 'engine.time'
    for t in tsplits:
        d, data = data.split(t)
        step1 = d.trim(d[timename][0]+tpre,
                       d[timename][0]+tpre+tstep1,
                       adjust=True)
        step2 = d.trim_left(t-tstep2, adjust=True)
        max1 = max(step1['ina.i_Na'], key=abs)
        max2 = max(step2['ina.i_Na'], key=abs)
        try:
            output = output + [max2/max1]
        except:
            output = output + [float('inf')]
    return output

def sakakibara_rec140_sum_stats(data):
    return sakakibara_rec_sum_stats_wrapper(data, tsplits140)
def sakakibara_rec120_sum_stats(data):
    return sakakibara_rec_sum_stats_wrapper(data, tsplits120)
def sakakibara_rec100_sum_stats(data):
    return sakakibara_rec_sum_stats_wrapper(data, tsplits100)

sakakibara_rec140 = Experiment(
    dataset=sakakibara_rec140_dataset,
    protocol=sakakibara_rec140_protocol,
    conditions=sakakibara_conditions,
    sum_stats=sakakibara_rec140_sum_stats,
    description=sakakibara_rec140_desc,
    Q10=None,
    Q10_factor=0)
sakakibara_rec120 = Experiment(
    dataset=sakakibara_rec120_dataset,
    protocol=sakakibara_rec120_protocol,
    conditions=sakakibara_conditions,
    sum_stats=sakakibara_rec120_sum_stats,
    description=sakakibara_rec120_desc,
    Q10=None,
    Q10_factor=0)
sakakibara_rec100 = Experiment(
    dataset=sakakibara_rec100_dataset,
    protocol=sakakibara_rec100_protocol,
    conditions=sakakibara_conditions,
    sum_stats=sakakibara_rec100_sum_stats,
    description=sakakibara_rec100_desc,
    Q10=None,
    Q10_factor=0)
