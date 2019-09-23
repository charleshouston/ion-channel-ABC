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


#
# IV curve [Sakakibara1992]
#
#sakakibara_iv_name = "IV"
sakakibara_iv_desc ="""
    describes the protocol used to measure the IV peak-current curve in the Sakakibara Paper figure 1B

    page 3 of the paper : The holding potential (Vh) was -140 mV. Depolarizing pulses in 10-mVsteps were applied from
     -70 to +20mV. Testpulses were applied at 0.1 Hz.

    protocol used in figure 1B: single test pulse at a frequency of 0.1Hz : every 10s, the voltage step occurs.
    """
vsteps, peaks, _ = data.IV_Sakakibara()
sakakibara_iv_dataset = np.asarray([vsteps, peaks, [0.,]*len(vsteps)])

tperiod_iv_saka = 10000 # ms
tstep = 1000 # ms (not precised in the paper, 1s seems enough to mesure the peak current)
tpreMeasuring_iv_saka = tperiod_iv_saka - tstep # before the measurement
Vhold = -140 # mV
Vlower = -100.001 # modified to not go through V = 0 that makes the nygren model crash ( expression of I_na)
dV = 10
Vupper = 20.001+dV
sakakibara_iv_protocol = myokit.pacing.steptrain_linear(
    Vlower,Vupper, dV, Vhold, tpreMeasuring_iv_saka, tstep)

sakakibara_conditions = {'na_conc.Na_o': 5, # mM
                         'na_conc.Na_i': 5, # mM
                         'phys.T': 290.15}  # K

def sakakibara_iv_sum_stats(data):
    output = []
    for d in data.split_periodic(tperiod_iv_saka, adjust=True):
        d = d.trim_left(tpreMeasuring_iv_saka, adjust=True)
        current = d['ina.i_Na']
        output = output+[max(current, key=abs)]
    return output

sakakibara_iv = Experiment(
    #name = sakakibara_iv_name,
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
#sakakibara_iv_Nao2_name = "IV  [Nao] = 2mM"
sakakibara_iv_Nao2_desc ="""
    describes the protocol used to measure the differents IV-curves in the Sakakibara Paper (figure 3A)
    this protocol is for measuring the curve with Nao = 2mM

    page 5 of the paper :
    Test pulses were applied at 0.1 Hz
    """
Na_o = 2
vsteps, peaks, _ = data.IV_Sakakibara_fig3A_all(Na_o)
sakakibara_iv_Nao2_dataset = np.asarray([vsteps, peaks, [0.,]*len(vsteps)])

tperiod_iv_Nao2_saka = 10000 # ms
tstep = 1000 # ms (not precised in the paper, 1s seems enough to mesure the peak current)
tpreMeasuring_iv_Nao2_saka = tperiod_iv_Nao2_saka - tstep # before the measurement

Vhold = -140 # mV
Vlower = -100.001 # modified to not go through V = 0 that makes the nygren model crash ( expression of I_na)
dV = 10
Vupper = 40+dV
sakakibara_iv_Nao2_protocol = myokit.pacing.steptrain_linear(
    Vlower, Vupper, dV, Vhold, tpreMeasuring_iv_Nao2_saka, tstep)

sakakibara_iv_Nao2_conditions = {'na_conc.Na_o': Na_o,
                                 'na_conc.Na_i': 5,
                                 'phys.T': 290.15}
sakakibara_iv_Nao2 = Experiment(
    #name = sakakibara_iv_Nao2_name,
    dataset=sakakibara_iv_Nao2_dataset,
    protocol=sakakibara_iv_Nao2_protocol,
    conditions=sakakibara_iv_Nao2_conditions,
    sum_stats=sakakibara_iv_sum_stats,
    description=sakakibara_iv_Nao2_desc,
    Q10=Q10_cond,
    Q10_factor=1)


#sakakibara_iv_Nao5_name = "IV  [Nao] = 5mM"
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
    #name = sakakibara_iv_Nao5_name,
    dataset=sakakibara_iv_Nao5_dataset,
    protocol=sakakibara_iv_Nao2_protocol,
    conditions=sakakibara_iv_Nao5_conditions,
    sum_stats=sakakibara_iv_sum_stats,
    description=sakakibara_iv_Nao5_desc,
    Q10=Q10_cond,
    Q10_factor=1)

#sakakibara_iv_Nao20_name = "IV  [Nao] = 20mM"
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
    #name = sakakibara_iv_Nao20_name,
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
#sakakibara_act_name = "activation"
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
    for d in data.split_periodic(tperiod_iv_saka, adjust=True):
        d = d.trim_left(tpreMeasuring_iv_saka, adjust=True)
        act_gate = d['ina.g']
        output = output+[max(act_gate, key=abs)]
    norm = output[-1]
    for i in range(len(output)):
        output[i] /= norm
    return output

sakakibara_act = Experiment(
    #name = sakakibara_act_name,
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
#sakakibara_inact_name = "inactivation"
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

tperiod_inact_saka = 10000 # ms
tstep = 1000
twait = 0
ttest = 30
tpre = tperiod_inact_saka - tstep - twait - ttest
tpreMeasuring_inact_saka = tperiod_inact_saka - ttest # tperiod - ttest
tMeasuring_inact_saka = ttest

Vhold = -140 # mV
Vtest = -20
Vlower = -140
dV = 10
Vupper = -30

sakakibara_inact_protocol = availability_linear(
    Vlower, Vupper, dV, Vhold, Vtest, tpre, tstep, twait, ttest)

def sakakibara_inact_sum_stats(data):
    output = []
    for d in data.split_periodic(tperiod_inact_saka, adjust=True):
        d = d.trim_left(tpreMeasuring_inact_saka, adjust = True)
        inact_gate = d['ina.g']
        index = np.argmax(np.abs(inact_gate))
        output = output+[np.abs(inact_gate[index])]
    Norm = output[0]
    for i in range(len(output)):
        output[i] /= Norm
    return output

sakakibara_inact = Experiment(
    #name = sakakibara_inact_name,
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
#sakakibara_inact_kin_1_name = "Inactivation Kinetics"
sakakibara_inact_kin_1_desc =   """
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
dataset1 = np.asarray([vsteps_th1, th1, variances_th1])
# Slow inactivation kinetics
vsteps_th2, th2, sd_th2 = data.TauS_Inactivation_Sakakibara()
variances_th2 = [(sd_)**2 for sd_ in sd_th2]
dataset2 = np.asarray([vsteps_th2, th2, variances_th2])
sakakibara_inact_kin_1_dataset = [dataset1, dataset2]

tstep = 100 # ms
tpre = 10000 # before the first pulse occurs
Vhold = -140 # mV
Vlower = -50
dV = 10
Vupper = -20+dV
sakakibara_inact_kin_1_protocol = myokit.pacing.steptrain_linear(
    Vlower, Vupper, dV, Vhold, tpre, tstep)

def sakakibara_inact_kin_1_sum_stats(data):
    #def double_exp(t, tauh, taus, Ah, As):
    #    return Ah*np.exp(-t/tauh) + As*np.exp(-t/taus)
    def double_exp(t, tauh, taus, Ah, As, A0):
        return Ah*np.exp(-t/tauh) + As*np.exp(-t/taus) + A0

    output1 = []
    output2 =  []
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
                if len(time)<=1 or len(current)<=1:
                    raise Exception('Failed simulation')

                popt, _ = so.curve_fit(double_exp,
                                       time,
                                       current,
                                       p0=[2,20,1.,1.,0],
                                       bounds=([0.,0.,-np.inf,-np.inf,-np.inf],
                                               [50.,100.,np.inf,np.inf,np.inf]))
                fit = [double_exp(t,popt[0],popt[1],popt[2],popt[3],popt[4]) for t in time]

                # Calculate r2
                ss_res = np.sum((np.array(current)-np.array(fit))**2)
                ss_tot = np.sum((np.array(current)-np.mean(np.array(current)))**2)
                r2 = 1 - (ss_res / ss_tot)

                tauh = min(popt[0],popt[1])
                taus = max(popt[0],popt[1])

                if r2 > 0.99:
                    output1 = output1+[tauh]
                    output2 = output2+[taus]
                else:
                    raise RuntimeWarning('scipy.optimize.curve_fit found a poor fit')
            except:
                output1 = output1+[float('inf')]
                output2 = output2+[float('inf')]
    output = output1+output2
    return output

sakakibara_inact_kin_1 = Experiment(
    #name = sakakibara_inact_kin_1_name,
    dataset=sakakibara_inact_kin_1_dataset,
    protocol=sakakibara_inact_kin_1_protocol,
    conditions=sakakibara_conditions,
    sum_stats=sakakibara_inact_kin_1_sum_stats,
    description=sakakibara_inact_kin_1_desc,
    Q10 = Q10_tau,
    Q10_factor = -1)


#
# Inactivation kinetics [Sakakibara1992]
#
#sakakibara_inact_kin_2_name = "Inactivation Kinetics w/ availability protocol"
#sakakibara_inact_kin_80_name = "Availability protocol : HP = -80mV"
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
#sakakibara_recov_name = "recovery (3 Holding Potentials)"
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
    #name = sakakibara_recov_name,
    dataset=sakakibara_rec140_dataset,
    protocol=sakakibara_rec140_protocol,
    conditions=sakakibara_conditions,
    sum_stats=sakakibara_rec140_sum_stats,
    description=sakakibara_rec140_desc,
    Q10=None,
    Q10_factor=0)
sakakibara_rec120 = Experiment(
    #name = sakakibara_recov_name,
    dataset=sakakibara_rec120_dataset,
    protocol=sakakibara_rec120_protocol,
    conditions=sakakibara_conditions,
    sum_stats=sakakibara_rec120_sum_stats,
    description=sakakibara_rec120_desc,
    Q10=None,
    Q10_factor=0)
sakakibara_rec100 = Experiment(
    #name = sakakibara_recov_name,
    dataset=sakakibara_rec100_dataset,
    protocol=sakakibara_rec100_protocol,
    conditions=sakakibara_conditions,
    sum_stats=sakakibara_rec100_sum_stats,
    description=sakakibara_rec100_desc,
    Q10=None,
    Q10_factor=0)


#
# BELOW IS UNCHECKED - Charles 2019-09-20
#

#######################################################################################################################
###  kinetics of recovery curves - Sakakibara 1992
#sakakibara_recov_kin_name = "recovery kinetics"
#sakakibara_recov_kin_desc = """
#    describes the protocol used to measure the fast time constant from the Recovery of I_na in the Sakakibara Paper (figure 9)
#
#    the Vhold used here is -140mV ,-120mV -110mV, -100mV and -90mV
#
#    The protocol is not decribed but was assumed to be the same as for the recovery protocol (fig 8A)
#    The protocol is a double pulse protocol at the frequency of 0.1Hz
#
#    """
#
## DATA
#vsteps_th_depol, th_depol, _ = dataSaka.TauF_Inactivation_Sakakibara_Depol()
#variances_depol = [0.]*len(th_depol)
#sakakibara_recov_kin_dataset = np.asarray([vsteps_th_depol, th_depol, variances_depol])
#
#
## PROTOCOL
#tmp_protocol = []
#tpreMeasuringList1_recov_kin = []
#
#tperiod_recov_kin = 10000 # ms
#tstep1 = 1000
#tstep2 = 1000
#
#tMeasuring1_recov_kin = tstep1
#tpreMeasuring2_recov_kin = tperiod_recov_kin - tstep2
#
#Vstep1 = -20
#Vstep2 = -20
#
#twaitList_recov_kin = [2,5,10,15,20,25,30,35,40,45,50,75,100,200,300,400,500,600,700,800,900,1000]
#Split_list_recov_kin = [len(twaitList_recov_kin)]
#
#for Vhold in [-140, -120, -110, -100, -90]:
#
#    tpreList = []
#    for twait in twaitList_recov_kin:
#        tpre = tperiod_recov_kin - tstep1 - twait - tstep2
#        tpreList.append(tpre)
#        tpreMeasuringList1_recov_kin.append(tpre)
#
#    protocol = recovery_tpreList(twaitList_recov_kin,Vhold,Vstep1, Vstep2,tpreList,tstep1,tstep2)
#
#    tmp_protocol.append(protocol)
#
## Fuse all the protocols into one
#sakakibara_recov_kin_protocol = tmp_protocol[0]
#for p in tmp_protocol[1:]:
#    for e in p.events():
#        sakakibara_recov_kin_protocol.add_step(e.level(), e.duration())
#
## CONDITIONS
#sakakibara_conditions = {'membrane.Na_o': 5,
#                        'membrane.Na_i': 5,
#                        'membrane.T': 290.15}
#
## SUMMARY STATISTICS
#def sakakibara_recov_kin_sum_stats(data):
#    def simple_exp(t, tauh):
#        return np.exp(-t/tauh)
#    output = []
#    loop = 0
#    sub_loop = 0
#
#
#    # spliting back the 5 protocols
#    dProtocolOne,dProtocoltwofive = data.split(tperiod_recov_kin*Split_list_recov_kin[0])
#    dProtocolTwo,dProtocolThreefive  = dProtocoltwofive.split(tperiod_recov_kin*2*Split_list_recov_kin[0])
#    dProtocolThree,dProtocolfourfive = dProtocolThreefive.split(tperiod_recov_kin*3*Split_list_recov_kin[0])
#    dProtocolFour,dProtocolFive  = dProtocolfourfive.split(tperiod_recov_kin*4*Split_list_recov_kin[0])
#    dProtocols = [dProtocolOne,dProtocolTwo,dProtocolThree,dProtocolFour,dProtocolFive]
#
#    for dOneProtocol in dProtocols:
#        rec = []
#
#        d_split = dOneProtocol.split_periodic(tperiod_recov, adjust = True)
#
#        d_split = d_split[loop*Split_list_recov_kin[0]:]    # specific to split_periodic function
#        #( if the time begins at t0 >0 it will create empty arrays from 0 to t0 : here we are getting rid of them)
#
#        for d in d_split:
#
#            dcond = d.trim(tpreMeasuringList1_recov_kin[sub_loop], tpreMeasuringList1_recov_kin[sub_loop]+tMeasuring1_recov_kin, adjust = True)
#            dtest = d.trim_left(tpreMeasuring2_recov_kin, adjust = True)
#
#            current_cond = dcond['ina.i_Na'][:-1]
#            current_test = dtest['ina.i_Na'][:-1]
#
#            index_cond = np.argmax(np.abs(current_cond))
#            index_test = np.argmax(np.abs(current_test))
#            try :
#                rec.append(current_test[index_test] / current_cond[index_cond])
#                sub_loop += 1
#            except :
#                rec.append(float('inf'))
#                sub_loop += 1
#        with warnings.catch_warnings():
#            warnings.simplefilter('error', OptimizeWarning)
#            warnings.simplefilter('error', RuntimeWarning)
#            try:
#            # Fit simple exponential to recovery curve
#
#                popt, _ = so.curve_fit(simple_exp, twaitList_recov_kin, 1.-np.asarray(rec),p0=[5], bounds=([0.1], [50.0]))
#                tauh = popt[0]
#
#                output = output + [tauh]
#            except:
#                output = output + [float('inf')]
#        loop += 1
#
#    return output
#
## Experiment
#sakakibara_recov_kin = Experiment(
#    name = sakakibara_recov_kin_name,
#    dataset=sakakibara_recov_kin_dataset,
#    protocol=sakakibara_recov_kin_protocol,
#    conditions=sakakibara_conditions,
#    sum_stats=sakakibara_recov_kin_sum_stats,
#    description=sakakibara_recov_kin_desc,
#     Q10 = Q10_tau,
#    Q10_factor = -1)
