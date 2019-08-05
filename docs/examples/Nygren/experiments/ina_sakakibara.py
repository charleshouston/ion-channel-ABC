#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 15:38:28 2019

@author: brm18
"""

from ionchannelABC.experiment import Experiment

import data.ina.Sakakibara1992.data_Sakakibara1992 as dataSaka
from ionchannelABC.protocol import recovery_tpreList ,manual_steptrain_linear, availability_linear,varying_test_duration_double_pulse,availability
import numpy as np
import pandas as pd
import myokit
import matplotlib.pyplot as plt
import warnings
from scipy.optimize import OptimizeWarning
import scipy.optimize as so



def temperature_adjust(R0, T0, T1, Q10):
    return R0*Q10**((T0-T1)/10)


# All experiments use conditions as defined in the
# Sakakibara and Schneider paper. Data is adjusted to these conditions.
# Experimental conditions are included in experiment setup below
# for reference.
    
Q10_tau = 2.79 # Ten Tusscher paper, 2004
Q10_cond = 1.5 # Correa paper, 1991 : To adjust the datas to 37°
#######################################################################################################################
### IV curve - Sakakibara 1992
sakakibara_iv_desc ="""
    describes the protocol used to measure the IV peak-current curve in the Sakakibara Paper figure 1B

    page 3 of the paper : The holding potential (Vh) was -140 mV. Depolarizing pulses in 10-mVsteps were applied from
     -70 to +20mV. Testpulses were applied at 0.1 Hz.
    
    protocol used in figure 1B: single test pulse at a frequency of 0.1Hz : every 10s, the voltage step occurs. 
    """

# DATA
# already converted in pA/pF
vsteps, peaks, _ = dataSaka.IV_Sakakibara()
peaks = [temperature_adjust(p,290.15,306.15,Q10_cond) for p in peaks]
sakakibara_iv_max_peak = np.max(np.abs(peaks)) # for later normalising
peaks = [p / sakakibara_iv_max_peak for p in peaks]

variances = [0 for p in peaks] # no error reported
sakakibara_iv_dataset = np.asarray([vsteps, peaks, variances])

# PROTOCOL
tperiod_iv_saka = 10000 # ms
tstep = 1000 # ms (not precised in the paper, 1s seems enough to mesure the peak current)
tpreMeasuring_iv_saka = tperiod_iv_saka - tstep # before the measurement

Vhold = -140 # mV
Vlower = -100.001 # modified to not go through V = 0 that makes the nygren model crash ( expression of I_na)
dV = 10
Vupper = 20+dV # need to add dV to Vupper to be sure that the step at 20mV happens (since Vlower has been shifted from 0.001)


sakakibara_iv_protocol = myokit.pacing.steptrain_linear(Vlower,Vupper, dV, Vhold, tpreMeasuring_iv_saka, tstep) 

# CONDITIONS
sakakibara_conditions = {'membrane.Na_o': 5,
                        'membrane.Na_i': 5,
                        'membrane.T': 290.15}

# SUMMARY STATISTICS
def sakakibara_iv_sum_stats(data):
    output = []
    for d in data.split_periodic(tperiod_iv_saka, adjust=True):
        d = d.trim_left(tpreMeasuring_iv_saka, adjust=True)
        current = d['ina.i_Na'][:-1] # the last value is sometimes a nan 
        # (because V =0 at the end of the simulation and that I in nygren model is not defined for V = 0)
        output = output+[max(current, key = abs)/sakakibara_iv_max_peak]
    return output

# Experiment
sakakibara_iv = Experiment(
    dataset=sakakibara_iv_dataset,
    protocol=sakakibara_iv_protocol,
    conditions=sakakibara_conditions,
    sum_stats=sakakibara_iv_sum_stats,
    description=sakakibara_iv_desc)


#######################################################################################################################
### activation curve - Sakakibara 1992
sakakibara_act_desc = """
    describes the protocol used to measure the activation curve in the Sakakibara Paper (figure 2)
    
    
    The protocol is not described !
    By hypothesis, this will be the same as the protocol used in figure 1B :
    single test pulse at a frequency of 0.1Hz
    """


# DATA
vsteps_act, act, sd_act = dataSaka.Act_Sakakibara()
variances_act = [(sd_)**2 for sd_ in sd_act]
sakakibara_act_dataset = np.asarray([vsteps_act, act, variances_act])

# PROTOCOL
tperiod_act_saka = 10000 # ms
tstep = 1000 # ms (not precised in the paper, 1s seems enough to mesure the peak current)
tpre = tperiod_act_saka - tstep # before the first pulse occurs
tpreMeasuring_act_saka = tperiod_act_saka - tstep # before the measurement

Vhold = -140 # mV
Vlower = -100.001 # modified to not go through V = 0 that makes the nygren model crash ( expression of I_na)
dV = 10
Vupper = 20 + dV# need to add dV to Vupper to be sure that the step at 20mV happens (since Vlower has been shifted from 0.001)


sakakibara_act_protocol = myokit.pacing.steptrain_linear(Vlower,Vupper, dV, Vhold, tpre, tstep) 

# CONDITIONS
sakakibara_conditions = {'membrane.Na_o': 5,
                        'membrane.Na_i': 5,
                        'membrane.T': 290.15}

# SUMMARY STATISTICS
def sakakibara_act_sum_stats(data):
    output = []
    for d in data.split_periodic(tperiod_act_saka, adjust=True):
        d = d.trim_left(tpreMeasuring_act_saka, adjust=True)
        act_gate = d['ina.G_Na_norm']
        index = np.argmax(np.abs(act_gate))
        output = output+[np.abs(act_gate[index])]
    return output

# Experiment
sakakibara_act = Experiment(
    dataset=sakakibara_act_dataset,
    protocol=sakakibara_act_protocol,
    conditions=sakakibara_conditions,
    sum_stats=sakakibara_act_sum_stats,
    description=sakakibara_act_desc)

#######################################################################################################################
### Inactivation curve - Sakakibara 1992
sakakibara_inact_desc = """
    describes the protocol used to measure the activation curve in the Sakakibara Paper (figure 2)
    
    page 7 of the paper : 
    The voltage dependence of h, was studied using a double-pulse protocol consisting of a
    1-second conditioning pulse from holding a potential of-140 mV 0.1 Hz (inset at lower left).
    Current amplitude elicited during the test pulse was normalized to that in absence of a conditioning pulse.

    TODO : The corresponding block in summary statistics function need to be changed to normalize the conditionning pulse

    The protocol is a double pulse protocol at the frequency of 0.1Hz
    
    """

# DATA
vsteps_inact, inact, sd_inact = dataSaka.Inact_Sakakibara()
variances_inact = [(sd_)**2 for sd_ in sd_inact]
sakakibara_inact_dataset = np.asarray([vsteps_inact, inact, variances_inact])

# PROTOCOL
tperiod_inact_saka = 10000 # ms
tstep = 1000 # ms (not precised in the paper, 1s seems enough to mesure the peak current)
twait = 2 
ttest = 30
tpre = tperiod_inact_saka - tstep - twait - ttest
tpreMeasuring_inact_saka = tperiod_inact_saka - tstep - twait - ttest # tperiod - ttest
tMeasuring_inact_saka = ttest 

Vhold = -140 # mV
Vtest = -20
Vlower = -140 
dV = 10
Vupper = -40 + dV # check why the +dv is required
    
sakakibara_inact_protocol = availability_linear(Vlower,Vupper, dV,Vhold, Vtest, tpre, tstep, twait, ttest)


# CONDITIONS
sakakibara_conditions = {'membrane.Na_o': 5,
                        'membrane.Na_i': 5,
                        'membrane.T': 290.15}

# SUMMARY STATISTICS
def sakakibara_inact_sum_stats(data):
    output = []
    for d in data.split_periodic(tperiod_inact_saka, adjust=True):
        d = d.trim(tpreMeasuring_inact_saka,tpreMeasuring_inact_saka  + tMeasuring_inact_saka, adjust = True)
        inact_gate = d['ina.G_Na_norm']
        index = np.argmax(np.abs(inact_gate))
        output = output+[np.abs(inact_gate[index])]
    return output

# Experiment
sakakibara_inact = Experiment(
    dataset=sakakibara_inact_dataset,
    protocol=sakakibara_inact_protocol,
    conditions=sakakibara_conditions,
    sum_stats=sakakibara_inact_sum_stats,
    description=sakakibara_inact_desc)

#######################################################################################################################
### IV curve : potential reversal as function of Na_c - Sakakibara 1992
##### NOT USED SO FAR
#sakakibara_rever_desc ="""
#    describes the protocol used to measure the reversal potential curve in the Sakakibara Paper (figure 3B)
#
#    page 5 of the paper :
#
#    The reversal potential was determined by the intersecting point on the current-voltage relation curve
#
#    the IV curve was plotted every 1mV and the protocol used for the IV curve is the same than the fig 1B
#    (single pulse protocol with a frequence of 0.1Hz)
#       
#    """
#
## DATA
#Na_c, rever, sd_rever = dataSaka.Reversal_potential_Sakakibara()
#max_rever = np.max(np.abs(rever)) # normalising
#rever = [rever_ / max_rever for rever_ in rever]
#variances_rever = [(sd_/max_rever)**2 for sd_ in sd_rever]
#sakakibara_rever_dataset = np.asarray([Na_c, rever, variances_rever])
#
## PROTOCOL
#conditions_list, tmp_protocol = [], []
#tpreMeasuring_list, tMeasuring_list = [], []
#
#tperiod_rever = 3000 # ms
#tstep = 500 # ms (not precised in the paper, 1s seems enough to mesure the peak current)
#tpre = tperiod_rever - tstep # before the first pulse occurs
#tpreMeasuring_rever = tperiod_rever - tstep # before the measurement
#
#Vhold = -140 # mV
#Vlower = -100.001 # modified to not go through V = 0 that makes the nygren model crash ( expression of I_na)
#Vupper = 40
#dV = 1 # needs to be small to ensure a smooth curve for the detection of V such that Ina(V) = 0
#
#protocol = myokit.pacing.steptrain_linear(Vlower,Vupper + dV, dV, Vhold, tpre, tstep) 
#
#for Na_c in [2,5,10,20]:
#    sakakibara_conditions = {'membrane.Na_o': Na_c,
#                            'membrane.Na_i': 5,
#                            'membrane.T': 290.15}
#
#    tmp_protocol.append(protocol)
#    conditions_list.append(sakakibara_conditions)
#    tpreMeasuring_list.append(tpreMeasuring_rever)
#
#sakakibara_rever_protocol = myokit.pacing.steptrain_linear(Vlower,Vupper, dV, Vhold, tpre, tstep) 
#
## SUMMARY STATISTICS
#def sakakibara_rever_sum_stats(data):
#    output = []
#    for i in range(4):
#        nomalized_peak = []
#        for d in data[i].split_periodic(tperiod_rever, adjust = True):
#
#            d = d.trim_left(tpreMeasuring_rever, adjust = True)
#            current = d['ina.i_Na']
#            current = current[:-1]           
#            index = np.argmax(np.abs(current))
#            nomalized_peak.append(current[index])
#        nomalized_peak_cut_50 = nomalized_peak[50:] 
#        # the reversal potential is after -50 mV and since the protocol starts at -100mV with dV = 1mV
#        # the cut was choosen to be at index 50
#        index = np.argmin(np.abs(nomalized_peak_cut_50))
#        output = output+ [(index - 50)/ max_rever] # shifting back the index
#    return output
#
## Experiment
#sakakibara_rever = Experiment(
#    dataset=sakakibara_rever_dataset,
#    protocol=sakakibara_rever_protocol,
#    conditions=sakakibara_conditions,
#    sum_stats=sakakibara_rever_sum_stats,
#    description=sakakibara_rever_desc)

#######################################################################################################################
### IV curves as function of Na_o = 2mM - Sakakibara 1992
sakakibara_iv_Nao2_desc ="""
    describes the protocol used to measure the differents IV-curves in the Sakakibara Paper (figure 3A)
    this protocol is for measuring the curve with Nao = 2mM

    page 5 of the paper :

    Test pulses were applied at 0.1 Hz
       
    """

# DATA
sakakibara_iv_Nao2_dataset = []
# already converted in pA/pF
Na_o = 2
vsteps, peaks, _ = dataSaka.IV_Sakakibara_fig3A_all(Na_o)
peaks = [temperature_adjust(p,290.15,306.15,Q10_cond) for p in peaks]
sakakibara_iv_max_peak_Na_o = np.max(np.abs(peaks)) # for later normalising
peaks = [p / sakakibara_iv_max_peak_Na_o for p in peaks]

variances_iv_curves = [0 for p in peaks] # no error reported
sakakibara_iv_Nao2_dataset.append(np.asarray([vsteps, peaks, variances_iv_curves]))


# PROTOCOL
tperiod_iv_Nao2_saka = 10000 # ms
tstep = 1000 # ms (not precised in the paper, 1s seems enough to mesure the peak current)
tpreMeasuring_iv_Nao2_saka = tperiod_iv_Nao2_saka - tstep # before the measurement

Vhold = -140 # mV
Vlower = -100.001 # modified to not go through V = 0 that makes the nygren model crash ( expression of I_na)
dV = 10
Vupper = 40+dV


sakakibara_iv_Nao2_protocol = myokit.pacing.steptrain_linear(Vlower,Vupper, dV, Vhold, tpreMeasuring_iv_Nao2_saka, tstep) 

# CONDITIONS
sakakibara_iv_Nao2_conditions = {'membrane.Na_o': Na_o,
                                'membrane.Na_i': 5,
                                'membrane.T': 290.15}

# SUMMARY STATISTICS
def sakakibara_iv_Nao2_sum_stats(data):
    output = []
    for d in data.split_periodic(tperiod_iv_Nao2_saka, adjust=True):
        d = d.trim_left(tpreMeasuring_iv_Nao2_saka, adjust=True)
        current = d['ina.i_Na'][:-1] # the last value is sometimes a nan 
        # (because V =0 at the end of the simulation and that I in nygren model is not defined for V = 0)
        output = output+[max(current, key = abs)/sakakibara_iv_max_peak]
    return output

# Experiment
sakakibara_iv_Nao2 = Experiment(
    dataset=sakakibara_iv_Nao2_dataset,
    protocol=sakakibara_iv_Nao2_protocol,
    conditions=sakakibara_iv_Nao2_conditions,
    sum_stats=sakakibara_iv_Nao2_sum_stats,
    description=sakakibara_iv_Nao2_desc)

#######################################################################################################################
### IV curves as function of Na_o = 5mM - Sakakibara 1992
sakakibara_iv_Nao5_desc ="""
    describes the protocol used to measure the differents IV-curves in the Sakakibara Paper (figure 3A)
    this protocol is for measuring the curve with Nao = 5mM

    page 5 of the paper :

    Test pulses were applied at 0.1 Hz
       
    """

# DATA
sakakibara_iv_Nao5_dataset = []
# already converted in pA/pF
Na_o = 5
vsteps, peaks, _ = dataSaka.IV_Sakakibara_fig3A_all(Na_o)
peaks = [temperature_adjust(p,290.15,306.15,Q10_cond) for p in peaks]
sakakibara_iv_max_peak_Na_o = np.max(np.abs(peaks)) # for later normalising
peaks = [p / sakakibara_iv_max_peak_Na_o for p in peaks]

variances_iv_curves = [0 for p in peaks] # no error reported
sakakibara_iv_Nao5_dataset.append(np.asarray([vsteps, peaks, variances_iv_curves]))


# PROTOCOL
tperiod_iv_Nao5_saka = 10000 # ms
tstep = 1000 # ms (not precised in the paper, 1s seems enough to mesure the peak current)
tpreMeasuring_iv_Nao5_saka = tperiod_iv_Nao5_saka - tstep # before the measurement

Vhold = -140 # mV
Vlower = -100.001 # modified to not go through V = 0 that makes the nygren model crash ( expression of I_na)
dV = 10
Vupper = 40+dV


sakakibara_iv_Nao5_protocol = myokit.pacing.steptrain_linear(Vlower,Vupper, dV, Vhold, tpreMeasuring_iv_Nao5_saka, tstep) 


sakakibara_iv_Nao5_conditions = {'membrane.Na_o': Na_o,
                                'membrane.Na_i': 5,
                                'membrane.T': 290.15}

# SUMMARY STATISTICS
def sakakibara_iv_Nao5_sum_stats(data):
    output = []
    for d in data.split_periodic(tperiod_iv_Nao5_saka, adjust=True):
        d = d.trim_left(tpreMeasuring_iv_Nao5_saka, adjust=True)
        current = d['ina.i_Na'][:-1] # the last value is sometimes a nan 
        # (because V =0 at the end of the simulation and that I in nygren model is not defined for V = 0)
        output = output+[max(current, key = abs)/sakakibara_iv_max_peak]
    return output

# Experiment
sakakibara_iv_Nao5 = Experiment(
    dataset=sakakibara_iv_Nao5_dataset,
    protocol=sakakibara_iv_Nao5_protocol,
    conditions=sakakibara_iv_Nao5_conditions,
    sum_stats=sakakibara_iv_Nao5_sum_stats,
    description=sakakibara_iv_Nao5_desc)

#######################################################################################################################
### IV curves as function of Na_o = 20mM - Sakakibara 1992
sakakibara_iv_Nao20_desc ="""
    describes the protocol used to measure the differents IV-curves in the Sakakibara Paper (figure 3A)
    this protocol is for measuring the curve with Nao = 20mM

    page 5 of the paper :

    Test pulses were applied at 0.1 Hz
       
    """

# DATA
sakakibara_iv_Nao20_dataset = []
# already converted in pA/pF
Na_o = 20
vsteps, peaks, _ = dataSaka.IV_Sakakibara_fig3A_all(Na_o)
peaks = [temperature_adjust(p,290.15,306.15,Q10_cond) for p in peaks]
sakakibara_iv_max_peak_Na_o = np.max(np.abs(peaks)) # for later normalising
peaks = [p / sakakibara_iv_max_peak_Na_o for p in peaks]

variances_iv_curves = [0 for p in peaks] # no error reported
sakakibara_iv_Nao2_dataset.append(np.asarray([vsteps, peaks, variances_iv_curves]))


# PROTOCOL
tperiod_iv_Nao20_saka = 10000 # ms
tstep = 1000 # ms (not precised in the paper, 1s seems enough to mesure the peak current)
tpreMeasuring_iv_Nao20_saka = tperiod_iv_Nao20_saka - tstep # before the measurement

Vhold = -140 # mV
Vlower = -100.001 # modified to not go through V = 0 that makes the nygren model crash ( expression of I_na)
dV = 10
Vupper = 40+dV


sakakibara_iv_Nao20_protocol = myokit.pacing.steptrain_linear(Vlower,Vupper, dV, Vhold, tpreMeasuring_iv_Nao20_saka, tstep) 


sakakibara_iv_Nao20_conditions = {'membrane.Na_o': Na_o,
                                'membrane.Na_i': 5,
                                'membrane.T': 290.15}

# SUMMARY STATISTICS
def sakakibara_iv_Nao20_sum_stats(data):
    output = []
    for d in data.split_periodic(tperiod_iv_Nao20_saka, adjust=True):
        d = d.trim_left(tpreMeasuring_iv_Nao20_saka, adjust=True)
        current = d['ina.i_Na'][:-1] # the last value is sometimes a nan 
        # (because V =0 at the end of the simulation and that I in nygren model is not defined for V = 0)
        output = output+[max(current, key = abs)/sakakibara_iv_max_peak]
    return output

# Experiment
sakakibara_iv_Nao20 = Experiment(
    dataset=sakakibara_iv_Nao20_dataset,
    protocol=sakakibara_iv_Nao20_protocol,
    conditions=sakakibara_iv_Nao20_conditions,
    sum_stats=sakakibara_iv_Nao20_sum_stats,
    description=sakakibara_iv_Nao20_desc)

#######################################################################################################################
###  Inactivation kinetics part 1 - Sakakibara 1992
sakakibara_inact_kin_1_desc =   """
    describes the protocol used to measure the inactivation kinetics (tau_f and tau_s) in the Sakakibara Paper (figure 5B)
    
    the Voltage goes from -50mV to -20mV for this function with a dV = 10 mV.

    page 5 of the paper :
    Figure 5A shows INa elicited at holding potentials of -140 to -40 mV (top)
    and -20 mV (bottom). 


    single test pulse at a frequency of 1Hz (since the step is a 100 msec test pulse)
    """

# DATA
tmp_dataset = []
# 5 : Fast inactivation kinetics : tau h1
vsteps_th1, th1, sd_th1 = dataSaka.TauF_Inactivation_Sakakibara()
th1 = [temperature_adjust(th1_,290.15,306.15,Q10_tau) for th1_ in th1]
max_th1 = np.max(np.abs(th1)) # normalising
th1 = [th1_ / max_th1 for th1_ in th1]
variances_th1 = [(sd_/max_th1)**2 for sd_ in sd_th1]
dataset1 = np.asarray([vsteps_th1, th1, variances_th1])

# 5bis : slow inactivation kinetics : tau h2
vsteps_th2, th2, sd_th2 = dataSaka.TauS_Inactivation_Sakakibara()
th2 = [temperature_adjust(th2_,290.15,306.15,Q10_tau) for th2_ in th2]
max_th2 = np.max(np.abs(th2)) # normalising
th2 = [th2_ / max_th2 for th2_ in th2]
variances_th2 = [(sd_/max_th2)**2 for sd_ in sd_th2]
dataset2 = np.asarray([vsteps_th2, th2, variances_th2])

sakakibara_inact_kin_1_dataset = [dataset1,dataset2]

# PROTOCOL
tperiod_kin_1_saka = 1000 # ms
tstep = 100 # ms 
tpre = tperiod_kin_1_saka - tstep # before the first pulse occurs
tpreMeasuring_kin_1_saka = tperiod_kin_1_saka - tstep # before the measurement

Vhold = -140 # mV
Vlower = -50 
Vupper = -20
dV = 10

sakakibara_inact_kin_1_protocol = myokit.pacing.steptrain_linear(Vlower,Vupper + dV, dV, Vhold, tpre, tstep) 


# CONDITIONS
sakakibara_conditions = {'membrane.Na_o': 5,
                        'membrane.Na_i': 5,
                        'membrane.T': 290.15}

# SUMMARY STATISTICS
def sakakibara_inact_kin_1_sum_stats(data):

    def double_exp(t, tauh,taus,Ah,As):
        return Ah*np.exp(-t/tauh) + As*np.exp(-t/taus)

    output = []
    ss_list =  []
    for d in data.split_periodic(tperiod_kin_1_saka, adjust = True):
        d = d.trim_left(tpreMeasuring_kin_1_saka, adjust = True)
        
        current = d['ina.i_Na'][:-1] # sometimes, the last value is nan and crashes the following,
                                                # so getting rid of the last value is perhaps the solution
        time = d['environment.time'][:-1]
        index = np.argmax(np.abs(current))

        # Set time zero to peak current
        current = current[index:]
        time = time[index:]
        t0 = time[0]
        time = [t-t0 for t in time]

        with warnings.catch_warnings():
            warnings.simplefilter('error',OptimizeWarning)
            warnings.simplefilter('error',RuntimeWarning)
            try:
                imax = max(current, key=abs)
                current = [c_/imax for c_ in current]


                if len(time)<=1 or len(current)<=1:
                    raise Exception('Failed simulation')
                popt, _ = so.curve_fit(double_exp, time, current,p0=[5,1,1,1], bounds=([0.01,0.01,0.01,0.01], [100.0,10,10,10]))

                tauh = min(popt[0],popt[1])
                taus = max(popt[0],popt[1])
                output = output+[tauh/max_th1]
                ss_list = ss_list + [taus/max_th2]
            except:
                output = output+[float('inf')]
                ss_list = ss_list+[float('inf')]
                
    output = output+ss_list
    return output

# Experiment
sakakibara_inact_kin_1 = Experiment(
    dataset=sakakibara_inact_kin_1_dataset,
    protocol=sakakibara_inact_kin_1_protocol,
    conditions=sakakibara_conditions,
    sum_stats=sakakibara_inact_kin_1_sum_stats,
    description=sakakibara_inact_kin_1_desc)

#######################################################################################################################
###  Inactivation kinetics part 2 - Sakakibara 1992
sakakibara_inact_kin_2_desc = """
    describes the protocol used to measure the inactivation kinetics in the Sakakibara Paper (figure 6)
    
    the Voltage used here is -100mV and -80mV
    
    page 7 of the paper : 
    Inactivation was induced by increasing
    the conditioning pulse duration. Conditioning
    pulses were applied to -80 and -100 mV Pulse
    frequency was 0.1 Hz. Current magnitude during
    the test pulse was normalized to levels obtained in
    the absence of a conditioning pulse.

    The protocol is a double pulse protocol at the frequency of 0.1Hz
    
    """

# DATA
sakakibara_inact_kin_2_dataset = []
for Vcond in [-100,-80]:
    timecourse, inact_timecourse,_ = dataSaka.Time_course_Inactivation_Sakakibara_all(Vcond)
    variances_time_course = [0 for t in timecourse]
    sakakibara_inact_kin_2_dataset.append(np.asarray([timecourse, inact_timecourse, variances_time_course]))



# PROTOCOL
conditions_list, tmp_protocol = [], []
tpreMeasuring_list, tMeasuring_list = [], []



# This is the main part of the protocol

tperiod_kin_2_saka = 10000 # ms


for Vcond in [-100,-80]:
    tstepList,_,_ = dataSaka.Time_course_Inactivation_Sakakibara_all(Vcond)
    
    twait = 2 
    ttest = 30

    tpreList = []
    for tstep in tstepList :
        tpre = tperiod_kin_2_saka - tstep - twait - ttest
        tpreList.append(tpre)
    
    Vhold = -140 # mV
    Vtest = -20        
        
    protocol = varying_test_duration_double_pulse(Vcond,Vhold, Vtest, tpreList, tstepList, twait, ttest)
    
    tmp_protocol.append(protocol)

## This part is for initializing the protocol : to determine the max I_Na when there is no conditionning pulse

twait = 0
tstep = 0
ttest = 30 
tpre = tperiod_kin_2_saka - twait - ttest - tstep

tpreMeasuring_kin_2_saka = tperiod_kin_2_saka - ttest

Vhold = -140 # mV
Vtest = -20        
protocol = availability([Vhold],Vhold, Vtest, tpre, tstep, twait, ttest)

tmp_protocol.append(protocol)

# Fuse all the protocols into one
sakakibara_inact_kin_2_protocol = tmp_protocol[0]
for p in tmp_protocol[1:]:
    for e in p.events():
        sakakibara_inact_kin_2_protocol.add_step(e.level(), e.duration())

# CONDITIONS
sakakibara_conditions = {'membrane.Na_o': 5,
                        'membrane.Na_i': 5,
                        'membrane.T': 290.15}

# SUMMARY STATISTICS
def sakakibara_inact_kin_2_sum_stats(data):
    output = []

    # retrieving the current max in absence of conditionning pulse
    dtwoProtocol,d_split_init = data.split(tperiod_kin_2_saka*23)

    d_init = d_split_init.trim_left(tpreMeasuring_kin_2_saka, adjust = True)
    current = d_init['ina.i_Na']

    current = current[:-1]
    index = np.argmax(np.abs(current))
    normalizing_peak = current[index] 

    # 6 : Fast inactivation kinetics : tau h1 part 2 
    loop = 0
    # tperiod_kin_2_saka*12 corresponds to the split between the vCond = -100 and the vcond = -80 (there is 12 data points for vCond = -100,
    # but 11 for vcond = -80 )
    for dOneProtocol in dtwoProtocol.split(tperiod_kin_2_saka*12): 
        D_split = dOneProtocol.split_periodic(tperiod_kin_2_saka, adjust = True)
        if loop == 1 :
            D_split = D_split[12:] # specific to split_periodic function
        for d in D_split:

                d = d.trim_left(tpreMeasuring_kin_2_saka, adjust = True)
            
                current = d['ina.i_Na']
                current = current[:-1]

                index = np.argmax(np.abs(current))
                try :
                    output = output+ [current[index] / normalizing_peak]
                except : 
                    output = output+ [float('inf')]
        loop += 1
    return output

# Experiment
sakakibara_inact_kin_2 = Experiment(
    dataset=sakakibara_inact_kin_2_dataset,
    protocol=sakakibara_inact_kin_2_protocol,
    conditions=sakakibara_conditions,
    sum_stats=sakakibara_inact_kin_2_sum_stats,
    description=sakakibara_inact_kin_2_desc)

#######################################################################################################################
###  recovery curves - Sakakibara 1992
sakakibara_recov_desc =    """
    describes the protocol used to measure the Recovery of I_na in the Sakakibara Paper (figure 8A)
    
    the Vhold used here is -140mV ,-120mV and -100mV
    
    page 8 of the paper : 
    The double-pulseprotocol shown in the inset was applied at various recovery potentials at a frequency of 0.1 Hz. The magnitude of the fast
    Na+ current during the test pulse was normalized to that
    induced by the conditioning pulse. 


    The protocol is a double pulse protocol at the frequency of 0.1Hz
    
    """  

# DATA
sakakibara_recov_dataset = []
for Vhold in [-140,-120,-100]:
    time,recov,_ = dataSaka.Recovery_Sakakibara_all(Vhold)
    variances_time = [0 for t in time]
    sakakibara_recov_dataset.append(np.asarray([time, recov, variances_time]))


# PROTOCOL
tmp_protocol = []
tpreMeasuringList1_recov = []

tperiod_recov = 10000 # ms
tstep1 = 1000
tstep2 = 1000

tMeasuring1_recov = tstep1 
tpreMeasuring2_recov = tperiod_recov - tstep2 

Vstep1 = -20
Vstep2 = -20  

Split_list_recov = [] # for the summary statistics function

for Vhold in [-140,-120,-100] :
    twaitList,_,_ = dataSaka.Recovery_Sakakibara_all(Vhold)

    Split_list_recov.append(len(twaitList))


    tpreList = []
    for twait in twaitList:
        tpre = tperiod_recov - tstep1 - twait - tstep2
        tpreList.append(tpre)
        tpreMeasuringList1_recov.append(tpre)

    protocol = recovery_tpreList(twaitList,Vhold,Vstep1, Vstep2,tpreList,tstep1,tstep2)
    
    tmp_protocol.append(protocol)

# Fuse all the protocols into one
sakakibara_recov_protocol = tmp_protocol[0]
for p in tmp_protocol[1:]:
    for e in p.events():
        sakakibara_recov_protocol.add_step(e.level(), e.duration())

# CONDITIONS
sakakibara_conditions = {'membrane.Na_o': 5,
                        'membrane.Na_i': 5,
                        'membrane.T': 290.15}

# SUMMARY STATISTICS
def sakakibara_recov_sum_stats(data):
    output = []
    loop = 0
    sub_loop = 0

    # spliting back the protocols
    dProtocolOne,dProtocoltwothree = data.split(tperiod_recov*Split_list_recov[0])
    dProtocolTwo,dProtocolThree  = dProtocoltwothree.split(tperiod_recov*(Split_list_recov[1]+Split_list_recov[0]))
    dProtocols = [dProtocolOne,dProtocolTwo,dProtocolThree]
   
    for dOneProtocol in dProtocols:
        d_split = dOneProtocol.split_periodic(tperiod_recov, adjust = True)
        if loop == 1 :
            d_split = d_split[Split_list_recov[0]:]    # specific to split_periodic function
        if loop == 2:
            d_split = d_split[Split_list_recov[0]+Split_list_recov[1]:]  


        for d in d_split:

            dcond = d.trim(tpreMeasuringList1_recov[sub_loop], tpreMeasuringList1_recov[sub_loop]+tMeasuring1_recov, adjust = True)
            dtest = d.trim_left(tpreMeasuring2_recov, adjust = True)
            
            current_cond = dcond['ina.i_Na'][:-1]
            current_test = dtest['ina.i_Na'][:-1]



            index_cond = np.argmax(np.abs(current_cond))
            index_test = np.argmax(np.abs(current_test))
            try :
                output = output + [current_test[index_test] / current_cond[index_cond]]  
                sub_loop += 1
            except :
                output = output + [float('inf')]  
                sub_loop += 1
        loop += 1
    return output

# Experiment
sakakibara_recov = Experiment(
    dataset=sakakibara_recov_dataset,
    protocol=sakakibara_recov_protocol,
    conditions=sakakibara_conditions,
    sum_stats=sakakibara_recov_sum_stats,
    description=sakakibara_recov_desc)

#######################################################################################################################
###  kinetics of recovery curves - Sakakibara 1992
sakakibara_recov_kin_desc = """
    describes the protocol used to measure the fast time constant from the Recovery of I_na in the Sakakibara Paper (figure 9)
    
    the Vhold used here is -140mV ,-120mV -110mV, -100mV and -90mV
    
    The protocol is not decribed but was assumed to be the same as for the recovery protocol (fig 8A)
    The protocol is a double pulse protocol at the frequency of 0.1Hz
    
    """  

# DATA
vsteps_th_depol, th_depol, _ = dataSaka.TauF_Inactivation_Sakakibara_Depol()
th_depol = [temperature_adjust(th_, 290.15, 306.15, Q10_tau) for th_ in th_depol]
max_th_depol = np.max(np.abs(th_depol))
th_depol = [th_ / max_th_depol for th_ in th_depol]
variances = [0.]*len(th_depol)
sakakibara_recov_kin_dataset = np.asarray([vsteps_th_depol, th_depol, variances])


# PROTOCOL
tmp_protocol = []
tpreMeasuringList1_recov_kin = []

tperiod_recov_kin = 10000 # ms
tstep1 = 1000
tstep2 = 1000

tMeasuring1_recov_kin = tstep1 
tpreMeasuring2_recov_kin = tperiod_recov_kin - tstep2 

Vstep1 = -20
Vstep2 = -20  

twaitList_recov_kin = [2,5,10,15,20,25,30,35,40,45,50,75,100,200,300,400,500,600,700,800,900,1000]
Split_list_recov_kin = [len(twaitList_recov_kin)]

for Vhold in [-140, -120, -110, -100, -90]:

    tpreList = []
    for twait in twaitList_recov_kin:
        tpre = tperiod_recov_kin - tstep1 - twait - tstep2
        tpreList.append(tpre)
        tpreMeasuringList1_recov_kin.append(tpre)

    protocol = recovery_tpreList(twaitList_recov_kin,Vhold,Vstep1, Vstep2,tpreList,tstep1,tstep2)
    
    tmp_protocol.append(protocol)

# Fuse all the protocols into one
sakakibara_recov_kin_protocol = tmp_protocol[0]
for p in tmp_protocol[1:]:
    for e in p.events():
        sakakibara_recov_kin_protocol.add_step(e.level(), e.duration())

# CONDITIONS
sakakibara_conditions = {'membrane.Na_o': 5,
                        'membrane.Na_i': 5,
                        'membrane.T': 290.15}

# SUMMARY STATISTICS
def sakakibara_recov_kin_sum_stats(data):
    def simple_exp(t, tauh):
        return np.exp(-t/tauh)
    output = []
    loop = 0
    sub_loop = 0
    

    # spliting back the 5 protocols
    dProtocolOne,dProtocoltwofive = data.split(tperiod_recov_kin*Split_list_recov_kin[0])
    dProtocolTwo,dProtocolThreefive  = dProtocoltwofive.split(tperiod_recov_kin*2*Split_list_recov_kin[0])
    dProtocolThree,dProtocolfourfive = dProtocolThreefive.split(tperiod_recov_kin*3*Split_list_recov_kin[0])
    dProtocolFour,dProtocolFive  = dProtocolfourfive.split(tperiod_recov_kin*4*Split_list_recov_kin[0])
    dProtocols = [dProtocolOne,dProtocolTwo,dProtocolThree,dProtocolFour,dProtocolFive]
   
    for dOneProtocol in dProtocols:
        rec = []

        d_split = dOneProtocol.split_periodic(tperiod_recov, adjust = True)

        d_split = d_split[loop*Split_list_recov_kin[0]:]    # specific to split_periodic function 
        #( if the time begins at t0 >0 it will create empty arrays from 0 to t0 : here we are getting rid of them)
 
        for d in d_split:

            dcond = d.trim(tpreMeasuringList1_recov_kin[sub_loop], tpreMeasuringList1_recov_kin[sub_loop]+tMeasuring1_recov_kin, adjust = True)
            dtest = d.trim_left(tpreMeasuring2_recov_kin, adjust = True)
            
            current_cond = dcond['ina.i_Na'][:-1]
            current_test = dtest['ina.i_Na'][:-1]

            index_cond = np.argmax(np.abs(current_cond))
            index_test = np.argmax(np.abs(current_test))
            try :
                rec.append(current_test[index_test] / current_cond[index_cond])
                sub_loop += 1
            except :
                rec.append(float('inf'))
                sub_loop += 1
        with warnings.catch_warnings():
            warnings.simplefilter('error', OptimizeWarning)
            warnings.simplefilter('error', RuntimeWarning)
            try:
            # Fit simple exponential to recovery curve

                popt, _ = so.curve_fit(simple_exp, twaitList_recov_kin, 1.-np.asarray(rec),p0=[5], bounds=([0.1], [50.0]))
                tauh = popt[0]
                
                output = output + [tauh/max_th_depol]
            except:
                output = output + [float('inf')]
        loop += 1

    return output

# Experiment
sakakibara_recov_kin = Experiment(
    dataset=sakakibara_recov_kin_dataset,
    protocol=sakakibara_recov_kin_protocol,
    conditions=sakakibara_conditions,
    sum_stats=sakakibara_recov_kin_sum_stats,
    description=sakakibara_recov_kin_desc)