#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 15:38:28 2019

@author: brm18
"""

from ionchannelABC.experiment import Experiment

import data.ina.Sakakibara1992.data_Sakakibara1992 as dataSaka
from ionchannelABC.protocol import availability_linear,availability
from custom_protocols import recovery_tpreList, varying_test_duration_double_pulse
import numpy as np
import myokit
import warnings
from scipy.optimize import OptimizeWarning
import scipy.optimize as so



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
        output = output+[max(current, key = abs)]
    return output

# Experiment
sakakibara_iv = Experiment(
    dataset=sakakibara_iv_dataset,
    protocol=sakakibara_iv_protocol,
    conditions=sakakibara_conditions,
    sum_stats=sakakibara_iv_sum_stats,
    description=sakakibara_iv_desc,
    Q10 = Q10_cond,
    Q10_factor = 1)


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
tpreMeasuring_inact_saka = tperiod_inact_saka - ttest # tperiod - ttest
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
        d = d.trim_left(tpreMeasuring_inact_saka, adjust = True)
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
        output = output+[max(current, key = abs)]
    return output

# Experiment
sakakibara_iv_Nao2 = Experiment(
    dataset=sakakibara_iv_Nao2_dataset,
    protocol=sakakibara_iv_Nao2_protocol,
    conditions=sakakibara_iv_Nao2_conditions,
    sum_stats=sakakibara_iv_Nao2_sum_stats,
    description=sakakibara_iv_Nao2_desc,
    Q10 = Q10_cond,
    Q10_factor = 1)

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
        output = output+[max(current, key = abs)]
    return output

# Experiment
sakakibara_iv_Nao5 = Experiment(
    dataset=sakakibara_iv_Nao5_dataset,
    protocol=sakakibara_iv_Nao5_protocol,
    conditions=sakakibara_iv_Nao5_conditions,
    sum_stats=sakakibara_iv_Nao5_sum_stats,
    description=sakakibara_iv_Nao5_desc,
    Q10 = Q10_cond,
    Q10_factor = 1)

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
variances_iv_curves = [0 for p in peaks] # no error reported
sakakibara_iv_Nao20_dataset.append(np.asarray([vsteps, peaks, variances_iv_curves]))


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
        output = output+[max(current, key = abs)]
    return output

# Experiment
sakakibara_iv_Nao20 = Experiment(
    dataset=sakakibara_iv_Nao20_dataset,
    protocol=sakakibara_iv_Nao20_protocol,
    conditions=sakakibara_iv_Nao20_conditions,
    sum_stats=sakakibara_iv_Nao20_sum_stats,
    description=sakakibara_iv_Nao20_desc,
    Q10 = Q10_cond,
    Q10_factor = 1)

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
variances_th1 = [(sd_)**2 for sd_ in sd_th1]
dataset1 = np.asarray([vsteps_th1, th1, variances_th1])

# 5bis : slow inactivation kinetics : tau h2
vsteps_th2, th2, sd_th2 = dataSaka.TauS_Inactivation_Sakakibara()
variances_th2 = [(sd_)**2 for sd_ in sd_th2]
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
                output = output+[tauh]
                ss_list = ss_list + [taus]
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
    description=sakakibara_inact_kin_1_desc,
    Q10 = Q10_tau,
    Q10_factor = -1)

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
                    output = output+ [current[index] / normalizing_peak] #should I still normalize since it's in the protocol itself ?
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
    description=sakakibara_inact_kin_2_desc,
    Q10 = Q10_tau,
    Q10_factor = -1)

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
                output = output + [current_test[index_test] / current_cond[index_cond]]  # should I still normalize ?
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
variances_depol = [0.]*len(th_depol)
sakakibara_recov_kin_dataset = np.asarray([vsteps_th_depol, th_depol, variances_depol])


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
                
                output = output + [tauh]
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
    description=sakakibara_recov_kin_desc,
     Q10 = Q10_tau,
    Q10_factor = -1)