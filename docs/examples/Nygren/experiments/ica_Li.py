#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 17:42:47 2019

@author: brm18
"""

from ionchannelABC.experiment import Experiment

import data.ina.Li1997.data_Li1997 as data
from ionchannelABC.protocol import availability_linear,availability
from custom_protocols import recovery_tpreList, varying_test_duration_double_pulse
import numpy as np
import myokit
import warnings
from scipy.optimize import OptimizeWarning
import scipy.optimize as so


  
Q10_tau = 2.1 # Ten Tusscher paper, 2004
Q10_cond = 2.3 # Kiyosue paper, 1993 
room_temp = 296 # K
#######################################################################################################################
### IV curve - Li 1997
Li_iv_80_desc ="""
    describes the protocol used to measure the IV peak-current curve in the Li Paper figure 1C

    page 2 of the paper : I-V relations were determined using 300-ms depolarizing steps every 10s 
    The magnitude was measured as the difference between the peak inward current and the steady state current at the end of the depolarizing step
    
    TODO : check that the substraction of both is still negative ! 
    
    protocol used in figure 1C: single test pulse at a frequency of 0.1Hz : every 10s, the voltage step occurs. 
    """

# DATA
# already converted in pA/pF
Vhold = 80 # mV
vsteps, peaks, sd_iv = data.IV_Li_all(Vhold)
variances =  [(sd_)**2 for sd_ in sd_iv]
Li_iv_80_dataset = np.asarray([vsteps, peaks, variances])

# PROTOCOL
tperiod_iv_Li = 10000 # ms
tstep = 300 # ms (not precised in the paper, 1s seems enough to mesure the peak current)
tpreMeasuring_iv_Li = tperiod_iv_Li - tstep # before the measurement


Vlower = -70 
dV = 10
Vupper = 60+dV # need to add dV to Vupper to be sure that the step at 20mV happens (since Vlower has been shifted from 0.001)


Li_iv_80_protocol = myokit.pacing.steptrain_linear(Vlower,Vupper, dV, Vhold, tpreMeasuring_iv_Li, tstep) 

# CONDITIONS 
#TODO : find the correct conditions of the paper for the concentrations of the different species.
Li_conditions = {'membrane.T': room_temp}

# SUMMARY STATISTICS
def Li_iv_80_sum_stats(data):
    output = []
    for d in data.split_periodic(tperiod_iv_Li, adjust=True):
        d = d.trim_left(tpreMeasuring_iv_Li, adjust=True)
        current = d['L_type_Ca.i_Ca_L'][:-1] # the last value is sometimes a nan 
        # (because V =0 at the end of the simulation and that I in nygren model is not defined for V = 0)
        output = output+[max(current, key = abs)-current[-1]]
    return output

# Experiment
Li_iv_80 = Experiment(
    dataset=Li_iv_80_dataset,
    protocol=Li_iv_80_protocol,
    conditions=Li_conditions,
    sum_stats=Li_iv_80_sum_stats,
    description=Li_iv_80_desc,
    Q10 = Q10_cond,
    Q10_factor = 1)

#######################################################################################################################
### IV curve - Li 1997
Li_iv_60_desc ="""
    describes the protocol used to measure the IV peak-current curve in the Li Paper figure 1C

    page 2 of the paper : I-V relations were determined using 300-ms depolarizing steps every 10s 
    The magnitude was measured as the difference between the peak inward current and the steady state current at the end of the depolarizing step
    
    TODO : check that the substraction of both is still negative ! 
    
    protocol used in figure 1C: single test pulse at a frequency of 0.1Hz : every 10s, the voltage step occurs. 
    """

# DATA
# already converted in pA/pF
Vhold = 60 # mV
vsteps, peaks, sd_iv = data.IV_Li_all(Vhold)
variances =  [(sd_)**2 for sd_ in sd_iv]
Li_iv_60_dataset = np.asarray([vsteps, peaks, variances])

# PROTOCOL
tperiod_iv_Li = 10000 # ms
tstep = 300 # ms (not precised in the paper, 1s seems enough to mesure the peak current)
tpreMeasuring_iv_Li = tperiod_iv_Li - tstep # before the measurement


Vlower = -50 
dV = 10
Vupper = 60+dV # need to add dV to Vupper to be sure that the step at 20mV happens (since Vlower has been shifted from 0.001)


Li_iv_60_protocol = myokit.pacing.steptrain_linear(Vlower,Vupper, dV, Vhold, tpreMeasuring_iv_Li, tstep) 

# CONDITIONS 
#TODO : find the correct conditions of the paper for the concentrations of the different species.
Li_conditions = {'membrane.T': room_temp}

# SUMMARY STATISTICS
def Li_iv_60_sum_stats(data):
    output = []
    for d in data.split_periodic(tperiod_iv_Li, adjust=True):
        d = d.trim_left(tpreMeasuring_iv_Li, adjust=True)
        current = d['L_type_Ca.i_Ca_L'][:-1] # the last value is sometimes a nan 
        # (because V =0 at the end of the simulation and that I in nygren model is not defined for V = 0)
        output = output+[max(current, key = abs)-current[-1]]
    return output

# Experiment
Li_iv_60 = Experiment(
    dataset=Li_iv_60_dataset,
    protocol=Li_iv_60_protocol,
    conditions=Li_conditions,
    sum_stats=Li_iv_60_sum_stats,
    description=Li_iv_60_desc,
    Q10 = Q10_cond,
    Q10_factor = 1)

#######################################################################################################################
### IV curve - Li 1997
Li_iv_40_desc ="""
    describes the protocol used to measure the IV peak-current curve in the Li Paper figure 1C

    page 2 of the paper : I-V relations were determined using 300-ms depolarizing steps every 10s 
    The magnitude was measured as the difference between the peak inward current and the steady state current at the end of the depolarizing step
    
    TODO : check that the substraction of both is still negative ! 
    
    protocol used in figure 1C: single test pulse at a frequency of 0.1Hz : every 10s, the voltage step occurs. 
    """

# DATA
# already converted in pA/pF
Vhold = 40 # mV
vsteps, peaks, sd_iv = data.IV_Li_all(Vhold)
variances =  [(sd_)**2 for sd_ in sd_iv]
Li_iv_40_dataset = np.asarray([vsteps, peaks, variances])

# PROTOCOL
tperiod_iv_Li = 10000 # ms
tstep = 300 # ms (not precised in the paper, 1s seems enough to mesure the peak current)
tpreMeasuring_iv_Li = tperiod_iv_Li - tstep # before the measurement


Vlower = -30 
dV = 10
Vupper = 60+dV # need to add dV to Vupper to be sure that the step at 20mV happens (since Vlower has been shifted from 0.001)


Li_iv_60_protocol = myokit.pacing.steptrain_linear(Vlower,Vupper, dV, Vhold, tpreMeasuring_iv_Li, tstep) 

# CONDITIONS 
#TODO : find the correct conditions of the paper for the concentrations of the different species.
Li_conditions = {'membrane.T': room_temp}

# SUMMARY STATISTICS
def Li_iv_40_sum_stats(data):
    output = []
    for d in data.split_periodic(tperiod_iv_Li, adjust=True):
        d = d.trim_left(tpreMeasuring_iv_Li, adjust=True)
        current = d['L_type_Ca.i_Ca_L'][:-1] # the last value is sometimes a nan 
        # (because V =0 at the end of the simulation and that I in nygren model is not defined for V = 0)
        output = output+[max(current, key = abs)-current[-1]]
    return output

# Experiment
Li_iv_60 = Experiment(
    dataset=Li_iv_60_dataset,
    protocol=Li_iv_60_protocol,
    conditions=Li_conditions,
    sum_stats=Li_iv_60_sum_stats,
    description=Li_iv_60_desc,
    Q10 = Q10_cond,
    Q10_factor = 1)


#######################################################################################################################
### activation curve - Li 1997
Li_act_desc = """
    describes the protocol used to measure the activation curve in the Li Paper (figure 2B)
    
    TODO
    single test pulse at a frequency of 0.1Hz
    """


# DATA
vsteps_act, act, sd_act = data.Act_Li()
variances_act = [(sd_)**2 for sd_ in sd_act]
Li_act_dataset = np.asarray([vsteps_act, act, variances_act])

# PROTOCOL
tperiod_act_Li = 10000 # ms
tstep = 1000 # ms (not precised in the paper, 1s seems enough to mesure the peak current)
tpre = tperiod_act_Li - tstep # before the first pulse occurs
tpreMeasuring_act_Li = tperiod_act_Li - tstep # before the measurement

Vhold = -80 #mV
Vlower = -80
dV = 10
Vupper = 20 + dV# need to add dV to Vupper to be sure that the step at 20mV happens (since Vlower has been shifted from 0.001)


Li_act_protocol = myokit.pacing.steptrain_linear(Vlower,Vupper, dV, Vhold, tpre, tstep) 

# CONDITIONS
#TODO : find the correct conditions of the paper for the concentrations of the different species.
Li_conditions = {'membrane.T': room_temp}

# SUMMARY STATISTICS
def Li_act_sum_stats(data):
    output = []
    for d in data.split_periodic(tperiod_act_Li, adjust=True):
        d = d.trim_left(tpreMeasuring_act_Li, adjust=True)
        act_gate = d['L_type_Ca.G_Na_norm']
        index = np.argmax(np.abs(act_gate))
        output = output+[np.abs(act_gate[index])]
    return output

# Experiment
Li_act = Experiment(
    dataset=Li_act_dataset,
    protocol=Li_act_protocol,
    conditions=Li_conditions,
    sum_stats=Li_act_sum_stats,
    description=Li_act_desc)


#######################################################################################################################
### Inactivation curve - Li 1997
Li_inact_1000_desc = """
    describes the protocol used to measure the activation curve in the Li Paper (figure 2B)

    The protocol is a double pulse protocol at the frequency of 0.1Hz
    
    """

# DATA
tstep = 1000 # ms
vsteps_inact, inact, sd_inact = data.Inact_Li(tstep)
variances_inact = [(sd_)**2 for sd_ in sd_inact]
Li_inact_1000_dataset = np.asarray([vsteps_inact, inact, variances_inact])

# PROTOCOL
tperiod_inact_Li_1000 = 10000 # ms
twait = 0 
ttest = 300
tpre = tperiod_inact_Li_1000 - tstep - twait - ttest
tpreMeasuring_inact_Li_1000 = tperiod_inact_Li_1000 - ttest # tperiod - ttest

Vhold = -80 # mV
Vtest = 10
Vlower = -80
dV = 10
Vupper = 50 + dV # check why the +dv is required
    
Li_inact_1000_protocol = availability_linear(Vlower,Vupper, dV,Vhold, Vtest, tpre, tstep, twait, ttest)


# CONDITIONS
#TODO : find the correct conditions of the paper for the concentrations of the different species.
Li_conditions = {'membrane.T': room_temp}

# SUMMARY STATISTICS
def Li_inact_1000_sum_stats(data):
    output = []
    for d in data.split_periodic(tperiod_inact_Li_1000, adjust=True):
        d = d.trim_left(tpreMeasuring_inact_Li_1000, adjust = True)
        inact_gate = d['L_type_Ca.G_Na_norm']
        index = np.argmax(np.abs(inact_gate))
        output = output+[np.abs(inact_gate[index])]
    return output

# Experiment
Li_inact = Experiment(
    dataset=Li_inact_1000_dataset,
    protocol=Li_inact_1000_protocol,
    conditions=Li_conditions,
    sum_stats=Li_inact_1000_sum_stats,
    description=Li_inact_1000_desc)

#######################################################################################################################
### Inactivation curve - Li 1997
Li_inact_300_desc = """
    describes the protocol used to measure the activation curve in the Li Paper (figure 2B)

    The protocol is a double pulse protocol at the frequency of 0.1Hz
    
    """

# DATA
tstep = 300 # ms
vsteps_inact, inact, sd_inact = data.Inact_Li(tstep)
variances_inact = [(sd_)**2 for sd_ in sd_inact]
Li_inact_300_dataset = np.asarray([vsteps_inact, inact, variances_inact])

# PROTOCOL
tperiod_inact_Li_300 = 10000 # ms
twait = 0 
ttest = 300
tpre = tperiod_inact_Li_300 - tstep - twait - ttest
tpreMeasuring_inact_Li_300 = tperiod_inact_Li_300 - ttest # tperiod - ttest

Vhold = -80 # mV
Vtest = 10
Vlower = -80
dV = 10
Vupper = 50 + dV 
    
Li_inact_300_protocol = availability_linear(Vlower,Vupper, dV,Vhold, Vtest, tpre, tstep, twait, ttest)


# CONDITIONS
#TODO : find the correct conditions of the paper for the concentrations of the different species.
Li_conditions = {'membrane.T': room_temp}

# SUMMARY STATISTICS
def Li_inact_300_sum_stats(data):
    output = []
    for d in data.split_periodic(tperiod_inact_Li_300, adjust=True):
        d = d.trim_left(tpreMeasuring_inact_Li_300, adjust = True)
        inact_gate = d['L_type_Ca.G_Na_norm']
        index = np.argmax(np.abs(inact_gate))
        output = output+[np.abs(inact_gate[index])]
    return output

# Experiment
Li_inact = Experiment(
    dataset=Li_inact_300_dataset,
    protocol=Li_inact_300_protocol,
    conditions=Li_conditions,
    sum_stats=Li_inact_300_sum_stats,
    description=Li_inact_300_desc)

#######################################################################################################################
### Inactivation curve - Li 1997
Li_inact_150_desc = """
    describes the protocol used to measure the activation curve in the Li Paper (figure 2B)

    The protocol is a double pulse protocol at the frequency of 0.1Hz
    
    """

# DATA
tstep = 150 # ms
vsteps_inact, inact, sd_inact = data.Inact_Li(tstep)
variances_inact = [(sd_)**2 for sd_ in sd_inact]
Li_inact_150_dataset = np.asarray([vsteps_inact, inact, variances_inact])

# PROTOCOL
tperiod_inact_Li_150 = 10000 # ms
twait = 0 
ttest = 300
tpre = tperiod_inact_Li_150 - tstep - twait - ttest
tpreMeasuring_inact_Li_150 = tperiod_inact_Li_150 - ttest # tperiod - ttest

Vhold = -80 # mV
Vtest = 10
Vlower = -80
dV = 10
Vupper = 50 + dV 
    
Li_inact_150_protocol = availability_linear(Vlower,Vupper, dV,Vhold, Vtest, tpre, tstep, twait, ttest)


# CONDITIONS
#TODO : find the correct conditions of the paper for the concentrations of the different species.
Li_conditions = {'membrane.T': room_temp}

# SUMMARY STATISTICS
def Li_inact_150_sum_stats(data):
    output = []
    for d in data.split_periodic(tperiod_inact_Li_150, adjust=True):
        d = d.trim_left(tpreMeasuring_inact_Li_150, adjust = True)
        inact_gate = d['L_type_Ca.G_Na_norm']
        index = np.argmax(np.abs(inact_gate))
        output = output+[np.abs(inact_gate[index])]
    return output

# Experiment
Li_inact = Experiment(
    dataset=Li_inact_150_dataset,
    protocol=Li_inact_150_protocol,
    conditions=Li_conditions,
    sum_stats=Li_inact_150_sum_stats,
    description=Li_inact_150_desc)

#######################################################################################################################
###  Inactivation kinetics Hp -80 - Li 1997
Li_inact_kin_80_desc =   """
    describes the protocol used to measure the inactivation kinetics (tau_f and tau_s) in the Sakakibara Paper (figure 5B)
    
    the Voltage goes from -50mV to -20mV for this function with a dV = 10 mV.

    page 5 of the paper :
    Figure 5A shows INa elicited at holding potentials of -140 to -40 mV (top)
    and -20 mV (bottom). 


    single test pulse at a frequency of 1Hz (since the step is a 100 msec test pulse)
    """

# DATA
Vhold = -80 # mV
# 5 : Fast inactivation kinetics : tau h1
vsteps_th1, th1, sd_th1 = data.Tau1_Li_all(Vhold)
variances_th1 = [(sd_)**2 for sd_ in sd_th1]
dataset1 = np.asarray([vsteps_th1, th1, variances_th1])

# 5bis : slow inactivation kinetics : tau h2
vsteps_th2, th2, sd_th2 = data.Tau2_Li_all(Vhold)
variances_th2 = [(sd_)**2 for sd_ in sd_th2]
dataset2 = np.asarray([vsteps_th2, th2, variances_th2])

Li_inact_kin_80_dataset = [dataset1,dataset2]

# PROTOCOL
tperiod_kin_80_Li = 10000 # ms
tstep = 300 # ms 
tpre = tperiod_kin_80_Li - tstep # before the first pulse occurs
tpreMeasuring_kin_80_Li = tperiod_kin_80_Li - tstep # before the measurement


Vlower = -10
Vupper = -30
dV = 10

Li_inact_kin_80_protocol = myokit.pacing.steptrain_linear(Vlower,Vupper + dV, dV, Vhold, tpre, tstep) 


# CONDITIONS
#TODO : find the correct conditions of the paper for the concentrations of the different species.
Li_conditions = {'membrane.T': room_temp}


# SUMMARY STATISTICS
def Li_inact_kin_80_sum_stats(data):

    def double_exp(t, tauh,taus,Ah,As):
        return Ah*np.exp(-t/tauh) + As*np.exp(-t/taus)

    output = []
    ss_list =  []
    for d in data.split_periodic(tperiod_kin_80_Li, adjust = True):
        d = d.trim_left(tpreMeasuring_kin_80_Li, adjust = True)
        
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
Li_inact_kin_80 = Experiment(
    dataset=Li_inact_kin_80_dataset,
    protocol=Li_inact_kin_80_protocol,
    conditions=Li_conditions,
    sum_stats=Li_inact_kin_80_sum_stats,
    description=Li_inact_kin_80_desc,
    Q10 = Q10_tau,
    Q10_factor = -1)

#######################################################################################################################
###  Inactivation kinetics Hp -60 - Li 1997
Li_inact_kin_60_desc =   """
    describes the protocol used to measure the inactivation kinetics (tau_f and tau_s) in the Sakakibara Paper (figure 5B)
    
    the Voltage goes from -50mV to -20mV for this function with a dV = 10 mV.

    page 5 of the paper :
    Figure 5A shows INa elicited at holding potentials of -140 to -40 mV (top)
    and -20 mV (bottom). 


    single test pulse at a frequency of 1Hz (since the step is a 100 msec test pulse)
    """

# DATA
Vhold = -60 # mV
# 5 : Fast inactivation kinetics : tau h1
vsteps_th1, th1, sd_th1 = data.Tau1_Li_all(Vhold)
variances_th1 = [(sd_)**2 for sd_ in sd_th1]
dataset1 = np.asarray([vsteps_th1, th1, variances_th1])

# 5bis : slow inactivation kinetics : tau h2
vsteps_th2, th2, sd_th2 = data.Tau2_Li_all(Vhold)
variances_th2 = [(sd_)**2 for sd_ in sd_th2]
dataset2 = np.asarray([vsteps_th2, th2, variances_th2])

Li_inact_kin_60_dataset = [dataset1,dataset2]

# PROTOCOL
tperiod_kin_60_Li = 10000 # ms
tstep = 300 # ms 
tpre = tperiod_kin_60_Li - tstep # before the first pulse occurs
tpreMeasuring_kin_60_Li = tperiod_kin_60_Li - tstep # before the measurement


Vlower = -10
Vupper = -30
dV = 10

Li_inact_kin_60_protocol = myokit.pacing.steptrain_linear(Vlower,Vupper + dV, dV, Vhold, tpre, tstep) 


# CONDITIONS
#TODO : find the correct conditions of the paper for the concentrations of the different species.
Li_conditions = {'membrane.T': room_temp}


# SUMMARY STATISTICS
def Li_inact_kin_60_sum_stats(data):

    def double_exp(t, tauh,taus,Ah,As):
        return Ah*np.exp(-t/tauh) + As*np.exp(-t/taus)

    output = []
    ss_list =  []
    for d in data.split_periodic(tperiod_kin_60_Li, adjust = True):
        d = d.trim_left(tpreMeasuring_kin_60_Li, adjust = True)
        
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
Li_inact_kin_60 = Experiment(
    dataset=Li_inact_kin_60_dataset,
    protocol=Li_inact_kin_60_protocol,
    conditions=Li_conditions,
    sum_stats=Li_inact_kin_60_sum_stats,
    description=Li_inact_kin_60_desc,
    Q10 = Q10_tau,
    Q10_factor = -1)

#######################################################################################################################
###  Inactivation kinetics Hp -40 - Li 1997
Li_inact_kin_40_desc =   """
    describes the protocol used to measure the inactivation kinetics (tau_f and tau_s) in the Sakakibara Paper (figure 5B)
    
    the Voltage goes from -50mV to -20mV for this function with a dV = 10 mV.

    page 5 of the paper :
    Figure 5A shows INa elicited at holding potentials of -140 to -40 mV (top)
    and -20 mV (bottom). 


    single test pulse at a frequency of 1Hz (since the step is a 100 msec test pulse)
    """

# DATA
Vhold = -40 # mV
# 5 : Fast inactivation kinetics : tau h1
vsteps_th1, th1, sd_th1 = data.Tau1_Li_all(Vhold)
variances_th1 = [(sd_)**2 for sd_ in sd_th1]
dataset1 = np.asarray([vsteps_th1, th1, variances_th1])

# 5bis : slow inactivation kinetics : tau h2
vsteps_th2, th2, sd_th2 = data.Tau2_Li_all(Vhold)
variances_th2 = [(sd_)**2 for sd_ in sd_th2]
dataset2 = np.asarray([vsteps_th2, th2, variances_th2])

Li_inact_kin_40_dataset = [dataset1,dataset2]

# PROTOCOL
tperiod_kin_40_Li = 10000 # ms
tstep = 300 # ms 
tpre = tperiod_kin_40_Li - tstep # before the first pulse occurs
tpreMeasuring_kin_40_Li = tperiod_kin_40_Li - tstep # before the measurement


Vlower = -10
Vupper = -30
dV = 10

Li_inact_kin_40_protocol = myokit.pacing.steptrain_linear(Vlower,Vupper + dV, dV, Vhold, tpre, tstep) 


# CONDITIONS
#TODO : find the correct conditions of the paper for the concentrations of the different species.
Li_conditions = {'membrane.T': room_temp}


# SUMMARY STATISTICS
def Li_inact_kin_40_sum_stats(data):

    def double_exp(t, tauh,taus,Ah,As):
        return Ah*np.exp(-t/tauh) + As*np.exp(-t/taus)

    output = []
    ss_list =  []
    for d in data.split_periodic(tperiod_kin_40_Li, adjust = True):
        d = d.trim_left(tpreMeasuring_kin_40_Li, adjust = True)
        
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
Li_inact_kin_40 = Experiment(
    dataset=Li_inact_kin_40_dataset,
    protocol=Li_inact_kin_40_protocol,
    conditions=Li_conditions,
    sum_stats=Li_inact_kin_40_sum_stats,
    description=Li_inact_kin_40_desc,
    Q10 = Q10_tau,
    Q10_factor = -1)