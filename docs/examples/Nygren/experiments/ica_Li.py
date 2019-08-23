#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 17:42:47 2019

@author: brm18
"""

from ionchannelABC.experiment import Experiment

import data.ica.Li1997.data_Li1997 as data
from ionchannelABC.protocol import availability_linear,availability
from custom_protocols import recovery_tpreList, varying_test_duration_double_pulse, manual_steptrain_linear
import numpy as np
import myokit
import warnings
from scipy.optimize import OptimizeWarning
import scipy.optimize as so
import matplotlib.pyplot as plt


  
Q10_tau = 2.1 # Ten Tusscher paper, 2004
Q10_cond = 2.3 # Kiyosue paper, 1993 
room_temp = 296 # K
#######################################################################################################################
### IV curve - Li 1997
Li_iv_80_name = "IV   HP = -80mV"
Li_iv_80_desc ="""
    describes the protocol used to measure the IV peak-current curve in the Li Paper figure 1C

    page 2 of the paper : I-V relations of Ica were determined using 300-ms depolarizing steps every 10s from HP of -80,-60, and -40 mV
    The magnitude was measured as the difference between the peak inward current and the steady state current at the end of the depolarizing step
    
    protocol used in figure 1C: single test pulse at a frequency of 0.1Hz : every 10s, the voltage step occurs. 
    """

# DATA
# already converted in pA/pF
Vhold = -80 # mV
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
        current = d['i_caL.i_Ca_L'][:-1] # the last value is sometimes a nan 
        # (because V =0 at the end of the simulation and that I in nygren model is not defined for V = 0)
        output = output+[max(current, key = abs)-current[-1]]
    return output

# Experiment
Li_iv_80 = Experiment(
    name = Li_iv_80_name,
    dataset=Li_iv_80_dataset,
    protocol=Li_iv_80_protocol,
    conditions=Li_conditions,
    sum_stats=Li_iv_80_sum_stats,
    description=Li_iv_80_desc,
    Q10 = Q10_cond,
    Q10_factor = 1)



#######################################################################################################################
### IV curve - Li 1997
Li_iv_60_name = "IV   HP = -60mV"
Li_iv_60_desc ="""
    describes the protocol used to measure the IV peak-current curve in the Li Paper figure 1C

    page 2 of the paper : I-V relations of Ica were determined using 300-ms depolarizing steps every 10s from HP of -80,-60, and -40 mV
    The magnitude was measured as the difference between the peak inward current and the steady state current at the end of the depolarizing step
    
    protocol used in figure 1C: single test pulse at a frequency of 0.1Hz : every 10s, the voltage step occurs. 
    """

# DATA
# already converted in pA/pF
Vhold = -60 # mV
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
        current = d['i_caL.i_Ca_L'][:-1] # the last value is sometimes a nan 
        # (because V =0 at the end of the simulation and that I in nygren model is not defined for V = 0)
        output = output+[max(current, key = abs)-current[-1]]
    return output

# Experiment
Li_iv_60 = Experiment(
    name = Li_iv_60_name,
    dataset=Li_iv_60_dataset,
    protocol=Li_iv_60_protocol,
    conditions=Li_conditions,
    sum_stats=Li_iv_60_sum_stats,
    description=Li_iv_60_desc,
    Q10 = Q10_cond,
    Q10_factor = 1)

#######################################################################################################################
### IV curve - Li 1997
Li_iv_40_name = "IV   HP = -40mV"
Li_iv_40_desc ="""
    describes the protocol used to measure the IV peak-current curve in the Li Paper figure 1C

    page 2 of the paper : I-V relations of Ica were determined using 300-ms depolarizing steps every 10s from HP of -80,-60, and -40 mV
    The magnitude was measured as the difference between the peak inward current and the steady state current at the end of the depolarizing step
    
    protocol used in figure 1C: single test pulse at a frequency of 0.1Hz : every 10s, the voltage step occurs. 
    """

# DATA
# already converted in pA/pF
Vhold = -40 # mV
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


Li_iv_40_protocol = myokit.pacing.steptrain_linear(Vlower,Vupper, dV, Vhold, tpreMeasuring_iv_Li, tstep) 

# CONDITIONS 
#TODO : find the correct conditions of the paper for the concentrations of the different species.
Li_conditions = {'membrane.T': room_temp}

# SUMMARY STATISTICS
def Li_iv_40_sum_stats(data):
    output = []
    for d in data.split_periodic(tperiod_iv_Li, adjust=True):
        d = d.trim_left(tpreMeasuring_iv_Li, adjust=True)
        current = d['i_caL.i_Ca_L'][:-1] # the last value is sometimes a nan 
        # (because V =0 at the end of the simulation and that I in nygren model is not defined for V = 0)
        output = output+[max(current, key = abs)-current[-1]]
    return output

# Experiment
Li_iv_40 = Experiment(
    name = Li_iv_40_name,
    dataset=Li_iv_40_dataset,
    protocol=Li_iv_40_protocol,
    conditions=Li_conditions,
    sum_stats=Li_iv_40_sum_stats,
    description=Li_iv_40_desc,
    Q10 = Q10_cond,
    Q10_factor = 1)


#######################################################################################################################
### activation curve - Li 1997
Li_act_name = "Activation"
Li_act_desc = """
    the protocol used to measure the activation curve in the Li Paper (figure 2B) is not decribed (whereas the inactivation ones are)
    
    the protocol is assumed to be the same than for the IV curves

    single 300ms test pulse at a frequency of 0.1Hz
    """


# DATA
vsteps_act, act, sd_act = data.Act_Li()
variances_act = [(sd_)**2 for sd_ in sd_act]
Li_act_dataset = np.asarray([vsteps_act, act, variances_act])

# PROTOCOL
tperiod_act_Li = 10000 # ms
tstep = 300 # ms (not precised in the paper, 1s seems enough to mesure the peak current)
tpre = tperiod_act_Li - tstep # before the first pulse occurs
tpreMeasuring_act_Li = tperiod_act_Li - tstep # before the measurement

Vhold = -80 #mV
Vlower = -80
dV = 10
Vupper = 20 + dV


Li_act_protocol = manual_steptrain_linear(vsteps_act, Vhold, tpre, tstep) 

# CONDITIONS
#TODO : find the correct conditions of the paper for the concentrations of the different species.
Li_conditions = {'membrane.T': room_temp}

# SUMMARY STATISTICS
def Li_act_sum_stats(data):
    output = []
    for d in data.split_periodic(tperiod_act_Li, adjust=True):
        d = d.trim_left(tpreMeasuring_act_Li, adjust=True)
        act_gate = d['i_caL.G_Na_norm']
        index = np.argmax(np.abs(act_gate))
        output = output+[np.abs(act_gate[index])]
    return output

# Experiment
Li_act = Experiment(
    name = Li_act_name,
    dataset=Li_act_dataset,
    protocol=Li_act_protocol,
    conditions=Li_conditions,
    sum_stats=Li_act_sum_stats,
    description=Li_act_desc)


#######################################################################################################################
### Inactivation curve - Li 1997
Li_inact_1000_name = "Inactivation tstep = 1000 ms"
Li_inact_1000_desc = """
    describes the protocol used to measure the activation curve in the Li Paper (figure 2B)
    
    page 3 of the paper :
        Prepulses of varying durations ... absence of a prepulse

    The protocol is a double pulse protocol at the frequency of 0.1Hz
    """

# DATA
tstep = 1000 # ms
vsteps_inact, inact, sd_inact = data.inact_Li_all(tstep)
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
        inact_gate = d['i_caL.G_Na_norm']
        index = np.argmax(np.abs(inact_gate))
        output = output+[np.abs(inact_gate[index])]
    Norm = output[0]
    for i in range(len(output)):
        output[i] /= Norm
    return output

# Experiment
Li_inact_1000 = Experiment(
    name = Li_inact_1000_name,
    dataset=Li_inact_1000_dataset,
    protocol=Li_inact_1000_protocol,
    conditions=Li_conditions,
    sum_stats=Li_inact_1000_sum_stats,
    description=Li_inact_1000_desc)

#######################################################################################################################
### Inactivation curve - Li 1997
Li_inact_300_name = "Inactivation tstep = 300 ms"
Li_inact_300_desc = """
    describes the protocol used to measure the activation curve in the Li Paper (figure 2B)
    
    page 3 of the paper :
        Prepulses of varying durations ... absence of a prepulse

    The protocol is a double pulse protocol at the frequency of 0.1Hz
    """

# DATA
tstep = 300 # ms
vsteps_inact, inact, sd_inact = data.inact_Li_all(tstep)
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
        inact_gate = d['i_caL.G_Na_norm']
        index = np.argmax(np.abs(inact_gate))
        output = output+[np.abs(inact_gate[index])]
    Norm = output[0]
    for i in range(len(output)):
        output[i] /= Norm
    return output

# Experiment
Li_inact_300 = Experiment(
    name = Li_inact_300_name,
    dataset=Li_inact_300_dataset,
    protocol=Li_inact_300_protocol,
    conditions=Li_conditions,
    sum_stats=Li_inact_300_sum_stats,
    description=Li_inact_300_desc)

#######################################################################################################################
### Inactivation curve - Li 1997
Li_inact_150_name = "Inactivation tstep = 150 ms"
Li_inact_150_desc = """
    describes the protocol used to measure the activation curve in the Li Paper (figure 2B)
    
    page 3 of the paper :
        Prepulses of varying durations ... absence of a prepulse

    The protocol is a double pulse protocol at the frequency of 0.1Hz
    """

# DATA
tstep = 150 # ms
vsteps_inact, inact, sd_inact = data.inact_Li_all(tstep)
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
        inact_gate = d['i_caL.G_Na_norm']
        index = np.argmax(np.abs(inact_gate))
        output = output+[np.abs(inact_gate[index])]
    Norm = output[0]
    for i in range(len(output)):
        output[i] /= Norm
    return output

# Experiment
Li_inact_150 = Experiment(
    name = Li_inact_150_name,
    dataset=Li_inact_150_dataset,
    protocol=Li_inact_150_protocol,
    conditions=Li_conditions,
    sum_stats=Li_inact_150_sum_stats,
    description=Li_inact_150_desc)

#######################################################################################################################
###  Inactivation kinetics Hp -80 - Li 1997
Li_inact_kin_80_name = "Inactivation Kinetics, HP = -80 mV"
Li_inact_kin_80_desc = """
    describes the protocol used to measure the inactivation kinetics (tau_f and tau_s) in the Li Paper (figure 3B)
    
    the Voltage goes from -10mV to 30mV for this function with a dV = 10 mV.

    page 4 of the paper :

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
Vupper = 30
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
        #d = d.trim(tpreMeasuring_kin_80_Li,tpreMeasuring_kin_80_Li+2, adjust = True)
        
        current = d['i_caL.i_Ca_L'][:-1] # sometimes, the last value is nan and crashes the following,
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
                
                #debug
                #plt.plot(time,current,time,double_exp(np.asarray(time), popt[0], popt[1], popt[2], popt[3]))
                #plt.show()
            except:
                output = output+[float('inf')]
                ss_list = ss_list+[float('inf')]
                
    output = output+ss_list
    return output

# Experiment
Li_inact_kin_80 = Experiment(
    name = Li_inact_kin_80_name,
    dataset=Li_inact_kin_80_dataset,
    protocol=Li_inact_kin_80_protocol,
    conditions=Li_conditions,
    sum_stats=Li_inact_kin_80_sum_stats,
    description=Li_inact_kin_80_desc,
    Q10 = Q10_tau,
    Q10_factor = -1)

#######################################################################################################################
###  Inactivation kinetics Hp -60 - Li 1997
Li_inact_kin_60_name = "Inactivation Kinetics, HP = -60 mV"
Li_inact_kin_60_desc = """
    describes the protocol used to measure the inactivation kinetics (tau_f and tau_s) in the Li Paper (figure 3B)
    
    the Voltage goes from -10mV to 30mV for this function with a dV = 10 mV.

    page 4 of the paper :

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
Vupper = 30
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
        
        current = d['i_caL.i_Ca_L'][:-1] # sometimes, the last value is nan and crashes the following,
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
    name = Li_inact_kin_60_name,
    dataset=Li_inact_kin_60_dataset,
    protocol=Li_inact_kin_60_protocol,
    conditions=Li_conditions,
    sum_stats=Li_inact_kin_60_sum_stats,
    description=Li_inact_kin_60_desc,
    Q10 = Q10_tau,
    Q10_factor = -1)

#######################################################################################################################
###  Inactivation kinetics Hp -40 - Li 1997
Li_inact_kin_40_name = "Inactivation Kinetics, HP = -40 mV"
Li_inact_kin_40_desc =  """
    describes the protocol used to measure the inactivation kinetics (tau_f and tau_s) in the Li Paper (figure 3B)
    
    the Voltage goes from -10mV to 30mV for this function with a dV = 10 mV.

    page 4 of the paper :

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
Vupper = 30
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
        
        current = d['i_caL.i_Ca_L'][:-1] # sometimes, the last value is nan and crashes the following,
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
    name = Li_inact_kin_40_name,
    dataset=Li_inact_kin_40_dataset,
    protocol=Li_inact_kin_40_protocol,
    conditions=Li_conditions,
    sum_stats=Li_inact_kin_40_sum_stats,
    description=Li_inact_kin_40_desc,
    Q10 = Q10_tau,
    Q10_factor = -1)


#######################################################################################################################
###  recovery curves - Li 1997
Li_recov_80_name = "Recovery, HP = -80 mV"
Li_recov_80_desc =   """
    describes the protocol used to measure the Recovery of I_ca_L in the Li Paper (figure 4B)
    
    the Vhold used here is -80mV
    
    The protocol is a double pulse protocol at the frequency of 0.1Hz
    
    """  

# DATA
Li_recov_80_dataset = []
Vhold = -80
time,recov,sd_recov = data.Recov_Li_all(Vhold)
variances_recov = [(sd_)**2 for sd_ in sd_recov]
Li_recov_80_dataset.append(np.asarray([time, recov, variances_recov]))


# PROTOCOL
tpreMeasuringList1_recov_80 = []

tperiod_recov_80 = 10000 # ms
tstep1 = 300
tstep2 = 300

tMeasuring1_recov_80 = tstep1 
tpreMeasuring2_recov_80 = tperiod_recov_80 - tstep2 

Vstep1 = 10
Vstep2 = 10  

Split_list_recov_80 = [] # for the summary statistics function

twaitList,_,_ = data.Recov_Li_all(Vhold)

Split_list_recov_80.append(len(twaitList))


tpreList = []
for twait in twaitList:
    tpre = tperiod_recov_80 - tstep1 - twait - tstep2
    tpreList.append(tpre)
    tpreMeasuringList1_recov_80.append(tpre)

Li_recov_80_protocol = recovery_tpreList(twaitList,Vhold,Vstep1, Vstep2,tpreList,tstep1,tstep2)

# CONDITIONS
#TODO : find the correct conditions of the paper for the concentrations of the different species.
Li_conditions = {'membrane.T': room_temp}

# SUMMARY STATISTICS
def Li_recov_80_sum_stats(data):
    output = []
    sub_loop = 0

    d_split = data.split_periodic(tperiod_recov_80, adjust = True)

    for d in d_split:

        dcond = d.trim(tpreMeasuringList1_recov_80[sub_loop], tpreMeasuringList1_recov_80[sub_loop]+tMeasuring1_recov_80, adjust = True)
        dtest = d.trim_left(tpreMeasuring2_recov_80, adjust = True)
        
        current_cond = dcond['i_caL.i_Ca_L'][:-1]
        current_test = dtest['i_caL.i_Ca_L'][:-1]


        index_cond = np.argmax(np.abs(current_cond))
        index_test = np.argmax(np.abs(current_test))
        try :
            output = output + [current_test[index_test] / current_cond[index_cond]]  # should I still normalize ?
            sub_loop += 1
        except :
            output = output + [float('inf')]  
            sub_loop += 1

    return output

# Experiment
Li_recov_80 = Experiment(
    name = Li_recov_80_name,
    dataset=Li_recov_80_dataset,
    protocol=Li_recov_80_protocol,
    conditions=Li_conditions,
    sum_stats=Li_recov_80_sum_stats,
    description=Li_recov_80_desc)