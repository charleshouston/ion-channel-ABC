#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 15:38:28 2019

@author: brm18
"""

from ionchannelABC.experiment import Experiment
import data.ina.Schneider1994.data_Schneider1994 as dataSch
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
Q_10_cond = 1.5 # Correa paper, 1991 : To adjust the datas to 37°


#######################################################################################################################
### IV curve - Schneider 1994
schneider_iv_desc = """
    describes the protocol used to measure the IV peak-current curve in the Schneider Paper figure 1B
    
    page 2 of the paper :

    depolarizing the membrane from -135 mV to test
    potentials varying from -85 to +55 mV in 10-mV.

    simple pulse protocol at the frequency 1Hz (the frequency of the pulses is not given)
    """

# DATA
vsteps, peaks, sd = dataSch.IV_Schneider()
cm_mean = 89 # pF
cm_sd = 26.7
# convert nA to pA/pF
peaks = np.array(peaks)
peaks = peaks*1000/cm_mean
sd = [(cm_sd/cm_mean)*p for p in peaks]
schneider_iv_max_peak = np.max(np.abs(peaks)) # normalising
peaks = [p / schneider_iv_max_peak for p in peaks]
variances = [(sd_ / schneider_iv_max_peak)**2 for sd_ in sd]
schneider_iv_dataset = np.asarray([vsteps, peaks, variances])

# PROTOCOL
tperiod_iv_sch = 1000 # ms (not precised in the paper, 1s seems enough)
tstep = 12 # ms 
tpre = tperiod_iv_sch - tstep # before the first pulse occurs
tpreMeasuring_iv_sch = tperiod_iv_sch - tstep # before the measurement

Vhold = -135 # mV
Vlower = -85.001 # modified to not go through V = 0 that makes the nygren model crash ( expression of I_na)
Vupper = 55
dV = 10

schneider_iv_protocol = myokit.pacing.steptrain_linear(Vlower,Vupper + dV, dV, Vhold, tpre, tstep) 

# CONDITIONS
schneider_conditions = {'membrane.Na_o': 120,
            'membrane.Na_i': 70,
            'membrane.T': 297.15}

# SUMMARY STATISTICS
def schneider_iv_sum_stats(data):
    output = []
    for d in data.split_periodic(tperiod_iv_sch, adjust=True):
        d = d.trim_left(tperiod_iv_sch, adjust=True)
        current = d['ina.i_Na']
        index = np.argmax(np.abs(current))
        output = output+[current[index]/schneider_iv_max_peak]
    return output

schneider_iv = Experiment(
    dataset=schneider_iv_dataset,
    protocol=schneider_iv_protocol,
    conditions=schneider_conditions,
    sum_stats=schneider_iv_sum_stats,
    description=schneider_iv_desc
)



#######################################################################################################################
### Activation kinetics Schneider 1994
schneider_taum_desc =     """
    describes the protocol used to measure the activation curve in the Schneider Paper (figure 3C)

    The protocol is not described !
    By hypothesis, this will be the same as the protocol used in figure 1B (IV curve)

    Temperature adjusted from 297 to 310 using Q10_tau of 2.79."""

# DATA
vsteps_tm, tm, sd_tm = dataSch.TauM_Activation_Schneider()
tm = [temperature_adjust(tm_, 297, 310, Q10_tau) for tm_ in tm]
max_tm = np.max(np.abs(tm)) # normalising
tm = [tm_ / max_tm for tm_ in tm]
variances_tm = [(sd_/max_tm)**2 for sd_ in sd_tm]
schneider_taum_dataset = np.asarray([vsteps_tm, tm, variances_tm])

# PROTOCOL
tperiod_act_kin_sch = 1000 # ms
tstep = 12 # ms (not precised in the paper, 1s seems enough)
tpre = tperiod_act_kin_sch - tstep # before the first pulse occurs
tpreMeasuring_act_kin_sch = tperiod_act_kin_sch - tstep # before the measurement

Vhold = -135 # mV

schneider_taum_protocol = manual_steptrain_linear(vsteps_tm,Vhold,tpre,tstep) 

# CONDITIONS
schneider_conditions = {'membrane.Na_o': 120,
            'membrane.Na_i': 70,
            'membrane.T': 297.15}
 
# SUMMARY STATISTICS
def schneider_taum_sum_stats(data):
    def sum_of_exp(t, taum, tauh):
        return ((1-np.exp(-t/taum))**3 *
                np.exp(-t/tauh))
    output = []
    for d in data.split_periodic(tperiod_act_kin_sch, adjust=True):
        d = d.trim_left(tpreMeasuring_act_kin_sch, adjust=True)
        current = d['ina.i_Na']
        time = d['environment.time']

        # Remove constant
        c0 = d['ina.i_Na'][0]
        current = [(c_-c0) for c_ in current]


        with warnings.catch_warnings():
            warnings.simplefilter('error', so.OptimizeWarning)
            warnings.simplefilter('error', RuntimeWarning)
            try:
                imax = max(current, key=abs)
                current = [c_/imax for c_ in current]
                if len(time)<=1 or len(current)<=1:
                    raise Exception('Failed simulation')
                popt, _ = so.curve_fit(sum_of_exp, time, current,
                                       p0=[0.5, 1.],
                                       bounds=([0., 0.],
                                               [1., 100.]))
                output = output+[popt[0]/max_tm]
            except:
                output = output+[float('inf')]
    return output

schneider_taum = Experiment(
    dataset=schneider_taum_dataset,
    protocol=schneider_taum_protocol,
    conditions=schneider_conditions,
    sum_stats=schneider_taum_sum_stats,
    description=schneider_taum_desc
)

#######################################################################################################################
### Fast Inactivation kinetics Schneider 1994
schneider_tauf_desc =     """
    describes the protocol used to measure the Inactivation curve in the Schneider Paper (figure 3C)

    The protocol is not described !
    By hypothesis, this will be the same as the protocol used in figure 1B (IV curve)

    Temperature adjusted from 297 to 310 using Q10_tau of 2.79."""

# DATA
vsteps_tf, tf, sd_tf = dataSch.TauF_Inactivation_Schneider()
tf = [temperature_adjust(tf_, 297, 310, Q10_tau) for tf_ in tf]
max_tf = np.max(np.abs(tf)) # normalising
tf = [tf_ / max_tf for tf_ in tf]
variances_tf = [(sd_/max_tf)**2 for sd_ in sd_tf]
schneider_tauf_dataset = np.asarray([vsteps_tf, tf, variances_tf])

# PROTOCOL
tperiod_inact_kin_sch = 1000 # ms
tstep = 12 # ms (not precised in the paper, 1s seems enough)
tpre = tperiod_inact_kin_sch - tstep # before the first pulse occurs
tpreMeasuring_inact_kin_sch = tperiod_inact_kin_sch - tstep # before the measurement

Vhold = -135 # mV

schneider_tauf_protocol = manual_steptrain_linear(vsteps_tf,Vhold,tpre,tstep) 


# CONDITIONS
schneider_conditions = {'membrane.Na_o': 120,
            'membrane.Na_i': 70,
            'membrane.T': 297.15}
 
# SUMMARY STATISTICS
def schneider_tauf_sum_stats(data):
    def sum_of_exp(t, taum, tauh):
        return ((1-np.exp(-t/taum))**3 *
                np.exp(-t/tauh))
    output = []
    for d in data.split_periodic(tperiod_inact_kin_sch, adjust=True):
        d = d.trim_left(tpreMeasuring_inact_kin_sch, adjust=True)
        current = d['ina.i_Na']
        time = d['environment.time']

        # Remove constant
        c0 = d['ina.i_Na'][0]
        current = [(c_-c0) for c_ in current]

        with warnings.catch_warnings():
            warnings.simplefilter('error', so.OptimizeWarning)
            warnings.simplefilter('error', RuntimeWarning)
            try:
                imax = max(current, key=abs)
                current = [c_/imax for c_ in current]
                if len(time)<=1 or len(current)<=1:
                    raise Exception('Failed simulation')
                popt, _ = so.curve_fit(sum_of_exp, time, current,
                                       p0=[0.5, 1.],
                                       bounds=([0., 0.],
                                               [1., 100.]))
                output = output+[popt[1]/max_tf]
            except:
                output = output+[float('inf')]
    return output

schneider_tauf = Experiment(
    dataset=schneider_tauf_dataset,
    protocol=schneider_tauf_protocol,
    conditions=schneider_conditions,
    sum_stats=schneider_tauf_sum_stats,
    description=schneider_tauf_desc
)
    

#######################################################################################################################
### Slow Inactivation kinetics Schneider 1994
schneider_taus_desc =     """
    describes the protocol used to measure the Inactivation curve in the Schneider Paper (figure 5B)
    
    
    page 4 of the paper :
    Slow inactivation was investigated in the potential
    range from -115 to -55 mV with prepulses of variable
    duration up to 1024 ms. The fraction of available Na
    channels was again determined with test pulses to
    -20 mV

    By hypothesis, this will be the same as the protocol used in figure 1B (IV curve)
    
    Temperature adjusted from 297 to 310 using Q10_tau of 2.79."""

# DATA
vsteps_ts, ts, sd_ts = dataSch.TauS_Inactivation_Schneider()
ts = [temperature_adjust(ts_, 297, 310, Q10_tau) for ts_ in ts]
max_ts = np.max(np.abs(ts)) # normalising
ts = [ts_ / max_ts for ts_ in ts]
variances_ts = [(sd_/max_ts)**2 for sd_ in sd_ts]
schneider_taus_dataset = np.asarray([vsteps_ts, ts, variances_ts])

# PROTOCOL
tperiod_inact_kin_slow_sch = 1000 # ms
tstep = 12 # ms (not precised in the paper, 1s seems enough)
tpre = tperiod_inact_kin_slow_sch - tstep # before the first pulse occurs
tpreMeasuring_inact_kin_slow_sch = tperiod_inact_kin_slow_sch - tstep # before the measurement

Vhold = -135 # mV

schneider_taus_protocol = manual_steptrain_linear(vsteps_ts,Vhold,tpre,tstep) 


# CONDITIONS
schneider_conditions = {'membrane.Na_o': 120,
            'membrane.Na_i': 70,
            'membrane.T': 297.15}
 
# SUMMARY STATISTICS
def schneider_taus_sum_stats(data):
    def sum_of_exp(t, taum, tauh):
        return ((1-np.exp(-t/taum))**3 *
                np.exp(-t/tauh))
    output = []
    for d in data.split_periodic(tperiod_inact_kin_slow_sch, adjust=True):
        d = d.trim_left(tpreMeasuring_inact_kin_slow_sch, adjust=True)
        current = d['ina.i_Na']
        time = d['environment.time']

        # Remove constant
        c0 = d['ina.i_Na'][0]
        current = [(c_-c0) for c_ in current]


        with warnings.catch_warnings():
            warnings.simplefilter('error', so.OptimizeWarning)
            warnings.simplefilter('error', RuntimeWarning)
            try:
                imax = max(current, key=abs)
                current = [c_/imax for c_ in current]
                if len(time)<=1 or len(current)<=1:
                    raise Exception('Failed simulation')
                popt, _ = so.curve_fit(sum_of_exp, time, current,
                                       p0=[0.5, 1.],
                                       bounds=([0., 0.],
                                               [1., 100.]))
                output = output+[popt[1]/max_ts]
            except:
                output = output+[float('inf')]
    return output

schneider_taus = Experiment(
    dataset=schneider_taus_dataset,
    protocol=schneider_taus_protocol,
    conditions=schneider_conditions,
    sum_stats=schneider_taus_sum_stats,
    description=schneider_taus_desc
)

#######################################################################################################################
### Inactivation Schneider 1994
schneider_inact_desc = """
    describes the protocol used to measure the activation curve in the Schneider Paper (figure 4)
    

    page 4 of the paper :

    Starting from a conditioning
    potential of -135 mV, the cells were depolarized
    with prepulses varying from -135 mV to +5 mV in
    10-mV steps with a subsequent test pulse to -20 mV.
    To test for fast and intermediate inactivation, h infinity curves
    were determined with five different prepulse durations
    (increasing from 32 ms to 512ms by factors of 2,
    Fig. 4). 

    the double pulse protocol is not more described so the sakakibara one has been applied to complete the lack of information
    """

# DATA
schneider_inact_dataset = []
for tprepulse in [32,64,128,256,512]:
    t_prepulse,inact,_ = dataSch.Inact_Schneider_all(tprepulse)
    variances_inact = [0 for t in t_prepulse]
    schneider_inact_dataset.append(np.asarray([t_prepulse,inact, variances_inact]))

# PROTOCOL
tmp_protocol = []
    
tperiod_inact_sch = 3000 # ms
tstep = 500 # ms (not precised in the paper, 0.5s seems enough to mesure the peak current)
twait = 2 
ttest = 30
tpreMeasuring_inact_sch = tperiod_inact_sch - ttest -twait -tstep
tMeasuring_inact_sch = tstep

Vhold = -135 # mV
Vtest = -20
Vlower = -135.001 
Vupper = 5
dV = 10

for tstep in [32,64,128,256,512]:

    tpre = tperiod_inact_sch - tstep - twait - ttest
    protocol = availability_linear(Vlower,Vupper, dV,Vhold, Vtest, tpre, tstep, twait, ttest)

    tmp_protocol.append(protocol)

 
# Fuse all the protocols into one
schneider_inact_protocol = tmp_protocol[0]
for p in tmp_protocol[1:]:
    for e in p.events():
        schneider_inact_protocol.add_step(e.level(), e.duration())

# CONDITIONS
schneider_conditions = {'membrane.Na_o': 120,
            'membrane.Na_i': 70,
            'membrane.T': 297.15}
 
# SUMMARY STATISTICS
def schneider_inact_sum_stats(data):

    output = []
    for d in data.split_periodic(tperiod_inact_sch, adjust=True):
        d = d.trim(tpreMeasuring_inact_sch,tpreMeasuring_inact_sch  + tMeasuring_inact_sch, adjust = True)
        inact_gate = d['ina.h_infinity']
        index = np.argmax(np.abs(inact_gate))
        output = output+[np.abs(inact_gate[index])]
    return output

schneider_inact = Experiment(
    dataset=schneider_inact_dataset,
    protocol=schneider_inact_protocol,
    conditions=schneider_conditions,
    sum_stats=schneider_inact_sum_stats,
    description=schneider_inact_desc
)
    
#######################################################################################################################
### Reduction Schneider 1994
schneider_reduc_desc ="""
    describes the protocol used to measure the Relative_peak IV curve in the Schneider Paper (figure 5A)
    
    page 5 of the paper :

    Reduction of the Na + current peak
    amplitude elicited by pulses to -20 mV following prepotentials
    varying in time (abscissa) and magnitude. Symbols represent
    different inactivation potential values: -105 mV to -65 mV

    the double pulse protocol is not more described so the sakakibara one has been applied to complete the lack of information
    """

# DATA
schneider_reduc_dataset = []
for Vinact in [-105,-95,-85,-75,-65]:
    tstepList,reduc,_ = dataSch.Reduction_Schneider_all(Vinact)
    variances_reduc = [0 for t in tstepList]
    schneider_reduc_dataset.append(np.asarray([tstepList,reduc, variances_reduc]))

# PROTOCOL
tmp_protocol = []

## This part is for initializing the protocol : to determine the max I_Na when there is no conditionning pulse
tperiod_reduc_sch = 3000 # ms seems enough for a double pulse protocol
twait = 0
tstep = 0
ttest = 30
tpre = tperiod_reduc_sch - twait - ttest - tstep

tpreMeasuring_reduc_sch = tperiod_reduc_sch - ttest
tMeasuring_reduc_sch = ttest

Vhold = -135 # mV
Vtest = -20   
       
protocol = availability([Vhold],Vhold, Vtest, tpre, tstep, twait, ttest)
tmp_protocol.append(protocol)

## This part is the main protocol
twait = 2 
tstep = 1000
ttest = 30

for Vinact in [-105,-95,-85,-75,-65]:
    tstepList,_,_ = dataSch.Reduction_Schneider_all(Vinact)

    tpreList = []
    for tstep in tstepList :
        tpre = tperiod_reduc_sch - tstep - twait - ttest
        tpreList.append(tpre)
                      
    protocol = varying_test_duration_double_pulse(Vinact,Vhold, Vtest, tpreList, tstepList, twait, ttest)
    tmp_protocol.append(protocol)

# Fuse all the protocols into one
schneider_reduc_protocol = tmp_protocol[0]
for p in tmp_protocol[1:]:
    for e in p.events():
        schneider_reduc_protocol.add_step(e.level(), e.duration())

# CONDITIONS
schneider_conditions = {'membrane.Na_o': 120,
            'membrane.Na_i': 70,
            'membrane.T': 297.15}
 
# SUMMARY STATISTICS
def schneider_reduc_sum_stats(data):

    output = []
    d_split,d_init = data.split(tperiod_reduc_sch)

    d_init = d_init.trim_left(tpreMeasuring_reduc_sch, adjust = True)
    current = d_init['ina.i_Na']

    current = current[:-1]
    index = np.argmax(np.abs(current))
    normalizing_peak = current[index] 

    for d in d_split.split_periodic(tperiod_reduc_sch, adjust=True):
        d = d.trim(tpreMeasuring_reduc_sch,tpreMeasuring_inact_sch  + tMeasuring_reduc_sch, adjust = True)
        current = d['ina.i_Na'][:-1]
        index = np.argmax(np.abs(current))
        try :
            output = output+[np.abs(current[index])/normalizing_peak]
        except :
            output = output + [float('inf')]
    return output

schneider_reduc = Experiment(
    dataset=schneider_reduc_dataset,
    protocol=schneider_reduc_protocol,
    conditions=schneider_conditions,
    sum_stats=schneider_reduc_sum_stats,
    description=schneider_reduc_desc
)
    

#######################################################################################################################
###  recovery curves - Schneider 1992
schneider_recov_desc =    """
    describes the protocol used to measure the Recovery of I_na in the Sakakibara Paper (figure 8A)
    
    the Vhold used here is -140mV ,-120mV and -100mV
    
    page 8 of the paper : 
    The double-pulseprotocol shown in the inset was applied at various recovery potentials at a frequency of 0.1 Hz. The magnitude of the fast
    Na+ current during the test pulse was normalized to that
    induced by the conditioning pulse. 


    The protocol is a double pulse protocol at the frequency of 0.1Hz
    
    """  

# DATA
schneider_recov_dataset = []
for Vhold in [-135,-125,-115,-105,-95,-85,-75] :
    time,recov,_ = dataSch.Recovery_Schneider_all(Vhold)
    variances_time = [0 for t in time]
    schneider_recov_dataset.append(np.asarray([time, recov, variances_time]))


# PROTOCOL
tmp_protocol = []
tpreMeasuringList1_recov_sch = []

tperiod_recov_sch = 5000 # ms
tstep1 = 200
tstep2 = 12

tMeasuring1_recov_sch = tstep1 
tpreMeasuring2_recov_sch = tperiod_recov_sch - tstep2 

Vstep1 = -20
Vstep2 = -20  

Split_list_recov_sch = [] # for the summary statistics function

for Vhold in [-135,-125,-115,-105,-95,-85,-75] :
    twaitList,_,_ = dataSch.Recovery_Schneider_all(Vhold)

    Split_list_recov_sch.append(len(twaitList))


    tpreList = []
    for twait in twaitList:
        tpre = tperiod_recov_sch - tstep1 - twait - tstep2
        tpreList.append(tpre)
        tpreMeasuringList1_recov_sch.append(tpre)

    protocol = recovery_tpreList(twaitList,Vhold,Vstep1, Vstep2,tpreList,tstep1,tstep2)
    
    tmp_protocol.append(protocol)

# Fuse all the protocols into one
schneider_recov_protocol = tmp_protocol[0]
for p in tmp_protocol[1:]:
    for e in p.events():
        schneider_recov_protocol.add_step(e.level(), e.duration())

# CONDITIONS
schneider_conditions = {'membrane.Na_o': 120,
            'membrane.Na_i': 70,
            'membrane.T': 297.15}

# SUMMARY STATISTICS
def schneider_recov_sum_stats(data):
    output = []
    loop = 0
    sub_loop = 0

    # spliting back the protocols
    Cumulated_len = Split_list_recov_sch[0]
    dProtocolOne,dProtocoltwoseven = data.split(tperiod_recov*Cumulated_len)
    Cumulated_len += Split_list_recov_sch[1]
    dProtocolTwo,dProtocolThreeseven  = dProtocoltwoseven.split(tperiod_recov*Cumulated_len)
    Cumulated_len += Split_list_recov_sch[2]
    dProtocolThree,dProtocolfourseven = dProtocolThreeseven.split(tperiod_recov*Cumulated_len)
    Cumulated_len += Split_list_recov_sch[3]
    dProtocolFour,dProtocolfiveseven  = dProtocolfourseven.split(tperiod_recov*Cumulated_len)
    Cumulated_len += Split_list_recov_sch[4]
    dProtocolFive,dProtocolsixseven = dProtocolfiveseven.split(tperiod_recov*Cumulated_len)
    Cumulated_len += Split_list_recov_sch[5]
    dProtocolSix,dProtocolSeven  = dProtocolsixseven.split(tperiod_recov*Cumulated_len)

    dProtocols = [dProtocolOne,dProtocolTwo,dProtocolThree,dProtocolFour,dProtocolFive,dProtocolSix,dProtocolSeven]
    Cumulated_len = 0
    for dOneProtocol in dProtocols:
        d_split = dOneProtocol.split_periodic(tperiod_recov, adjust = True)

        if loop > 0 :
            Cumulated_len += Split_list_recov_sch[loop-1]

        d_split = d_split[Cumulated_len:]  


        for d in d_split:

            dcond = d.trim(tpreMeasuringList1_recov_sch[sub_loop],
             tpreMeasuringList1_recov_sch[sub_loop]+tMeasuring1_recov_sch, adjust = True)
            dtest = d.trim_left(tpreMeasuring2_recov_sch, adjust = True)
            
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
schneider_recov = Experiment(
    dataset=schneider_recov_dataset,
    protocol=schneider_recov_protocol,
    conditions=schneider_conditions,
    sum_stats=schneider_recov_sum_stats,
    description=schneider_recov_desc)