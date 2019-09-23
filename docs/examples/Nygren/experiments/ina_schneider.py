#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 15:38:28 2019

@author: brm18
Updated: Charles Houston 2019-09-20
"""

from ionchannelABC.experiment import Experiment
import data.ina.Schneider1994.data_Schneider1994 as dataSch
from ionchannelABC.protocol import  availability, availability_linear, recovery
from custom_protocols import recovery_tpreList, varying_test_duration_double_pulse,manual_steptrain_linear
import numpy as np
import myokit
import warnings
import scipy.optimize as so
import matplotlib.pyplot as plt


Q10_tau = 2.79 # [tenTusscher2004]
Q10_cond = 1.5 # [Correa1991]

fit_threshold = 0.95 # for curve_fit


#
# IV curve [Schneider1994]
#
schneider_iv_desc = """
    describes the protocol used to measure the IV peak-current curve in the Schneider Paper figure 1B

    page 2 of the paper :

    depolarizing the membrane from -135 mV to test
    potentials varying from -85 to +55 mV in 10-mV.
    """
vsteps_iv, peaks_iv, _ = dataSch.IV_Schneider()
cm_mean = 89 # pF
cm_sd = 26.7
# convert nA to pA/pF
peaks_iv = np.array(peaks_iv)
peaks_iv = peaks_iv*1000/cm_mean
sd_iv = [abs(cm_sd/cm_mean*p) for p in peaks_iv]
variances_iv = [(sd_)**2 for sd_ in sd_iv]
schneider_iv_dataset = np.asarray([vsteps_iv, peaks_iv, variances_iv])

norm_peak = np.max(np.abs(peaks_iv))
peaks_iv_norm = peaks_iv/norm_peak
sd_iv_norm = [abs(cm_sd/cm_mean*p) for p in peaks_iv_norm]
variances_iv_norm = [(sd_)**2 for sd_ in sd_iv_norm]
schneider_iv_normalised_dataset = np.asarray([vsteps_iv, peaks_iv_norm, variances_iv_norm])

tpre = 10000 # ms
tstep = 12 # ms
vhold = -135 # mV
vlower = -85+1e-5 # modified to not go through V = 0 that makes the nygren model crash ( expression of I_na)
vupper = 55+1e-5
dv = 10
schneider_iv_protocol = myokit.pacing.steptrain_linear(
    vlower, vupper+dv, dv, vhold, tpre, tstep)

schneider_conditions = {'na_conc.Na_o': 120, # mM
                        'na_conc.Na_i': 70,  # mM
                        'phys.T': 297.15}    # K

def schneider_iv_sum_stats(data):
    output = []
    for d in data.split_periodic(10012, adjust=True):
        d = d.trim_left(10000, adjust=True)
        current = d['ina.i_Na'][:-1]
        output = output+[max(current, key=abs)]
    return output

def schneider_iv_normalised_sum_stats(data):
    output = []
    for d in data.split_periodic(10012, adjust=True):
        d = d.trim_left(10000, adjust=True)
        current = d['ina.i_Na'][:-1]
        output = output+[max(current, key=abs)]
    output = np.array(output)
    norm_out = np.max(np.abs(output))
    output /= norm_out
    return output.tolist()

schneider_iv = Experiment(
    name = schneider_iv_name,
    dataset=schneider_iv_dataset,
    protocol=schneider_iv_protocol,
    conditions=schneider_conditions,
    sum_stats=schneider_iv_sum_stats,
    description=schneider_iv_desc,
    Q10=Q10_cond,
    Q10_factor=1)
schneider_iv_normalised = Experiment(
    dataset=schneider_iv_normalised_dataset,
    protocol=schneider_iv_protocol,
    conditions=schneider_conditions,
    sum_stats=schneider_iv_normalised_sum_stats,
    description=schneider_iv_desc,
    Q10=None,
    Q10_factor=0)


#
# Fast Activation and Inactivation kinetics [Schneider1994]
#
schneider_tau_desc = """
    describes the protocol used to measure the activation and fast inactivation time
    constants in Schneider Paper (figure 3C)

    The protocol is not described !
    By hypothesis, this will be the same as the protocol used in figure 1B (IV curve)
    """
vsteps_tm, tm, sd_tm = dataSch.TauM_Activation_Schneider()
variances_tm = [(sd_)**2 for sd_ in sd_tm]
schneider_taum_dataset = np.asarray([vsteps_tm, tm, variances_tm])

vsteps_tf, tf, sd_tf = dataSch.TauF_Inactivation_Schneider()
variances_tf = [(sd_)**2 for sd_ in sd_tf]
schneider_tauf_dataset = np.asarray([vsteps_tf, tf, variances_tf])

tpre = 10000 # ms
tstep = 12
vhold = -135 # mV
vlower = -65+1e-5
vupper = 105+1e-5
dv = 10
schneider_tau_protocol = myokit.pacing.steptrain_linear(
    vlower, vupper+dv, dv, vhold, tpre, tstep)

def schneider_tau_sum_stats(data):
    def sum_of_exp(t, taum, tauh, Imax, C):
        return Imax*(1-np.exp(-t/taum))**3*np.exp(-t/tauh)+C
    output1 = []
    output2 = []
    for i,d in enumerate(data.split_periodic(10012, adjust=True)):
        d = d.trim_left(10000, adjust=True)

        current = d['ina.i_Na'][:-1]
        time = d['engine.time'][:-1]

        with warnings.catch_warnings():
            warnings.simplefilter('error', so.OptimizeWarning)
            warnings.simplefilter('error', RuntimeWarning)
            try:
                if len(time)<=1 or len(current)<=1:
                    raise Exception('Failed simulation')
                popt, _ = so.curve_fit(sum_of_exp,
                                       time,
                                       current,
                                       p0=[0.5, 1., max(current,key=abs), 0.],
                                       bounds=([0., 0., -np.inf, -np.inf],
                                                np.inf),
                                       max_nfev=1000)

                fit = [sum_of_exp(t,popt[0],popt[1],popt[2],popt[3]) for t in time]
                # Calculate r2
                ss_res = np.sum((np.array(current)-np.array(fit))**2)
                ss_tot = np.sum((np.array(current)-np.mean(np.array(current)))**2)
                r2 = 1 - (ss_res / ss_tot)

                taum = popt[0]
                tauh = popt[1]
                if r2 > fit_threshold:
                    if i < 9: # only some data for activation time constants
                        output1 = output1+[taum]
                    output2 = output2+[tauh]
                else:
                    raise RuntimeWarning('scipy.optimize.curve_fit found a poor fit')
            except:
                if i < 9:
                    output1 = output1+[float('inf')]
                output2 = output2+[float('inf')]
    output = output1+output2
    return output

schneider_tau = Experiment(
    dataset=[schneider_taum_dataset,
             schneider_tauf_dataset],
    protocol=schneider_tau_protocol,
    conditions=schneider_conditions,
    sum_stats=schneider_tau_sum_stats,
    description=schneider_tau_desc,
    Q10=Q10_tau,
    Q10_factor=-1)


#
# Slow inactivation kinetics [Schneider1994]
#
schneider_taus_desc =     """
    describes the protocol used to measure slow inactivation kinetics
    in the Schneider Paper (figure 5B)

    page 4 of the paper :
    Slow inactivation was investigated in the potential
    range from -115 to -55 mV with prepulses of variable
    duration up to 1024 ms. The fraction of available Na
    channels was again determined with test pulses to
    -20 mV
    """
vsteps_ts, ts, sd_ts = dataSch.TauS_Inactivation_Schneider()
variances_ts = [(sd_)**2 for sd_ in sd_ts]
schneider_taus_dataset = np.asarray([vsteps_ts, ts, variances_ts])

# Assume logarithmic prepulses up to 1024ms
tprepulse = [2**i for i in range(1,11)] # ms
tpre = 10000
twait = 0
ttest = 12 # assumed same as IV curve
vhold = -135 # mV
vtest = -20
tmp_protocols = []
for vstep in vsteps_ts:
    tmp_protocols.append(
        varying_test_duration_double_pulse(
            vstep, vhold, vtest, tpre, tprepulse, twait, ttest)
    )
schneider_taus_protocol = tmp_protocols[0]
tsplit_tests = tmp_protocols[0].characteristic_time()
for p in tmp_protocols[1:]:
    for e in p.events():
        schneider_taus_protocol.add_step(e.level(), e.duration())

tsplits = [t+ttest+twait+tpre for t in tprepulse]
for i in range(len(tsplits)-1):
    tsplits[i+1] += tsplits[i]

def schneider_taus_sum_stats(data):
    def single_exp(t, tau, A, C):
        return A*np.exp(-t/tau)+C
    output = []
    # Split by test pulse potential
    for d in data.split_periodic(tsplit_tests, adjust=True):
        peaks = []
        for t in tsplits:
            d_, d = d.split(t)
            d_ = d_.trim_left(t-ttest, adjust=True)
            current = d_['ina.i_Na'][:-1]
            peaks = peaks + [max(current, key=abs)]
        norm = peaks[0]
        for i in range(len(peaks)):
            peaks[i] /= norm

        # fit to single exponential
        with warnings.catch_warnings():
            warnings.simplefilter('error', so.OptimizeWarning)
            warnings.simplefilter('error', RuntimeWarning)
            try:
                popt, _ = so.curve_fit(single_exp,
                                       tprepulse,
                                       peaks,
                                       p0=[40, 1., 0.],
                                       bounds=([0., -np.inf, -np.inf],
                                               np.inf),
                                       max_nfev=1000)

                fit = [single_exp(t,popt[0],popt[1],popt[2]) for t in tprepulse]
                # Calculate r2
                ss_res = np.sum((np.array(peaks)-np.array(fit))**2)
                ss_tot = np.sum((np.array(peaks)-np.mean(np.array(peaks)))**2)
                r2 = 1 - (ss_res / ss_tot)

                taus = popt[0]
                if r2 > fit_threshold:
                    output = output+[taus]
                else:
                    raise RuntimeWarning('scipy.optimize.curve_fit found a poor fit')
            except:
                output = output+[float('inf')]
    return output

schneider_taus = Experiment(
    name = schneider_taus_name,
    dataset=schneider_taus_dataset,
    protocol=schneider_taus_protocol,
    conditions=schneider_conditions,
    sum_stats=schneider_taus_sum_stats,
    description=schneider_taus_desc,
    Q10=Q10_tau,
    Q10_factor=-1)


#
# Inactivation [Schneider1994]
#
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
    """
prepulses, inflexion, sd_inflexion = dataSch.Inact_Schneider_Vh()
variances_inflex = [sd_**2 for sd_ in sd_inflexion]
schneider_inact_Vh_dataset = np.array([prepulses, inflexion, variances_inflex])
prepulses, slope, sd_slope = dataSch.Inact_Schneider_k()
variances_slope = [sd_**2 for sd_ in sd_slope]
schneider_inact_k_dataset = np.array([prepulses, slope, variances_slope])

vhold = -135 # mV
vtest = -20
tpre = 10000 # ms
tsteps_inact = prepulses
twait = 0
ttest = 12
vstart = -135
vend = 5
dv = 5
vsteps_inact = np.arange(-135, 10, 5).tolist()

tmp_protocols = []
for tstep in tsteps_inact:
    tmp_protocols.append(
        availability_linear(vstart, vend+dv, dv, vhold, vtest,
                            tpre, tstep, twait, ttest)
    )
tsplits_inact = [p.characteristic_time() for p in tmp_protocols]
for i in range(len(tsplits_inact)-1):
    tsplits_inact[i+1] += tsplits_inact[i]

schneider_inact_protocol = tmp_protocols[0]
for p in tmp_protocols[1:]:
    for e in p.events():
        schneider_inact_protocol.add_step(e.level(), e.duration())

def schneider_inact_sum_stats(data):
    def boltzmann_fn(V, Vh, k):
        return 1/(1+np.exp((V-Vh)/k))
    output_Vh = []
    output_k = []
    for i,t in enumerate(tsplits_inact):
        inact = []
        d, data = data.split(t)
        if i > 0:
            d = d.trim_left(tsplits_inact[i-1], adjust=True)
        for d_ in d.split_periodic(10012+tsteps_inact[i], adjust=True, closed_intervals=False):
            d_ = d_.trim_left(10000+tsteps_inact[i], adjust=True)
            inact = inact + [max(d_['ina.g'], key=abs)]
        norm = inact[0]
        for j in range(len(inact)):
            inact[j] /= norm

        # fit to boltzmann function
        with warnings.catch_warnings():
            warnings.simplefilter('error', so.OptimizeWarning)
            warnings.simplefilter('error', RuntimeWarning)
            try:
                popt, _ = so.curve_fit(boltzmann_fn,
                                       vsteps_inact,
                                       inact,
                                       p0=[-60, 5],
                                       bounds=(-np.inf,
                                               np.inf))

                fit = [boltzmann_fn(v,popt[0],popt[1]) for v in vsteps_inact]
                # Calculate r2
                ss_res = np.sum((np.array(inact)-np.array(fit))**2)
                ss_tot = np.sum((np.array(inact)-np.mean(np.array(inact)))**2)
                r2 = 1 - (ss_res / ss_tot)

                Vh = popt[0]
                k = popt[1]
                if r2 > fit_threshold:
                    output_Vh = output_Vh+[Vh]
                    output_k = output_k+[k]
                else:
                    raise RuntimeWarning('scipy.optimize.curve_fit found a poor fit')
            except:
                output_Vh = output_Vh+[float('inf')]
                output_k = output_k+[float('inf')]
    output = output_Vh+output_k
    return output

schneider_inact = Experiment(
    dataset=[schneider_inact_Vh_dataset,
             schneider_inact_k_dataset],
    protocol=schneider_inact_protocol,
    conditions=schneider_conditions,
    sum_stats=schneider_inact_sum_stats,
    description=schneider_inact_desc,
    Q10=None,
    Q10_factor=0
)


#
# Recovery [Schneider1994]
#
schneider_recov_desc =    """
    describes the protocol used to measure the Recovery of I_na in the Sakakibara Paper (figure 8A)

    the Vhold used here is -140mV ,-120mV and -100mV

    page 8 of the paper :
    The double-pulseprotocol shown in the inset was applied at various recovery potentials at a frequency of 0.1 Hz. The magnitude of the fast
    Na+ current during the test pulse was normalized to that
    induced by the conditioning pulse.


    The protocol is a double pulse protocol at the frequency of 0.1Hz
    """

prepulse_recov_r1,tau_r1,sd_r1 = dataSch.Recovery_Schneider_tau_r1()
variances_r1 = [sd_**2 for sd_ in sd_r1]
schneider_taur1_dataset = np.array([prepulse_recov_r1,tau_r1,variances_r1])
prepulse_recov_r2,tau_r2,sd_r2 = dataSch.Recovery_Schneider_tau_r2()
variances_r2 = [sd_**2 for sd_ in sd_r2]
schneider_taur2_dataset = np.array([prepulse_recov_r2,tau_r2,variances_r2])

tpre = 10000 #ms
tstep1 = 200
twaits_recov = [2**i for i in range(1,11)]
tstep2 = 12
vstep1 = -20
vstep2 = -20
vhold = -135

tmp_protocols = []
for v in prepulse_recov_r2:
    tmp_protocols.append(
        recovery(twaits_recov,vhold,vstep1,vstep2,tpre,tstep1,tstep2,v)
    )
schneider_recov_protocol = tmp_protocols[0]
tsplit_recov = tmp_protocols[0].characteristic_time()
for p in tmp_protocols[1:]:
    for e in p.events():
        schneider_recov_protocol.add_step(e.level(), e.duration())

tsplits_recov = [t+tstep1+tstep2+tpre for t in twaits_recov]
for i in range(len(tsplits_recov)-1):
    tsplits_recov[i+1] += tsplits_recov[i]

def schneider_recov_sum_stats(data):
    def double_exp(t, tau_r1, tau_r2, A0, A1, A2):
        return A0-A1*np.exp(-t/tau_r1)-A2*np.exp(-t/tau_r2)
    # single_exp fn makes sense when looking at [Schneider1994]
    def single_exp(t, tau_r2, A0, A2):
        return A0-A2*np.exp(-t/tau_r2)
    output1 = []
    output2 = []
    timename = 'engine.time'
    for i,d in enumerate(data.split_periodic(tsplit_recov, adjust=True, closed_intervals=True)):
        recov = []
        for t in tsplits_recov:
            d_, d = d.split(t)
            step1 = d_.trim(d_[timename][0]+tpre,
                            d_[timename][0]+tpre+tstep1,
                            adjust=True)
            step2 = d_.trim_left(t-tstep2, adjust=True)
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
            if i < len(prepulse_recov_r1):
                try:
                    popt, _ = so.curve_fit(double_exp,
                                           twaits_recov,
                                           recov,
                                           p0=[5.,20.,0.5,0.5,0.],
                                           bounds=([0.,0.,-np.inf,-np.inf,-np.inf],
                                                   np.inf),
                                           max_nfev=1000)

                    fit = [double_exp(t,popt[0],popt[1],popt[2],popt[3],popt[4])
                           for t in twaits_recov]

                    # Calculate r2
                    ss_res = np.sum((np.array(recov)-np.array(fit))**2)
                    ss_tot = np.sum((np.array(recov)-np.mean(np.array(recov)))**2)
                    r2 = 1 - (ss_res / ss_tot)

                    taur1 = min(popt[0],popt[1])
                    taur2 = max(popt[0],popt[1])
                    if r2 > fit_threshold:
                        output1 = output1+[taur1]
                        output2 = output2+[taur2]
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
                                           p0=[20.,0.5,0.],
                                           bounds=([0.,-np.inf,-np.inf],
                                                   np.inf),
                                           max_nfev=1000)

                    fit = [single_exp(t,popt[0],popt[1],popt[2])
                           for t in twaits_recov]

                    # Calculate r2
                    ss_res = np.sum((np.array(recov)-np.array(fit))**2)
                    ss_tot = np.sum((np.array(recov)-np.mean(np.array(recov)))**2)
                    r2 = 1 - (ss_res / ss_tot)

                    taur2 = popt[0]
                    if r2 > fit_threshold:
                        output2 = output2+[taur2]
                    else:
                        raise RuntimeWarning('scipy.optimize.curve_fit found a poor fit')
                except:
                    output2 = output2+[float('inf')]
    output = output1+output2
    return output

schneider_recov = Experiment(
    dataset=[schneider_taur1_dataset,
             schneider_taur2_dataset],
    protocol=schneider_recov_protocol,
    conditions=schneider_conditions,
    sum_stats=schneider_recov_sum_stats,
    description=schneider_recov_desc,
    Q10=None,
    Q10_factor=0.)

########################################################################################################################
####  recovery curves - Schneider 1992
#
## DATA
#schneider_recov_dataset = []
#for Vhold in [-135,-125,-115,-105,-95,-85,-75] :
#    time,recov,_ = dataSch.Recovery_Schneider_all(Vhold)
#    variances_time = [0 for t in time]
#    schneider_recov_dataset.append(np.asarray([time, recov, variances_time]))
#
#
## PROTOCOL
#tmp_protocol = []
#tpreMeasuringList1_recov_sch = []
#
#tperiod_recov_sch = 5000 # ms
#tstep1 = 200
#tstep2 = 12
#
#tMeasuring1_recov_sch = tstep1
#tpreMeasuring2_recov_sch = tperiod_recov_sch - tstep2
#
#Vstep1 = -20
#Vstep2 = -20
#
#Split_list_recov_sch = [] # for the summary statistics function
#
#for Vhold in [-135,-125,-115,-105,-95,-85,-75] :
#    twaitList,_,_ = dataSch.Recovery_Schneider_all(Vhold)
#
#    Split_list_recov_sch.append(len(twaitList))
#
#
#    tpreList = []
#    for twait in twaitList:
#        tpre = tperiod_recov_sch - tstep1 - twait - tstep2
#        tpreList.append(tpre)
#        tpreMeasuringList1_recov_sch.append(tpre)
#
#    protocol = recovery_tpreList(twaitList,Vhold,Vstep1, Vstep2,tpreList,tstep1,tstep2)
#
#    tmp_protocol.append(protocol)
#
## Fuse all the protocols into one
#schneider_recov_protocol = tmp_protocol[0]
#for p in tmp_protocol[1:]:
#    for e in p.events():
#        schneider_recov_protocol.add_step(e.level(), e.duration())
#
## CONDITIONS
#schneider_conditions = {'membrane.Na_o': 120,
#            'membrane.Na_i': 70,
#            'membrane.T': 297.15}
#
## SUMMARY STATISTICS
#def schneider_recov_sum_stats(data):
#    output = []
#    loop = 0
#    sub_loop = 0
#
#    # spliting back the protocols
#    Cumulated_len = Split_list_recov_sch[0]
#    dProtocolOne,dProtocoltwoseven = data.split(tperiod_recov_sch*Cumulated_len)
#    Cumulated_len += Split_list_recov_sch[1]
#    dProtocolTwo,dProtocolThreeseven  = dProtocoltwoseven.split(tperiod_recov_sch*Cumulated_len)
#    Cumulated_len += Split_list_recov_sch[2]
#    dProtocolThree,dProtocolfourseven = dProtocolThreeseven.split(tperiod_recov_sch*Cumulated_len)
#    Cumulated_len += Split_list_recov_sch[3]
#    dProtocolFour,dProtocolfiveseven  = dProtocolfourseven.split(tperiod_recov_sch*Cumulated_len)
#    Cumulated_len += Split_list_recov_sch[4]
#    dProtocolFive,dProtocolsixseven = dProtocolfiveseven.split(tperiod_recov_sch*Cumulated_len)
#    Cumulated_len += Split_list_recov_sch[5]
#    dProtocolSix,dProtocolSeven  = dProtocolsixseven.split(tperiod_recov_sch*Cumulated_len)
#
#    dProtocols = [dProtocolOne,dProtocolTwo,dProtocolThree,dProtocolFour,dProtocolFive,dProtocolSix,dProtocolSeven]
#    Cumulated_len = 0
#    for dOneProtocol in dProtocols:
#        d_split = dOneProtocol.split_periodic(tperiod_recov_sch, adjust = True)
#        if loop > 0 :
#            Cumulated_len += Split_list_recov_sch[loop-1]
#
#        d_split = d_split[Cumulated_len:]
#        if loop == 6:
#            d_split = d_split[:-1]
#
#        for d in d_split:
##            print(sub_loop)
##
##            plt.plot(d['environment.time'],d['membrane.V'])
##            plt.show()
#            dcond = d.trim(tpreMeasuringList1_recov_sch[sub_loop],
#             tpreMeasuringList1_recov_sch[sub_loop]+tMeasuring1_recov_sch, adjust = True)
#            dtest = d.trim_left(tpreMeasuring2_recov_sch, adjust = True)
#
#            current_cond = dcond['ina.i_Na'][:-1]
#            current_test = dtest['ina.i_Na'][:-1]
#
#            index_cond = np.argmax(np.abs(current_cond))
#            index_test = np.argmax(np.abs(current_test))
#            try :
#                output = output + [current_test[index_test] / current_cond[index_cond]]
#            except :
#                output = output + [float('inf')]
#            sub_loop += 1
#        loop += 1
#    return output
#
## Experiment
#schneider_recov = Experiment(
#    dataset=schneider_recov_dataset,
#    protocol=schneider_recov_protocol,
#    conditions=schneider_conditions,
#    sum_stats=schneider_recov_sum_stats,
#    description=schneider_recov_desc)

########################################################################################################################
#### Reduction Schneider 1994
#schneider_reduc_desc ="""
#    describes the protocol used to measure the Relative_peak IV curve in the Schneider Paper (figure 5A)
#
#    page 5 of the paper :
#
#    Reduction of the Na + current peak
#    amplitude elicited by pulses to -20 mV following prepotentials
#    varying in time (abscissa) and magnitude. Symbols represent
#    different inactivation potential values: -105 mV to -65 mV
#
#    the double pulse protocol is not more described so the sakakibara one has been applied to complete the lack of information
#    """
#
## DATA
#schneider_reduc_dataset = []
#for Vinact in [-105,-95,-85,-75,-65]:
#    tstepList,reduc,_ = dataSch.Reduction_Schneider_all(Vinact)
#    variances_reduc = [0 for t in tstepList]
#    schneider_reduc_dataset.append(np.asarray([tstepList,reduc, variances_reduc]))
#
## PROTOCOL
#tmp_protocol = []
#
### This part is for initializing the protocol : to determine the max I_Na when there is no conditionning pulse
#tperiod_reduc_sch = 3000 # ms seems enough for a double pulse protocol
#Split_list_reduc = [] # for the sum statistics function
#tpreMeasuringReducList = []
#twait = 0
#tstep = 0
#ttest = 12
#tpre = tperiod_reduc_sch - twait - ttest - tstep
#
#tpreMeasuring_reduc_sch = tperiod_reduc_sch - ttest
#tMeasuring_reduc_sch = ttest
#
#Vhold = -135 # mV
#Vtest = -20
#
#protocol = availability([Vhold],Vhold, Vtest, tpre, tstep, twait, ttest)
#tmp_protocol.append(protocol)
#
#
### This part is the main protocol
#twait = 2
#tstep = 1000
#ttest = 12
#
#for Vinact in [-105,-95,-85,-75,-65]:
#    tstepList,_,_ = dataSch.Reduction_Schneider_all(Vinact)
#
#    tpreList = []
#    for tstep in tstepList :
#        tpre = tperiod_reduc_sch - tstep - twait - ttest
#        tpreList.append(tpre)
#        tpreMeasuringReducList.append(tpre)
#        Split_list_reduc.append(len(tstepList))
#
#    protocol = varying_test_duration_double_pulse(Vinact,Vhold, Vtest, tpreList, tstepList, twait, ttest)
#    tmp_protocol.append(protocol)
#
#
## Fuse all the protocols into one
#schneider_reduc_protocol = tmp_protocol[0]
#for p in tmp_protocol[1:]:
#    for e in p.events():
#        schneider_reduc_protocol.add_step(e.level(), e.duration())
#
## CONDITIONS
#schneider_conditions = {'membrane.Na_o': 120,
#            'membrane.Na_i': 70,
#            'membrane.T': 297.15}
#
## SUMMARY STATISTICS
#def schneider_reduc_sum_stats(data):
#
#    output = []
#    loop = 0
#
#    # spliting back the 5 protocols and the initialisation
#    cumulated_shift = 1
#    dProtocolInit,dProtocolOnefive = data.split(tperiod_reduc_sch*cumulated_shift)
#    cumulated_shift += Split_list_reduc[0]
#    dProtocolOne,dProtocoltwofive = dProtocolOnefive.split(tperiod_reduc_sch*cumulated_shift)
#    cumulated_shift += Split_list_reduc[1]
#    dProtocolTwo,dProtocolThreefive  = dProtocoltwofive.split(tperiod_reduc_sch*cumulated_shift)
#    cumulated_shift += Split_list_reduc[2]
#    dProtocolThree,dProtocolfourfive = dProtocolThreefive.split(tperiod_reduc_sch*cumulated_shift)
#    cumulated_shift += Split_list_reduc[3]
#    dProtocolFour,dProtocolFive  = dProtocolfourfive.split(tperiod_reduc_sch*cumulated_shift)
#    dProtocols = [dProtocolOne,dProtocolTwo,dProtocolThree,dProtocolFour,dProtocolFive]
#
#
#
#    dProtocolInit = dProtocolInit.trim_left(tpreMeasuring_reduc_sch, adjust = True)
#    current = dProtocolInit['ina.i_Na']
#
#    current = current[:-1]
#    index = np.argmax(np.abs(current))
#    normalizing_peak = current[index]
#
#    cumulated_shift = 1
#    for dOneProtocol in dProtocols:
#        d_split = dOneProtocol.split_periodic(tperiod_reduc_sch, adjust = True)
#        if loop > 0:
#            cumulated_shift += Split_list_reduc[loop-1]
#        d_split = d_split[cumulated_shift:]    # specific to split_periodic function
#        #( if the time begins at t0 >0 it will create empty arrays from 0 to t0 : here we are getting rid of them)
#
#        for d in d_split:
#            d = d.trim(tpreMeasuring_reduc_sch,tpreMeasuring_reduc_sch  + tMeasuring_reduc_sch, adjust = True)
#            index = np.argmax(np.abs(current))
#            try :
#                output = output+[np.abs(current[index])/normalizing_peak]
#            except :
#                output = output + [float('inf')]
#        loop += 1
#    return output
#
#schneider_reduc = Experiment(
#    dataset=schneider_reduc_dataset,
#    protocol=schneider_reduc_protocol,
#    conditions=schneider_conditions,
#    sum_stats=schneider_reduc_sum_stats,
#    description=schneider_reduc_desc
#)
#
#
