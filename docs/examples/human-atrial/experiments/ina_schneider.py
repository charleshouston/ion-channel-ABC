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

fit_threshold = 0.9 # for curve_fit

# optional adjustments to match papers
nyg_adjust_act = 22.8
nyg_adjust_inact = 32.2
cou_adjust = 20


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

vsteps_tm_nyg_adjust = [v+nyg_adjust_act for v in vsteps_tm]
schneider_taum_nyg_adjust_dataset = np.asarray(
    [vsteps_tm_nyg_adjust, tm, variances_tm])
vsteps_tm_cou_adjust = [v+cou_adjust for v in vsteps_tm]
schneider_taum_cou_adjust_dataset = np.asarray(
    [vsteps_tm_cou_adjust, tm, variances_tm])


vsteps_tf, tf, sd_tf = dataSch.TauF_Inactivation_Schneider()
variances_tf = [(sd_)**2 for sd_ in sd_tf]
schneider_tauf_dataset = np.asarray([vsteps_tf, tf, variances_tf])

tpre = 10000 # ms
tstep = 12
vhold = -135 # mV
vlower = -65+1e-5
vupper = 105+1e-5
vupper_tm = 15+1e-5
dv = 10
schneider_tau_protocol = myokit.pacing.steptrain_linear(
    vlower, vupper+dv, dv, vhold, tpre, tstep)
schneider_taum_protocol = myokit.pacing.steptrain_linear(
    vlower, vupper_tm+dv, dv, vhold, tpre, tstep)
schneider_taum_nyg_adjust_protocol = myokit.pacing.steptrain_linear(
    vlower+nyg_adjust_act, vupper_tm+dv+nyg_adjust_act, dv,
    vhold+nyg_adjust_act, tpre, tstep)
schneider_taum_cou_adjust_protocol = myokit.pacing.steptrain_linear(
    vlower+cou_adjust, vupper_tm+dv+cou_adjust, dv,
    vhold+cou_adjust, tpre, tstep)


def schneider_tau_sum_stats(data):
    def sum_of_exp(t, taum, tauh, C):
        return (1-np.exp(-t/taum))**3*np.exp(-t/tauh)+C
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
                maxcurr = max(current, key=abs)
                current = [c/maxcurr for c in current]
                if len(time)<=1 or len(current)<=1:
                    raise Exception('Failed simulation')
                popt, _ = so.curve_fit(sum_of_exp,
                                       time,
                                       current,
                                       p0=[0.5, 1., 0.],
                                       bounds=([0., 0., -1.0],
                                               [np.inf, np.inf, 1.0]),
                                       max_nfev=1000)

                fit = [sum_of_exp(t,popt[0],popt[1],popt[2]) for t in time]
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

def schneider_taum_sum_stats(data):
    def sum_of_exp(t, taum, tauh, C):
        return (1-np.exp(-t/taum))**3*np.exp(-t/tauh)+C
    output = []
    for i,d in enumerate(data.split_periodic(10012, adjust=True)):
        d = d.trim_left(10000, adjust=True)

        current = d['ina.i_Na'][:-1]
        time = d['engine.time'][:-1]

        with warnings.catch_warnings():
            warnings.simplefilter('error', so.OptimizeWarning)
            warnings.simplefilter('error', RuntimeWarning)
            try:
                maxcurr = max(current, key=abs)
                current = [c/maxcurr for c in current]
                if len(time)<=1 or len(current)<=1:
                    raise Exception('Failed simulation')
                popt, _ = so.curve_fit(sum_of_exp,
                                       time,
                                       current,
                                       p0=[0.05, 1., 0.],
                                       bounds=([0., 0., -1.0],
                                               [np.inf, np.inf, 1.0]),
                                       max_nfev=1000)

                fit = [sum_of_exp(t,popt[0],popt[1],popt[2]) for t in time]
                # Calculate r2
                ss_res = np.sum((np.array(current)-np.array(fit))**2)
                ss_tot = np.sum((np.array(current)-np.mean(np.array(current)))**2)
                r2 = 1 - (ss_res / ss_tot)

                taum = popt[0]
                if r2 > fit_threshold:
                    output = output+[taum]
                else:
                    raise RuntimeWarning('scipy.optimize.curve_fit found a poor fit')
            except:
                output = output+[float('inf')]
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
schneider_taum = Experiment(
    dataset=schneider_taum_dataset,
    protocol=schneider_taum_protocol,
    conditions=schneider_conditions,
    sum_stats=schneider_taum_sum_stats,
    description=schneider_tau_desc,
    Q10=Q10_tau,
    Q10_factor=-1)

schneider_taum_nyg_adjust = Experiment(
    dataset=schneider_taum_nyg_adjust_dataset,
    protocol=schneider_taum_nyg_adjust_protocol,
    conditions=schneider_conditions,
    sum_stats=schneider_taum_sum_stats,
    description=schneider_tau_desc,
    Q10=Q10_tau,
    Q10_factor=-1)
schneider_taum_cou_adjust = Experiment(
    dataset=schneider_taum_cou_adjust_dataset,
    protocol=schneider_taum_cou_adjust_protocol,
    conditions=schneider_conditions,
    sum_stats=schneider_taum_sum_stats,
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
    #def single_exp(t, tau, A, C):
    #    return A*np.exp(-t/tau)+C
    def single_exp(t, tau):
        return np.exp(-t/tau)
    output = []
    # Split by test pulse potential
    for d in data.split_periodic(tsplit_tests, adjust=True):
        peaks = []
        for t in tsplits:
            d_, d = d.split(t)
            d_ = d_.trim_left(t-ttest, adjust=True)
            current = d_['ina.i_Na'][:-1]
            peaks = peaks + [max(current, key=abs)]

        # fit to single exponential
        with warnings.catch_warnings():
            warnings.simplefilter('error', so.OptimizeWarning)
            warnings.simplefilter('error', RuntimeWarning)
            try:
                norm = peaks[0]
                for i in range(len(peaks)):
                    peaks[i] /= norm

                peaks = [p-peaks[-1] for p in peaks]
                maxpk = max(peaks, key=abs)
                peaks = [p/maxpk for p in peaks]
                popt, _ = so.curve_fit(single_exp,
                                       tprepulse,
                                       peaks,
                                       p0 = 1.,
                                       bounds = (0., np.inf),
                                       #p0=[1., peaks[0]-peaks[-1], peaks[0]],
                                       #bounds=(0.,
                                       #        [np.inf, 1.0, 1.0]),
                                       max_nfev=1000)

                #fit = [single_exp(t,popt[0],popt[1],popt[2]) for t in tprepulse]
                fit = [single_exp(t,popt[0]) for t in tprepulse]

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

        # fit to boltzmann function
        with warnings.catch_warnings():
            warnings.simplefilter('error', so.OptimizeWarning)
            warnings.simplefilter('error', RuntimeWarning)
            try:
                norm = inact[0]
                for j in range(len(inact)):
                    inact[j] /= norm
                popt, _ = so.curve_fit(boltzmann_fn,
                                       vsteps_inact,
                                       inact,
                                       p0=[-70, 5],
                                       bounds=([-100., 0.],
                                               100),
                                       max_nfev=1000)

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
                                           p0=[5.,20.,0.,0.5,0.5],
                                           bounds=([0.],
                                                   [np.inf,np.inf,1.0,1.0,1.0]),
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
                                           p0=[20.,0.,0.5],
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
