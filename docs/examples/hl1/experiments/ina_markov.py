from ionchannelABC.protocol import availability_linear, recovery
import data.ina.data_ina as data
import numpy as np
import pandas as pd
import myokit
from ionchannelABC.experiment import Experiment
import warnings
import scipy.optimize as so


room_temp = 296
Q10_cond = 1.5 # [Correa1991]
Q10_tau = 2.79 # [tenTusscher2004]

fit_threshold = 0.9


#
# IV curve [Dias2014]
#
dias_iv_desc = """IV curve for iNa from Dias 2014.
Measurements taken at room temperature.
"""
vsteps, peaks, sd = data.IV_Dias()
variances = [sd**2 for sd in sd]
dias_iv_dataset = np.asarray([vsteps, peaks, variances])

vsteps_taua, taua, sd_taua = data.TauAct_Dias()
variances_taua = [sd**2 for sd in sd_taua]
dias_taua_dataset = np.asarray([vsteps_taua, taua, variances_taua])

vsteps_taui, taui, sd_taui = data.TauInact_Dias()
variances_taui = [sd**2 for sd in sd_taui]
dias_taui_dataset = np.asarray([vsteps_taui, taui, variances_taui])

dias_iv_protocol = myokit.pacing.steptrain_linear(-100, 50, 10, -80, 5000, 100)
dias_conditions = {'extra.Na_o': 140e3,
                   'sodium.Na_i': 10e3,
                   'extra.K_o': 4000,
                   'potassium.K_i': 130e3,
                   'phys.T': room_temp}

#def dias_iv_sum_stats(data):
#    output = []
#    for d in data.split_periodic(5100, adjust=True):
#        d = d.trim(5000, 5100, adjust=True)
#        current = d['ina.i_Na']
#        index = np.argmax(np.abs(current))
#        output = output+[current[index]]
#    return output

def dias_iv_tau_sum_stats(data, ss=True, taua=True, taui=True):
    out1 = []
    out2 = []
    out3 = []
    def activation_exp(t, tau, A, A0):
        return A*(1-np.exp(-t/tau))**3+A0
    def inactivation_exp(t, tau, A, A0):
        return A*np.exp(-t/tau)+A0
    for d in data.split_periodic(5100, adjust=True):
        d = d.trim(5000, 5100, adjust=True)
        curr = d['ina.i_Na']
        index = np.argmax(np.abs(curr))
        if ss:
            out1 = out1+[curr[index]]
        if (taua and
            d['membrane.V'][0] >= vsteps_taua[0] and
            d['membrane.V'][0] <= vsteps_taua[-1]):
            # Separate activation portion
            time = d['engine.time']
            rise = curr[:index]
            time = time[:index]
            # fit to activation exponential
            with warnings.catch_warnings():
                warnings.simplefilter('error', so.OptimizeWarning)
                warnings.simplefilter('error', RuntimeWarning)
                try:
                    popt, _ = so.curve_fit(activation_exp, time, rise,
                                           p0=[0.5, 1., 0.],
                                           bounds=([0., -np.inf, -np.inf],
                                                   np.inf))
                    fit = [activation_exp(t,popt[0],popt[1],popt[2]) for t in time]

                    # calculate r squared of fit
                    ss_res = np.sum((np.array(rise)-np.array(fit))**2)
                    ss_tot = np.sum((np.array(rise)-np.mean(np.array(rise)))**2)
                    r2 = 1 - (ss_res / ss_tot)
                    taua = popt[0]

                    # only accept reasonable curve fits
                    if r2 > fit_threshold:
                        out2 = out2 + [taua]
                    else:
                        out2 = out2 + [float('inf')]
                except:
                    out2 = out2 + [float('inf')]
        if (taui and
            d['membrane.V'][0] >= vsteps_taui[0] and
            d['membrane.V'][0] <= vsteps_taui[-1]):
            # Separate decay portion
            time = d['engine.time']
            decay = curr[index:]
            time = time[index:]
            t0 = time[0]
            time = [t-t0 for t in time]
            # fit to single exponential
            with warnings.catch_warnings():
                warnings.simplefilter('error', so.OptimizeWarning)
                warnings.simplefilter('error', RuntimeWarning)
                try:
                    popt, _ = so.curve_fit(inactivation_exp, time, decay,
                                           p0=[2., 1., 0.],
                                           bounds=([0., -np.inf, -np.inf],
                                                   np.inf))
                    fit = [inactivation_exp(t,popt[0],popt[1],popt[2]) for t in time]

                    # calculate r squared of fit
                    ss_res = np.sum((np.array(decay)-np.array(fit))**2)
                    ss_tot = np.sum((np.array(decay)-np.mean(np.array(decay)))**2)
                    r2 = 1 - (ss_res / ss_tot)
                    taui = popt[0]

                    # only accept reasonable curve fits
                    if r2 > fit_threshold:
                        out3 = out3 + [taui]
                    else:
                        out3 = out3 + [float('inf')]
                except:
                    out3 = out3 + [float('inf')]
    return out1+out2+out3

def dias_iv_sum_stats(data):
    return dias_iv_tau_sum_stats(data, ss=True, tau=False)
def dias_taua_sum_stats(data):
    return dias_iv_tau_sum_stats(data, ss=False, taua=True, taui=False)
def dias_taui_sum_stats(data):
    return dias_iv_tau_sum_stats(data, ss=False, taua=False, taui=True)

dias_iv = Experiment(
    dataset=dias_iv_dataset,
    protocol=dias_iv_protocol,
    conditions=dias_conditions,
    sum_stats=dias_iv_sum_stats,
    description=dias_iv_desc,
    Q10=Q10_cond,
    Q10_factor=1
)
dias_tau_act = Experiment(
    dataset=dias_taua_dataset,
    protocol=dias_iv_protocol,
    conditions=dias_conditions,
    sum_stats=dias_taui_sum_stats,
    description=dias_iv_desc,
    Q10=Q10_tau,
    Q10_factor=-1
)
dias_tau_inact = Experiment(
    dataset=dias_taui_dataset,
    protocol=dias_iv_protocol,
    conditions=dias_conditions,
    sum_stats=dias_taui_sum_stats,
    description=dias_iv_desc,
    Q10=Q10_tau,
    Q10_factor=-1
)
dias_iv_tau = Experiment(
    dataset=[dias_iv_dataset,
             dias_taua_dataset,
             dias_taui_dataset],
    protocol=dias_iv_protocol,
    conditions=dias_conditions,
    sum_stats=dias_iv_tau_sum_stats,
    description=dias_iv_desc,
    Q10=[Q10_cond,Q10_tau,Q10_tau],
    Q10_factor=[1,-1,-1]
)


#
# Inactivation [Nakajima2010]
#
nakajima_desc = """Inactivation curve from Nakajima 2010.
Measurements taken at room temperature.
"""
vsteps_inact, inact, sd_inact = data.Inact_Nakajima()
variances_inact = [sd**2 for sd in sd_inact]
nakajima_inact_dataset = np.asarray([vsteps_inact, inact, variances_inact])

nakajima_inact_protocol = availability_linear(
    -130, -20, 10, -120, -20, 5000, 500, 0, 100
)
nakajima_conditions = {'extra.Na_o': 145e3,
                       'sodium.Na_i': 10e3,
                       'extra.K_o': 0., # Cs used to avoid contamination
                       'potassium.K_i': 0.,
                       'phys.T': room_temp}

def nakajima_inact_sum_stats(data):
    output = []
    for d in data.split_periodic(5600, adjust=True):
        d = d.trim(5500, 5600, adjust=True)
        current = d['ina.i_Na']
        output = output+[max(current, key=abs)]
    for i in range(1, len(output)):
        output[i] = output[i]/output[0]
    output[0] = 1.
    return output


nakajima_inactivation = Experiment(
    dataset=nakajima_inact_dataset,
    protocol=nakajima_inact_protocol,
    conditions=nakajima_conditions,
    sum_stats=nakajima_inact_sum_stats,
    description=nakajima_desc,
    Q10=None,
    Q10_factor=0
)


#
# Recovery [Zhang2013]
#
zhang_rec_desc = """Recovery curve for iNa in Zhang 2013.
Experiments conducted at room temperature.
"""

tsteps_rec, rec, sd_rec = data.Recovery_Zhang()
variances_rec = [sd_**2 for sd_ in sd_rec]
zhang_rec_dataset = np.asarray([tsteps_rec, rec, variances_rec])

zhang_rec_protocol = recovery(
    tsteps_rec, -120, -30, -30, 3000, 20, 20
)
zhang_conditions = {'extra.Na_o': 136e3,
                    'sodium.Na_i': 10e3,
                    'extra.K_o': 0., # Cs used to avoid contamination
                    'potassium.K_i': 0.,
                    'phys.T': room_temp}

split_times = [3040+tw for tw in tsteps_rec]
for i, time in enumerate(split_times[:-1]):
    split_times[i+1] += split_times[i]
def zhang_rec_sum_stats(data):
    dsplit = []
    tkey = 'engine.time'
    # First need to split data by differing time periods
    for i, time in enumerate(split_times):
        split_temp = data.split(time)
        dsplit.append(
            split_temp[0].trim(split_temp[0][tkey][0]+3000,
                               split_temp[0][tkey][0]+3040+tsteps_rec[i],
                               adjust=True)
        )
        data = split_temp[1]

    output = []
    # Process data in each time period
    for d in dsplit:
        # Interested in two 20ms pulses
        pulse1 = d.trim(0, 20, adjust=True)['ina.i_Na']
        endtime = d[tkey][-1]
        pulse2 = d.trim(endtime-20, endtime, adjust=True)['ina.i_Na']

        max1 = np.max(np.abs(pulse1))
        max2 = np.max(np.abs(pulse2))
        output = output+[max2/max1]
        #with warnings.catch_warnings():
        #    warnings.simplefilter('error', RuntimeWarning)
        #    try:
        #        wait = d.trim(20, endtime-20, adjust=True)['ina.i_Na']
        #        if np.max(np.abs(wait))>=max2:
        #            output = output+[float('inf')]
        #        else:
        #            output = output+[max2/max1]
        #    except:
        #        output = output+[float('inf')]
    return output

zhang_recovery = Experiment(
    dataset=zhang_rec_dataset,
    protocol=zhang_rec_protocol,
    conditions=zhang_conditions,
    sum_stats=zhang_rec_sum_stats,
    description=zhang_rec_desc,
    Q10=None,
    Q10_factor=0
)
