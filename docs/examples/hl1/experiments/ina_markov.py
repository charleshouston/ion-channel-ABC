from ionchannelABC.protocol import availability_linear, recovery
import data.ina.data_ina as data
import numpy as np
import pandas as pd
import myokit
from ionchannelABC.experiment import Experiment


room_temp = 295


### IV curve Dias 2014
dias_iv_description = """IV curve for iNa from Dias 2014.
Measurements taken at room temperature so no adjustment.
"""
vsteps, peaks, sd = data.IV_Dias()
max_observed_peak = np.max(np.abs(peaks))
peaks = [p / max_observed_peak for p in peaks]
variances = [(sd / max_observed_peak)**2 for sd in sd]
dias_iv_dataset = np.asarray([vsteps, peaks, variances])

dias_iv_protocol = myokit.pacing.steptrain_linear(-100, 50, 10, -80, 5000, 100)
dias_conditions = {'membrane.Na_o': 140e3,
                   'membrane.Na_i': 10e3,
                   'membrane.K_o': 4000,
                   'membrane.K_i': 130e3,
                   'membrane.T': room_temp}

def dias_iv_sum_stats(data):
    output = []
    for d in data.split_periodic(5100, adjust=True):
        d = d.trim(5000, 5100, adjust=True)
        current = d['ina.i_Na']
        index = np.argmax(np.abs(current))
        output = output+[current[index]/max_observed_peak]
    return output

dias2014_iv = Experiment(dataset=dias_iv_dataset,
                         protocol=dias_iv_protocol,
                         conditions=dias_conditions,
                         sum_stats=dias_iv_sum_stats,
                         description=dias_iv_description)


### Inactivation
nakajima_desc = """Inactivation curve from Nakajima 2009.
Measurements taken at room temperature so no adjustment.
"""
vsteps_inact, inact, sd_inact = data.Inact_Nakajima()
variances_inact = [sd**2 for sd in sd_inact]
nakajima_inact_dataset = np.asarray([vsteps_inact, inact, variances_inact])

nakajima_inact_protocol = availability_linear(
    -130, -20, 10, -120, -20, 5000, 500, 0, 100
)
nakajima_conditions = {'membrane.Na_o': 145e3,
                       'membrane.Na_i': 10e3,
                       'membrane.K_o': 0., # Cs used to avoid contamination
                       'membrane.K_i': 0.,
                       'membrane.T': room_temp}

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
    description=nakajima_desc
)


### Recovery Zhang 2013
zhang_rec_desc = """Recovery curve for iNa in Zhang 2013."""

tsteps_rec, rec, _ = data.Recovery_Zhang()
zhang_rec_dataset = np.asarray([tsteps_rec, rec, [0.]*len(rec)])

zhang_rec_protocol = recovery(
    tsteps_rec, -120, -30, -30, 3000, 20, 20
)
zhang_conditions = {'membrane.Na_o': 136e3,
                    'membrane.Na_i': 10e3,
                    'membrane.K_o': 0., # Cs used to avoid contamination
                    'membrane.K_i': 0.,
                    'membrane.T': room_temp}

split_times = [3040+tw for tw in tsteps_rec]
for i, time in enumerate(split_times[:-1]):
    split_times[i+1] += split_times[i]
def zhang_rec_sum_stats(data):
    dsplit = []
    tkey = "environment.time"
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
    return output

zhang_recovery = Experiment(
    dataset=zhang_rec_dataset,
    protocol=zhang_rec_protocol,
    conditions=zhang_conditions,
    sum_stats=zhang_rec_sum_stats,
    description=zhang_rec_desc
)
