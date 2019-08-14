from ionchannelABC.protocol import recovery, availability_linear
import data.icat.data_icat as data
import numpy as np
import pandas as pd
import myokit
from ionchannelABC.experiment import Experiment


room_temp = 296
# Q10 values for iCaL MAY NOT BE VALID for iCaT
Q10_cond = 2.3 # [Kiyosue1993]
Q10_tau = 2.1 # [tenTusscher2004]


#
# IV curve [Nguyen2013]
#
nguyen_iv_desc = """IV curve from Nguyen 2013 Fig 5B.
Measurements at room temperature.
"""

vsteps, peaks, sd = data.IV_Nguyen()
variances = [sd**2 for sd in sd]
nguyen_iv_dataset = np.asarray([vsteps, peaks, variances])

nguyen_iv_protocol = myokit.pacing.steptrain_linear(
    -75, 40, 5, -75, 5000, 300
)
nguyen_conditions = {'extra.Ca_o': 5000,
                     'calcium.Ca_i': 0.2, # estimated [LR1994]
                     'phys.T': room_temp}

def nguyen_iv_sum_stats(data):
    output = []
    for d in data.split_periodic(5300, adjust=True):
        d = d.trim(5000, 5300, adjust=True)
        current = d['icat.i_CaT']
        index = np.argmax(np.abs(current))
        output = output + [current[index]]
    return output

nguyen_iv = Experiment(
    dataset=nguyen_iv_dataset,
    protocol=nguyen_iv_protocol,
    conditions=nguyen_conditions,
    sum_stats=nguyen_iv_sum_stats,
    description=nguyen_iv_desc,
    Q10=Q10_cond,
    Q10_factor=1
)


#
# Steady-state inactivation [Nguyen2013]
#
nguyen_inact_desc = """Inactivation curve for icat from Nguyen 2013 Fig 5E.
Recordings at room temperature.
"""

vsteps_inact, inact, sd_inact = data.Inact_Nguyen()
variances_inact = [sd**2 for sd in sd_inact]
nguyen_inact_dataset = np.asarray([vsteps_inact, inact, variances_inact])

nguyen_inact_protocol = availability_linear(
    -80, -20, 5, -90, -10, 5000, 1000, 0, 150
)

def nguyen_inact_sum_stats(data):
    output = []
    for d in data.split_periodic(6150, adjust=True):
        d = d.trim(6000, 6150, adjust=True)
        current = d['icat.i_CaT']
        output = output + [max(current, key=abs)]
    for i in range(1, len(output)):
        output[i] = output[i]/output[0]
    output[0] = 1.
    return output

nguyen_inact = Experiment(
    dataset=nguyen_inact_dataset,
    protocol=nguyen_inact_protocol,
    conditions=nguyen_conditions,
    sum_stats=nguyen_inact_sum_stats,
    description=nguyen_inact_desc,
    Q10=None,
    Q10_factor=0
)


#
# Recovery kinetics [Deng2009]
#
deng_rec_desc = """Recovery curve in HL-1 from Deng 2009 Fig 4B.
Measurements taken at room temperature so no temp adjustment.
"""

tsteps_rec, rec, sd_rec = data.Rec_Deng()
variances_rec = [sd**2 for sd in sd_rec]
deng_rec_dataset = np.asarray([tsteps_rec, rec, variances_rec])

deng_rec_protocol = recovery(
    tsteps_rec, -80, -20, -20, 5000, 300, 300
)
deng_conditions = {'extra.Ca_o': 5000,
                   'calcium.Ca_i': 0.2, # estimated LR1994
                   'phys.T': room_temp}

split_times = [5600+tw for tw in tsteps_rec]
for i, time in enumerate(split_times[:-1]):
    split_times[i+1] += split_times[i]

def deng_rec_sum_stats(data):
    pulse_traces = []
    for i, time in enumerate(split_times):
        d_, data = data.split(time)
        pulse_traces.append(
            d_.trim(d_['engine.time'][0]+5000,
                    d_['engine.time'][0]+5600+tsteps_rec[i],
                    adjust=True)
        )
    output = []
    for d in pulse_traces:
        # Interested in two 300ms pulses
        pulse1 = d.trim(0, 300, adjust=True)['icat.i_CaT']
        endtime = d['engine.time'][-1]
        pulse2 = d.trim(endtime-300, endtime, adjust=True)['icat.i_CaT']

        max1 = np.max(np.abs(pulse1))
        max2 = np.max(np.abs(pulse2))

        output = output + [max2/max1]
    return output

deng_rec = Experiment(
    dataset=deng_rec_dataset,
    protocol=deng_rec_protocol,
    conditions=deng_conditions,
    sum_stats=deng_rec_sum_stats,
    description=deng_rec_desc,
    Q10=None,
    Q10_factor=0
)
