import myokit
import data.ical.data_ical as data
import numpy as np
from ionchannelABC.experiment import Experiment
from ionchannelABC.protocol import availability_linear, recovery


room_temp = 295
Q10_cond = 2.3 # Kiyosue 1993
Q10_tau = 2.1 # ten Tusscher 2004

def temperature_adjust_cond(R0, T0, T1, Q10):
    return R0*Q10**((T1-T0)/10)

def temperature_adjust_tau(R0, T0, T1, Q10):
    return R0*Q10**((T0-T1)/10)


### IV curve Dias 2014
dias_iv_desc = """IV curve measured in HL1-6 myocytes from Dias 2014.
Experiments carried out at room temperature so no adjustment.
"""

vsteps_iv, peaks, sd_iv = data.IV_Dias()
max_iv_peak = np.max(np.abs(peaks))
peaks = [p/max_iv_peak for p in peaks]
variances_iv = [(sd/max_iv_peak)**2 for sd in sd_iv]
dias_iv_dataset = np.asarray([vsteps_iv, peaks, variances_iv])

dias_iv_protocol = myokit.Protocol()
time = 0.
vpre = -80.
tpre = 5000.
vhold = -40.
thold = 200.
tstep = 250.
for v in vsteps_iv:
    dias_iv_protocol.schedule(vpre, time, tpre)
    time += tpre
    dias_iv_protocol.schedule(vhold, time, thold)
    time += thold
    dias_iv_protocol.schedule(v, time, tstep)
    time += tstep

dias_conditions = {'membrane.Ca_o': 2000,
                   'membrane.Ca_i': 7.6e-3,
                   'membrane.T': room_temp}

def dias_iv_sum_stats(data):
    output = []
    for d in data.split_periodic(5450, adjust=True):
        d = d.trim(5200, 5450, adjust=True)
        output = output + [max(d['ical.i_CaL'], key=abs)/max_iv_peak]
    return output

dias_iv = Experiment(
    dataset=dias_iv_dataset,
    protocol=dias_iv_protocol,
    conditions=dias_conditions,
    sum_stats=dias_iv_sum_stats,
    description=dias_iv_desc
)


### Inactivation curve Rao 2009
rao_inact_desc = """Inactivation curve for iCaL in HL-1 from Rao 2009.
Measurements taken at room temperature so no temperature adjustment."""

vsteps_inact, inact, sd_inact = data.Inact_Rao()
variances_inact = [sd**2 for sd in sd_inact]
rao_inact_dataset = np.asarray([vsteps_inact, inact, variances_inact])

rao_inact_protocol = availability_linear(
    -100, 20, 10, -80, -20, 5000, 1000, 0, 400
)

rao_conditions = {'membrane.Ca_o': 5000,
                  'membrane.Ca_i': 0.2, #estimate from LR1994
                  'membrane.T': room_temp}

def rao_inact_sum_stats(data):
    output = []
    for d in data.split_periodic(6400, adjust=True):
        d = d.trim(6000, 6400, adjust=True)
        current = d['ical.i_CaL']
        output = output + [max(current, key=abs)]
    for i in range(1, len(output)):
        output[i] = output[i]/output[0]
    output[0] = 1.
    return output

rao_inact = Experiment(
    dataset=rao_inact_dataset,
    protocol=rao_inact_protocol,
    conditions=rao_conditions,
    sum_stats=rao_inact_sum_stats,
    description=rao_inact_desc
)


### Recovery curve Rao 2009
rao_rec_desc = """Recovery curve for iCaL in HL-1 cells from Rao 2009.
Measurements carried out at room temperature."""

times_rec, rec, sd_rec = data.Rec_Rao()
variances_rec = [sd**2 for sd in sd_rec]
rao_rec_dataset = np.asarray([times_rec, rec, variances_rec])

rao_rec_protocol = recovery(
    times_rec, -80, -20, -20, 5000, 400, 400
)

split_times = [5800+tw for tw in times_rec]
for i, time in enumerate(split_times[:-1]):
    split_times[i+1] += split_times[i]

def rao_rec_sum_stats(data):
    pulse_traces = []
    for i, time in enumerate(split_times):
        d_, data = data.split(time)
        pulse_traces.append(
            d_.trim(d_['environment.time'][0]+5000,
                    d_['environment.time'][0]+5800+times_rec[i],
                    adjust=True)
        )
    output = []
    for d in pulse_traces:
        pulse1 = d.trim(0, 400, adjust=True)['ical.i_CaL']
        endtime = d['environment.time'][-1]
        pulse2 = d.trim(endtime-400, endtime, adjust=True)['ical.i_CaL']

        max1 = np.max(np.abs(pulse1))
        max2 = np.max(np.abs(pulse2))

        output = output + [max2/max1]
    return output

rao_rec = Experiment(
    dataset=rao_rec_dataset,
    protocol=rao_rec_protocol,
    conditions=rao_conditions,
    sum_stats=rao_rec_sum_stats,
    description=rao_rec_desc
)
