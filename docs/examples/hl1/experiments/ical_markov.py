import myokit
import data.ical.data_ical as data
import numpy as np
from ionchannelABC.experiment import Experiment
from ionchannelABC.protocol import availability_linear, recovery
import warnings
import scipy.optimize as so


room_temp = 296
Q10_cond = 2.3 # Kiyosue 1993
Q10_tau = 2.1 # ten Tusscher 2004

#
# IV curve [Dias2014]
#
dias_iv_desc = """IV curve measured in HL1-6 myocytes from Dias 2014.
Experiments carried out at room temperature so no adjustment.
"""

vsteps_iv, peaks, sd_iv = data.IV_Dias()
variances_iv = [sd**2 for sd in sd_iv]
dias_iv_dataset = np.asarray([vsteps_iv, peaks, variances_iv])

vsteps_tau, taui, _ = data.TauInact_Dias()
dias_tau_dataset = np.asarray([vsteps_tau, taui, [0.]*len(taui)])

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

dias_conditions = {'extra.Ca_o': 2e3,
                   'calcium.Ca_i': 7.6e-3,
                   'phys.T': room_temp}

def dias_iv_sum_stats(data):
    output = []
    for d in data.split_periodic(5450, adjust=True):
        d = d.trim(5200, 5450, adjust=True)
        # ical should peak within the 250ms step
        curr = d['ical.i_CaL']
        index = np.argmax(np.abs(curr))
        if index == len(curr)-1:
            output = output + [float('inf')]
        else:
            output = output + [curr[index]]
    return output

def dias_iv_tau_sum_stats(data):
    out1 = []
    out2 = []
    def single_exp(t, tau, A, A0):
        return A*(1-np.exp(-t/tau))+A0
    for d in data.split_periodic(5450, adjust=True):
        d = d.trim(5200, 5450, adjust=True)
        curr = d['ical.i_CaL']
        index = np.argmax(np.abs(curr))
        out1 = out1 + [curr[index]]
        if (d['membrane.V'][0] >= vsteps_tau[0] and
            d['membrane.V'][0] <= vsteps_tau[-1]):
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
                    popt, _ = so.curve_fit(single_exp, time, decay,
                                           p0=[5., 1., 0.],
                                           bounds=([0., -np.inf, -np.inf],
                                                   np.inf))
                    taui = popt[0]
                    out2 = out2 + [taui]
                except:
                    out2 = out2 + [float('inf')]
    return out1+out2

dias_iv = Experiment(
    dataset=dias_iv_dataset,
    protocol=dias_iv_protocol,
    conditions=dias_conditions,
    sum_stats=dias_iv_sum_stats,
    description=dias_iv_desc,
    Q10=Q10_cond,
    Q10_factor=1
)

dias_iv_tau = Experiment(
    dataset=[dias_iv_dataset,
             dias_tau_dataset],
    protocol=dias_iv_protocol,
    conditions=dias_conditions,
    sum_stats=dias_iv_tau_sum_stats,
    description=dias_iv_desc,
    Q10=Q10_cond, # TODO: fix to use different Q10 values?
    Q10_factor=[1,-1]
)


#
# Steady-state inactivation [Rao2009]
#
rao_inact_desc = """Inactivation curve for iCaL in HL-1 from Rao 2009.
Measurements taken at room temperature.
"""

vsteps_inact, inact, sd_inact = data.Inact_Rao()
variances_inact = [sd**2 for sd in sd_inact]
rao_inact_dataset = np.asarray([vsteps_inact, inact, variances_inact])

rao_inact_protocol = availability_linear(
    -100, 20, 10, -80, -20, 5000, 1000, 0, 400
)

rao_conditions = {'extra.Ca_o': 5e3,
                  'calcium.Ca_i': 7.6e-3, # assume same as Dias2014
                  'phys.T': room_temp}

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
    description=rao_inact_desc,
    Q10=None,
    Q10_factor=0
)


#
# Recovery [Rao2009]
#
rao_rec_desc = """Recovery curve for iCaL in HL-1 cells from Rao 2009.
Measurements taken at room temperature.
"""

times_rec, rec, sd_rec = data.Rec_Rao()
variances_rec = [sd**2 for sd in sd_rec]
rao_rec_dataset = np.asarray([times_rec, rec, variances_rec])

rao_rec_protocol = recovery(
    times_rec, -80, -20, -20, 10000, 400, 400
)

split_times = [10800+tw for tw in times_rec]
for i, time in enumerate(split_times[:-1]):
    split_times[i+1] += split_times[i]

def rao_rec_sum_stats(data):
    pulse_traces = []
    for i, time in enumerate(split_times):
        d_, data = data.split(time)
        pulse_traces.append(
            d_.trim(d_['engine.time'][0]+10000,
                    d_['engine.time'][0]+10800+times_rec[i],
                    adjust=True)
        )
    output = []
    for i,d in enumerate(pulse_traces):
        pulse1 = d.trim(0, 400, adjust=True)['ical.i_CaL']
        endtime = 800+times_rec[i]
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
    description=rao_rec_desc,
    Q10=None,
    Q10_factor=0
)


#
# Inactivation kinetics [Rao2009]
#
rao_taui_desc = """Inactivation kinetics for iCaL in HL-1 from Rao 2009.
Measurements taken at room temperature.
"""

vsteps_taui, taui, sd_taui = data.TauInact_Rao()
variances_taui = [sd**2 for sd in sd_taui]
rao_taui_dataset = np.asarray([vsteps_taui, taui, variances_taui])

rao_taui_protocol = myokit.pacing.steptrain(
    [v+1e-5 for v in vsteps_taui], -80., 5000, 100)

def rao_taui_sum_stats(data):
    out = []
    def single_exp(t, tau, A):
        return A*np.exp(-t/tau)
    for d in data.split_periodic(5100, adjust=True):
        d = d.trim(5000, 5100, adjust=True)
        curr = d['ical.i_CaL']
        index = np.argmax(np.abs(curr))
        time = d['engine.time']

        # separate decay portion
        decay = curr[index:]
        time = time[index:]
        t0 = time[0]
        time = [t-t0 for t in time]

        with warnings.catch_warnings():
            warnings.simplefilter('error', so.OptimizeWarning)
            warnings.simplefilter('error', RuntimeWarning)
            try:
                # fit to single exponential
                popt, _ = so.curve_fit(single_exp, time, decay,
                                       p0=[50., 1.],
                                       bounds=([0., -np.inf],
                                               np.inf))
                fit = [single_exp(t,popt[0],popt[1]) for t in time]

                # calculate r squared for fit
                ss_res = np.sum((np.array(decay) - np.array(fit)) ** 2)
                ss_tot = np.sum((np.array(decay) - np.mean(np.array(decay))) ** 2)
                r2 = 1 - (ss_res / ss_tot)

                taui = popt[0]
                if r2 > 0.99:
                    out = out + [taui]
                else:
                    out = out + [float('inf')]
            except:
                out = out + [float('inf')]
    return out

rao_taui = Experiment(
    dataset=rao_taui_dataset,
    protocol=rao_taui_protocol,
    conditions=rao_conditions,
    sum_stats=rao_taui_sum_stats,
    description=rao_taui_desc,
    Q10=Q10_tau,
    Q10_factor=-1
)
