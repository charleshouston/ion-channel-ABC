from ionchannelABC.protocol import recovery, availability_linear
from ionchannelABC.experiment import Experiment
import data.ina.crn_ina as data
import numpy as np
import pandas as pd
import myokit
import warnings
import scipy.optimize as so


def temperature_adjust(R0, T0, T1, Q10):
    return R0*Q10**((T0-T1)/10)


# All experiments use conditions as defined in the
# Courtemanche paper. Data is adjusted to these conditions.
# Experimental conditions are included in experiment setup below
# for reference.
crn_conditions = {'membrane.Na_o': 140000,
                  'membrane.Na_i': 11200,
                  'membrane.T': 310}
Q10 = 3.


### IV curve - Schneider 1994
schneider_iv_desc = """IV curve from Schneider et al 1994."""

vsteps, peaks, sd = data.IV_Schneider()
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

schneider_iv_protocol = myokit.pacing.steptrain_linear(
    -85, 65, 10, -135, 500, 12
)
schneider_conditions = {'membrane.Na_o': 120000,
                        'membrane.Na_i': 70000,
                        'membrane.T': 297.15}

def schneider_iv_sum_stats(data):
    output = []
    for d in data.split_periodic(512, adjust=True):
        d = d.trim(500, 512, adjust=True)
        current = d['ina.i_Na']
        index = np.argmax(np.abs(current))
        output = output+[current[index]/schneider_iv_max_peak]
    return output

schneider_iv = Experiment(
    dataset=schneider_iv_dataset,
    protocol=schneider_iv_protocol,
    conditions=crn_conditions,
    sum_stats=schneider_iv_sum_stats,
    description=schneider_iv_desc
)


### IV curve Sakakibara 1992 (voltage shifted +20mV)
sakakibara_iv_desc = """IV curve from Sakakibara 1992.
Voltage shifted +20mV as in Courtemanche 1998."""

vsteps, peaks, _ = data.IV_Sakakibara()
# Propagate errors in capacitance measure to IV
cm_mean = 126.8 # pF
cm_sem = 10.3 # pF
cm_N = 46 # cells
cm_sd = cm_sem * np.sqrt(cm_N) # pF
# convert nA to pA/pF
peaks = np.array(peaks)
peaks = peaks*1000/cm_mean
sd = [(cm_sd/cm_mean)*p for p in peaks]
sakakibara_iv_max_peak = np.max(np.abs(peaks)) # normalising
peaks = [p / sakakibara_iv_max_peak for p in peaks]
variances = [(sd_ / sakakibara_iv_max_peak)**2 for sd_ in sd]
sakakibara_iv_dataset = np.asarray([vsteps, peaks, variances])

sakakibara_act_protocol = myokit.pacing.steptrain_linear(
    -100, 30, 10, -140, 9900, 100
)
sakakibara_conditions = {'membrane.Na_o': 5000,
                         'membrane.Na_i': 5000,
                         'membrane.T': 290.15}

def sakakibara_iv_sum_stats(data):
    output = []
    for d in data.split_periodic(10000, adjust=True):
        d = d.trim(9900, 10000, adjust=True)
        current = d['ina.i_Na']
        index = np.argmax(np.abs(current))
        output = output + [current[index]/sakakibara_iv_max_peak]
    return output

sakakibara_iv = Experiment(
    dataset=sakakibara_iv_dataset,
    protocol=sakakibara_act_protocol,
    conditions=crn_conditions,
    sum_stats=sakakibara_iv_sum_stats,
    description=sakakibara_iv_desc
)


### Activation Sakakibara (+20mV voltage shift as in CRN 1998)
sakakibara_act_desc = """Activation curve Sakakibara 1992.
Note: Curve is voltage shifted +20mV as in Courtemanche 1998."""

vsteps_act, act, sd_act = data.Act_Sakakibara()
vsteps_act = [v_+20 for v_ in vsteps_act]
variances_act = [sd**2 for sd in sd_act]
sakakibara_act_dataset = np.asarray([vsteps_act, act, variances_act])

def sakakibara_act_sum_stats(data):
    output = []
    for d in data.split_periodic(10000, adjust=True):
        d = d.trim(9900, 10000, adjust=True)
        gate = d['ina.g']
        output = output + [max(gate)]
    for i in range(len(output)):
        output[i] = output[i] / output[-1]
    return output

sakakibara_act = Experiment(
    dataset=sakakibara_act_dataset,
    protocol=sakakibara_act_protocol,
    conditions=crn_conditions,
    sum_stats=sakakibara_act_sum_stats,
    description=sakakibara_act_desc
)


### Inactivation Sakakibara 1992 (voltage shifted)
sakakibara_inact_desc = """Inactivation curve from Sakakibara 1992.
Voltage shifted +20mV as in Courtemanche 1998."""
vsteps_inact, inact, sd_inact = data.Inact_Sakakibara()
vsteps_inact = [v_+20 for v_ in vsteps_inact]
variances_inact = [sd**2 for sd in sd_inact]
sakakibara_inact_dataset = np.asarray([vsteps_inact, inact, variances_inact])

sakakibara_inact_protocol = availability_linear(
   -120, -10, 10, -140, -20, 10000, 1000, 0, 30
)

def sakakibara_inact_sum_stats(data):
    output = []
    for d in data.split_periodic(11030, adjust=True):
        d = d.trim(11000, 11030, adjust=True).npview()
        gate = d['ina.g']
        output = output + [max(gate)]
    for i in range(1, len(output)):
        output[i] = output[i] / output[0]
    output[0] = 1.
    return output

sakakibara_inact = Experiment(
    dataset=sakakibara_inact_dataset,
    protocol=sakakibara_inact_protocol,
    conditions=crn_conditions,
    sum_stats=sakakibara_inact_sum_stats,
    description=sakakibara_inact_desc
)


### Activation kinetics Schneider 1994
schneider_taum_desc = """Activation kinetics from Schneider 1994.
Temperature adjusted from 297 to 310 using Q10 of 3."""

vsteps_tm, tm, sd_tm = data.TauM_Activation_Schneider()
tm = [temperature_adjust(tm_, 297, 310, Q10) for tm_ in tm]
max_tm = np.max(np.abs(tm)) # normalising
tm = [tm_ / max_tm for tm_ in tm]
variances_tm = [(sd_/max_tm)**2 for sd_ in sd_tm]
schneider_taum_dataset = np.asarray([vsteps_tm, tm, variances_tm])

schneider_taum_protocol = myokit.pacing.steptrain_linear(
    -65, 25, 10, -135, 500, 100
)

def schneider_taum_sum_stats(data):
    output = []
    for d in data.split_periodic(600, adjust=True):
        d = d.trim(500, 600, adjust=True)
        current = d['ina.i_Na']
        time = d['environment.time']

        # Remove constant
        c0 = d['ina.i_Na'][0]
        current = [(c_-c0) for c_ in current]

        def sum_of_exp(t, taum, tauh):
            return ((1-np.exp(-t/taum))**3 *
                    np.exp(-t/tauh))
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
    conditions=crn_conditions,
    sum_stats=schneider_taum_sum_stats,
    description=schneider_taum_desc
)


### Inactivation kinetics Sakakibara 1992 (voltage shifted)
sakakibara_tauh_desc = """Inactivation time constants from Sakakibara 1992.
Voltage shifted and temperature adjusted as in Courtemanche 1998."""

vsteps_th, th, sd_th = data.TauH_Inactivation_Sakakibara()
vsteps_th = [v_+20 for v_ in vsteps_th]
th = [temperature_adjust(th_, 290, 310, Q10) for th_ in th]
max_th = np.max(np.abs(th)) # normalising
th = [th_ / max_th for th_ in th]
variances_th = [(sd_/max_th)**2 for sd_ in sd_th]
sakakibara_tauh_dataset = np.asarray([vsteps_th, th, variances_th])

sakakibara_tauh_protocol = myokit.pacing.steptrain_linear(
   -30, 10, 10, -140, 1000, 100
)

def sakakibara_tauh_sum_stats(data):
    output = []
    def simple_exp(t, tauh):
        return np.exp(-t/tauh)

    for d in data.split_periodic(1100, adjust=True):
        d = d.trim(1000, 1100, adjust=True)
        current = d['ina.i_Na']
        time = d['environment.time']
        index = np.argmax(np.abs(current))

        # Set time zero to peak current
        current = current[index:]
        time = time[index:]
        t0 = time[0]
        time = [t-t0 for t in time]

        # Keep only decay phase (i.e. not beyond when current=0)
        index = np.argwhere(np.isclose(current,0.0))
        if len(index) != 0:
            current = current[:index[0][0]]
            time = time[:index[0][0]]

        with warnings.catch_warnings():
            warnings.simplefilter('error', so.OptimizeWarning)
            warnings.simplefilter('error', RuntimeWarning)
            try:
                imax = max(current, key=abs)
                current = [c_/imax for c_ in current]
                if len(time)<=1 or len(current)<=1:
                    raise Exception('Failed simulation')
                popt, _ = so.curve_fit(simple_exp, time, current,
                                       p0=[5], bounds=([0.01], [20.0]))
                tauh = popt[0]
                output = output+[tauh/max_th]
            except:
                output = output+[float('inf')]
    return output

sakakibara_tauh = Experiment(
    dataset=sakakibara_tauh_dataset,
    protocol=sakakibara_tauh_protocol,
    conditions=crn_conditions,
    sum_stats=sakakibara_tauh_sum_stats,
    description=sakakibara_tauh_desc
)


### Recovery kinetics Sakakibara 1992 (voltage shifted)
sakakibara_tauh_depol_desc = """Inactivation time constants obtained at low
voltage using recovery protocols.
Voltage and temperature adjusted as in Courtemanche 1998."""

vsteps_th_depol, th_depol, _ = data.TauH_Inactivation_Sakakibara_Depol()
vsteps_th_depol = [v_+20 for v_ in vsteps_th_depol]
th_depol = [temperature_adjust(th_, 290, 310, Q10) for th_ in th_depol]
max_th_depol = np.max(np.abs(th_depol))
th_depol = [th_ / max_th_depol for th_ in th_depol]
variances_th_depol = [0.]*len(th_depol)
sakakibara_tauh_depol_dataset = np.asarray([vsteps_th_depol, th_depol, variances_th_depol])

# Protocol creation a little more complicated as multiple recovery protocols
# need to be combined. We therefore create the protocols for each depolarisation
# potential separately then combine them with a large time gap.
twaits = [0,2,5,10,15,20,25,30,35,40,45,50,75,100,200,300,400,500,600,700,800,900,1000]
tmp_protocols = []
for v in vsteps_th_depol:
    tmp_protocols.append(
        recovery(twaits, v, -20, -20, 1000, 1000, 1000)
    )
sakakibara_tauh_depol_protocol = tmp_protocols[0]
for p in tmp_protocols[1:]:
    for e in p.events():
        sakakibara_tauh_depol_protocol.add_step(e.level(), e.duration())

twaits_split = [t+3000 for t in twaits]
for i in range(len(twaits_split)-1):
    twaits_split[i+1] += twaits_split[i]

def sakakibara_tauh_depol_sum_stats(data):
    output = []
    def simple_exp(t, tauh):
           return np.exp(-t/tauh)

    for d in data.split_periodic(sum(twaits)+len(twaits)*3000, adjust=True):
        with warnings.catch_warnings():
            warnings.simplefilter('error', so.OptimizeWarning)
            warnings.simplefilter('error', RuntimeWarning)
            try:
                # Get recovery curve
                rec = []
                trim1, trim2, trim3 = 1000, 2000, 3000
                for i, t in enumerate(twaits_split):
                    trace, data = d.split(t)
                    peak1 = max(trace.trim(trim1, trim2)['ina.i_Na'], key=abs)
                    peak2 = max(trace.trim(trim2+twaits[i], trim3+twaits[i])['ina.i_Na'],
                                key=abs)
                    rec.append(peak2/peak1)
                    # Update trim times for next iteration (not adjusting time)
                    trim1 += twaits[i]+3000
                    trim2 += twaits[i]+3000
                    trim3 += twaits[i]+3000

                # Fit double exponential to recovery curve
                popt, _ = so.curve_fit(simple_exp, twaits, 1.-np.asarray(rec),
                                       p0=[1],
                                       bounds=([0],[100]))
                tauh = popt[0]
                output = output+[tauh/max_th_depol]
            except:
                output = output+[float('inf')]
    return output

sakakibara_tauh_depol = Experiment(
    dataset=sakakibara_tauh_depol_dataset,
    protocol=sakakibara_tauh_depol_protocol,
    conditions=crn_conditions,
    sum_stats=sakakibara_tauh_depol_sum_stats,
    description=sakakibara_tauh_depol_sum_stats
)
