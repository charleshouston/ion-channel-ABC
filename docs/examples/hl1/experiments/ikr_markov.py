import data.ikr.data_ikr as data
import numpy as np
from functools import partial
import scipy.optimize as so
import myokit
import warnings
from ionchannelABC.experiment import Experiment
from ionchannelABC.protocol import varying_test_duration


room_temp = 295
Q10_cond = 1.5 # Correa 1991
Q10_tau = 2.79 # ten Tusscher 2004

def temperature_adjust_cond(R0, T0, T1, Q10):
    return R0*Q10**((T1-T0)/10)

def temperature_adjust_tau(R0, T0, T1, Q10):
    return R0*Q10**((T0-T1)/10)


### IV curve
toyoda_iv_desc = """IV curve from Figure 1E of Toyoda 2005.
Current measured just before the end of the 1s pulse.

Data recorded at 308K and adjusted to room_temp using Q10 factor of 1.5
(Correa 1991)."""

vsteps_iv, peaks, sd_iv = data.IV_Toyoda()
peaks = [temperature_adjust_cond(pk, 308, room_temp, Q10_cond) for pk in peaks]
sd_iv = [temperature_adjust_cond(sd, 308, room_temp, Q10_cond) for sd in sd_iv]
max_observed_peak_iv = np.max(np.abs(peaks))
peaks = [p / max_observed_peak_iv for p in peaks]
variances_iv = [(sd / max_observed_peak_iv)**2 for sd in sd_iv]
toyoda_iv_dataset = np.asarray([vsteps_iv, peaks, variances_iv])

toyoda_iv_protocol = myokit.pacing.steptrain_linear(
    -80, 50, 10, -50, 5000, 1000, 5000
)

toyoda_conditions = {'membrane.K_o': 5.4e3,
                     'membrane.K_i': 130e3,
                     'membrane.T': 295}

def toyoda_iv_sum_stats(data):
    output = []
    for d in data.split_periodic(11000, adjust=True):
        d = d.trim(5000, 6000, adjust=True)
        output = output + [d['ikr.i_Kr'][-1] / max_observed_peak_iv]
    return output

toyoda_iv = Experiment(
    dataset=toyoda_iv_dataset,
    protocol=toyoda_iv_protocol,
    conditions=toyoda_conditions,
    sum_stats=toyoda_iv_sum_stats,
    description=toyoda_iv_desc
)


### Activation kinetics Toyoda 2005
toyoda_taua_desc = """Activation kinetics Toyoda 2005 (Fig 3C).
Assessed by envelope-of-tails protocol. Series of depolarising
steps of varying duration with single exponential fit to peak
tail current elicited upon repolarisation to holding potential.

Data recorded at 308K and temperature adjusted to 295K using a Q10
factor of 2.79 (ten Tusscher 2004).
"""

vsteps_taua, taua, sd_taua = data.ActKin_Toyoda()
taua = [temperature_adjust_tau(ta, 308, room_temp, Q10_tau) for ta in taua]
sd_taua = [temperature_adjust_tau(sd, 308, room_temp, Q10_tau) for sd in sd_taua]
max_ta = np.max(np.abs(taua))
taua = [ta / max_ta for ta in taua]
variances_ta = [(sd/max_ta)**2 for sd in sd_taua]
toyoda_taua_dataset = np.asarray([vsteps_taua, taua, variances_ta])

# Generate single protocol to cover all voltage steps
intervals = np.arange(25, 975+50, 50)
tmp_protocols = []
for v in vsteps_taua:
    tmp_protocols.append(
        varying_test_duration(intervals, -50, v, 5000, 5000)
    )
characteristic_time = tmp_protocols[0].characteristic_time()
toyoda_taua_protocol = tmp_protocols[0]
for p in tmp_protocols[1:]:
    for e in p.events():
        toyoda_taua_protocol.add_step(e.level(), e.duration())

tsplit = [i+10000 for i in intervals]
for i in range(len(tsplit)-1):
    tsplit[i+1] += tsplit[i]

# Fitting single exponential to peak tail currents with step
# duration as independent variable
def toyoda_taua_sum_stats(data):
    output = []
    def simple_exp(t, taua, A):
        return A*(1-np.exp(-t/taua))

    for d in data.split_periodic(characteristic_time, adjust=True):
        peak_tail = []
        for i, t in enumerate(tsplit):
            trace, d = d.split(t)
            trace = trace.trim(t-5000, t)
            peak_tail.append(max(trace['ikr.i_Kr'])-trace['ikr.i_Kr'][-1])

        with warnings.catch_warnings():
            warnings.simplefilter('error', so.OptimizeWarning)
            warnings.simplefilter('error', RuntimeWarning)
            try:
                popt, _ = so.curve_fit(simple_exp,
                                       intervals,
                                       peak_tail,
                                       p0=[500, 1.],
                                       bounds=([0.01, 0.],
                                               [2000, np.inf]))
                taua = popt[0]
                output = output+[taua/max_ta]
            except:
                output = output+[float('inf')]
    return output

toyoda_taua = Experiment(
    dataset=toyoda_taua_dataset,
    protocol=toyoda_taua_protocol,
    conditions=toyoda_conditions,
    sum_stats=toyoda_taua_sum_stats,
    description=toyoda_taua_desc
)


### Deactivation kinetics Toyoda 2005
toyoda_deact_desc = """Deactivation kinetics of ikr from Toyoda 2005 (Fig 3C).
Measured by fitting sum of exponentials to current decay phase.

Adjusted to 295K from 308K using Q10 factor 2.79 (ten Tusscher 2004).
"""

# Multiple datasets for fast, slow tau and relative amplitude
vsteps_taui, taui_f, sd_taui_f = data.DeactKinFast_Toyoda()
taui_f = [temperature_adjust_tau(ti, 308, room_temp, Q10_tau) for ti in taui_f]
max_tf = np.max(np.abs(taui_f))
sd_taui_f = [temperature_adjust_tau(sd, 308, room_temp, Q10_tau) for sd in sd_taui_f]
taui_f = [ti_/max_tf for ti_ in taui_f]
variances_taui_f = [(sd_/max_tf)**2 for sd_ in sd_taui_f]

_, taui_s, sd_taui_s = data.DeactKinSlow_Toyoda()
taui_s = [temperature_adjust_tau(ti, 308, room_temp, Q10_tau) for ti in taui_s]
max_ts = np.max(np.abs(taui_s))
sd_taui_s = [temperature_adjust_tau(sd, 308, room_temp, Q10_tau) for sd in sd_taui_s]
taui_s = [ti_/max_ts for ti_ in taui_s]
variances_taui_s = [(sd_/max_ts)**2 for sd_ in sd_taui_s]

_, taui_relamp, sd_taui_relamp = data.DeactKinRelAmp_Toyoda()
variances_taui_relamp = [sd_**2 for sd_ in sd_taui_relamp]

toyoda_taui_f_dataset = np.asarray([vsteps_taui, taui_f, variances_taui_f])
toyoda_taui_s_dataset = np.asarray([vsteps_taui, taui_s, variances_taui_s])
toyoda_taui_relamp_dataset = np.asarray([vsteps_taui, taui_relamp, variances_taui_relamp])

toyoda_deact_protocol = myokit.pacing.steptrain_linear(
    -120, -20, 10, 20, 5000, 1000, 5000
)

def toyoda_deact_sum_stats(data):
    out1 = []
    out2 = []
    out3 = []

    def double_exp(t, tauf, taus, Arel):
        return (Arel*np.exp(-t/tauf) + (1-Arel)*np.exp(-t/taus))

    for d in data.split_periodic(11000, adjust=True):
        d = d.trim(5000, 6000, adjust=True)
        current = d['ikr.i_Kr']
        time = d['environment.time']
        index = np.argmax(np.abs(current))

        # set time zero to peak current
        current = current[index:]
        time = time[index:]
        t0 = time[0]
        time = [t-t0 for t in time]

        with warnings.catch_warnings():
            warnings.simplefilter('error', so.OptimizeWarning)
            warnings.simplefilter('error', RuntimeWarning)
            try:
                imax = max(current, key=abs)
                current = [c_/imax for c_ in current]
                if len(time)<=1 or len(current)<=1:
                    raise Exception('failed simulation')
                popt, _ = so.curve_fit(double_exp, time, current,
                                       p0=[50, 400, 0.7],
                                       bounds=([0.01,200,0.],
                                               [300,3000,1.0]))
                taui_f = popt[0]
                taui_s = popt[1]
                taui_relamp = popt[2]
                out1 = out1 + [taui_f/max_tf]
                out2 = out2 + [taui_s/max_ts]
                out3 = out3 + [taui_relamp]
            except:
                out1 = out1 + [float('inf')]
                out2 = out2 + [float('inf')]
                out3 = out3 + [float('inf')]
    output = out1 + out2 + out3
    return output

toyoda_deact = Experiment(
    dataset=[toyoda_taui_f_dataset,
             toyoda_taui_s_dataset,
             toyoda_taui_relamp_dataset],
    protocol=toyoda_deact_protocol,
    conditions=toyoda_conditions,
    sum_stats=toyoda_deact_sum_stats,
    description=toyoda_deact_desc
)


### Recovery kinetics Toyoda 2005
toyoda_trec_desc = """Recovery kinetics of iKr from Toyoda 2005 (Fig 5B).
Measured by fitting increment phases of tail currents.

Carried out at 298K so no temperature adjustment made.
"""

vsteps_trec, trec, sd_trec = data.InactKin_Toyoda()
max_tr = np.max(np.abs(trec))
trec = [tr_/max_tr for tr_ in trec]
variances_trec = [(sd_/max_tr)**2 for sd_ in sd_trec]
toyoda_trec_dataset = np.asarray([vsteps_trec, trec, variances_trec])

# Custom protocol (see inset Fig 5A)
toyoda_trec_protocol = myokit.Protocol()
time_ = 0.
vhold = -50.
vpre = 20.
vsteps = np.arange(-120, 0, 10)
thold = 5000.
tpre = 1000.
tstep = 1000.
for v in vsteps:
    toyoda_trec_protocol.schedule(vhold, time_, thold)
    time_ += thold
    toyoda_trec_protocol.schedule(vpre, time_, tpre)
    time_ += tpre
    toyoda_trec_protocol.schedule(v, time_, tstep)
    time_ += tstep

def toyoda_trec_sum_stats(data):
    output = []
    def simple_exp(t, taur):
        return 1-np.exp(-t/taur)
    for d in data.split_periodic(7000, adjust=True):
        d = d.trim(6000, 7000, adjust=True)
        current = d['ikr.i_Kr']
        time = d['environment.time']
        # only initial increasing phase of tail current
        index = np.argmax(np.abs(current))
        current = current[:index+1]
        time = time[:index+1]

        with warnings.catch_warnings():
            warnings.simplefilter('error', so.OptimizeWarning)
            warnings.simplefilter('error', RuntimeWarning)
            try:
                imax = max(current, key=abs)
                current = [c_/imax for c_ in current]
                if len(time)<=1 or len(current)<=1:
                    raise Exception('failed sim')
                popt, _ = so.curve_fit(simple_exp, time, current,
                                       p0=[1.],
                                       bounds=([0.],
                                               [50.]))
                taur = popt[0]/max_tr
                output = output + [taur]
            except:
                output = output + [float('inf')]
    return output

toyoda_trec = Experiment(
    dataset=toyoda_trec_dataset,
    protocol=toyoda_trec_protocol,
    conditions=toyoda_conditions,
    sum_stats=toyoda_trec_sum_stats,
    description=toyoda_trec_desc
)


### Inactivation curve Toyoda 2005
toyoda_inact_desc = """Inactivation curve from Toyoda 2005.
Data is the raw measured values from Fig 6B (i.e. not the author's
corrected curve).

Measurements taken at 298K so no temperature adjustment applied."""

vsteps_inact, inact, _ = data.Inact_Toyoda()
toyoda_inact_dataset = np.asarray([vsteps_inact, inact, [0.]*len(inact)])

toyoda_inact_protocol = myokit.Protocol()
time_ = 0.
vhold = -50.
vpre = 20.
vsteps = np.arange(-120, 20, 10)
vpost = 20.
thold = 5000.
tpre = 1000.
tstep = 10.
tpost = 5000.
for v in vsteps:
    toyoda_inact_protocol.schedule(vhold, time_, thold)
    time_ += thold
    toyoda_inact_protocol.schedule(vpre, time_, tpre)
    time_ += tpre
    toyoda_inact_protocol.schedule(v, time_, tstep)
    time_ += tstep
    toyoda_inact_protocol.schedule(vpost, time_, tpost)
    time_ += tpost

def toyoda_inact_sum_stats(data):
    output = []
    for d in data.split_periodic(11010, adjust=True):
        d = d.trim(6010, 11010, adjust=True)
        output = output + [max(d['ikr.i_Kr'])]
    max_curr = np.max(output)
    output = [curr/max_curr for curr in output]
    return output

toyoda_inact = Experiment(
    dataset=toyoda_inact_dataset,
    protocol=toyoda_inact_protocol,
    conditions=toyoda_conditions,
    sum_stats=toyoda_inact_sum_stats,
    description=toyoda_inact_desc
)
