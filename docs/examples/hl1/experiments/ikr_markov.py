import data.ikr.data_ikr as data
import numpy as np
from functools import partial
import scipy.optimize as so
import myokit
import warnings
from ionchannelABC.experiment import Experiment
from ionchannelABC.protocol import varying_test_duration


room_temp = 296
Q10_cond = 4.4 # [Kiyosue1993]
Q10_tau = 4.1  # [Kiyosue1993]


#
# IV curve [Toyoda2010]
#
toyoda_iv_desc = """IV curve from Figure 1E of Toyoda 2010.
Current measured just before the end of the 1s pulse.

Data recorded at 308K."""

vsteps_iv, peaks, sd_iv = data.IV_Toyoda()
variances_iv = [sd**2 for sd in sd_iv]
toyoda_iv_dataset = np.asarray([vsteps_iv, peaks, variances_iv])

toyoda_iv_protocol = myokit.pacing.steptrain_linear(
    -80, 50, 10, -50, 5000, 1000, 5000
)

toyoda_conditions = {'extra.K_o': 5.4e3,
                     'potassium.K_i': 130e3,
                     'phys.T': 308}

def toyoda_iv_sum_stats(data):
    output = []
    for d in data.split_periodic(11000, adjust=True):
        d = d.trim(5000, 6000, adjust=True)
        output = output + [d['ikr.i_Kr'][-1]]# / max_observed_peak_iv]
    return output

toyoda_iv = Experiment(
    dataset=toyoda_iv_dataset,
    protocol=toyoda_iv_protocol,
    conditions=toyoda_conditions,
    sum_stats=toyoda_iv_sum_stats,
    description=toyoda_iv_desc,
    Q10=Q10_cond,
    Q10_factor=1
)


#
# Activation kinetics [Toyoda2010]
#
toyoda_taua_desc = """Activation kinetics Toyoda 2010 (Fig 3C).
Assessed by envelope-of-tails protocol. Series of depolarising
steps of varying duration with single exponential fit to peak
tail current elicited upon repolarisation to holding potential.

Data recorded at 308K.
"""

vsteps_taua, taua, sd_taua = data.ActKin_Toyoda()
variances_ta = [sd**2 for sd in sd_taua]
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
                                       bounds=([0., 0.],
                                               [np.inf, np.inf]))
                taua = popt[0]
                output = output+[taua]
            except:
                output = output+[float('inf')]
    return output

toyoda_taua = Experiment(
    dataset=toyoda_taua_dataset,
    protocol=toyoda_taua_protocol,
    conditions=toyoda_conditions,
    sum_stats=toyoda_taua_sum_stats,
    description=toyoda_taua_desc,
    Q10=Q10_tau,
    Q10_factor=-1
)


#
# Deactivation kinetics [Toyoda2010]
#
toyoda_deact_desc = """Fast deactivation kinetics of ikr from Toyoda 2010 (Fig 3C).
Assuming single exponential fit to decay phase.

Carried out at 308K.
"""

# Multiple datasets for fast, slow tau and relative amplitude
vsteps_taui, taui_f, sd_taui_f = data.DeactKinFast_Toyoda()
variances_taui_f = [sd**2 for sd in sd_taui_f]

_, taui_s, sd_taui_s = data.DeactKinSlow_Toyoda()
variances_taui_s = [sd**2 for sd in sd_taui_s]

_, taui_relamp, sd_taui_relamp = data.DeactKinRelAmp_Toyoda()
variances_taui_relamp = [sd**2 for sd in sd_taui_relamp]

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

    def simple_exp(t, tauf, A):
        return A*np.exp(-t/tauf)

    def double_exp(t, tauf, taus, Frac, Amp):
        return Amp*(Frac*np.exp(-t/tauf) + (1-Frac)*np.exp(-t/taus))

    for d in data.split_periodic(11000, adjust=True):
        d = d.trim(5000, 6000, adjust=True)
        current = d['ikr.i_Kr']
        time = d['engine.time']
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
                                       p0=[10.,1000.,0.5,1.0],
                                       bounds=([1.,100.,0.0,0.],
                                               [1000.,5000.,1.0,np.inf]))
                taui_f = popt[0]
                taui_s = popt[1]
                taui_relamp = popt[2]
                out1 = out1 + [taui_f]
                out2 = out2 + [taui_s]
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
    description=toyoda_deact_desc,
    Q10=Q10_tau,
    Q10_factor=[-1,-1,0]
)


#
# Recovery kinetics [Toyoda2010]
#
toyoda_trec_desc = """Recovery kinetics of iKr from Toyoda 2010 (Fig 5B).
Measured by fitting increment phases of tail currents.

Carried out at 298K.
"""

vsteps_trec, trec, sd_trec = data.InactKin_Toyoda()
variances_trec = [sd**2 for sd in sd_trec]
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
        time = d['engine.time']
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
                                               [np.inf]))
                taur = popt[0]
                output = output + [taur]
            except:
                output = output + [float('inf')]
    return output

toyoda_conditions_cold = {'extra.K_o': 5.4e3,
                          'potassium.K_i': 130e3,
                          'phys.T': 298}

toyoda_trec = Experiment(
    dataset=toyoda_trec_dataset,
    protocol=toyoda_trec_protocol,
    conditions=toyoda_conditions_cold,
    sum_stats=toyoda_trec_sum_stats,
    description=toyoda_trec_desc,
    Q10=Q10_tau,
    Q10_factor=-1
)


#
# Steady-state inactivation [Toyoda2010]
#
toyoda_inact_desc = """Inactivation curve from Toyoda 2010.
Data is the raw measured values from Fig 6B (i.e. not the author's
corrected curve).

Measurements taken at 298K.
"""

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
    conditions=toyoda_conditions_cold,
    sum_stats=toyoda_inact_sum_stats,
    description=toyoda_inact_desc,
    Q10=None,
    Q10_factor=0
)
