import data.iha.data_iha as data
import numpy as np
import scipy.optimize as so
import warnings
import myokit
from ionchannelABC.experiment import Experiment


room_temp = 295
Q10_cond = 1.5 # Correa 1991
Q10_tau = 2.79 # ten Tusscher 2004

def temperature_adjust_cond(R0, T0, T1, Q10):
    return R0*Q10**((T1-T0)/10)

def temperature_adjust_tau(R0, T0, T1, Q10):
    return R0*Q10**((T0-T1)/10)


### IV curve Sartiani 2002
sartiani_iv_desc = """IV curve for iha in HL-1 cells from Sartiani 2002.
Measurements taken at room temperature."""

vsteps_iv, peaks, sd_iv = data.IV_Sartiani()
max_iv_peak = np.max(np.abs(peaks))
peaks = [p/max_iv_peak for p in peaks]
variances_iv = [(sd/max_iv_peak)**2 for sd in sd_iv]
sartiani_iv_dataset = np.asarray([vsteps_iv, peaks, variances_iv])

# non-standard IV curve protocol
sartiani_iv_protocol = myokit.Protocol()
time_ = 0.
vpre = -40
tpre = 5000
vhold = -120
thold = 1500
tstep = 1000
for v in vsteps_iv:
    sartiani_iv_protocol.schedule(vpre, time_, tpre)
    time_ += tpre
    sartiani_iv_protocol.schedule(vhold, time_, thold)
    time_ += thold
    sartiani_iv_protocol.schedule(v, time_, tstep)
    time_ += tstep

sartiani_conditions = {'membrane.K_o': 25e3,
                       'membrane.K_i': 120e3,
                       'membrane.Na_o': 140e3,
                       'membrane.Na_i': 10.8e3,
                       'membrane.T': room_temp}

def sartiani_iv_sum_stats(data):
    output = []
    def simple_exp(t, tau, A):
        return A*(np.exp(-t/tau)-1)
    for d in data.split_periodic(7500, adjust=True):
        d = d.trim(6500, 7500, adjust=True)
        curr = d['iha.i_ha']
        time = d['environment.time']

        c0 = curr[0]
        curr = [c_-c0 for c_ in curr]

        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            warnings.simplefilter('error', so.OptimizeWarning)
            try:
                popt, _ = so.curve_fit(simple_exp,
                                       time,
                                       curr,
                                       p0 = [100., 1.],
                                       bounds=([0., -np.inf],
                                              [10e3, np.inf]))
                amp = popt[1]
                output = output + [amp/max_iv_peak]
            except:
                output = output + [float('inf')]
    return output

sartiani_iv = Experiment(
    dataset=sartiani_iv_dataset,
    protocol=sartiani_iv_protocol,
    conditions=sartiani_conditions,
    sum_stats=sartiani_iv_sum_stats,
    description=sartiani_iv_desc
)


### Activation steady-state and kinetics
sartiani_act_desc = """Activation SS and kinetics for i_ha in HL-1 from Sartiani 2002.
Measurements recorded at room temperature."""

vsteps_act, act, sd_act = data.Act_Sartiani()
variances_act = [sd**2 for sd in sd_act]

_, tau_a, sd_tau_a = data.ActTau_Sartiani()
max_taua = np.max(tau_a)*1000. # convert to ms
tau_a = [ta*1000./max_taua for ta in tau_a]
variances_ta = [(sd/max_taua)**2 for sd in sd_tau_a]
sartiani_act_dataset = [np.asarray([vsteps_act, act, variances_act]),
                        np.asarray([vsteps_act, tau_a, variances_ta])]

sartiani_act_protocol = myokit.Protocol()
time_ = 0.
tpre = 5000.
vpre = -40.
tstep = 1500.
ttest = 1000.
vtest = 40.
for v in vsteps_act:
    sartiani_act_protocol.schedule(vpre, time_, tpre)
    time_ += tpre
    sartiani_act_protocol.schedule(v, time_, tstep)
    time_ += tstep
    sartiani_act_protocol.schedule(vtest, time_, ttest)
    time_ += ttest

def sartiani_act_sum_stats(data):
    out_ss = []
    out_tau = []
    def simple_exp(t, tau, A):
        return A*(1-np.exp(-t/tau))
    for d in data.split_periodic(7500, adjust=True):
        d = d.trim(5000, 6500, adjust=True)
        cond = d['iha.g']
        time = d['environment.time']
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            warnings.simplefilter('error', so.OptimizeWarning)
            try:
                popt, _ = so.curve_fit(simple_exp,
                                       time,
                                       cond,
                                       p0=[1000., 1.],
                                       bounds=([0., -np.inf],
                                               np.inf))
                tau_a = popt[0]
                max_cond = popt[1]
                out_ss = out_ss + [max_cond]
                out_tau = out_tau + [tau_a/max_taua]
            except:
                out_ss = out_ss + [float('inf')]
                out_tau = out_tau + [float('inf')]

    max_out_ss = max(out_ss)
    out_ss = [o/max_out_ss for o in out_ss]
    return out_ss + out_tau

sartiani_act = Experiment(
    dataset=sartiani_act_dataset,
    protocol=sartiani_act_protocol,
    conditions=sartiani_conditions,
    sum_stats=sartiani_act_sum_stats,
    description=sartiani_act_desc
)
