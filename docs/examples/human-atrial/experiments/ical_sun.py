from ionchannelABC.experiment import Experiment
import data.ical.Sun1997.data_Sun1997 as data
from ionchannelABC.protocol import availability_linear, recovery
import numpy as np
import myokit
import warnings
from scipy.optimize import OptimizeWarning
import scipy.optimize as so


Q10_cond = 1.6      # [Li1997]
Q10_tau_act = 1.7   # [Li1997]
Q10_tau_inact = 1.3 # [Li1997]


fit_threshold = 0.9


#
# Inactivation kinetics [Sun1997]
#
sun_inact_kin_desc = """
    Inactivation kinetics measured using bi-exponential funcion

    cf Fig 4B [Sun1997]
    """
vsteps_tf, tf, sd_tf = data.inact_tauf_Sun()
variances_tf = [sd_**2 for sd_ in sd_tf]
sun_inact_kin_tf_dataset = np.array([vsteps_tf, tf, variances_tf])

vsteps_ts, ts, sd_ts = data.inact_taus_Sun()
variances_ts = [sd_**2 for sd_ in sd_ts]
sun_inact_kin_ts_dataset = np.array([vsteps_ts, ts, variances_ts])

vsteps_rel_inact, rel_inact, sd_rel_inact = data.rel_inact_Sun()
variances_rel_inact = [sd_**2 for sd_ in sd_rel_inact]
sun_inact_kin_rel_inact_dataset = np.array([vsteps_rel_inact, rel_inact, variances_rel_inact])

# all voltage protocols are preceded by 500ms prepulse to -40mV from HP -80mV
tpre = 10000 # ms
tstep = 1000
vlower = -10
dv = 10
vupper = 30+dv

sun_inact_kin_protocol = myokit.Protocol()
for v in vsteps_tf:
    sun_inact_kin_protocol.add_step(-80, tpre-500)
    sun_inact_kin_protocol.add_step(-40, 500)
    sun_inact_kin_protocol.add_step(v, 1000)

sun_conditions = {'phys.T': 296.15,  # K
                  'ca_conc.Ca_o': 1} # mM

def sun_inact_kin_sum_stats(data, fast=True, slow=True):
    def double_exp(t, tauh, taus, Ah, As, A0):
        return A0 + Ah*np.exp(-t/tauh) + As*np.exp(-t/taus)
    output_tf = []
    output_ts = []
    for d in data.split_periodic(11000, adjust=True, closed_intervals=False):
        d = d.trim_left(10000, adjust=True)

        current = d['ical.i_CaL']
        time = d['engine.time']
        index = np.argmax(np.abs(current))

        # Set time zero to peak current
        current = current[index:]
        time = time[index:]
        t0 = time[0]
        time = [t-t0 for t in time]

        with warnings.catch_warnings():
            warnings.simplefilter('error', OptimizeWarning)
            warnings.simplefilter('error', RuntimeWarning)
            try:
                current = [c/current[0] for c in current]
                if len(time)<=1 or len(current)<=1:
                    raise Exception('Failed simulation')
                popt, _ = so.curve_fit(double_exp,
                                       time,
                                       current,
                                       p0=[10,200,0.5,0.5,0],
                                       bounds=(0.,
                                               [np.inf, np.inf, 1.0, 1.0, 1.0]),
                                       max_nfev=1000)
                fit = [double_exp(t,popt[0],popt[1],popt[2],popt[3],popt[4]) for t in time]
                # Calculate r2
                ss_res = np.sum((np.array(current)-np.array(fit))**2)
                ss_tot = np.sum((np.array(current)-np.mean(np.array(current)))**2)
                r2 = 1 - (ss_res / ss_tot)

                tauf = min(popt[0],popt[1])
                taus = max(popt[0],popt[1])

                if r2 > fit_threshold:
                    if fast:
                        output_tf = output_tf+[tauf]
                    if slow:
                        output_ts = output_ts+[taus]
                else:
                    raise RuntimeWarning('scipy.optimize.curve_fit found a poor fit')
            except:
                if fast:
                    output_tf = output_tf+[float('inf')]
                if slow:
                    output_ts = output_ts+[float('inf')]
    output = output_tf+output_ts
    return output

def sun_inact_kin_sum_stats_tf(data):
    return sun_inact_kin_sum_stats(data, fast=True, slow=False)

def sun_inact_kin_sum_stats_ts(data):
    return sun_inact_kin_sum_stats(data, fast=False, slow=True)

def sun_rel_inact_sum_stats(data):
    output = []
    for d in data.split_periodic(11000, adjust=True, closed_intervals=False):
        d = d.trim_left(10000, adjust=True)
        current = d['ical.i_CaL']
        peak = max(current, key=abs)
        ss = current[-1]

        try:
            output = output + [1-ss/peak]
        except:
            output = output + [float('inf')]
    return output

sun_inact_kin = Experiment(
    dataset=[sun_inact_kin_tf_dataset,
             sun_inact_kin_ts_dataset],
    protocol=sun_inact_kin_protocol,
    conditions=sun_conditions,
    sum_stats=sun_inact_kin_sum_stats,
    description=sun_inact_kin_desc,
    Q10=Q10_tau_inact,
    Q10_factor=-1)

sun_inact_kin_fast = Experiment(
    dataset=sun_inact_kin_tf_dataset,
    protocol=sun_inact_kin_protocol,
    conditions=sun_conditions,
    sum_stats=sun_inact_kin_sum_stats_tf,
    description=sun_inact_kin_desc,
    Q10=Q10_tau_inact,
    Q10_factor=-1)

sun_inact_kin_slow = Experiment(
    dataset=sun_inact_kin_ts_dataset,
    protocol=sun_inact_kin_protocol,
    conditions=sun_conditions,
    sum_stats=sun_inact_kin_sum_stats_ts,
    description=sun_inact_kin_desc,
    Q10=Q10_tau_inact,
    Q10_factor=-1)

sun_rel_inact = Experiment(
    dataset=sun_inact_kin_rel_inact_dataset,
    protocol=sun_inact_kin_protocol,
    conditions=sun_conditions,
    sum_stats=sun_rel_inact_sum_stats,
    description=sun_inact_kin_desc,
    Q10=None,
    Q10_factor=0)


#
# Inactivation kinetics using monovalent cation [Sun1997]
#
sun_v_inact_kin_desc = """
    Voltage-dependent inactivation kinetics measured using bi-exponential funcion

    cf Fig 6B-C [Sun1997]
    """
vsteps_tf, tf, sd_tf = data.inact_tauf_Sun()
variances_tf = [sd_**2 for sd_ in sd_tf]
sun_inact_kin_tf_dataset = np.array([vsteps_tf, tf, variances_tf])

vsteps_ts, ts, sd_ts = data.inact_taus_Sun()
variances_ts = [sd_**2 for sd_ in sd_ts]
sun_inact_kin_ts_dataset = np.array([vsteps_ts, ts, variances_ts])

vsteps_rel_inact, rel_inact, sd_rel_inact = data.rel_inact_Sun()
variances_rel_inact = [sd_**2 for sd_ in sd_rel_inact]
sun_inact_kin_rel_inact_dataset = np.array([vsteps_rel_inact, rel_inact, variances_rel_inact])

# all voltage protocols are preceded by 500ms prepulse to -40mV from HP -80mV
tpre = 10000 # ms
tstep = 1000
vlower = -10
dv = 10
vupper = 30+dv

sun_inact_kin_protocol = myokit.Protocol()
for v in vsteps_tf:
    sun_inact_kin_protocol.add_step(-80, tpre-500)
    sun_inact_kin_protocol.add_step(-40, 500)
    sun_inact_kin_protocol.add_step(v, 1000)

sun_conditions = {'phys.T': 296.15,  # K
                  'ca_conc.Ca_o': 1} # mM

def sun_inact_kin_sum_stats(data, fast=True, slow=True):
    def double_exp(t, tauh, taus, Ah, As, A0):
        return A0 + Ah*np.exp(-t/tauh) + As*np.exp(-t/taus)
    output_tf = []
    output_ts = []
    for d in data.split_periodic(11000, adjust=True, closed_intervals=False):
        d = d.trim_left(10000, adjust=True)

        current = d['ical.i_CaL']
        time = d['engine.time']
        index = np.argmax(np.abs(current))

        # Set time zero to peak current
        current = current[index:]
        time = time[index:]
        t0 = time[0]
        time = [t-t0 for t in time]

        with warnings.catch_warnings():
            warnings.simplefilter('error', OptimizeWarning)
            warnings.simplefilter('error', RuntimeWarning)
            try:
                current = [c/current[0] for c in current]
                if len(time)<=1 or len(current)<=1:
                    raise Exception('Failed simulation')
                popt, _ = so.curve_fit(double_exp,
                                       time,
                                       current,
                                       p0=[10,200,0.5,0.5,0],
                                       bounds=(0.,
                                               [np.inf, np.inf, 1.0, 1.0, 1.0]),
                                       max_nfev=1000)
                fit = [double_exp(t,popt[0],popt[1],popt[2],popt[3],popt[4]) for t in time]
                # Calculate r2
                ss_res = np.sum((np.array(current)-np.array(fit))**2)
                ss_tot = np.sum((np.array(current)-np.mean(np.array(current)))**2)
                r2 = 1 - (ss_res / ss_tot)

                tauf = min(popt[0],popt[1])
                taus = max(popt[0],popt[1])

                if r2 > fit_threshold:
                    if fast:
                        output_tf = output_tf+[tauf]
                    if slow:
                        output_ts = output_ts+[taus]
                else:
                    raise RuntimeWarning('scipy.optimize.curve_fit found a poor fit')
            except:
                if fast:
                    output_tf = output_tf+[float('inf')]
                if slow:
                    output_ts = output_ts+[float('inf')]
    output = output_tf+output_ts
    return output

def sun_inact_kin_sum_stats_tf(data):
    return sun_inact_kin_sum_stats(data, fast=True, slow=False)

def sun_inact_kin_sum_stats_ts(data):
    return sun_inact_kin_sum_stats(data, fast=False, slow=True)

def sun_rel_inact_sum_stats(data):
    output = []
    for d in data.split_periodic(11000, adjust=True, closed_intervals=False):
        d = d.trim_left(10000, adjust=True)
        current = d['ical.i_CaL']
        peak = max(current, key=abs)
        ss = current[-1]

        try:
            output = output + [1-ss/peak]
        except:
            output = output + [float('inf')]
    return output

sun_inact_kin = Experiment(
    dataset=[sun_inact_kin_tf_dataset,
             sun_inact_kin_ts_dataset],
    protocol=sun_inact_kin_protocol,
    conditions=sun_conditions,
    sum_stats=sun_inact_kin_sum_stats,
    description=sun_inact_kin_desc,
    Q10=Q10_tau_inact,
    Q10_factor=-1)


