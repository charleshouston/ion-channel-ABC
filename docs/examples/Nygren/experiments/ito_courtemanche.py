"""
Experiments from [Courtemanche1998]
Charles Houston 2019-10-18
"""
from ionchannelABC.experiment import Experiment
from ionchannelABC.protocol import recovery
import data.ito.Courtemanche1998.data_Courtemanche1998 as data
import numpy as np
import myokit

import warnings
from scipy.optimize import OptimizeWarning
import scipy.optimize as so


Q10_tau = 2.2 # [Wang1993]


fit_threshold = 0.9


#
# Activation and inactivation kinetics [Courtemanche1998]
#
courtemanche_kin_desc = """
    Activation kinetics for ito in human
    atrial myocytes [Courtemanche1998] cf Fig 5d.

    It is not clear where these data are from. We assume
    the same activation curve fitting protocol as described
    in [Wang1993] (same lab).

    We assume the SD is 10% of the given time constants as
    no error is reported.

    100 ms pulses from holding potential of -50 mV to
    various test potentials.
    """
vsteps_taua, tau_a, _ = data.Act_Kin_Courtemanche()
sd_taua = np.asarray(tau_a)*0.1
variances_taua = [sd_**2 for sd_ in sd_taua]
courtemanche_act_kin_dataset = np.asarray([vsteps_taua, tau_a, variances_taua])

vsteps_taui, tau_i, _ = data.Inact_Kin_Courtemanche()
sd_taui = np.asarray(tau_i)*0.1
variances_taui = [sd_**2 for sd_ in sd_taui]
courtemanche_inact_kin_dataset = np.asarray([vsteps_taui, tau_i, variances_taui])

courtemanche_kin_protocol = myokit.pacing.steptrain(
    vsteps_taui, -50, 20000, 100)

wang_conditions = {'phys.T': 295.15,  # K
                   'k_conc.K_i': 130, # mM
                   'k_conc.K_o': 5.4
                  }

def courtemanche_kin_sum_stats(data, act=True, inact=True):
    def single_exp_act(t, tau, A, A0):
        return A0 - A*np.exp(-t/tau)
    def single_exp_inact(t, tau, A, A0):
        return A0 + A*np.exp(-t/tau)
    output_act = []
    output_inact = []

    for d in data.split_periodic(20100, adjust=True, closed_intervals=False):
        d = d.trim_left(20000, adjust=True)

        current = d['ito.i_to']
        time = d['engine.time']
        index = np.argmax(np.abs(current))
        if act:
            # Set time zero to peak current
            act = current[:index]
            time_act = time[:index]

            with warnings.catch_warnings():
                warnings.simplefilter('error', OptimizeWarning)
                warnings.simplefilter('error', RuntimeWarning)
                try:
                    norm = act[-1]
                    act = [c/norm for c in act]
                    if len(time_act)<=1 or len(act)<=1:
                        raise Exception('Failed simulation')
                    popt, _ = so.curve_fit(single_exp_act,
                                           time_act,
                                           act,
                                           p0=[10,0.5,0.],
                                           bounds=(0.,
                                                   [np.inf, 1.0, 1.0]),
                                           max_nfev=1000)
                    fit_act = [single_exp_act(t,popt[0],popt[1],popt[2]) for t in time_act]

                    # Calculate r2
                    ss_res = np.sum((np.array(act)-np.array(fit_act))**2)
                    ss_tot = np.sum((np.array(act)-np.mean(np.array(act)))**2)
                    r2 = 1 - (ss_res / ss_tot)

                    taua = popt[0]

                    if r2 > fit_threshold:
                        output_act = output_act+[taua]
                    else:
                        raise RuntimeWarning('scipy.optimize.curve_fit found a poor fit')
                except:
                    output_act = output_act+[float('inf')]

        if inact:
            # Set time zero to peak current
            inact = current[index:]
            time_inact = time[index:]
            t0 = time_inact[0]
            time_inact = [t-t0 for t in time_inact]

            with warnings.catch_warnings():
                warnings.simplefilter('error', OptimizeWarning)
                warnings.simplefilter('error', RuntimeWarning)
                try:
                    norm = inact[0]
                    inact = [c/norm for c in inact]
                    if len(time_inact)<=1 or len(inact)<=1:
                        raise Exception('Failed simulation')
                    popt, _ = so.curve_fit(single_exp_inact,
                                           time_inact,
                                           inact,
                                           p0=[100,0.5,0.],
                                           bounds=(0.,
                                                   [np.inf, 1.0, 1.0]),
                                           max_nfev=1000)
                    fit_inact = [single_exp_inact(t,popt[0],popt[1],popt[2]) for t in time_inact]

                    # Calculate r2
                    ss_res = np.sum((np.array(inact)-np.array(fit_inact))**2)
                    ss_tot = np.sum((np.array(inact)-np.mean(np.array(inact)))**2)
                    r2 = 1 - (ss_res / ss_tot)

                    taui = popt[0]

                    if r2 > fit_threshold:
                        output_inact = output_inact+[taui]
                    else:
                        raise RuntimeWarning('scipy.optimize.curve_fit found a poor fit')
                except:
                    output_inact = output_inact+[float('inf')]
    return output_act+output_inact

def courtemanche_act_kin_protocol(data):
    return courtemanche_kin_protocol(data, act=True, inact=False)

def courtemanche_inact_kin_protocol(data):
    return courtemanche_kin_protocol(data, act=False, inact=True)

courtemanche_kin = Experiment(
    dataset=[courtemanche_act_kin_dataset,
             courtemanche_inact_kin_dataset],
    protocol=courtemanche_kin_protocol,
    conditions=wang_conditions,
    sum_stats=courtemanche_kin_sum_stats,
    description=courtemanche_kin_desc,
    Q10=Q10_tau,
    Q10_factor=-1)
courtemanche_act_kin = Experiment(
    dataset=courtemanche_act_kin_dataset,
    protocol=courtemanche_act_kin_protocol,
    conditions=wang_conditions,
    sum_stats=courtemanche_kin_sum_stats,
    description=courtemanche_kin_desc,
    Q10=Q10_tau,
    Q10_factor=-1)
courtemanche_inact_kin = Experiment(
    dataset=courtemanche_inact_kin_dataset,
    protocol=courtemanche_inact_kin_protocol,
    conditions=wang_conditions,
    sum_stats=courtemanche_kin_sum_stats,
    description=courtemanche_kin_desc,
    Q10=Q10_tau,
    Q10_factor=-1)


#
# Recovery kinetics [Nygren1998]
#
courtemanche_rec_desc = """
    Recovery kinetics for ito in human
    atrial myocytes [Courtemanche1998] cf Fig 5d.

    It is not clear where these data are from. We assume
    the same recovery protocol as described
    in [Wang1993] (same lab).

    We assume the SD is 10% of the given time constants as
    no error is reported.

    The reactivation process was assessed by a paired-pulse
    protocol consisting of two identical pulses to +50 mV from
    a holding potential of -60 to +40 mV for 200 ms, at a
    variety of P1-P2 intervals.
    """
prepulses, tau_r, _ = data.Rec_Courtemanche()
sd_taur = 0.1*np.asarray(tau_r)
variances_taur = [sd_**2 for sd_ in sd_taur]
courtemanche_rec_dataset = np.asarray([prepulses, tau_r, variances_taur])

twaits = [2**i for i in range(1,8)]
tmp_protocols = []
for v in prepulses:
    tmp_protocols.append(
        recovery(twaits,v,50,50,20000,200,200)
    )
courtemanche_rec_protocol = tmp_protocols[0]
tsplit_rec = tmp_protocols[0].characteristic_time()
for p in tmp_protocols[1:]:
    for e in p.events():
        courtemanche_rec_protocol.add_step(e.level(), e.duration())

tsplits_rec = [t+200+200+20000 for t in twaits]
for i in range(len(tsplits_rec)-1):
    tsplits_rec[i+1] += tsplits_rec[i]

def courtemanche_rec_sum_stats(data):
    def single_exp(t, tau, A, A0):
        return A0-A*np.exp(-t/tau)
    output = []
    timename = 'engine.time'
    for i, d in enumerate(data.split_periodic(tsplit_rec, adjust=True, closed_intervals=False)):
        recov = []
        for t in tsplits_rec:
            d_, d = d.split(t)
            step1 = d_.trim(d_[timename][0]+20000,
                            d_[timename][0]+20000+200,
                            adjust=True)
            step2 = d_.trim_left(t-200, adjust=True)
            try:
                max1 = max(step1['ito.i_to'], key=abs)
                max2 = max(step2['ito.i_to'], key=abs)
                recov = recov + [max2/max1]
            except:
                recov = recov + [float('inf')]

        # now fit to exponential curve
        with warnings.catch_warnings():
            warnings.simplefilter('error', OptimizeWarning)
            warnings.simplefilter('error', RuntimeWarning)
            try:
                popt, _ = so.curve_fit(single_exp,
                                       twaits,
                                       recov,
                                       p0=[50,0.5,0.],
                                       bounds=(0.,
                                               [np.inf, 1.0, 1.0]),
                                       max_nfev=1000)
                fit = [single_exp(t,popt[0],popt[1],popt[2]) for t in twaits]
                # Calculate r2
                ss_res = np.sum((np.array(recov)-np.array(fit))**2)
                ss_tot = np.sum((np.array(recov)-np.mean(np.array(recov)))**2)
                r2 = 1 - (ss_res / ss_tot)

                tau = popt[0]

                if r2 > fit_threshold:
                    output = output+[tau]
                else:
                    raise RuntimeWarning('scipy.optimize.curve_fit found a poor fit')
            except:
                output = output+[float('inf')]
    return output

courtemanche_rec = Experiment(
    dataset=courtemanche_rec_dataset,
    protocol=courtemanche_rec_protocol,
    conditions=wang_conditions,
    sum_stats=courtemanche_rec_sum_stats,
    description=courtemanche_rec_desc,
    Q10=Q10_tau,
    Q10_factor=-1)


#
# Deactivation kinetics [Courtemanche1998]
#
courtemanche_deact_desc = """
    Deactivation kinetics for ito in human
    atrial myocytes [Courtemanche1998] cf Fig 5d.

    It is not clear where these data are from. We assume
    the same protocol as described
    in Fig 9A [Wang1993] (same lab).

    We assume the SD is 10% of the given time constants as
    no error is reported.

    Deactivation assessed by depolarising pulse to +50 mV from
    holding potential of -50 mV followed by hyperpolarising pulse
    to a range of test voltages.
    """
vsteps_taud, tau_d, _ = data.Deact_Kin_Courtemanche()
sd_taud = 0.1*np.asarray(tau_d)
variances_taud = [sd_**2 for sd_ in sd_taud]
courtemanche_deact_dataset = np.asarray([vsteps_taud, tau_d, variances_taud])

courtemanche_deact_protocol = myokit.Protocol()
for v in vsteps_taud:
    courtemanche_deact_protocol.add_step(-50, 20000)
    courtemanche_deact_protocol.add_step(50, 10)
    courtemanche_deact_protocol.add_step(v, 100)

def courtemanche_deact_sum_stats(data):
    def single_exp(t, tau, A, A0):
        return A0+A*np.exp(-t/tau)
    output = []
    for d in data.split_periodic(20110, adjust=True, closed_intervals=False):
        d = d.trim_left(20010, adjust=True)

        current = d['ito.i_to']
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
                norm = current[0]
                current = [c/norm for c in current]
                if len(time)<=1 or len(current)<=1:
                    raise Exception('Failed simulation')
                popt, _ = so.curve_fit(single_exp,
                                       time,
                                       current,
                                       p0=[10,0.5,0.],
                                       bounds=(0.,
                                               [np.inf, 1.0, 1.0]),
                                       max_nfev=1000)
                fit= [single_exp(t,popt[0],popt[1],popt[2]) for t in time]

                # Calculate r2
                ss_res = np.sum((np.array(current)-np.array(fit))**2)
                ss_tot = np.sum((np.array(current)-np.mean(np.array(current)))**2)
                r2 = 1 - (ss_res / ss_tot)

                taud = popt[0]

                if r2 > fit_threshold:
                    output = output+[taud]
                else:
                    raise RuntimeWarning('scipy.optimize.curve_fit found a poor fit')
            except:
                output = output+[float('inf')]
    return output

courtemanche_deact = Experiment(
    dataset=courtemanche_deact_dataset,
    protocol=courtemanche_deact_protocol,
    conditions=wang_conditions,
    sum_stats=courtemanche_deact_sum_stats,
    description=courtemanche_deact_desc,
    Q10=Q10_tau,
    Q10_factor=-1)
