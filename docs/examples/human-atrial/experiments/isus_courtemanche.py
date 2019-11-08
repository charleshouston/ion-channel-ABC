"""
Experiments from [Courtemanche1998]
Charles Houston 2019-10-21
"""
from ionchannelABC.experiment import Experiment
from ionchannelABC.protocol import recovery
import data.isus.Courtemanche1998.data_Courtemanche1998 as data
import numpy as np
import myokit

import warnings
from scipy.optimize import OptimizeWarning
import scipy.optimize as so


Q10_tau = 2.2 # [Wang1993]


fit_threshold = 0.9


#
# Inactivation kinetics [Courtemanche1998]
#
courtemanche_inact_kin_desc = """
    Inactivation kinetics for isus in human
    atrial myocytes [Courtemanche1998] cf Fig 5b.

    It is not clear where these data are from. We assume
    the same inactivation protocol as described
    in [Wang1993] (same lab).

    We assume the SD is 10% of the given time constants as
    no error is reported.

    2 s depolarising pulses following conditioning pulses.
    """
vsteps_taui, tau_i, _ = data.Inact_Kin_Courtemanche()
sd_taui = np.asarray(tau_i)*0.1
variances_taui = [sd_**2 for sd_ in sd_taui]
courtemanche_inact_kin_dataset = np.asarray([vsteps_taui, tau_i, variances_taui])

courtemanche_kin_protocol = myokit.pacing.steptrain(
    vsteps_taui, -50, 20000, 2000)

wang_conditions = {'phys.T': 295.15,  # K
                   'k_conc.K_i': 130, # mM
                   'k_conc.K_o': 5.4
                  }

def courtemanche_inact_kin_sum_stats(data):
    def single_exp(t, tau, A, A0):
        return A0 + A*np.exp(-t/tau)
    output = []

    for d in data.split_periodic(22000, adjust=True, closed_intervals=False):
        d = d.trim_left(20000, adjust=True)

        current = d['isus.i_sus']
        time = d['engine.time']
        index = np.argmax(np.abs(current))

        # Set time zero to peak current
        current_inact = current[index:]
        time_inact = time[index:]
        t0 = time_inact[0]
        time_inact = [t-t0 for t in time_inact]

        with warnings.catch_warnings():
            warnings.simplefilter('error', OptimizeWarning)
            warnings.simplefilter('error', RuntimeWarning)
            try:
                norm = current_inact[0]
                current_inact = [c/norm for c in current_inact]
                if len(time_inact)<=1 or len(current_inact)<=1:
                    raise Exception('Failed simulation')
                popt, _ = so.curve_fit(single_exp,
                                       time_inact,
                                       current_inact,
                                       p0=[100,0.5,0.],
                                       bounds=(0.,
                                               [np.inf, 1.0, 1.0]),
                                       max_nfev=1000)
                fit_inact = [single_exp(t,popt[0],popt[1],popt[2]) for t in time_inact]

                # Calculate r2
                ss_res = np.sum((np.array(current_inact)-np.array(fit_inact))**2)
                ss_tot = np.sum((np.array(current_inact)-np.mean(np.array(current_inact)))**2)
                r2 = 1 - (ss_res / ss_tot)

                taui = popt[0]

                if r2 > fit_threshold:
                    output = output+[taui]
                else:
                    raise RuntimeWarning('scipy.optimize.curve_fit found a poor fit')
            except:
                output = output+[float('inf')]
    return output

courtemanche_inact_kin = Experiment(
    dataset=courtemanche_inact_kin_dataset,
    protocol=courtemanche_kin_protocol,
    conditions=wang_conditions,
    sum_stats=courtemanche_inact_kin_sum_stats,
    description=courtemanche_inact_kin_desc,
    Q10=Q10_tau,
    Q10_factor=-1)


#
# Deactivation kinetics [Courtemanche1998]
#
courtemanche_deact_desc = """
    Deactivation kinetics for isus in human
    atrial myocytes [Courtemanche1998] cf Fig 5b.

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

        current = d['isus.i_sus']
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
