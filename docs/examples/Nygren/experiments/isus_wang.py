"""
Experiments from [Wang1993]
Charles Houston 2019-10-18
"""
from ionchannelABC.experiment import Experiment
from ionchannelABC.protocol import availability
import data.isus.Wang1993.data_Wang1993 as data
import numpy as np
import myokit

import warnings
from scipy.optimize import OptimizeWarning
import scipy.optimize as so


Q10_tau = 2.2 # [Wang1993]


fit_threshold = 0.9


#
# Activation [Wang1993]
#
wang_act_desc = """
    Steady-state activation curve for isus in human
    atrial myocytes [Wang1993] cf Fig 8e.

    Isus was obtained by depolarizing the membrane for
    100 ms to test potentials ranging from -40 to +80 mV
    from a holding potential of -50 mV, then repolarizing
    to -10 mV at a frequency of 0.1 Hz. A 1-second
    pre-pulse was introduced 20 ms before each test pulse to
    inactivate the 4AP-sensitive transient outward current.
    """
vsteps_act, act, sd_act = data.Act_Wang()
variances_act = [sd_**2 for sd_ in sd_act]
wang_act_dataset = np.asarray([vsteps_act, act, variances_act])

vsteps_taua, taua, sd_taua = data.Act_Kin_Wang()
variances_taua = [sd_**2 for sd_ in sd_taua]
wang_act_kin_dataset = np.asarray([vsteps_taua, taua, variances_taua])

wang_act_protocol = myokit.Protocol()
for v in vsteps_act:
    wang_act_protocol.add_step(-50, 10000) # holding potential
    wang_act_protocol.add_step(50, 1000)   # 1 s pre-pulse
    wang_act_protocol.add_step(-50, 20)    # back to holding for 20 ms
    wang_act_protocol.add_step(v, 100)     # test pulse
    wang_act_protocol.add_step(-10, 100)   # measure pulse

wang_conditions = {'phys.T': 295.15,  # K
                   'k_conc.K_i': 130, # mM
                   'k_conc.K_o': 5.4
                  }

def wang_act_and_kin_sum_stats(data, ss=True, tau=True):
    def single_exp(t, tau, A, A0):
        return A0 - A*np.exp(-t/tau)

    output_ss = []
    output_tau = []

    for d in data.split_periodic(11220, adjust=True, closed_intervals=False):
        if ss:
            d_ss = d.trim_left(11220-100, adjust=True)
            act_gate = d_ss['isus.g']
            output_ss = output_ss + [max(act_gate, key=abs)]

        if tau:
            d_tau = d.trim(11020, 11120, adjust=True)

            # only -20 mV to +50 mV in tau_a dataset
            if d_tau['membrane.V'][0] < -20:
                continue

            current = d_tau['isus.i_sus']
            time = d_tau['engine.time']
            index = np.argmax(np.abs(current))

            with warnings.catch_warnings():
                warnings.simplefilter('error', OptimizeWarning)
                warnings.simplefilter('error', RuntimeWarning)
                try:
                    current = current[:index]
                    time = time[:index]
                    norm = current[-1]
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
                    fit = [single_exp(t,popt[0],popt[1],popt[2]) for t in time]
                    # Calculate r2
                    ss_res = np.sum((np.array(current)-np.array(fit))**2)
                    ss_tot = np.sum((np.array(current)-np.mean(np.array(current)))**2)
                    r2 = 1 - (ss_res / ss_tot)
                    taua = popt[0]

                    if r2 > fit_threshold:
                        output_tau = output_tau + [taua]
                    else:
                        raise RuntimeWarning('scipy.optimize.curve_fit found a poor fit')
                except:
                    output_tau = output_tau + [float('inf')]

    if ss:
        norm = max(output_ss)
        try:
            for i in range(len(output_ss)):
                output_ss[i] /= norm
        except:
            output_ss = [float('inf'),]*len(output_ss)

    return output_ss+output_tau

def wang_act_sum_stats(data):
    return wang_act_and_kin_sum_stats(data, ss=True, tau=False)

def wang_act_kin_sum_stats(data):
    return wang_act_and_kin_sum_stats(data, ss=False, tau=True)

wang_act_and_kin = Experiment(
    dataset=[wang_act_dataset,
             wang_act_kin_dataset],
    protocol=wang_act_protocol,
    conditions=wang_conditions,
    sum_stats=wang_act_and_kin_sum_stats,
    description=wang_act_desc,
    Q10=Q10_tau,
    Q10_factor=[0,-1])

wang_act = Experiment(
    dataset=wang_act_dataset,
    protocol=wang_act_protocol,
    conditions=wang_conditions,
    sum_stats=wang_act_sum_stats,
    description=wang_act_desc,
    Q10=None,
    Q10_factor=0)

wang_act_kin = Experiment(
    dataset=wang_act_kin_dataset,
    protocol=wang_act_protocol,
    conditions=wang_conditions,
    sum_stats=wang_act_kin_sum_stats,
    description=wang_act_desc,
    Q10=Q10_tau,
    Q10_factor=-1)

#
# Inactivation [Wang1993]
#
wang_inact_desc = """
    Steady-state inactivation curve for isus in human
    atrial myocytes [Wang1993] cf Fig 7a.

    Isus is measured from the steady-state current at the end
    of a 2000 ms test pulse at +40 mV preceded by a 1000 ms
    conditioning pulse at various potentials delivered from a
    holding potential of -60 mV. Current amplitude is normalised
    to the maximum value obtained with a conditioning voltage of
    -90 mV.
    """
vsteps_inact, inact, sd_inact = data.Inact_Wang()
variances_inact = [sd_**2 for sd_ in sd_inact]
wang_inact_dataset = np.asarray([vsteps_inact, inact, variances_inact])

wang_inact_protocol = availability(
    vsteps_inact, -60, 40, 20000, 1000, 0, 2000)

def wang_inact_sum_stats(data):
    output = []
    for d in data.split_periodic(23000, adjust=True, closed_intervals=False):
        d = d.trim_left(21000, adjust=True)
        inact_gate = d['isus.g']
        output = output + [inact_gate[-1]]
    norm = output[0]
    try:
        for i in range(len(output)):
            output[i] /= norm
    except:
        output = [float('inf'),]*len(output)
    return output

wang_inact = Experiment(
    dataset=wang_inact_dataset,
    protocol=wang_inact_protocol,
    conditions=wang_conditions,
    sum_stats=wang_inact_sum_stats,
    description=wang_inact_desc,
    Q10=None,
    Q10_factor=0.)
