from experiment import (Experiment, ExperimentData,
                        ExperimentStimProtocol)
from channel import Channel
import numpy as np


modelfile = 'models/HL1.mmt'

params = {'icat.g_CaT': (0, 2),
          'icat.E_CaT': (0, 50),
          'icat.p1': (0, 100),
          'icat.p2': (1, 10),
          'icat.p3': (0, 1),
          'icat.p4': (0, 10),
          'icat.p5': (0, 0.1),
          'icat.p6': (0, 200),
          'icat.q1': (0, 100),
          'icat.q2': (1, 10),
          'icat.q3': (0, 10),
          'icat.q4': (0, 100),
          'icat.q5': (0, 0.1),
          'icat.q6': (0, 100),
          'ical.g_CaL': (0, 0.001),
          'ical.p1': (-50, 50),
          'ical.p2': (0, 50),
          'ical.p3': (-100, 50),
          'ical.p4': (-50, 50),
          'ical.p5': (-50, 50),
          'ical.p6': (-50, 50),
          'ical.p7': (0, 200),
          'ical.p8': (0, 200),
          'ical.q1': (0, 100),
          'ical.q2': (0, 50),
          'ical.q3': (0, 10000),
          'ical.q4': (0, 100),
          'ical.q5': (0, 1000),
          'ical.q6': (0, 1000),
          'ical.q7': (0, 100),
          'ical.q8': (0, 100),
          'ical.q9': (-500, 500),
          'ina.g_Na': (0, 100),
          'ina.p1': (0, 100),
          'ina.p2': (-10, 0),
          'ina.p3': (0, 1),
          'ina.p4': (0, 100),
          'ina.p5': (-1, 0),
          'ina.p6': (0, 1),
          'ina.p7': (0, 100),
          'ina.q1': (0, 100),
          'ina.q2': (0, 10),
          'iha.k_yss1': (0, 100),
          'iha.k_yss2': (1, 10),
          'iha.k_ytau1': (0, 10),
          'iha.k_ytau2': (0, 1.0),
          'iha.k_ytau3': (0, 100),
          'iha.k_ytau4': (1, 100),
          'iha.k_ytau5': (0, 1.0),
          'iha.k_ytau6': (0, 100),
          'iha.k_ytau7': (0, 100),
          'iha.k_i_haNa': (0, 1.0),
          'iha.g_ha': (0, 0.1),
          'ikr.g_Kr': (0, 1),
          'ikr.p1': (0, 0.1),
          'ikr.p2': (0, 0.1),
          'ikr.p3': (0, 0.1),
          'ikr.p4': (0, 0.1),
          'ikr.p5': (-0.1, 0.1),
          'ikr.p6': (0, 0.1),
          'ikr.q1': (0, 0.1),
          'ikr.q2': (-0.1, 0),
          'ikr.q3': (0, 0.1),
          'ikr.q4': (-0.1, 0),
          'ikr.q5': (0, 0.01),
          'ikr.q6': (-0.1, 0),
          'ikr.k_f': (0, 0.5),
          'ikr.k_b': (0, 0.1),
          'ikur.g_Kur': (0, 1),
          'ikur.k_ass1': (0, 100),
          'ikur.k_ass2': (0, 100),
          'ikur.k_atau1': (0, 100),
          'ikur.k_atau2': (0, 100),
          'ikur.k_atau3': (0, 10),
          'ikur.k_iss1': (0, 100),
          'ikur.k_iss2': (0, 100),
          'ikur.k_itau1': (0, 10),
          'ikur.k_itau2': (0, 100),
          'ito.g_to': (0, 1),
          'ito.k_xss1': (0, 10),
          'ito.k_xss2': (0, 100),
          'ito.k_xtau1': (0, 10),
          'ito.k_xtau2': (0, 10),
          'ito.k_xtau3': (0, 100),
          'ito.k_yss1': (0, 100),
          'ito.k_yss2': (0, 100),
          'ito.k_ytau1': (0, 100),
          'ito.k_ytau2': (0, 100),
          'ito.k_ytau3': (0, 100),
          'ito.k_ytau4': (0, 100),
          'ik1.g_K1': (0, 0.2),
          'ik1.k_1': (-500, 500),
          'ik1.k_2': (0, 50),
          'ik1.k_3': (0, 1),
          'ik1.k_4': (0, 0.1),
          'icab.g_Cab': (0, 0.001),
          'incx.k_NCX': (0, 2.6e16),
          'inab.g_Nab': (0, 0.0026),
          'inak.i_NaK_max': (0, 2.7)}

# TODO: this is a bit of a hack of the Channel class...
hl1 = Channel('hl1', modelfile, params, vvar='membrane.i_stim')

stim_times = [500000]
stim_levels = [0.0]
time = np.linspace(0, stim_times[0], 1000)
# See resting state behaviour....
def interpolate_align(data_list):
    import numpy as np
    data = data_list[0]
    if len(data_list) > 1:
        for log in data_list[1:]:
            data = data.extend(log)
    simtime = data['environment.time']
    simtime_min = min(simtime)
    simtime = [t - simtime_min for t in simtime]
    data_output = dict()
    data_output['environment.time'] = simtime
    for var in data:
        data_output[var] = np.interp(time, simtime, data[var])
    return data_output
resting_prot = ExperimentStimProtocol(stim_times, stim_levels,
                                      ind_var=time,
                                      measure_index=range(len(stim_times)),
                                      measure_fn=interpolate_align)
dias_conditions = dict(T=305,
                       Ca_o=1800,
                       Na_o=1.4e5,
                       K_o=4e3)
resting_exp = Experiment(resting_prot, None, dias_conditions)
hl1.add_experiment(resting_exp)

n_pulses = 100
stim_times = [10000] + [2, 998]*n_pulses + [2, 998]
stim_levels = [0] + [40, 0]*n_pulses + [40, 0]
pace_time = np.linspace(0, 1998, 1998)
def interpolate_align_pace(data_list):
    import numpy as np
    data = data_list[0]
    if len(data_list) > 1:
        for log in data_list[1:]:
            data = data.extend(log)
    simtime = data['environment.time']
    simtime_min = min(simtime)
    simtime = [t - simtime_min for t in simtime]
    data_output = dict()
    data_output['environment.time'] = simtime
    for var in data:
        data_output[var] = np.interp(pace_time, simtime, data[var])
    return data_output

pulse_train_prot = ExperimentStimProtocol(stim_times, stim_levels,
                                          ind_var=pace_time,
                                          measure_index=[len(stim_times)-3,
                                                         len(stim_times)-2,
                                                         len(stim_times)-1],
                                          measure_fn=interpolate_align_pace)
pulse_train_exp = Experiment(pulse_train_prot, None, dias_conditions)
hl1.add_experiment(pulse_train_exp)

# Increasing pulse train
stims = range(0, 45, 5)
stim_times = [100000] + [2, 4998]*len(stims)
stim_levels = [0]
for lev in stims:
    stim_levels.append(lev)
    stim_levels.append(0)
pace_time = np.linspace(0, sum(stim_times[1:]), 2000)

increasing_pulse_train_prot = ExperimentStimProtocol(stim_times, stim_levels,
                                                     ind_var=pace_time,
                                                     measure_index = range(1, len(stim_times)),
                                                     measure_fn=interpolate_align_pace)
increasing_pulse_train_exp = Experiment(increasing_pulse_train_prot, None, dias_conditions)
hl1.add_experiment(increasing_pulse_train_exp)
