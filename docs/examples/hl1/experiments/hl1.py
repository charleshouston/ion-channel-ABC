from ionchannelABC import (Experiment,
                           ExperimentData,
                           ExperimentStimProtocol,
                           IonChannelModel)
import numpy as np


modelfile = 'models/HL1.mmt'

#params = {'ik1.g_K1': (0, 0.1),
#          'incx.k_NCX': (0, 10),
#          'icab.g_Cab': (0, 0.001),
#          'inab.g_Nab': (0, 0.01),
#          'inak.i_NaK_max': (0, 10)}

hl1 = IonChannelModel('hl1',
                      modelfile,
                      vvar='membrane.i_stim')

# Run cell with no stimulus for 1000 simulated seconds to reach steady state
# Then stimulate in current clamp mode using 2 ms pulses from 0 to 40 pA
# increasing in increments of 5 pA;  5, 10, 15, 20, 25, 30, 35, 40

pulses = range(5, 45, 5)
stim_times = [1000000] + [2, 4998]*len(pulses)
stim_levels = [0]
for level in pulses:
    stim_levels.append(level)
    stim_levels.append(0)
pace_time = np.linspace(0, sum(stim_times[1:]), sum(stim_times[1:]))

def ap_measurements(data_list):
    import numpy as np
    data = data_list[0]
    pulses = [0]
    counter = 1
    if len(data_list) > 1:
        for log in data_list[1:]:
            if counter % 2 == 0:
                # Log every other index as start of pulse
                pulses.append(len(data['environment.time']))
            counter = counter+1
            data = data.extend(log)
    pulses.append(len(data['environment.time']))

    # Set time to start at 0 after steady state is reached
    mintime = min(data['environment.time'])
    data['environment.time'] = [t - mintime for t in data['environment.time']]

    # Detect triggered action potentials
    V = data['membrane.V']
    triggers = [np.max(V[pulses[i]:pulses[i+1]])>0
                for i in range(len(pulses)-1)]
    if sum(triggers) == 0:
        # No action potentials
        return [float('inf')]*6
    
    Vrp = []
    Cai = []
    Nai = []
    Ki = []
    APA = []
    APD90 = []
    ca_amplitude = []
    t_ca_rise = []
    t_ca50 = []
    t_ca90 = []
    for i, fire in enumerate(triggers):
        tsubset = data['environment.time'][pulses[i]:pulses[i+1]]
        Vsubset = data['membrane.V'][pulses[i]:pulses[i+1]]
        Casubset = data['ca_conc.Ca_i'][pulses[i]:pulses[i+1]]
        Ksubset = data['k_conc.K_i'][pulses[i]:pulses[i+1]]
        Nasubset = data['na_conc.Na_i'][pulses[i]:pulses[i+1]]

        # Take last 500ms as resting V
        t_index = np.where(np.array(tsubset) > tsubset[0]+100)[0][0]
        Vrp.append(np.mean(Vsubset[t_index:]))
        Cai.append(np.mean(Casubset[t_index:]))
        Nai.append(np.mean(Nasubset[t_index:]))
        Ki.append(np.mean(Ksubset[t_index:]))

        if not fire:
            # if no action potential don't attempt to take measurements
            continue 

        # Record AP amplitude
        APAi = max(Vsubset) - Vrp[-1]
        APA.append(APAi)

        # Find APD90
        V_diff = np.diff(Vsubset)
        peak_index = np.nonzero(np.array(Vsubset) ==
                                max(Vsubset))[0][0]
        APD90i = 0.0
        for j in range(peak_index, pulses[i+1]):
            if Vsubset[j] <= (Vrp[-1] + 0.1*APAi):
                APD90i = tsubset[j] - tsubset[0]
                break
        APD90.append(APD90i)

        # CaT measurements
        ca_peak_index = np.nonzero(np.array(Casubset) ==
                                   max(Casubset))[0][0]
        ca_rp = Casubset[0]
        ca_amp = max(Casubset) - ca_rp
        ca_amplitude.append(ca_amp)
        t_ca_rise.append(tsubset[ca_peak_index] - tsubset[0])
        t_ca50i = 0.0
        t_ca90i = 0.0
        for j in range(ca_peak_index, pulses[i+1]):
            if (t_ca50i == 0.0 and
                Casubset[j] <= ca_rp + 0.5*ca_amp):
                t_ca50i = tsubset[j] - tsubset[0]
            if Casubset[j] <= ca_rp + 0.1*ca_amp:
                t_ca90i = tsubset[j] - tsubset[0]
                break

        if t_ca50i == 0.0:
            t_ca50.append(float('inf'))
        else:
            t_ca50.append(t_ca50i)
        if t_ca90i == 0.0:
            t_ca90.append(float('inf'))
        else:
            t_ca90.append(t_ca90i)
    data_output = [np.mean(Vrp),
#                   np.mean(Cai),
#                   np.mean(Ki),
#                   np.mean(Nai),
                   np.mean(APA),
                   np.mean(APD90),
#                   np.mean(ca_amplitude),
                   np.mean(t_ca_rise),
                   np.mean(t_ca50),
                   np.mean(t_ca90)]
    return data_output

def unwrap(data, ind_var):
    return data[0], False

dias_conditions = dict(T=305,
                       Ca_o=1800,
                       Na_o=1.4e5,
                       K_o=4e3)
pulse_train_prot = ExperimentStimProtocol(stim_times, stim_levels,
        ind_var=['vrp', #'ca_i', #'k_i', 'na_i', 
                 'apa',
                 'apd90',
                 #'ca_amp', 
                 't_ca_rise','t_ca50','t_ca90'],
        measure_index=range(1, len(stim_times)),
        measure_fn=ap_measurements,
        post_fn=unwrap)

pulse_train_data = ExperimentData(x=['vrp', #'ca_i', #'k_i', 'na_i', 
                                     'apa',
                                     'apd90',
                                     #'ca_amp',
                                     't_ca_rise','t_ca50','t_ca90'],
                                  y=[-67.0, #0.1, #1.5e5, 2.0e4, 
                                      105,
                                     42, 
                                     #0.7,
                                     52, 157, 397],
                                  errs=[-69, #0.05, #0.75e5, 1.0e4, 
                                       103,
                                        33,
                                        #0.35,
                                        50, 151, 383])
pulse_train_exp = Experiment(pulse_train_prot, pulse_train_data, dias_conditions)

hl1.add_experiments([pulse_train_exp])
