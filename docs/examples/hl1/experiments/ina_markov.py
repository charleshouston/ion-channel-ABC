from ionchannelABC.protocol import availability_linear, recovery
import data.ina.data_ina as data
import numpy as np
import pandas as pd
import myokit

modelfile = 'models/ina_markov.mmt'

### Get observations
# IV curve
datasets = []
vsteps, peaks, sd = data.IV_Dias()
max_observed_peak = np.max(np.abs(peaks))
peaks = [p / max_observed_peak for p in peaks]
variances = [(sd / max_observed_peak)**2 for sd in sd]
datasets.append([vsteps, peaks, variances])

# Inactivation
vsteps_inact, inact, sd_inact = data.Inact_Nakajima()
variances_inact = [sd**2 for sd in sd_inact]
datasets.append([vsteps_inact, inact, variances_inact])

# Recovery
tsteps_rec, rec, _ = data.Recovery_Zhang()
datasets.append([tsteps_rec, rec, [0.,]*len(tsteps_rec)])

observations = pd.DataFrame(columns=['x', 'y', 'variance', 'exp_id'])
for id, data in enumerate(datasets):
    data = np.asarray(data).T.tolist()
    data = [d + [str(id),] for d in data]
    observations = observations.append(
        pd.DataFrame(data, columns=['x','y','variance','exp_id']),
        ignore_index=True
    )


### Setup protocols
protocols, conditions = [], []

# IV curve
protocols.append(
    myokit.pacing.steptrain_linear(-100, 50, 10, -80, 5000, 100)
)
dias_conditions = {'membrane.Na_o': 140e3,
                   'membrane.Na_i': 10e3}
conditions.append(dias_conditions)

# Inactivation
protocols.append(
    availability_linear(-130, -20, 10, -120, -20, 5000, 500, 0, 100)
)
nakajima_conditions = {'membrane.Na_o': 145e3,
                       'membrane.Na_i': 10e3}
conditions.append(nakajima_conditions)

# Recovery
protocols.append(
    recovery(tsteps_rec, -120, -30, -30, 3000, 20, 20)
)
zhang_conditions = {'membrane.Na_o': 136e3,
                    'membrane.Na_i': 10e3}
conditions.append(zhang_conditions)


### Create model and simulations
m = myokit.load_model(modelfile)
v = m.get('membrane.V')
v.demote()
v.set_rhs(0)
v.set_binding('pace')

simulations, times = [], []
for p, c in zip(protocols, conditions):
    s = myokit.Simulation(m, p)
    for ci, vi, in c.items():
        s.set_constant(ci, vi)
    simulations.append(s)
    times.append(p.characteristic_time())

def summary_statistics(data):
    """Converts raw simulation output to measures as observations"""
    # Check for error
    if data is None:
        return {}

    # Process sensible results
    ss = {}
    cnt = 0

    # I-V curve (normalised)
    d0 = data[0].split_periodic(5100, adjust=True)
    for d in d0:
        d = d.trim(5000, 5100, adjust=True)
        current = d['ina.i_Na']
        index = np.argmax(np.abs(current))
        ss[str(cnt)] = current[index] / max_observed_peak
        cnt += 1

    # Inactivation data
    d1 = data[1].split_periodic(5600, adjust=True)
    cnt_normalise = cnt
    for d in d1:
        d = d.trim(5500, 5600, adjust=True)
        current = d['ina.i_Na']
        ss[str(cnt)] = max(current, key=abs)
        cnt += 1
    for i in range(1, len(d1)):
        ss[str(cnt_normalise+i)] = ss[str(cnt_normalise+i)]/ss[str(cnt_normalise)]
    ss[str(cnt_normalise)] = 1.

    # Recovery data
    d2_ = data[2]
    d2 = []
    split_times = [3040+tw for tw in tsteps_rec]
    for i, time in enumerate(split_times[:-1]):
        split_times[i+1] += split_times[i]
    for i, time in enumerate(split_times):
        split_data = d2_.split(time)
        d2.append(
            split_data[0].trim(split_data[0]['environment.time'][0]+3000,
                               split_data[0]['environment.time'][0]+3040+tsteps_rec[i],
                               adjust=True)
        )
        d2_ = split_data[1]
    for d in d2:
        # Interested in two 20ms pulses
        pulse1 = d.trim(0, 20, adjust=True)['ina.i_Na']
        endtime = d['environment.time'][-1]
        pulse2 = d.trim(endtime-20, endtime, adjust=True)['ina.i_Na']

        max1 = np.max(np.abs(pulse1))
        max2 = np.max(np.abs(pulse2))

        ss[str(cnt)] = max2/max1
        cnt += 1

    return ss
