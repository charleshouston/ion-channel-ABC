from ionchannelABC.protocol import recovery
import data.icat.data_icat as data
import numpy as np
import pandas as pd
import myokit


modelfile = 'models/icat_markov.mmt'

protocols, conditions = [], []

# IV curve and Activation curve protocol
protocols.append(
    myokit.pacing.steptrain_linear(-75, 40, 5, -75, 5000, 300)
)
nguyen_conditions = {'membrane.Ca_o': 5000,
                     'membrane.Ca_subSL': 318, # determined from RP=+35mV
                     'membrane.T': 295}
conditions.append(nguyen_conditions)

# Inactivation curve protocol
protocols.append(
    myokit.pacing.steptrain_linear(-80, -20, 5, -10, 1000, 1000, 500)
)
conditions.append(nguyen_conditions)

# Recovery protocol
twait, _, _ = data.Rec_Deng()
protocols.append(
    recovery(twait, -80, -20, -20, 5000, 300, 300)
)
deng_conditions = {'membrane.Ca_o': 5000,
                   'membrane.Ca_subSL': 318,
                   'membrane.T': 298}
conditions.append(deng_conditions)

# Current trace
#protocols.append(
#    myokit.pacing.steptrain([-20], -80, 5000, 300)
#)
#conditions.append(deng_conditions)

# Steps from -40, icat current by definition should not activate
#protocols.append(
#    myokit.pacing.steptrain_linear(-80, 50, 10, -40, 5000, 300)
#)
#conditions.append(deng_conditions)

# Grab all observational data
datasets = []
vsteps, peaks, variances = data.IV_Nguyen()
max_observed_peak = np.max(np.abs(peaks)) # for later normalising
peaks = [p / max_observed_peak for p in peaks]
variances = [v / max_observed_peak for v in variances]
datasets.append([vsteps, peaks, variances])
datasets.append(data.Act_Nguyen())
datasets.append(data.Inact_Nguyen())
datasets.append(data.Rec_Deng())
#trace_time, trace_curr, _ = data.CurrTrace_Deng()
#max_observed_curr_trace = np.max(np.abs(trace_curr))
#trace_curr = [c / max_observed_curr_trace for c in trace_curr]
#datasets.append([trace_time, trace_curr, [0.,]*len(trace_time)])
#vsteps, _, _ = data.IV_Deng()
#datasets.append([vsteps, [0.,]*len(vsteps), [0.,]*len(vsteps)])

observations = pd.DataFrame(columns=['x','y','variance','exp_id'])
for id, data in enumerate(datasets):
    data = np.asarray(data).T.tolist()
    data = [d + [str(id),] for d in data]
    observations = observations.append(
        pd.DataFrame(data, columns=['x','y','variance','exp_id']),
        ignore_index=True
    )

# Create model and simulations
m = myokit.load_model(modelfile)
v = m.get('membrane.V')
v.demote()
v.set_rhs(0)
v.set_binding('pace')

simulations, times = [], []
for p, c in zip(protocols, conditions):
    s = myokit.Simulation(m, p)
    for ci, vi in c.items():
        s.set_constant(ci, vi)
    simulations.append(s)
    times.append(p.characteristic_time())

def summary_statistics(data):
    """Converts raw simulation output to sensible results"""
    
    # Check for error
    if data is None:
        return {}
    
    # Process sensible results
    ss = {}
    cnt = 0
    
    # I-V curve (normalised)
    d0 = data[0].split_periodic(5300)
    for d in d0:
        d = d.trim_left(5000, adjust=True)
        current = d['icat.i_CaT']
        index = np.argmax(np.abs(current))
        ss[str(cnt)] = current[index] / max_observed_peak
        cnt += 1
    
    # Activation data
    # Same data as IV less final 6 steps
    for d in d0[:-6]:
        gate = d['icat.g']
        index = np.argmax(np.abs(gate))
        ss[str(cnt)] = np.abs(gate[index])
        cnt += 1

    # Inactivation data
    d1 = data[1].split_periodic(2500)
    for d in d1:
        d.trim_left(2000, adjust=True)
        gate = d['icat.g']
        index = np.argmax(np.abs(gate))
        ss[str(cnt)] = np.abs(gate[index])
        cnt += 1

    # Recovery data
    d2_ = data[2]
    d2 = []
    split_times = [5600+tw for tw in twait]
    for i, time in enumerate(split_times[:-1]):
        split_times[i+1] += split_times[i]
    for time in split_times:
        split_data = d2_.split(time)
        d2.append(split_data[0].trim_left(5000, adjust=True))
        d2_ = split_data[1]
    for d in d2:
        # Interested in two 300ms pulses
        pulse1 = d.trim_left(300, adjust=True)['icat.i_CaT']
        endtime = d['environment.time'][-1]
        pulse2 = d.trim(endtime-300, endtime, adjust=True)['icat.i_CaT']

        max1 = np.max(pulse1, key=abs)
        max2 = np.max(pulse2, key=abs)

        ss[str(cnt)] = max2/max1
        cnt += 1

    # Current trace
    #def interpolate_align(data, time):
    #    simtime = data['environment.time']
    #    simtime_min = min(simtime)
    #    simtime = [t - simtime_min for t in simtime]
    #    curr = data['icat.i_CaT']
    #    #max_curr = abs(max(curr, key=abs))
    #    curr = [c / max_observed_curr_trace for c in curr]
    #    return np.interp(time, simtime, curr)
    #for curr in interpolate_align(data[3], trace_time):
    #    ss[str(cnt)] = curr
    #    cnt += 1

    ## I-V curve
    #d4 = data[4].split_periodic(5300)
    #for d in d4:
    #    d = d.trim_left(5000, adjust=True)
    #    current = d['icat.i_CaT']
    #    index = np.argmax(np.abs(current))
    #    ss[str(cnt)] = current[index]
    #    cnt += 1

    return ss