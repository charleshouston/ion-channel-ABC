from ionchannelABC.protocol import recovery
import data.ina.Nygren1998.data_ina as data
import numpy as np
import pandas as pd
import myokit
import matplotlib.pyplot as plt

modelfile = 'models/nygren_Na.mmt'

protocols, conditions = [], []

# IV curve 
protocols.append(
    myokit.pacing.steptrain_linear(-70.0001,75, 5, -75, 5000, 500) # very sensitive : be careful !
)


nygren_conditions = dict()
conditions.append(nygren_conditions)

# Inactivation curve protocol
protocols.append(
    myokit.pacing.steptrain_linear(-100, -20 +5, 5, -10, 1000, 1000, 500)
)
conditions.append(nygren_conditions)

# Activation curve protocol
protocols.append(
    myokit.pacing.steptrain_linear(-80.0001, 40 +5, 5, -80, 5000, 500)
)
conditions.append(nygren_conditions)

# Grab all observational data
datasets = []
vsteps, peaks, _ = data.IV_Nygren()
max_observed_peak = np.max(np.abs(peaks)) # for later normalising
peaks = [p / max_observed_peak for p in peaks]
variances = [0 for p in peaks]
datasets.append([vsteps, peaks, variances])

vsteps_inact, inact, _,_= data.Inact_Nygren()
variances_inact = [0 for v in vsteps_inact]
datasets.append([vsteps_inact, inact, variances_inact])

vsteps_act, act, _,_ = data.Act_Nygren()
variances_act = [0 for v in vsteps_act]
datasets.append([vsteps_act, act, variances_act])


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
        print(ci,vi)
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
    d0 = data[0].split_periodic(5500)
    for d in d0:
        d = d.trim_left(5000, adjust=True)
              
        current = d['sodium_current.i_Na']
        current = current[:-1]
        
        index = np.argmax(np.abs(current))
        ss[str(cnt)] = current[index] / max_observed_peak
        cnt += 1

        
    # Inactivation and inactivation data
    d1 = data[1].split_periodic(2500)
    for d in d1:
        d.trim_left(2000, adjust=True)
        inact_gate = d['sodium_current.h_infinity']
        index = np.argmax(np.abs(inact_gate))
        ss[str(cnt)] = np.abs(inact_gate[index])
        cnt += 1
     

    d2 = data[2].split_periodic(5500)
    for d in d2:
        d = d.trim_left(5000, adjust=True)
        act_gate = d['sodium_current.m_infinity_cube']
        index = np.argmax(np.abs(act_gate))
        ss[str(cnt)] = np.abs(act_gate[index])
        cnt += 1


    return ss

#    d0 = data[0].split_periodic(5500)
#    ss["current"] = {}
#    for d in d0:
#        d = d.trim_left(5000, adjust=True)
#              
#        current = d['sodium_current.i_Na']
#        current = current[:-1]
#        
#        index = np.argmax(np.abs(current))
#        ss["current"][str(cnt)] = current[index] / max_observed_peak
#        cnt += 1
#        
#    cnt = 0
#    # Inactivation and inactivation data
#    d1 = data[1].split_periodic(2500)
#    ss["inact"] = {}
#    for d in d1:
#        d.trim_left(2000, adjust=True)
#        inact_gate = d['sodium_current.h_infinity']
#        index = np.argmax(np.abs(inact_gate))
#        ss["inact"][str(cnt)] = np.abs(inact_gate[index])
#        cnt += 1
#     
#        
#    cnt = 0
#    ss["act"] = {}
#    d2 = data[2].split_periodic(5500)
#    for d in d2:
#        d = d.trim_left(5000, adjust=True)
#        act_gate = d['sodium_current.m_infinity_cube']
#        index = np.argmax(np.abs(act_gate))
#        ss["act"][str(cnt)] = np.abs(act_gate[index])
#        cnt += 1

