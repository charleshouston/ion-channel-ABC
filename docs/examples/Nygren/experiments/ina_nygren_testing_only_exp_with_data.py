from ionchannelABC.protocol import recovery
import data.ina.Sakakibara1992.data_ina as data
import numpy as np
import pandas as pd
import myokit
import matplotlib.pyplot as plt
import warnings
from scipy.optimize import OptimizeWarning
import scipy.optimize as so


modelfile = 'models/nygren_Na.mmt'

protocols, conditions = [], []

sakakibara_conditions = {'cleft_space_ion_concentrations.Na_c': 5,
                         'intracellular_ion_concentrations.Na_i': 5,
                         'membrane.T': 290.15}

# Inactivation curve protocol
protocols.append(
    myokit.pacing.steptrain_linear(-140, -40 +10, 10, -140, 10000, 1000,2)
)
conditions.append(sakakibara_conditions)


# Activation curve protocol
protocols.append(
    myokit.pacing.steptrain_linear(-90.0001, 20 +10, 10, -140, 10000, 1000,2)
)
conditions.append(sakakibara_conditions)

# Inactivation kinetics
t_hold = 1000
t_act = 100
protocols.append(
    myokit.pacing.steptrain_linear(-50, -10, 10, -140, t_hold, t_act)
)
conditions.append(sakakibara_conditions)

# IV curve 
protocols.append(
        myokit.pacing.steptrain_linear(-100.0001,30, 10, -140, 10000, 1000) 
)
conditions.append(sakakibara_conditions)




# Grab all observational data
datasets = []
#vsteps, peaks, _ = data.IV_Nygren()
#max_observed_peak = np.max(np.abs(peaks)) # for later normalising
#peaks = [p / max_observed_peak for p in peaks]
#variances = [0 for p in peaks]
#datasets.append([vsteps, peaks, variances])

vsteps_inact, inact, sd_inact= data.Inact_Sakakibara()
variances_inact = [sd**2 for sd in sd_inact]
datasets.append([vsteps_inact, inact, variances_inact])

vsteps_act, act, sd_act = data.Act_Sakakibara()
variances_act = [sd**2 for sd in sd_act]
datasets.append([vsteps_act, act, variances_act])

vsteps_tf, tf, sd_tf = data.TauF_Inactivation_Sakakibara()
max_tf = np.max(np.abs(tf)) # normalising
tf = [tf_ / max_tf for tf_ in tf]
variances_tf = [(sd_/max_tf)**2 for sd_ in sd_tf]
datasets.append([vsteps_tf, tf, variances_tf])

vsteps, peaks, _ = data.IV_Sakakibara()
max_observed_peak = np.max(np.abs(peaks)) # for later normalising
peaks = [p / max_observed_peak for p in peaks]
variances = [0 for p in peaks]
datasets.append([vsteps, peaks, variances])


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

def simple_exp(t, tauf):
    return np.exp(-t/tauf)

def summary_statistics(data):
    """Converts raw simulation output to sensible results"""
    
    # Check for error
    if data is None:
        return {}
    
    # Process sensible results
    ss = {}
    cnt = 0
    
    # I-V curve (normalised)
#    d0 = data[0].split_periodic(5500)
#    for d in d0:
#        d = d.trim_left(5000, adjust=True)
#              
#        current = d['sodium_current.i_Na']
#        current = current[:-1]
#        
#        index = np.argmax(np.abs(current))
#        ss[str(cnt)] = current[index] / max_observed_peak
#        cnt += 1

        
    # Inactivation data
    d0 = data[0].split_periodic(11002)
    for d in d0:
        d.trim_left(10000, adjust=True)
        inact_gate = d['sodium_current.h_infinity']
        index = np.argmin(np.abs(inact_gate))
        ss[str(cnt)] = np.abs(inact_gate[index])
        cnt += 1
     
    # Activation data
    d1 = data[1].split_periodic(11002, adjust = True)
    for d in d1:
        d = d.trim_left(10000, adjust= True )
        act_gate = d['sodium_current.m_infinity_cube']
        index = np.argmax(np.abs(act_gate))
        ss[str(cnt)] = np.abs(act_gate[index])
        cnt += 1
        
    # Inactivation kinetics (tf)
    d2 = data[2].split_periodic(t_hold + t_act, adjust = True)
    for d in d2:

        d = d.trim_left(t_hold, adjust = True)

        current = d['sodium_current.i_Na']
        current = current[:-1]
        time = d['environment.time']
        time = time[:-1]
        index = np.argmax(np.abs(current))


        # Set time zero to peak current
        current = current[index:]
        time = time[index:]
        t0 = time[0]
        time = [t-t0 for t in time]

        # Remove constant
        c0 = current[-1]
        current = [(c_-c0) for c_ in current]
        
    
        
        with warnings.catch_warnings():
            warnings.simplefilter('error',OptimizeWarning)
            warnings.simplefilter('error',RuntimeWarning)
            try:
                imax = max(current, key=abs)
                current = [c_/imax for c_ in current]
                if len(time)<=1 or len(current)<=1:
                    raise Exception('Failed simulation')
                popt, _ = so.curve_fit(simple_exp, time, current)
    
                tauf = popt[0]
                ss[str(cnt)] = tauf/max_tf
                cnt += 1
            
 
            except:
                ss[str(cnt)] = float('inf')
                cnt += 1
    
    d3 = data[3].split_periodic(11000, adjust = True)
    for d in d3:
        d = d.trim_left(10000, adjust=True)
              
        current = d['sodium_current.i_Na']
        current = current[:-1]
        
        index = np.argmax(np.abs(current))
        ss[str(cnt)] = current[index] / max_observed_peak
        cnt += 1    
    return ss


