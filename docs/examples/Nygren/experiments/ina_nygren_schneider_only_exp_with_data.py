from ionchannelABC.protocol import manual_steptrain_linear, manual_availability, availability, recovery
import data.ina.Schneider1994.data_ina as data
import numpy as np
import pandas as pd
import myokit
import matplotlib.pyplot as plt
import warnings
from scipy.optimize import OptimizeWarning
import scipy.optimize as so
modelfile = 'models/nygren_Na.mmt'


### observations dataframe ###

#Load all the experimental data
datasets = []

# IV curve
vsteps, peaks, _ = data.IV_Schneider()
max_observed_peak = np.max(np.abs(peaks)) # for later normalising
peaks = [p / max_observed_peak for p in peaks]
variances = [0 for p in peaks] # no error reported
datasets.append([vsteps, peaks, variances])

# fast inactivation kinetics : tau h1
vsteps_th1, th1, sd_th1 = data.TauH1_Inactivation_Schneider()
max_th1 = np.max(np.abs(th1)) # normalising
th1 = [th1_ / max_th1 for th1_ in th1]
variances_th1 = [(sd_/max_th1)**2 for sd_ in sd_th1]
datasets.append([vsteps_th1, th1, variances_th1])

# slow inactivation kinetics : tau h2

vsteps_th2, th2, sd_th2 = data.TauH2_Inactivation_Schneider()
max_th2 = np.max(np.abs(th2)) # normalising
th2 = [th2_ / max_th2 for th2_ in th2]
variances_th2 = [(sd_/max_th2)**2 for sd_ in sd_th2]
datasets.append([vsteps_th2, th2, variances_th2])

# activation kinetics : tau m
vsteps_tm, tm, sd_tm = data.TauM_Activation_Schneider()
max_tm = np.max(np.abs(tm)) # normalising
tm = [tm_ / max_tm for tm_ in tm]
variances_tm = [(sd_/max_tm)**2 for sd_ in sd_tm]
datasets.append([vsteps_tm, tm, variances_tm])

# inactivation curve : h infinity
vsteps_inact, inact, _ = data.Inact_Schneider_512()
variances_inact = [0 for i in inact]
datasets.append([vsteps_inact, inact, variances_inact])
#print(datasets)

"""
# recovery curve with the recovery potential = -95 mV:
time_recov, recov, _ = data.Recovery_Schneider_95()
variances_recov = [0 for i in recov]
datasets.append([time_recov, recov, variances_recov])
"""


observations = pd.DataFrame(columns=['x','y','variance','exp_id'])
for id, datas in enumerate(datasets):
    datas = np.asarray(datas).T.tolist()
    datas = [d + [str(id),] for d in datas]
    #print(datas)
    observations = observations.append(
        pd.DataFrame(datas, columns=['x','y','variance','exp_id']),
        ignore_index=True)




### protocols and conditions ###
protocols, conditions = [], []

schneider_conditions = {'cleft_space_ion_concentrations.Na_c': 120,
                        'intracellular_ion_concentrations.Na_i': 70,
                        'membrane.T': 297.15}

# IV curve protocol (Schneider 1994)
protocols.append(myokit.pacing.steptrain_linear(-85, 65, 10, -135, 1000, 100))
conditions.append(schneider_conditions)


# Fast inactivation kinetics : tau_h1
protocols.append(manual_steptrain_linear(vlist = vsteps_th1,
                 vhold = -135,
                 tpre= 1000,
                 tstep= 100))
#protocols.append(myokit.pacing.steptrain_linear(-65, 25, 10, -135, 1000, 100))
conditions.append(schneider_conditions)

# Slow inactivation kinetics : tau_h2
protocols.append(manual_steptrain_linear(vlist = vsteps_th2,
                 vhold = -135,
                 tpre= 1000,
                 tstep= 100))
#protocols.append(myokit.pacing.steptrain_linear(-65, 25, 10, -135, 1000, 100))
conditions.append(schneider_conditions)

# activation kinetics : tau m
protocols.append(manual_steptrain_linear(vlist = vsteps_tm,
                 vhold = -135,
                 tpre= 1000,
                 tstep= 100))
#protocols.append(myokit.pacing.steptrain_linear(-65, 25, 10, -135, 1000, 100))
conditions.append(schneider_conditions)

# inactivation curve : h infinity tstep = 512 ms for this protocol, twait and ttest have been approximated with sakakibara's protocol
"""
protocols.append(manual_availability(vlist = vsteps_inact,
                 vhold = -135,
                 vtest = -20,
                 tpre = 1000,
                 tstep = 512,
                 twait = 2,
                 ttest = 12))
conditions.append(schneider_conditions)       
"""
protocols.append(manual_steptrain_linear(vlist = vsteps_inact,
                 vhold = -135,
                 tpre = 1000,
                 tstep = 512))
conditions.append(schneider_conditions) 

     

"""   
# recovery protocol
protocols.append(
    recovery(twait = time_recov ,
                 vhold = -95,
                 vstep1= -20,
                 vstep2= -20,
                 tpre= 1000,
                 tstep1= 1000,
                 tstep2= 12))
conditions.append(schneider_conditions)       
"""

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

def sum_of_exp(t,tauh1,tauh2,taum,Imax):
    return (Imax * (1-np.exp(-t/taum))**3 *(
            0.9*np.exp(-t/tauh1) + 0.1*np.exp(-t/tauh2))) # change this 
    
def summary_statistics(data):
    """Converts raw simulation output to sensible results"""
    
    # Check for error
    if data is None:
        return {}
    
    # Process sensible results
    ss = {}
    cnt = 0
    
    d0 = data[0].split_periodic(1100, adjust = True)
    for d in d0:
        d = d.trim_left(1000, adjust=True)
              
        current = d['sodium_current.i_Na']
        current = current[:-1]
        
        index = np.argmax(np.abs(current))
        ss[str(cnt)] = current[index] / max_observed_peak
        cnt += 1            

        
    # Inactivation kinetics th1
    d1 = data[1].split_periodic(1100, adjust = True)
    for d in d1:

        d = d.trim_left(1000, adjust = True)
        current = d['sodium_current.i_Na']
        current = current[:-1]
        time = d['environment.time']
        time = time[:-1]



        # Remove constant
        c0 = current[0]
        current = [(c_-c0) for c_ in current]
        
    
        
        with warnings.catch_warnings():
            warnings.simplefilter('error',OptimizeWarning)
            warnings.simplefilter('error',RuntimeWarning)

            try:
                if len(time)<=1 or len(current)<=1:
                    raise Exception('Failed simulation')
                popt, _ = so.curve_fit(sum_of_exp, time, current,
                                           p0=[0.5, 1.,1., max(current, key=abs)/2],
                                           bounds=([0., 0., 0, min(current)],
                                                   [1., 10.,10, max(current)]))
    

                tauh1 = popt[0]

                ss[str(cnt)] = tauh1/max_th1
                cnt += 1
            
                plt.plot(time,current,time,sum_of_exp(np.asarray(time),popt[0],popt[1],popt[2],popt[3]))
                plt.show()
            except:
                ss[str(cnt)] = float('inf')
                cnt += 1

    # Inactivation kinetics th2
    d2 = data[2].split_periodic(1100, adjust = True)
    for d in d2:

        d = d.trim_left(1000, adjust = True)
        current = d['sodium_current.i_Na']
        current = current[:-1]
        time = d['environment.time']
        time = time[:-1]



        # Remove constant
        c0 = current[0]
        current = [(c_-c0) for c_ in current]
        
    
        
        with warnings.catch_warnings():
            warnings.simplefilter('error',OptimizeWarning)
            warnings.simplefilter('error',RuntimeWarning)

            try:
                if len(time)<=1 or len(current)<=1:
                    raise Exception('Failed simulation')
                popt, _ = so.curve_fit(sum_of_exp, time, current,
                                           p0=[0.5, 1.,1., max(current, key=abs)/2],
                                           bounds=([0., 0., 0, min(current)],
                                                   [1., 10.,10, max(current)]))
    

                tauh2 = popt[1]

                ss[str(cnt)] = tauh2/max_th2
                cnt += 1
            
                plt.plot(time,current,time,sum_of_exp(np.asarray(time),popt[0],popt[1],popt[2],popt[3]))
                plt.show()
            except:
                ss[str(cnt)] = float('inf')
                cnt += 1    

    # Activation kinetics tm
    d3 = data[3].split_periodic(1100, adjust = True)
    for d in d3:

        d = d.trim_left(1000, adjust = True)
        current = d['sodium_current.i_Na']
        current = current[:-1]
        time = d['environment.time']
        time = time[:-1]



        # Remove constant
        c0 = current[0]
        current = [(c_-c0) for c_ in current]
        
    
        
        with warnings.catch_warnings():
            warnings.simplefilter('error',OptimizeWarning)
            warnings.simplefilter('error',RuntimeWarning)

            try:
                if len(time)<=1 or len(current)<=1:
                    raise Exception('Failed simulation')
                popt, _ = so.curve_fit(sum_of_exp, time, current,
                                           p0=[0.5, 1.,1., max(current, key=abs)/2],
                                           bounds=([0., 0., 0, min(current)],
                                                   [1., 10.,10, max(current)]))
    

                taum = popt[2]

                ss[str(cnt)] = taum/max_tm
                cnt += 1
            
                plt.plot(time,current,time,sum_of_exp(np.asarray(time),popt[0],popt[1],popt[2],popt[3]))
                plt.show()
            except:
                ss[str(cnt)] = float('inf')
                cnt += 1 
    
    # Inactivation curve           
    #d4 = data[4].split_periodic(1526, adjust = True)
    d4 = data[4].split_periodic(1512, adjust = True)
    for d in d4:
        d = d.trim_left(1000, adjust=True)
              
        inact_gate = d['sodium_current.h_infinity']
        index = np.argmin(np.abs(inact_gate))
        ss[str(cnt)] = np.abs(inact_gate[index])
        cnt += 1   

    return ss


