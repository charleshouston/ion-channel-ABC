from ionchannelABC.protocol import recovery, availability
import data.ina.cm_ina as data
import numpy as np
import pandas as pd
import myokit
import warnings
from scipy.optimize import OptimizeWarning
import scipy.optimize as so


modelfile = 'models/ina_markov.mmt'

protocols, conditions = [], []

''' No IV curve protocol as activation data includes it
# IV curve protocol (Schneider 1994)
protocols.append(
    myokit.pacing.steptrain_linear(-85, 65, 10, -135, 500, 12)
)
schneider_conditions = {'membrane.Na_o': 120000,
                        'membrane.Na_i': 70000,
                        'membrane.T': 297.15}
conditions.append(schneider_conditions)
'''

# Activation curve protocol (Sakakibara 1992)
protocols.append(
    myokit.pacing.steptrain_linear(-100, 30, 10, -140, 9900, 100)
)
sakakibara_conditions = {'membrane.Na_o': 5000,
                         'membrane.Na_i': 5000,
                         'membrane.T': 290.15}
conditions.append(sakakibara_conditions)

# Inactivation curve protocol (Sakakibara 1992, voltage-shifted +20mV see CM paper)
protocols.append(
    availability(-120, -10, 10, -140, -20, 10000-1000-2-30, 1000, 2, 30)
)
conditions.append(sakakibara_conditions)

# Activation kinetics
protocols.append(
    myokit.pacing.steptrain_linear(-65, 25, 10, -135, 500, 100)
)
schneider_conditions = {'membrane.Na_o': 120000,
                        'membrane.Na_i': 70000,
                        'membrane.T': 297.15}
conditions.append(schneider_conditions)


# Inactivation kinetics
protocols.append(
    myokit.pacing.steptrain_linear(-50, -10, 10, -140, 1000, 100)
)
conditions.append(sakakibara_conditions)

# Grab all observational data
datasets = []

'''
vsteps, peaks, sd = data.IV_Schneider()
cm_mean = 89 # pF
cm_sd = 26.7

vsteps, peaks, _ = data.IV_Sakakibara()
# Propagate errors in capacitance measure to IV
cm_mean = 126.8 # pF
cm_sem = 10.3 # pF
cm_N = 46 # cells
cm_sd = cm_sem * np.sqrt(cm_N) # pF

# convert nA to nA/pF
peaks = np.array(peaks)
peaks = peaks*1000/cm_mean
sd = [(cm_sd/cm_mean)*p for p in peaks]
max_observed_peak = np.max(np.abs(peaks)) # normalising
peaks = [p / max_observed_peak for p in peaks]
variances = [(sd_ / max_observed_peak)**2 for sd_ in sd]
datasets.append([vsteps, peaks, variances])
'''

vsteps_act, act, sd_act = data.Act_Sakakibara()
variances_act = [sd**2 for sd in sd_act]
datasets.append([vsteps_act, act, variances_act])

vsteps_inact, inact, sd_inact = data.Inact_Sakakibara()
vsteps_inact = [v_+20 for v_ in vsteps_inact]
variances_inact = [sd**2 for sd in sd_inact]
datasets.append([vsteps_inact, inact, variances_inact])

vsteps_tm, tm, sd_tm = data.TauM_Activation_Schneider()
max_tm = np.max(np.abs(tm)) # normalising
tm = [tm_ / max_tm for tm_ in tm]
variances_tm = [(sd_/max_tm)**2 for sd_ in sd_tm]
datasets.append([vsteps_tm, tm, variances_tm])

vsteps_tj, tj, sd_tj = data.TauJ_Inactivation_Sakakibara()
max_tj = np.max(np.abs(tj)) # normalising
tj = [tj_ / max_tj for tj_ in tj]
variances_tj = [(sd_/max_tj)**2 for sd_ in sd_tj]
datasets.append([vsteps_tj, tj, variances_tj])

'''
vsteps_th, th, sd_th = data.TauH_Inactivation_Sakakibara()
max_th = np.max(np.abs(th)) # normalising
th = [th_ / max_th for th_ in th]
variances_th = [(sd_/max_th)**2 for sd_ in sd_th]
datasets.append([vsteps_th, th, variances_th])

vsteps_rel, rel, sd_rel = data.Rel_Tf_Sakakibara()
variances_rel = [sd_**2 for sd_ in sd_rel]
datasets.append([vsteps_rel, rel, variances_rel])
'''

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

    '''Commented out - not using IV curve data
    # Summary statistic for Schneider data
    # I-V curve (normalised)
    d0 = data[0].split_periodic(512, adjust=True)
    for d in d0:
        d = d.trim_left(500, adjust=True)
        current = d['ina.i_Na']
        index = np.argmax(np.abs(current))
        ss[str(cnt)] = current[index] / max_observed_peak
        cnt += 1
    '''

    '''Commented out as using Schneider data above
    # IV curve (normalised)
    d0 = data[0].split_periodic(10000, adjust=True)
    for d in d0:
        d = d.trim_left(9900, adjust=True)
        current = d['ina.i_Na']
        index = np.argmax(np.abs(current))
        ss[str(cnt)] = current[index] / max_observed_peak
        cnt += 1
    '''

    # Activation data (same simulated data as IV)
    d0 = data[0].split_periodic(10000, adjust=True)
    for d in d0:
        d = d.trim_left(9900, adjust=True)
        gate = d.npview()['ina.m']**3
        index = np.argmax(np.abs(gate))
        ss[str(cnt)] = np.abs(gate[index])
        cnt += 1

    # Inactivation data
    d1 = data[1].split_periodic(10000, adjust=True)
    for d in d1:
        d = d.trim(10000-32, 10000-30, adjust=True).npview()
        gate = d['ina.h']*d['ina.j']
        #index = np.argmax(np.abs(gate))
        ss[str(cnt)] = gate[0]#np.abs(gate[index])
        cnt += 1
        
    # Activation kinetics (tm)
    d2 = data[2].split_periodic(600, adjust=True)
    for d in d2:
        d = d.trim_left(500, adjust=True)
        current = d['ina.i_Na']
        time = d['environment.time']

        # Remove constant
        c0 = d['ina.i_Na'][0]
        current = [(c_-c0) for c_ in current]

        def sum_of_exp(t, taum, tauh, Imax):
            return (Imax * (1-np.exp(-t/taum))**3 *
                    np.exp(-t/tauh))
        with warnings.catch_warnings():
            warnings.simplefilter('error',OptimizeWarning)
            warnings.simplefilter('error',RuntimeWarning)
            try:
                if len(time)<=1 or len(current)<=1:
                    raise Exception('Failed simulation')

                popt, _ = so.curve_fit(sum_of_exp, time, current,
                                       p0=[0.5, 1., max(current, key=abs)/2],
                                       bounds=([0., 0., min(current)],
                                               [1., 10., max(current)]))

                ss[str(cnt)] = popt[0]/max_tm
                cnt += 1
            except (Exception, RuntimeWarning, OptimizeWarning, RuntimeError):
                ss[str(cnt)] = float('inf')
                cnt += 1

    # Inactivation kinetics (tj and th)
    d3 = data[3].split_periodic(1100, adjust=True)
    for d in d3:
        d = d.trim_left(1000, adjust=True)
        current = d['ina.i_Na']
        time = d['environment.time']
        index = np.argmax(np.abs(current))

        # Set time zero to peak current
        current = current[index:]
        time = time[index:]
        t0 = time[0]
        time = [t-t0 for t in time]

        # Remove constant
        c0 = current[-1]
        current = [(c_-c0) for c_ in current]

        def simple_exp(t, tauj):
            return np.exp(-t/tauj)
        #def sum_of_exp2(t, tauj, tauh, Arel):
        #    return Arel*np.exp(-t/tauj) + (1-Arel)*np.exp(-t/tauh)
        with warnings.catch_warnings():
            warnings.simplefilter('error',OptimizeWarning)
            warnings.simplefilter('error',RuntimeWarning)
            try:
                imax = max(current, key=abs)
                current = [c_/imax for c_ in current]
                if len(time)<=1 or len(current)<=1:
                    raise Exception('Failed simulation')
                popt, _ = so.curve_fit(simple_exp, time, current,
                                       p0=[5], bounds=([0.1], [10.0]))
                '''
                popt, _ = so.curve_fit(sum_of_exp2, time, current,
                                       p0=[5, 50, 0.9],
                                       bounds=([0.1, 10, 0.0],
                                               [10, 100, 0.95]))
                tauj = popt[0]
                tauh = popt[1]
                Arel = popt[2]
                '''
                tauj = popt[0]

                ss[str(cnt)] = tauj/max_tj
                #ss[str(cnt+4)] = tauh/max_th
                #ss[str(cnt+8)] = Arel
                cnt += 1
            except:
                ss[str(cnt)] = float('inf')
                #ss[str(cnt+4)] = float('inf')
                #ss[str(cnt+8)] = float('inf')
                cnt += 1
    return ss
