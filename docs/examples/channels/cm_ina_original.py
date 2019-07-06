from ionchannelABC.protocol import recovery, availability_linear
import data.ina.cm_ina as data
import numpy as np
import pandas as pd
import myokit
import warnings
from scipy.optimize import OptimizeWarning
import scipy.optimize as so


def temperature_correct(R0, T0, T1, Q10):
    return R0*Q10**((T0-T1)/10)

modelfile = 'models/Courtemanche_iNa.mmt'

# Grab all observational data
datasets = []

''' IV datasets
vsteps, peaks, sd = data.IV_Schneider()
cm_mean = 89 # pF
cm_sd = 26.7
vsteps, peaks, _ = data.IV_Sakakibara()
# Propagate errors in capacitance measure to IV
cm_mean = 126.8 # pF
cm_sem = 10.3 # pF
cm_N = 46 # cells
cm_sd = cm_sem * np.sqrt(cm_N) # pF
# convert nA to pA/pF
peaks = np.array(peaks)
peaks = peaks*1000/cm_mean
sd = [(cm_sd/cm_mean)*p for p in peaks]
max_observed_peak = np.max(np.abs(peaks)) # normalising
peaks = [p / max_observed_peak for p in peaks]
variances = [(sd_ / max_observed_peak)**2 for sd_ in sd]
datasets.append([vsteps, peaks, variances])
'''

vsteps_act, act, sd_act = data.Act_Sakakibara()
vsteps_act = [v_+20 for v_ in vsteps_act]
variances_act = [sd**2 for sd in sd_act]
datasets.append([vsteps_act, act, variances_act])

vsteps_inact, inact, sd_inact = data.Inact_Sakakibara()
vsteps_inact = [v_+20 for v_ in vsteps_inact]
variances_inact = [sd**2 for sd in sd_inact]
datasets.append([vsteps_inact, inact, variances_inact])

'''
vsteps_inact, inact32, _ = data.Inact_Schneider_32()
variances_inact = [0.0]*len(vsteps_inact)
datasets.append([vsteps_inact, inact32, variances_inact])
_, inact64, _ = data.Inact_Schneider_64()
datasets.append([vsteps_inact, inact64, variances_inact])
_, inact128, _ = data.Inact_Schneider_128()
datasets.append([vsteps_inact, inact128, variances_inact])
_, inact256, _ = data.Inact_Schneider_256()
datasets.append([vsteps_inact, inact256, variances_inact])
_, inact512, _ = data.Inact_Schneider_512()
datasets.append([vsteps_inact, inact512, variances_inact])
'''

vsteps_tm, tm, sd_tm = data.TauM_Activation_Schneider()
#vsteps_tm = [v_ for v_ in vsteps_tm]
tm = [temperature_correct(tm_, 297, 310, 3) for tm_ in tm]
max_tm = np.max(np.abs(tm)) # normalising
tm = [tm_ / max_tm for tm_ in tm]
variances_tm = [(sd_/max_tm)**2 for sd_ in sd_tm]
datasets.append([vsteps_tm, tm, variances_tm])

'''
vsteps_th, th, sd_th = data.TauH_Inactivation_Schneider()
max_th = np.max(np.abs(th))
th = [th_ / max_th for th_ in th]
variances_th = [(sd_/max_th)**2 for sd_ in sd_th]
datasets.append([vsteps_th, th, variances_th])
'''

vsteps_th, th, sd_th = data.TauH_Inactivation_Sakakibara()
vsteps_th = [v_+20 for v_ in vsteps_th]
th = [temperature_correct(th_, 290, 310, 3) for th_ in th]
max_th = np.max(np.abs(th)) # normalising
th = [th_ / max_th for th_ in th]
variances_th = [(sd_/max_th)**2 for sd_ in sd_th]
datasets.append([vsteps_th, th, variances_th])

#vsteps_tj, tj, sd_tj = data.TauJ_Inactivation_Sakakibara()
#max_tj = np.max(np.abs(tj)) # normalising
#tj = [tj_ / max_tj for tj_ in tj]
#variances_tj = [(sd_/max_tj)**2 for sd_ in sd_tj]
#datasets.append([vsteps_tj, tj, variances_tj])
#
#vsteps_rel, rel, sd_rel = data.Rel_Tf_Sakakibara()
#variances_rel = [sd_**2 for sd_ in sd_rel]
#datasets.append([vsteps_rel, rel, variances_rel])

vsteps_th_depol, th_depol, _ = data.TauH_Inactivation_Sakakibara_Depol()
vsteps_th_depol = [v_+20 for v_ in vsteps_th_depol]
th_depol = [temperature_correct(th_, 290, 310, 3) for th_ in th_depol]
max_th_depol = np.max(np.abs(th_depol))
th_depol = [th_ / max_th_depol for th_ in th_depol]
variances = [0.]*len(th_depol)
datasets.append([vsteps_th_depol, th_depol, variances])

observations = pd.DataFrame(columns=['x','y','variance','exp_id'])
for id, data in enumerate(datasets):
    data = np.asarray(data).T.tolist()
    data = [d + [str(id),] for d in data]
    observations = observations.append(
        pd.DataFrame(data, columns=['x','y','variance','exp_id']),
        ignore_index=True
    )


# Create protocols for simulations
protocols, conditions = [], []

'''
# IV curve protocol (Schneider 1994)
protocols.append(
    myokit.pacing.steptrain_linear(-85, 65, 10, -135, 500, 12)
)
schneider_conditions = {'membrane.Na_o': 120000,
                        'membrane.Na_i': 70000,
                        'membrane.T': 297.15}
conditions.append(schneider_conditions)
'''

cm_conditions = {'membrane.Na_o': 140000,
                 'membrane.Na_i': 11200,
                 'membrane.T': 310}

# Activation protocol (Sakakibara 1992)
protocols.append(
    myokit.pacing.steptrain_linear(-100, 30, 10, -140, 9900, 100)
)
sakakibara_conditions = {'membrane.Na_o': 5000,
                         'membrane.Na_i': 5000,
                         'membrane.T': 290.15}
conditions.append(cm_conditions)

# Inactivation protocol voltage-shifted +20mV (Sakakibara 1992)
protocols.append(
    availability_linear(-120, -10, 10, -140, -20, 10000, 1000, 0, 30)
)
conditions.append(cm_conditions)

'''
# Inactivation protocols (Schneider 1994, pp 32ms)
protocols.append(
    availability(vsteps_inact, -135, -20, 10000, 32, 0, 30)
)
conditions.append(schneider_conditions)

# Inactivation protocols (Schneider 1994, pp 64ms)
protocols.append(
    availability(vsteps_inact, -135, -20, 10000, 64, 0, 30)
)
conditions.append(schneider_conditions)

# Inactivation protocols (Schneider 1994, pp 128ms)
protocols.append(
    availability(vsteps_inact, -135, -20, 10000, 128, 0, 30)
)
conditions.append(schneider_conditions)

# Inactivation protocols (Schneider 1994, pp 256ms)
protocols.append(
    availability(vsteps_inact, -135, -20, 10000, 256, 0, 30)
)
conditions.append(schneider_conditions)

# Inactivation protocols (Schneider 1994, pp 512ms)
protocols.append(
    availability(vsteps_inact, -135, -20, 10000, 512, 0, 30)
)
conditions.append(schneider_conditions)
'''

# Activation kinetics
protocols.append(
    myokit.pacing.steptrain_linear(-65, 25, 10, -135, 500, 100)
)
schneider_conditions = {'membrane.Na_o': 120000,
                        'membrane.Na_i': 70000,
                        'membrane.T': 297.15}
conditions.append(cm_conditions)

# Inactivation kinetics (Sakakibara voltage shifted)
protocols.append(
    myokit.pacing.steptrain_linear(-30, 10, 10, -140, 1000, 100)
)
conditions.append(cm_conditions)

# Recovery kinetics (Sakakibara voltage shifted)
# Time used doesn't matter as we aren't fitting to
# recovery curve directly
twaits = [0,2,5,10,15,20,25,30,35,40,45,50,75,100,200,300,400,500,600,700,800,900,1000]
for v in vsteps_th_depol:
    protocols.append(
        recovery(twaits, v, -20, -20, 1000, 1000, 1000)
    )
    conditions.append(cm_conditions)

# Create model and simulations
m = myokit.load_model(modelfile)
ena = m.get('na_conc.E_Na').value()
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

    '''
    # Summary statistic for Schneider data
    # I-V curve (normalised)
    d0 = data[0].split_periodic(512, adjust=True)
    for d in d0:
        d = d.trim(500, 512, adjust=True)
        current = d['ina.i_Na']
        index = np.argmax(np.abs(current))
        ss[str(cnt)] = current[index] / max_observed_peak
        cnt += 1
    '''

    ''' Summary statistic for Sakakibara IV curve (normalised)
    d0 = data[0].split_periodic(10000, adjust=True)
    for d in d0:
        d = d.trim(9900, 10000, adjust=True)
        current = d['ina.i_Na']
        index = np.argmax(np.abs(current))
        ss[str(cnt)] = current[index] / max_observed_peak
        cnt += 1
    '''

    # Activation data for Sakakibara
    d0 = data[0].split_periodic(10000, adjust=True)
    for d in d0:
        d = d.trim(9900, 10000, adjust=True)
        gate = d['ina.g']
        #gate = d.npview()['ina.m']**3
        ss[str(cnt)] = max(gate)
        cnt += 1
    for i in range(cnt):
        ss[str(i)] = ss[str(i)] / ss[str(cnt-1)]
    cnt0 = cnt

    # Inactivation data
    d1 = data[1].split_periodic(11030, adjust=True)
    for d in d1:
        d = d.trim(11000, 11030, adjust=True).npview()
        gate = d['ina.g']
        #gate = d['ina.h']*d['ina.j']
        ss[str(cnt)] = max(gate) #gate[0]
        cnt += 1
    for i in range(cnt0+1, cnt):
        ss[str(i)] = ss[str(i)] / ss[str(cnt0)]
    ss[str(cnt0)] = 1.0

    '''
    # Inactivation data
    d1 = data[1].split_periodic(10000+32+30, adjust=True)
    cnt_normalise = cnt
    for d in d1:
        d = d.trim(10032, 10062, adjust=True).npview()
        curr = d['ina.i_Na']
        ss[str(cnt)] = max(curr, key=abs)
        # normalise to maximum current
        if cnt != cnt_normalise:
            ss[str(cnt)] = ss[str(cnt)] / ss[str(cnt_normalise)]
        cnt += 1
    ss[str(cnt_normalise)] = 1.0 # finally normalise initial current

    d2 = data[2].split_periodic(10000+64+30, adjust=True)
    cnt_normalise = cnt
    for d in d2:
        d = d.trim(10064, 10094, adjust=True).npview()
        curr = d['ina.i_Na']
        ss[str(cnt)] = max(curr, key=abs)
        # normalise to maximum current
        if cnt != cnt_normalise:
            ss[str(cnt)] = ss[str(cnt)] / ss[str(cnt_normalise)]
        cnt += 1
    ss[str(cnt_normalise)] = 1.0 # finally normalise initial current

    d3 = data[3].split_periodic(10000+128+30, adjust=True)
    cnt_normalise = cnt
    for d in d3:
        d = d.trim(10128, 10158, adjust=True).npview()
        curr = d['ina.i_Na']
        ss[str(cnt)] = max(curr, key=abs)
        # normalise to maximum current
        if cnt != cnt_normalise:
            ss[str(cnt)] = ss[str(cnt)] / ss[str(cnt_normalise)]
        cnt += 1
    ss[str(cnt_normalise)] = 1.0 # finally normalise initial current

    d4 = data[4].split_periodic(10000+256+30, adjust=True)
    cnt_normalise = cnt
    for d in d4:
        d = d.trim(10256, 10286, adjust=True).npview()
        curr = d['ina.i_Na']
        ss[str(cnt)] = max(curr, key=abs)
        # normalise to maximum current
        if cnt != cnt_normalise:
            ss[str(cnt)] = ss[str(cnt)] / ss[str(cnt_normalise)]
        cnt += 1
    ss[str(cnt_normalise)] = 1.0 # finally normalise initial current

    d5 = data[5].split_periodic(10000+512+30, adjust=True)
    cnt_normalise = cnt
    for d in d5:
        d = d.trim(10512, 10542, adjust=True).npview()
        curr = d['ina.i_Na']
        ss[str(cnt)] = max(curr, key=abs)
        # normalise to maximum current
        if cnt != cnt_normalise:
            ss[str(cnt)] = ss[str(cnt)] / ss[str(cnt_normalise)]
        cnt += 1
    ss[str(cnt_normalise)] = 1.0 # finally normalise initial current
    '''

    # Activation and inactivation kinetics (tm and th)
    d2 = data[2].split_periodic(600, adjust=True)
    for i, d in enumerate(d2):
        d = d.trim(500, 600, adjust=True)
        current = d['ina.i_Na']
        time = d['environment.time']

        # Remove constant
        c0 = d['ina.i_Na'][0]
        current = [(c_-c0) for c_ in current]

        def sum_of_exp(t, taum, tauh):
            return ((1-np.exp(-t/taum))**3 *
                    np.exp(-t/tauh))
        with warnings.catch_warnings():
            warnings.simplefilter('error',OptimizeWarning)
            warnings.simplefilter('error',RuntimeWarning)
            try:
                imax = max(current, key=abs)
                current = [c_/imax for c_ in current]
                if len(time)<=1 or len(current)<=1:
                    raise Exception('Failed simulation')
                popt, _ = so.curve_fit(sum_of_exp, time, current,
                                       p0=[0.5, 1.], #max(current, key=abs)/2],
                                       bounds=([0., 0.], #min(current)],
                                               [1., 100.]))#, max(current)]))
                fit = [sum_of_exp(t, popt[0], popt[1]) for t in time]
                ss[str(cnt)] = popt[0]/max_tm
                #if i < 9:
                #    ss[str(cnt)] = popt[0]/max_tm
                #ss[str(cnt+9)] = popt[1]/max_th
                cnt += 1
            except (Exception, RuntimeWarning, OptimizeWarning, RuntimeError):
                ss[str(cnt)] = float('inf')
                #if i < 9:
                #    ss[str(cnt)] = float('inf')
                #ss[str(cnt+9)] = float('inf')
                cnt += 1

    # Inactivation kinetics (th)
    d3 = data[3].split_periodic(1100, adjust=True)
    for d in d3:
        d = d.trim(1000, 1100, adjust=True)
        current = d['ina.i_Na']
        time = d['environment.time']
        index = np.argmax(np.abs(current))

        # Set time zero to peak current
        current = current[index:]
        time = time[index:]
        t0 = time[0]
        time = [t-t0 for t in time]

        # Remove constant
        #c0 = current[-1]
        #current = [(c_-c0) for c_ in current]

        index = np.argwhere(np.isclose(current,0.0))
        if len(index) != 0:
            current = current[:index[0][0]]
            time = time[:index[0][0]]

        def simple_exp(t, tauh):
            return np.exp(-t/tauh)
        #def sum_of_exp2(t, tauh, tauj, Arel):
        #    return Arel*np.exp(-t/tauh) + (1-Arel)*np.exp(-t/tauj)
        with warnings.catch_warnings():
            warnings.simplefilter('error',OptimizeWarning)
            warnings.simplefilter('error',RuntimeWarning)
            try:
                imax = max(current, key=abs)
                current = [c_/imax for c_ in current]
                if len(time)<=1 or len(current)<=1:
                    raise Exception('Failed simulation')
                popt, _ = so.curve_fit(simple_exp, time, current,
                                       p0=[5], bounds=([0.01], [20.0]))
                #popt, _ = so.curve_fit(sum_of_exp2, time, current,
                #                       p0=[9, 60, 0.9],
                #                       bounds=([0.1, 10, 0.0],
                #                               [10, 100, 1.0]))
                tauh = popt[0]
                #tauj = popt[1]
                #Arel = popt[2]
                #fit = [sum_of_exp2(t, tauh, tauj, Arel) for t in time]

                ss[str(cnt)] = tauh/max_th
                #ss[str(cnt+4)] = tauj/max_tj
                #ss[str(cnt+8)] = Arel
                cnt += 1
            except:
                ss[str(cnt)] = float('inf')
                #ss[str(cnt+4)] = float('inf')
                #ss[str(cnt+8)] = float('inf')
                cnt += 1

    twaits_split = [t+3000 for t in twaits]
    for i in range(len(twaits_split)-1):
        twaits_split[i+1] += twaits_split[i]

    for d in data[4:]:
        rec = []
        trim1, trim2, trim3 = 1000, 2000, 3000

        with warnings.catch_warnings():
            warnings.simplefilter('error', OptimizeWarning)
            warnings.simplefilter('error', RuntimeWarning)
            try:
                # Get recovery curve
                for i, t in enumerate(twaits_split):
                    trace, d = d.split(t)
                    peak1 = max(trace.trim(trim1, trim2)['ina.i_Na'], key=abs)
                    peak2 = max(trace.trim(trim2+twaits[i], trim3+twaits[i])['ina.i_Na'],
                                key=abs)
                    rec.append(peak2/peak1)
                    trim1 += twaits[i]+3000
                    trim2 += twaits[i]+3000
                    trim3 += twaits[i]+3000

                # Fit double exponential to recovery curve
                popt, _ = so.curve_fit(simple_exp, twaits, 1.-np.asarray(rec),
                                       p0=[1],
                                       bounds=([0],[100]))
                tauh = popt[0]
                ss[str(cnt)] = tauh/max_th_depol
                cnt += 1
            except:
                ss[str(cnt)] = float('inf')
                cnt += 1
    return ss
