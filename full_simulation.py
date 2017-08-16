# full_simulation.py
# Runs the full cell model with random draws from ABC posterior results
# Plots results with(out) stimulating current for all simulations.

import myokit
import numpy as np
import math
import matplotlib.pyplot as plt
import ast # for literal_eval
import channel_setup as cs
import distributions as Dist
import pickle

# Helper function for rounding to 3 sig figs adapted from:
#  https://stackoverflow.com/questions/3410976/how-to-round-a-number-to-significant-figures-in-python
round_to_n = lambda x, n: round(x, -int(math.floor(math.log10(abs(x)))) + (n - 1)) if x != 0 else 0

# Load full model
m,p,x = myokit.load('models/Houston2017.mmt')
print myokit.step(m)

i_stim = m.get('membrane.i_stim')
i_stim.set_rhs(0)
i_stim.set_binding('pace')

s = myokit.Simulation(m)

# Channel list
channels = [cs.icat(),
            cs.iha(),
            cs.ikur(),
            cs.ikr(),
            cs.ina(),
            cs.ito()]

# Load results from ABC simulation for each channel
results = []
for ch in channels:
    results_file = 'final_results/results_' + ch.name + '.txt'
    f = open(results_file, 'r')
    lines = f.readlines()
    particles = ast.literal_eval(lines[0])
    weights = ast.literal_eval(lines[1])
    dists = Dist.Arbitrary(particles, weights)
    results.append(dists)

# Parameters from particle swarm optimisation
vary_params = ['ik1.g_K1','ipca.i_pCa_max','inak.i_NaK_max','incx.k_NaCa','icab.g_Cab','inab.g_Nab']
vary_vals =[0.9704410042718667, 6.422758877110399, 5.339669099989606, 255.23623247664673, 0.0005138224827351635, 0.00015230851189352624]
# vary_vals = [None,None,None,None,None,None,None,None]
for i,p in enumerate(vary_params):
    if vary_vals[i] is not None:
        s.set_constant(p, vary_vals[i])

# Prepare figure for plotting
# May need to edit nrows or ncols depending on how many variables are logged
fig, ax = plt.subplots(2,3,sharex=True,sharey='row',figsize=(9,4.6))
colors = ['#0072B2', '#009E73', '#D55E00', '#CC79A7', '#F0E442', '#56B4E9']
num_auto = 0
firing_rates = []
resting_potentials = []

# Population settings
pop_dvdtmax = []
pop_apd25 = []
pop_apd50 = []
pop_apd75 = []
pop_apd90 = []
pop_apa = []
pop_vrp = []
all_results = []

# Pacing current
p = myokit.pacing.blocktrain(498, 2, limit=0, level=-40)
# s.set_protocol(p)
s.reset()
out = []
for i in range(200):
    print str(i)
    s.reset()
    # Set simulation constants to results from ABC
    for k, c in enumerate(channels):
        params = results[k].draw()
        for j, p in enumerate(c.parameters):
            s.set_constant(c.parameters[j], params[j])
    res = s.run(105000,
                    log=['environment.time',
                     'membrane.V'])
                     'calcium_concentration.Cai',
                     'ina.i_Na',
                     'ikr.i_Kr',
                     'ikur.i_Kur',
                     'ito.i_to',
                     'iha.i_ha',
                     'ical.i_CaL',
                     'icat.i_CaT',
                     'ik1.i_K1',
                     'ipca.i_pCa',
                     'incx.i_NaCa',
                     'inak.i_NaK',
                     'icab.i_Cab',
                     'inab.i_Nab'],
                     log_interval=.05)
    res.trim_left(100000, adjust=True)

    # Find action potentials
    fire_times = []
    fire_volts = []
    going_up = True
    for j,V in enumerate(res['membrane.V'][1:]):
        if going_up and V > 0 and V < res['membrane.V'][j-1]:
            going_up = False
            fire_times.append(res['environment.time'][j])
            fire_volts.append(res['membrane.V'][j])
        if not going_up and V > res['membrane.V'][j-1]:
            going_up = True

    # Determine firing rate
    rate = 0.0
    fire_spacing = []
    if len(fire_times) > 1:
        fire_spacing = [fire_times[k+1] - fire_times[k] for k in range(len(fire_times[1:]))]
        rate = 1000/np.mean(fire_spacing)
        firing_rates.append(rate)
        num_auto += 1
    else:
        resting_potentials.append(np.mean(res['membrane.V']))

    print "Firing rate: " + str(rate) + " Hz"

    # Only consider first second now
    res.trim_right(1000)

    # Find resting potential before stim
    stim_time = 98.0
    stim_index = res.find(stim_time)
    v_rp = res['membrane.V'][stim_index]

    # Ignore cell with high resting potential
    if v_rp > -50:
        print "High Vrp!"
        continue

    # Find which peak is the stim
    fire_index = 0
    while fire_times[fire_index] < stim_time:
        fire_index += 1
    peak_time = fire_times[fire_index]
    peak_index = res.find(peak_time)

    # Check if automatic AP near stim
    # Skip if interference
    if len(fire_spacing) > 1:
        if abs(fire_spacing[fire_index]) < 100 or abs(fire_spacing[fire_index+1] < 100):
            print "Near stim!"
            continue

    # Calculate amplitude and upstroke rate
    APA = fire_volts[fire_index] - v_rp
    dvdt_max = APA / (peak_time - stim_time)

    # Find APDs
    rep25 = APA * 0.75 + v_rp
    rep50 = APA * 0.5 + v_rp
    rep75 = APA * 0.25 + v_rp
    rep90 = APA * 0.1 + v_rp
    v_curr = fire_volts[0]
    index = peak_index
    repolarised = True
    while v_curr > rep25:
        index += 1
        v_curr = res['membrane.V'][index]
        t = res['environment.time'][index]
        if t > 100 + peak_time:
            repolarised = False
            break
    APD25 = res['environment.time'][index] - stim_time
    index = peak_index
    while v_curr > rep50:
        index += 1
        v_curr = res['membrane.V'][index]
        t = res['environment.time'][index]
        if t > 100 + peak_time:
            repolarised = False
            break
    APD50 = res['environment.time'][index] - stim_time
    index = peak_index
    while v_curr > rep75:
        index += 1
        v_curr = res['membrane.V'][index]
        t = res['environment.time'][index]
        if t > 100 + peak_time:
            repolarised = False
            break
    APD75 = res['environment.time'][index] - stim_time
    index = peak_index
    while v_curr > rep90:
        index += 1
        v_curr = res['membrane.V'][index]
        t = res['environment.time'][index]
        if t > 100 + peak_time:
            repolarised = False
            break
    APD90 = res['environment.time'][index] - stim_time

    # If the cell stays depolarised, don't record it
    if not repolarised:
        continue

    print "Resting potential: " + str(v_rp) + " mV"
    pop_vrp.append(v_rp)
    print "Amplitude: " + str(APA) + " mV"
    pop_apa.append(APA)
    print "Max upstroke: " + str(dvdt_max) + " mV/ms"
    pop_dvdtmax.append(dvdt_max)
    print "APD25: " + str(APD25) + " ms"
    pop_apd25.append(APD25)
    print "APD50: " + str(APD50) + " ms"
    pop_apd50.append(APD50)
    print "APD75: " + str(APD75) + " ms"
    pop_apd75.append(APD75)
    print "APD90: " + str(APD90) + " ms"
    pop_apd90.append(APD90)

    # Trim to size for plotting
    res.trim_left(88, adjust=True)
    res.trim_right(100)
    all_results.append(res)

    # Save log
    res.save_csv('AP_stims/AP_'+str(i)+'.csv')

    # Plot current
    ax[0][i].plot(res['environment.time'], res['membrane.V'],color=colors[i],ls='--',lw=1)
    ax[0][i].ticklabel_format(useOffset=False)
    ax[0][i].annotate(str(round_to_n(rate,2))+"Hz",xy=(1,1),xycoords='axes fraction',fontsize=12, xytext = (-5,-5), textcoords = 'offset points',ha='right',va='top')
    ax[1][i].plot(res['environment.time'], res['ito.i_to'],color=colors[i],ls='--',lw=1)
    ax[1][i].ticklabel_format(useOffset=False)
    ax[1][i].set_xlabel('Time (ms)')

ax[0][0].set_ylabel('Membrane potential (mV)')
ax[1][0].set_ylabel('Intra. Ca2+ conc. (uM)')
plt.tight_layout()
fig.savefig('final_results/fig_APsample.pdf', bbox_inches='tight')

# Print summary to stdout
print "Proportion exhibiting automaticity:\n  " + str(round_to_n(num_auto/200.0 * 100,3)) + "%"
print "Firing rate:\n  " + str(round_to_n(np.mean(firing_rates),3)) + " +/- " + str(round_to_n(np.sqrt(np.var(firing_rates)),3)) + " Hz"
print "Resting potential:\n  " + str(round_to_n(np.mean(resting_potentials),3)) + " +/- "  + str(round_to_n(np.sqrt(np.var(resting_potentials)),3)) + " mV"
