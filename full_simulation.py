# full_simulation.py
# Runs the full cell model with random draws from ABC posterior results
# Plots results without stimulating current for all simulations.

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
m,_,_ = myokit.load('models/Houston2017b.mmt')
print myokit.step(m)

i_stim = m.get('membrane.i_stim')
i_stim.set_rhs(0)
i_stim.set_binding('pace')

s = myokit.Simulation(m)
# s.set_max_step_size(0.005)

# Channel list
channels = [cs.icat(),
            cs.iha(),
            cs.ikur(),
            cs.ikr(),
            cs.ina2(),
            cs.ito(),
            cs.ical(),
            cs.ik1()]

# Load results from ABC simulation for each channel
results = []
for ch in channels:
    results_file = 'results/updated-model/results_' + ch.name + '.txt'
    f = open(results_file, 'r')
    lines = f.readlines()
    particles = ast.literal_eval(lines[0])
    weights = ast.literal_eval(lines[1])
    dists = Dist.Arbitrary(particles, weights)
    results.append(dists)

# Parameters from particle swarm optimisation
vary_params = ['inak.i_NaK_max','incx.k_NCX','icab.g_Cab','inab.g_Nab','ryanodine_receptors.k_RyR','serca.V_max']
vary_vals = [6.218201215970397, 5.211334049027319e-16, 0.00029726874812888604, 0.00963563319999778, 0.06930901723850878, 1.9376164667772877] 

# Prepare figure for plotting
# May need to edit nrows or ncols depending on how many variables are logged
# fig, ax = plt.subplots(2,3,sharex=True,sharey='row',figsize=(9,4.6))
colors = ['#0072B2', '#009E73', '#D55E00', '#CC79A7', '#F0E442', '#56B4E9']
# num_auto = 0
# firing_rates = []
# resting_potentials = []

# Population results and traces
pop_results = []
traces = []

# Pacing current
prot = myokit.pacing.blocktrain(1000, 0.5, limit=0, level=-80)

# Number of simulations for uncertainty quantification
N = 1000
for sim_num in range(N):
    print str(sim_num)
    
    s.reset()
    s.set_protocol(prot)

    # Set simulation constants to results from ABC
    for k, c in enumerate(channels):
        params = results[k].draw()
        for j, p in enumerate(c.parameters):
            s.set_constant(c.parameters[j], params[j])

    # Set PSO optimised parameters to correct values
    for i, p in enumerate(vary_params):
        if vary_vals[i] is not None:
            s.set_constant(p, vary_vals[i])

    # Run results from simulation
    output = s.run(101000,
                log=myokit.LOG_ALL)
    output.trim_left(99800, adjust=True)

    # fig,ax = plt.subplots(nrows=2, ncols=3, sharex=True)
    # time = output['environment.time']
    # ax[0][0].plot(time, output['membrane.V'], label='V')
    # ax[0][0].legend()

    # ax[1][0].plot(time, output['icab.i_Cab'], label='i_Cab')
    # ax[1][0].plot(time, output['inab.i_Nab'], label='i_Nab')
    # ax[1][0].legend()

    # ax[0][1].plot(time, output['ina.i_Na'], label='i_Na')
    # ax[0][1].legend()

    # ax[0][2].plot(time, output['ical.i_CaL'], label='i_CaL')
    # ax[0][2].plot(time, output['icat.i_CaT'], label='i_CaT')
    # ax[0][2].legend()

    # ax[1][1].plot(time, output['incx.i_NCX'], label='i_NCX')
    # ax[1][1].plot(time, output['inak.i_NaK'], label='i_NaK')
    # ax[1][1].plot(time, output['iha.i_ha'], label='i_ha')
    # ax[1][1].legend()

    # ax[1][2].plot(time, output['ikr.i_Kr'], label='i_Kr')
    # ax[1][2].plot(time, output['ikur.i_Kur'], label='i_Kur')
    # ax[1][2].plot(time, output['ik1.i_K1'], label='i_K1')
    # ax[1][2].plot(time, output['ito.i_to'], label='i_to')
    # ax[1][2].legend()

    # for a in ax.flatten():
    #     a.get_yaxis().get_major_formatter().set_useOffset(False)

    # fig.show()

    # f2, a2 = plt.subplots(nrows=2, ncols=3, sharex=True)

    # a2[0][0].plot(time, output['ca_conc.Ca_i'],label='Ca_i')
    # a2[0][0].legend()

    # a2[1][0].plot(time, output['ca_conc_sr.Ca_SRrelease'],label='SR release')
    # a2[1][0].plot(time, output['ca_conc_sr.Ca_SRuptake'],label='SR uptake')
    # a2[1][0].plot(time, output['ca_conc_sr.Ca_SR'],label='SR')
    # a2[1][0].legend()

    # a2[0][1].plot(time, output['ca_conc.J_CaSR'], label='J_CaSR')
    # a2[0][1].plot(time, output['ca_conc.J_CaSL'], label='J_CaSL')
    # a2[0][1].legend()

    # a2[1][1].plot(time, output['ryanodine_receptors.J_RyR'],label='J_RyR')
    # a2[1][1].legend()

    # a2[0][2].plot(time, output['serca.J_SERCA'],label='J_SERCA')
    # a2[0][2].plot(time, output['ca_diffusion.J_tr'],label='J_tr')
    # a2[0][2].legend()

    # a2[1][2].plot(time,output['leak_flux.J_leak'],label='J_leak')
    # a2[1][2].legend()

    # for a in a2.flatten():
    #     a.get_yaxis().get_major_formatter().set_useOffset(False)

    # f2.show()

    # import pdb;pdb.set_trace()

    ####################################################
    # Take measurements for uncertainty quantification #
    ####################################################

    # Find action potentials
    # Find resting potential before stim
    stim_time = 200.0
    stim_index = output.find(stim_time)
    v_rp = output['membrane.V'][stim_index]

    # Ignore cell with high resting potential
    if v_rp > -60:
        print "High Vrp!"
        continue

    try:
        # Find voltage peak
        peak_index = output['membrane.V'].index(max(output['membrane.V']))
        peak_time = output['environment.time'][peak_index]
        if peak_time < stim_time:
            print "Automaticity!"
            continue

        v_max = output['membrane.V'][peak_index]
        APA = v_max - v_rp
        dvdt_max = APA / (peak_time - stim_time)

        # Find APDs
        rep25 = APA * 0.75 + v_rp
        rep50 = APA * 0.5 + v_rp
        rep75 = APA * 0.25 + v_rp
        rep90 = APA * 0.1 + v_rp
        rep_vals = [rep25, rep50, rep75, rep90]
        apd_vals = []

        v_curr = v_max
        index = peak_index
        for rep in rep_vals:
            while v_curr > rep:
                index += 1
                v_curr = output['membrane.V'][index]
            apd_vals.append(output['environment.time'][index] - stim_time)

        # Calcium measurements
        # Diastole
        ca_i_diastole = output['ca_conc.Ca_i'][stim_index]
        ca_sr_diastole = output['ca_conc_sr.Ca_SR'][stim_index]

        # Systole
        ca_i_systole = max(output['ca_conc.Ca_i'])
        peak_index_ca = output['ca_conc.Ca_i'].index(ca_i_systole)
        peak_time_ca = output['environment.time'][peak_index_ca]
        ca_sr_systole = min(output['ca_conc_sr.Ca_SR'])
        ca_time_to_peak  = peak_time_ca - stim_time

        # Decay measurements
        ca_amplitude = ca_i_systole - ca_i_diastole
        decay50 = ca_amplitude * 0.5 + ca_i_diastole
        decay90 = ca_amplitude * 0.1 + ca_i_diastole
        ca_decays = [decay50, decay90]
        cat_vals = []

        ca_curr = ca_i_systole
        index = peak_index_ca
        for decay in ca_decays:
            while ca_curr > decay:
                index += 1
                ca_curr = output['ca_conc.Ca_i'][index]
            cat_vals.append(output['environment.time'][index] - stim_time)

    except Exception as e:
        print "Exception! " + str(e)
        continue

    cell_results = []
    print "Resting potential: " + str(v_rp) + " mV"
    cell_results.append(v_rp)
    print "Action potential amplitude: " + str(APA) + " mV"
    cell_results.append(APA)
    print "Max upstroke: " + str(dvdt_max) + " mV/ms"
    cell_results.append(dvdt_max)
    print "APD measurements: " + str(apd_vals) + " ms"
    for apd in apd_vals:
        cell_results.append(apd)
    print "Ca2+ conc in diastole: " + str(ca_i_diastole) + " uM"
    cell_results.append(ca_i_diastole)
    print "Ca2+ conc in systole: " + str(ca_i_systole) + " uM"
    cell_results.append(ca_i_systole)
    print "Ca2+ time to peak: " + str(ca_time_to_peak) + " ms"
    cell_results.append(ca_time_to_peak)
    print "Ca2+ decay measurements: " + str(cat_vals) + " ms"
    for cat in cat_vals:
        cell_results.append(cat)

    # Add to population results
    pop_results.append(cell_results)

    # Add trace to output
    traces.append(output)

out_file = open('final_results.pkl', 'wb')
pickle.dump(pop_results, out_file, -1)
out_file.close()

out_file2 = open('final_traces.pkl', 'wb')
pickle.dump(traces, out_file2, -1)
out_file2.close()

# Output average and SD values
pop_results = np.array(pop_results)
print "Successful simulations: " + str(pop_results.shape[0]) + "/" + str(N)

print "Mean values: "
print str(np.mean(pop_results,0))

print "Std deviations: "
print str(np.std(pop_results,0))
