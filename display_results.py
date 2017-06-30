import matplotlib.pyplot as plt
import ast
import numpy as np
import math

import myokit

import channel_setup

# Import channel
channel = channel_setup.RapidlyActivatingDelayedPotassium()
sim_original = channel.simulate()

# Open results from ABC simulation
f = open('results/results_' + str(channel.name) + '.txt')
pool = f.readline()
pool = ast.literal_eval(pool)

# Fill with simulations from final distribution pool
sim_ABC = []
for i in range(len(sim_original)):
    sim_ABC.append(np.array([]))

# Generate output for each parameter in pool
for params in pool:
    channel.reset_params(params)
    sim_vals = channel.simulate()
    for i in range(len(sim_vals)):
        sim_reshaped = np.reshape(sim_vals[i], (len(sim_vals[i]), 1))
        sim_ABC[i] = np.hstack([sim_ABC[i], sim_reshaped]) if sim_ABC[i].size else sim_reshaped

# Calculate summary statistics
sim_ABC_mu = []
sim_ABC_sd = []
for i in range(len(sim_ABC)):
    sim_ABC_mu.append(np.mean(sim_ABC[i],axis=1))
    sim_ABC_sd.append(np.std(sim_ABC[i],axis=1))

# Plot the results
fig, ax = plt.subplots(nrows=1, ncols=len(sim_ABC))
fig.suptitle('Voltage clamp simulations for ' + str(channel.name) + ' in HL-1')
for i in range(len(sim_ABC)):
    ax[i].errorbar(channel.data_exp[i][0], sim_ABC_mu[i], yerr=sim_ABC_sd[i],
                 marker='o', color='b', ls='None', label='Simulation')
    ax[i].plot(channel.data_exp[i][0], channel.data_exp[i][1], 'rx', label='Experiment')
    ax[i].plot(channel.data_exp[i][0], sim_original[i], 'gs', label='Published model')
ax[-1].legend(loc='lower right')

fig.show()
input('Please press Enter to continue...')
