import matplotlib.pyplot as plt
import ast
import numpy as np
import math

import myokit

import channel_setup

# Import channel
channel = channel_setup.TTypeCalcium()
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
plt.figure
plt.suptitle('Voltage clamp simulations for ' + str(channel.name) + ' in HL-1')
for i in range(len(sim_ABC)):
    plt.subplot(1,len(sim_ABC),i+1)
    plt.errorbar(channel.data_exp[i][0], sim_ABC_mu[i], yerr=sim_ABC_sd[i],
                 marker='o', color='b', ls='None')
    plt.plot(channel.data_exp[i][0], channel.data_exp[i][1], 'rx')
    plt.plot(channel.data_exp[i][0], sim_original[i], 'gs')

plt.show()
