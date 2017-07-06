import matplotlib.pyplot as plt
import ast
import os
import numpy as np
import math
import pickle

import myokit

import channel_setup

# Import channel
channel = channel_setup.icat()

# Check whether to store sim results
plot_data_name = 'plotting/plotdata_'+channel.name
overwrite = True
if os.path.isfile(plot_data_name):
    overwrite = 'Y' == raw_input('Plot data already exists. Overwrite? (Y/N)')

if not overwrite:
    with open(plot_data_name, 'rb') as f:
        loaded_data = pickle.load(f)
        sim_original, sim_ABC = loaded_data[0], loaded_data[1]
else:
    # Generate result with original parameters
    channel.setup_simulations(continuous=True)
    sim_original = channel.simulate()

    # Open results from ABC simulation
    with open('results/results_' + channel.name + '.txt') as f:
        pool = f.readline()
        pool = ast.literal_eval(pool)

    # Fill with simulations from final distribution pool
    sim_ABC = []
    for i in range(len(channel.simulations)):
        sim_ABC.append(np.array([]))

    # Generate output for each parameter in pool
    sim_vals_all = []
    for params in pool:
        channel.reset_params(params)
        sim_vals = channel.simulate()
        if sim_vals is not None:
            # sim_vals_all.append(channel.simulate())
            for i in range(len(sim_vals)):
                sim_reshaped = np.reshape(sim_vals[i], (len(sim_vals[i]), 1))
                sim_ABC[i] = np.hstack([sim_ABC[i], sim_reshaped]) if sim_ABC[i].size else sim_reshaped

    # Save plotting data to avoid further simulations
    with open(plot_data_name, 'wb') as f:
        pickle.dump([sim_original, sim_ABC], f)

# Calculate summary statistics
sim_ABC_mu = []
sim_ABC_sd = []
for i in range(len(sim_ABC)):
    sim_ABC_mu.append(np.mean(sim_ABC[i],axis=1))
    sim_ABC_sd.append(np.std(sim_ABC[i],axis=1))


# Plot the results
plt.style.use('seaborn-colorblind')
fig, ax = plt.subplots(nrows=1, ncols=len(sim_ABC), figsize=(5*len(sim_ABC), 5))
for i in range(len(sim_ABC)):
    # x values for continuous simulations i.e. not experimental data
    x_cont1 = np.linspace(min(channel.data_exp[i][0]), max(channel.data_exp[i][0]),
                         len(sim_original[i]))
    x_cont2 = np.linspace(min(channel.data_exp[i][0]), max(channel.data_exp[i][0]),
                         len(sim_ABC_mu[i]))
    if len(sim_ABC) > 1:
        axi = ax[i]
    else:
        axi = ax
    axi.plot(x_cont2, sim_ABC_mu[i], '-', label='ABC simulations')
    # for s in sim_vals_all:
    #     axi.plot(x_cont2, s[i], '-', lw=0.1)
    axi.fill_between(x_cont2, sim_ABC_mu[i]-sim_ABC_sd[i], sim_ABC_mu[i]+sim_ABC_sd[i], alpha=0.25, lw=0)
    axi.plot(channel.data_exp[i][0], channel.data_exp[i][1], 'o', label='Experimental data')
    axi.plot(x_cont1, sim_original[i], '--', label=channel.publication)

if len(sim_ABC) > 1:
    ax[-1].legend(loc='lower right')
else:
    ax.legend(loc='lower right')
fig.show()
x = raw_input('...')
fig.savefig('results/fig_'+str(channel.name)+'.eps', bbox_inches="tight")
