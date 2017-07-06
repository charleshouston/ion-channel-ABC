import matplotlib.pyplot as plt
import ast
import os
import numpy as np
import math
import pickle

import myokit

import channel_setup
import distributions as Dist

# Import channel
channel = channel_setup.icat()

# Check whether to store sim results
plot_data_name = 'plotting/plotdata_'+channel.name
overwrite = True
if os.path.isfile(plot_data_name):
    overwrite = 'y' == raw_input('Plot data already exists. Overwrite? (y/n)')

# Open results from ABC simulation
with open('results/results_' + channel.name + '.txt') as f:
    pool = f.readline()
    pool = ast.literal_eval(pool)
    weights = f.readline()
    weights = ast.literal_eval(weights)

if not overwrite:
    with open(plot_data_name, 'rb') as f:
        loaded_data = pickle.load(f)
        sim_original, sim_ABC = loaded_data[0], loaded_data[1]
else:
    # Generate result with original parameters
    channel.setup_simulations(continuous=True)
    sim_original = channel.simulate()

    # Fill with simulations from final distribution pool
    sim_ABC = []
    for i in range(len(channel.simulations)):
        sim_ABC.append([])

    # Generate output for each parameter in pool
    sim_vals_all = []
    for params in pool:
        channel.reset_params(params)
        sim_vals = channel.simulate()
        if sim_vals is not None:
            for i in range(len(sim_vals)):
                sim_ABC[i].append(sim_vals[i])

    # Save plotting data to avoid further simulations
    with open(plot_data_name, 'wb') as f:
        pickle.dump([sim_original, sim_ABC], f)

# Calculate summary statistics
sim_ABC_mu = []
sim_ABC_sd = []
for i in range(len(sim_ABC)):
    d = Dist.Arbitrary(sim_ABC[i], weights)
    sim_ABC_mu.append(d.getmean())
    sim_ABC_sd.append(np.sqrt(d.getvar()))

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
    axi.fill_between(x_cont2, sim_ABC_mu[i]-sim_ABC_sd[i], sim_ABC_mu[i]+sim_ABC_sd[i], alpha=0.25, lw=0)
    axi.plot(channel.data_exp[i][0], channel.data_exp[i][1], 'o', label='Experimental data')
    axi.plot(x_cont1, sim_original[i], '--', label=channel.publication)
    axi.set_xlabel(channel.setup_exp[i]['xlabel'])
    axi.set_ylabel(channel.setup_exp[i]['ylabel'])

if len(sim_ABC) > 1:
    ax[-1].legend(loc='best')
else:
    ax.legend(loc='best')

fig.savefig('results/fig_'+str(channel.name)+'.pdf', bbox_inches="tight")
