import matplotlib.pyplot as plt
import ast
import os
import numpy as np
import math
import pickle

import myokit

import channel_setup
import distributions as Dist
from ion_channel_ABC import LossFunction

# Helper function for rounding to 3 sig figs adapted from:
#  https://stackoverflow.com/questions/3410976/how-to-round-a-number-to-significant-figures-in-python
round_to_n = lambda x, n: round(x, -int(math.floor(math.log10(abs(x)))) + (n - 1)) if x != 0 else 0

# Import channel
channel = channel_setup.ikur()

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
sim_ABC_med = []
sim_ABC_iqr = []
for i in range(len(sim_ABC)):
    d = Dist.Arbitrary(sim_ABC[i], weights)
    sim_ABC_mu.append(d.getmean())
    sim_ABC_sd.append(np.sqrt(d.getvar()))
    sim_ABC_med.append(d.getmedian())
    sim_ABC_iqr.append(np.array(d.getiqr())/2)

# Plot the results
plt.style.use('seaborn-colorblind')
fig, ax = plt.subplots(nrows=len(sim_ABC), ncols=2, figsize=(8, 3*len(sim_ABC)), sharey='row')
for i in range(len(sim_ABC)):
    # x values for continuous simulations i.e. not experimental data
    x_cont1 = np.linspace(min(channel.data_exp[i][0]), max(channel.data_exp[i][0]),
                         len(sim_original[i]))
    x_cont2 = np.linspace(min(channel.data_exp[i][0]), max(channel.data_exp[i][0]),
                         len(sim_ABC_mu[i]))
    if len(sim_ABC) > 1:
        ax1 = ax[i][0]
        ax2 = ax[i][1]
    else:
        ax1 = ax[0]
        ax2 = ax[1]

    for j in range(len(sim_ABC[i])):
        ax2.plot(x_cont2, sim_ABC[i][j], linestyle=':', color='#0072B2', lw=0.5, alpha = (weights[j]-min(weights))/(max(weights) - min(weights)))

    ax1.plot(x_cont2, sim_ABC_mu[i], '-', label='ABC simulations')
    ax1.fill_between(x_cont2, sim_ABC_mu[i]-sim_ABC_sd[i], sim_ABC_mu[i]+sim_ABC_sd[i], alpha=0.25, lw=0)

    # Uncomment below for median and IQR
    # axi.plot(x_cont2, sim_ABC_med[i], '-', label='ABC simulations')
    # axi.fill_between(x_cont2, sim_ABC_med[i]-sim_ABC_iqr[i], sim_ABC_med[i]+sim_ABC_iqr[i], alpha=0.25, lw=0)

    ax1.plot(channel.data_exp[i][0], channel.data_exp[i][1], 'o', label='Experimental data')
    ax1.plot(x_cont1, sim_original[i], '--', label=channel.publication)
    ax1.set_xlabel(channel.setup_exp[i]['xlabel'])
    ax2.set_xlabel(channel.setup_exp[i]['xlabel'])
    ax1.set_ylabel(channel.setup_exp[i]['ylabel'])
if len(sim_ABC) > 1:
    ax[int(len(sim_ABC))-1][0].legend(loc='best')
else:
    ax[0].legend(loc='best')
plt.tight_layout()
fig.savefig('results/fig_'+str(channel.name)+'.pdf', bbox_inches="tight")

# Save summary statistics
#  Get original parameters
parameters = channel.parameters
m_temp, _, _ = myokit.load('models/'+channel.model_name)
reported_vals = []
for p in parameters:
    reported_vals.append(m_temp.get(p).value())

# Calculate summary of each parameters
d = Dist.Arbitrary(pool, weights)
mean_vals = d.getmean()
var_vals = d.getvar()
median_vals = d.getmedian()
iqr_vals = d.getiqr()

# Run coarse simulations to calculate errors to experimental values
channel.setup_simulations(continuous=False)
channel.reset_params(reported_vals)
# Loss of original values
err_original = LossFunction(channel.simulate(), channel.data_exp)
# Loss of ABC values
err_ABC = []
for params in pool:
    channel.reset_params(params)
    err_ABC.append(LossFunction(channel.simulate(), channel.data_exp))

sig_figs = 3
with open('results/summary_'+channel.name+'.csv', 'w') as f:
    f.write('parameter,reported value,mean,variance,median,interquartile range\n')
    for i,p in enumerate(channel.parameter_names):
        f.write(p + ',')
        f.write(str(round_to_n(reported_vals[i], sig_figs)) + ',')
        f.write(str(round_to_n(mean_vals[i], sig_figs)) + ',')
        f.write(str(round_to_n(var_vals[i], sig_figs)) + ',')
        f.write(str(round_to_n(median_vals[i], sig_figs)) + ',')
        f.write(str(round_to_n(iqr_vals[i], sig_figs)))
        f.write('\n')

    f.write('original loss\n')
    f.write(str(round_to_n(err_original, sig_figs))+'\n')
    f.write('minimum posterior loss,maximum posterior loss\n')
    f.write(str(round_to_n(min(err_ABC), sig_figs)) + ',' + \
                str(round_to_n(max(err_ABC), sig_figs)))

# Plotting histograms of parameters
num_of_rows = int(math.ceil(len(parameters) / 3.0))
fig, ax = plt.subplots(nrows=num_of_rows, ncols=3, figsize = (12, num_of_rows*2))
param_dists = np.swapaxes(pool, 0, 1)
for i,d in enumerate(param_dists):
    j = i % 3
    k = int(math.floor(i / 3.0))
    ax[k][j].hist(d, bins=20,weights=weights, range=(channel.prior_intervals[i][0],channel.prior_intervals[i][1]))
    ax[k][j].axvline(reported_vals[i], color='#009E73', linestyle='dashed', lw=2)
    ax[k][j].set_yticks([])
    ax[k][j].margins(x=0)
    ax[k][j].set_xlabel(channel.parameter_names[i])
plt.tight_layout()
fig.savefig('results/hist_'+str(channel.name)+'.pdf', bbox_inches="tight")

