import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

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
channel = channel_setup.icat()
m,_,_ = myokit.load('models/'+channel.model_name)

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
    if channel.name == 'ina':
        remove = [10,57,78,92,150,186,218,220,253,279,323,380]
        pool = [i for j,i in enumerate(pool) if j not in remove]
        weights = [i for j,i in enumerate(weights) if j not in remove]

if not overwrite:
    with open(plot_data_name, 'rb') as f:
        loaded_data = pickle.load(f)
        sim_original, sim_ABC = loaded_data[0], loaded_data[1]
else:
    # Generate result with original parameters
    channel.setup_simulations()
    sim_original = channel.simulate()
    params_orig = [m.get(channel.name+'.'+p).value() for p in channel.parameter_names]

    # Fill with simulations from final distribution pool
    sim_ABC = [[] for i in range(len(pool))]

    # Generate output for each parameter in pool
    sim_vals_all = []
    for i,params in enumerate(pool):
        channel.reset_params(params)
        sim_vals = channel.simulate()
        if sim_vals is not None:
            sim_ABC[i] = sim_vals

    # Save plotting data to avoid further simulations
    with open(plot_data_name, 'wb') as f:
        pickle.dump([sim_original, sim_ABC], f)

# Calculate summary statistics
sim_ABC = np.array(sim_ABC)
sim_ABC_reshape = sim_ABC[:,:,1].swapaxes(0,1)
sim_ABC_mu = []
sim_ABC_sd = []
sim_ABC_med = []
sim_ABC_iqr = []

# Create a distribution for each experiment
for exper in sim_ABC_reshape:
    d = Dist.Arbitrary(exper, weights)
    sim_ABC_mu.append(d.getmean())
    sim_ABC_sd.append(np.sqrt(d.getvar()))
    sim_ABC_med.append(d.getmedian())
    sim_ABC_iqr.append(np.array(d.getiqr())/2)

# Plot the results
plt.style.use('seaborn-colorblind')

ncols = len(sim_ABC_reshape)
fig, ax = plt.subplots(nrows=1, ncols=ncols, figsize=(3*ncols, 2.8))

for i in range(ncols):
    if ncols > 1:
        axi = ax[i]
    else:
        axi = ax
    axi.plot(channel.simulations_x[i], sim_ABC_mu[i], '-', label='ABC posterior')
    axi.fill_between(channel.simulations_x[i], sim_ABC_mu[i]-sim_ABC_sd[i], sim_ABC_mu[i]+sim_ABC_sd[i], alpha=0.25, lw=0)

    axi.plot(channel.data_exp[i][0], channel.data_exp[i][1], 'o', label='experimental data')
    axi.plot(sim_original[i][0], sim_original[i][1], '--', label='original formulation')

    # set correct axis labels
    axi.set_xlabel(channel.setup_exp[i]['xlabel'])
    axi.set_ylabel(channel.setup_exp[i]['ylabel'])

    # Use legend from last plot
    if i == ncols-1:
        handles, labels = axi.get_legend_handles_labels()
        lgd = fig.legend(handles, labels, loc='lower center', ncol=3)
        bb = lgd.get_bbox_to_anchor().inverse_transformed(fig.transFigure)
        bb.y0 -= 0.1
        lgd.set_bbox_to_anchor(bb, transform=fig.transFigure)

uppercase_letters = map(chr, range(65,91))
if ncols > 1:
    for i,a in enumerate(ax.flatten()):
        a.text(-0.07, -0.06,
                uppercase_letters[i], transform=a.transAxes,
                fontsize=16, fontweight='bold', va='top', ha='right')
plt.tight_layout()
fig.savefig('results/fig_'+str(channel.name)+'.pdf', bbox_inches="tight")
plt.close(fig)

# Colormap using seaborn-colorblind first colour with transparency
# cmap = np.array([[0.0, 114/256.0, 178/256.0, i] for i in np.linspace(0, 1, 256)])
# cmap = ListedColormap(cmap)

# for i in range(len(sim_ABC_params)):
#     if len(sim_ABC) > 1:
#         ax1 = ax[i][0]
#         ax2 = ax[i][1]
#     else:
#         ax1 = ax[0]
#         ax2 = ax[1]

#     for j in range(len(sim_ABC_params[i])):
#         normalised_weight = (weights[j]-min(weights))/(max(weights) - min(weights))
#         ax2.plot(sim_ABC_params[i][j][0], sim_ABC_params[i][j][1], linestyle=':', c=cmap(normalised_weight), lw=0.5)

#     ax1.plot(channel.simulations_x[i], sim_ABC_mu[i], '-', label='ABC simulations')
#     ax1.fill_between(channel.simulations_x[i], sim_ABC_mu[i]-sim_ABC_sd[i], sim_ABC_mu[i]+sim_ABC_sd[i], alpha=0.25, lw=0)

    # Uncomment below for median and IQR
    # ax1.plot(x_cont2, sim_ABC_med[i], '-', label='ABC simulations')
    # ax1.fill_between(x_cont2, sim_ABC_med[i]-sim_ABC_iqr[i], sim_ABC_med[i]+sim_ABC_iqr[i], alpha=0.25, lw=0)

    # ax1.plot(channel.data_exp[i][0], channel.data_exp[i][1], 'o', label='Experimental data')
    # ax1.plot(sim_original[i][0], sim_original[i][1], '--', label=channel.publication)
    # ax1.set_xlabel(channel.setup_exp[i]['xlabel'])
    # ax2.set_xlabel(channel.setup_exp[i]['xlabel'])
    # ax1.set_ylabel(channel.setup_exp[i]['ylabel'])

# if len(sim_ABC) > 1:
#     ax[int(len(sim_ABC_params))-1][0].legend(loc='best')
# else:
#     ax[0].legend(loc='best')

# uppercase_letters = map(chr, range(65,91))
# for i,a in enumerate(ax.flatten()):
#     a.text(-0.01, -0.08,
#            uppercase_letters[i], transform=a.transAxes,
#            fontsize=16, fontweight='bold', va='top', ha='right')
# plt.tight_layout()
# fig.savefig('results/fig_'+str(channel.name)+'.pdf', bbox_inches="tight")

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

# Loss of original values
err_original = LossFunction(sim_original, channel.data_exp)
# Loss of ABC values
err_ABC = []
for sim in sim_ABC:
    err_ABC.append(LossFunction(sim, channel.data_exp))

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
    err_ABC = [err for err in err_ABC if not math.isnan(err)]
    f.write(str(round_to_n(min(err_ABC), sig_figs)) + ',' + \
                str(round_to_n(max(err_ABC), sig_figs)))
