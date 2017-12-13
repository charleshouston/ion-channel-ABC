import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import ast
import os
import numpy as np
import math
import pickle

import myokit
import channel_setup as cs
import distributions as Dist
from ion_channel_ABC import LossFunction

# Helper function for rounding to 3 sig figs adapted from:
#  https://stackoverflow.com/questions/3410976/how-to-round-a-number-to-significant-figures-in-python
round_to_n = lambda x, n: round(x, -int(math.floor(math.log10(abs(x)))) + (n - 1)) if x != 0 else 0

# Get parameters to process
results_path = raw_input("Enter path to results file: ")
assert os.path.exists(results_path), "File could not be found at " + str(results_path)

# Set path for processing output and grab results file name
output_dir, filename = os.path.split(str(results_path))
# Work out which channel from results filename
channel_code = (filename.split(".")[0]).split("_")[1]

# Import channel
channels = {'icat': cs.icat,
            'ina' : cs.ina2,
            'ik1' : cs.ik1,
            'iha' : cs.iha,
            'ical': cs.ical,
            'ikr' : cs.ikr,
            'ikur': cs.ikur,
            'ito' : cs.ito}
ch = channels[channel_code]()
m,_,_ = myokit.load('models/'+ch.model_name)

# Check whether to store sim results
if not os.path.isdir(output_dir+"/plotdata"):
    os.mkdir(output_dir+"/plotdata")
    
plot_data_name = output_dir+"/plotdata/plotdata_"+ch.name
overwrite = True
if os.path.isfile(plot_data_name):
    overwrite = 'y' == raw_input('Plot data already exists. Overwrite? (y/n)')

# Open results from ABC simulation
with open(results_path) as f:
    pool = f.readline()
    pool = ast.literal_eval(pool)
    weights = f.readline()
    weights = ast.literal_eval(weights)

ch.setup_simulations(continuous=True)
if not overwrite:
    with open(plot_data_name, 'rb') as f:
        loaded_data = pickle.load(f)
        sim_original, sim_ABC = loaded_data[0], loaded_data[1]
else:
    # Generate result with original parameters
    params_orig = [m.get(ch.name+'.'+p).value() for p in ch.parameter_names]
    ch.reset_params(params_orig)
    sim_original = ch.simulate()

    # Fill with simulations from final distribution pool
    sim_ABC = [[] for i in range(len(pool))]

    # Generate output for each parameter in pool
    sim_vals_all = []
    for i,params in enumerate(pool):
        ch.reset_params(params)
        sim_vals = ch.simulate()
        if sim_vals is not None:
            sim_vals = np.array(sim_vals)
            sim_ABC[i] = sim_vals

    # Save plotting data to avoid further simulations
    with open(plot_data_name, 'wb') as f:
        pickle.dump([sim_original, sim_ABC], f)

# Calculate summary statistics
weights = [w for i,w in enumerate(weights) if len(sim_ABC[i]) != 0]
sim_ABC = [s for s in sim_ABC if len(s) != 0]

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
    axi.plot(ch.simulations_x[i], sim_ABC_mu[i], '-', label='ABC posterior')
    axi.fill_between(ch.simulations_x[i], sim_ABC_mu[i]-sim_ABC_sd[i], sim_ABC_mu[i]+sim_ABC_sd[i], alpha=0.25, lw=0)

    axi.errorbar(ch.data_exp[i][0], ch.data_exp[i][1], yerr=ch.data_exp[i][2], fmt='o', label='experimental data')

    # set correct axis labels
    axi.set_xlabel(ch.setup_exp[i]['xlabel'])
    axi.set_ylabel(ch.setup_exp[i]['ylabel'])

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
if not os.path.isdir(output_dir+'/figures'):
    os.mkdir(output_dir+'/figures')
    
fig.savefig(output_dir + '/figures/fig_'+str(ch.name)+'.pdf', bbox_inches="tight")
plt.close(fig)

# Save summary statistics
#  Get original parameters
parameters = ch.parameters
m_temp, _, _ = myokit.load('models/'+ch.model_name)
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
err_original = LossFunction(sim_original, ch.data_exp)
# Loss of ABC values
err_ABC = []
for sim in sim_ABC:
    err_ABC.append(LossFunction(sim, ch.data_exp))

sig_figs = 3
if not os.path.isdir(output_dir+'/summary_data'):
    os.mkdir(output_dir+'/summary_data')

with open(output_dir+'/summary_data/summary_'+ch.name+'.csv', 'w') as f:
    f.write('parameter,reported value,mean,variance,median,interquartile range\n')
    for i,p in enumerate(ch.parameter_names):
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
