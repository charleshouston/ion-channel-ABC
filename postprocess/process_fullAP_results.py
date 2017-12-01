####################################################################################
# process_fullAP_results.py                                                        #
# Process the output from multiple stochastic simulations from ABC and PSO results #
# to produce summary figures of behaviour of the cell model.                       #
#                                                                                  #
# Author: C Houston                                                                #
# Date Last Edit: 1/12/17                                                          #
####################################################################################

import matplotlib.pyplot as plt

import pickle
import numpy as np
import math

import distributions as Dist

#######################
# Loading all results #
#######################

print "Loading traces..."
filename = 'final_traces.pkl'
all_results = pickle.load(open(filename,'r'))
time = all_results[0]['environment.time']

def get_label(s):
    return s.split('.')[1]

print "Extracting variables of interest..."
var_names = ['membrane.V', 'ca_conc.Ca_i']
all_sims = []
for sim in all_results:
    sim_vals = [[] for i in range(len(var_names))]
    for i, key in enumerate(var_names):
        sim_vals[i] = sim[key]
    all_sims.append(sim_vals)

############################################
# Code for loading in pre-processed traces #
#  in which case comment out code above.   #
############################################

# reduced_filename = 'reduced_sims.pkl'
# all_sims = pickle.load(open(reduced_filename,'r'))
# time = all_sims['environment.time']

# Convert to numpy array
all_sims = np.array(all_sims)

# Swap axes so first axis is variable
all_sims = all_sims.swapaxes(0,1)

print "Generating distributions..."
sim_dists = []
for var in all_sims:
    d = Dist.Arbitrary(var)
    sim_dists.append(d)

# Setup for plotting
plt.style.use('seaborn-colorblind')
colors = ['#0072B2',  '#D55E00', '#D55E00', '#D55E00','#009E73', '#D55E00','#D55E00','#D55E00','#D55E00','#D55E00','#D55E00','#D55E00']
fig, ax = plt.subplots(nrows = 2, ncols = 1, figsize = (2.5,6), sharex=True)
ax = ax.flatten()

for i, a in enumerate(ax):
    a.plot(time, sim_dists[i].getmean(),color=colors[i])
    a.fill_between(time, sim_dists[i].getmean() - np.sqrt(sim_dists[i].getvar()), sim_dists[i].getmean() + np.sqrt(sim_dists[i].getvar()), color=colors[i], alpha=0.25, lw=0)
    a.annotate(var_names[i],xy=(1,1),xycoords='axes fraction',fontsize=14, xytext = (-5,-8), textcoords = 'offset points',ha='right',va='top')
    if i==0:
        a.set_ylabel('Voltage (mV)')
    elif i==1:
        a.set_ylabel('Ca2+ conc. (uM)')
        a.set_xlabel('Time (ms)')

plt.tight_layout()
fig.savefig('AP-uncert.pdf',bbox_inches='tight')
