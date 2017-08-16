import matplotlib.pyplot as plt

import pickle
import numpy as np
import math

import distributions as Dist

filename = 'AP_traces_final'
all_results = pickle.load(open(filename,'r'))
time = all_results[0]['environment.time']
def get_label(s):
    return s.split('.')[1]

all_sims = []
for sim in all_results:
    sim_vals = [[] for i in range(len(sim.keys()))]
    for i, key in enumerate(sim.keys()):
        sim_vals[i] = sim[key]
    all_sims.append(sim_vals)
all_sims = np.array(all_sims)

# Swap axes so first axis is variable
all_sims = all_sims.swapaxes(0,1)
sim_dists = []
for var in all_sims:
    d = Dist.Arbitrary(var)
    sim_dists.append(d)

plt.style.use('seaborn-colorblind')
colors = ['#0072B2',  '#D55E00', '#D55E00', '#D55E00','#009E73', '#D55E00','#D55E00','#D55E00','#D55E00','#D55E00','#D55E00','#D55E00']
fig, ax = plt.subplots(nrows = 3, ncols = 4, figsize = (11,5), sharex=True)
ax = ax.flatten()
var_indices = [1,3,5,7,2,4,6,9,8,10,12,13]
var_names = [get_label(all_results[0].keys()[i]) for i in var_indices]

for i, a in enumerate(ax):
    index = var_indices[i]
    a.plot(time, sim_dists[index].getmean(),color=colors[i])
    a.fill_between(time, sim_dists[index].getmean() - np.sqrt(sim_dists[index].getvar()), sim_dists[index].getmean() + np.sqrt(sim_dists[index].getvar()), color=colors[i], alpha=0.25, lw=0)
    a.annotate(var_names[i],xy=(1,1),xycoords='axes fraction',fontsize=12, xytext = (-5,-8), textcoords = 'offset points',ha='right',va='top')
    if i==0:
        a.set_ylabel('Voltage (mV)')
    elif i==4:
        a.set_ylabel('Ca2+ conc. (uM)')
    elif i > 7:
        a.set_xlabel('Time (ms)')
        a.set_ylabel('Current (pA/pF)')
    else:
        a.set_ylabel('Current (pA/pF)')

plt.tight_layout()
fig.savefig('final_results/AP-uncert.pdf',bbox_inches='tight')
