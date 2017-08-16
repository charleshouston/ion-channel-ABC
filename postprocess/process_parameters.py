import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import ast
import numpy as np

import myokit
import channel_setup as cs
import distributions as Dist

plt.style.use('seaborn-colorblind')

# Setup
channel = cs.ikur()
v = np.arange(-100, 60, 0.01)
consts_ss = ['a_ur_ss','i_ur_ss']
consts_time = ['tau_a_ur','tau_i_ur']

parameters = channel.parameters
m, _, _ = myokit.load('models/'+channel.model_name)
reported_vals = [m.get(p).value() for p in parameters]

# Get results from ABC
with open('results/results_'+channel.name+'.txt') as f:
    pool = f.readline()
    pool = ast.literal_eval(pool)
    weights = f.readline()
    weights = ast.literal_eval(weights)

ss_orig = [m.get(channel.name+'.'+ss).pyfunc()(v) for ss in consts_ss]
time_orig = [m.get(channel.name+'.'+time).pyfunc()(v) for time in consts_time]

ss_ABC = [[] for i in range(len(consts_ss))]
time_ABC = [[] for i in range(len(consts_time))]
for params in pool:
    for i,p in enumerate(params):
        m.set_value(channel.name+'.'+channel.parameter_names[i], p)

    for i,ss in enumerate(consts_ss):
        ss_ABC[i].append(m.get(channel.name+'.'+ss).pyfunc()(v))
    for i,time in enumerate(consts_time):
        time_ABC[i].append(m.get(channel.name+'.'+time).pyfunc()(v))

colors = ['#0072B2', '#009E73', '#D55E00', '#CC79A7', '#F0E442', '#56B4E9']
colors_lighter = ['#33A5E5','#33D1A6']

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(9, 2.8))

for i in range(len(consts_ss)):
    for sim in ss_ABC[i]:
        ax[0].plot(v, sim, color=colors_lighter[i], ls=':', lw=0.2,alpha=0.4)
    ax[0].plot(v, ss_orig[i], color=colors[i], label=consts_ss[i])
ax[0].set_ylim([0,None])
ax[0].set_xlabel('Voltage (mV)')
ax[0].legend()

for i in range(len(consts_time)):
    for sim in time_ABC[i]:
        ax[i+1].plot(v, sim, color=colors_lighter[i], ls=':', lw=0.2,alpha=0.4)
    ax[i+1].plot(v, time_orig[i], color=colors[i],label=consts_time[i])
    ax[i+1].set_xlabel('Voltage (mV)')
    ax[i+1].legend()

ax[1].set_ylim([0, 500])
ax[2].set_ylim([0, None])

uppercase_letters = map(chr, range(65,91))
for i,a in enumerate(ax.flatten()):
    a.text(-0.07, -0.06,
            uppercase_letters[i], transform=a.transAxes,
            fontsize=16, fontweight='bold', va='top', ha='right')
plt.tight_layout()
fig.savefig('results_debug/consts_'+str(channel.name)+'.pdf', bbox_inches="tight")
plt.close(fig)

# Plot covariance between parameters
import pandas as pd
from pandas.plotting import scatter_matrix

high_var_params = ['k_ytau1','k_ytau2','k_ytau3','k_ytau4']
df = pd.DataFrame(pool, columns = channel.parameter_names)
param_dists = np.swapaxes(pool, 0, 1)

nparams = len(high_var_params)
sm = scatter_matrix(df[high_var_params], alpha=0.2, figsize=(nparams*1.1,nparams*1.1), diagonal='hist', hist_kwds={'bins':8, 'weights':weights})
plt.tight_layout()
plt.savefig('results_debug/cov_'+str(channel.name)+'.pdf', bbox_inches="tight")
plt.close()

# Also plot correlation matrix
fig, ax = plt.subplots()
corr = df.corr()
heatmap = ax.pcolor(corr, cmap=plt.cm.coolwarm, alpha=0.8, vmin=-1, vmax=1)
ax.set_aspect('equal')

fig = plt.gcf()
fig.set_size_inches(len(channel.parameter_names)/3, len(channel.parameter_names)/3)

row_labels = channel.parameter_names
column_labels = row_labels

# ax.set_frame_on(False)
ax.set_xticks(np.arange(corr.shape[0])+0.5, minor=False)
ax.set_yticks(np.arange(corr.shape[1])+0.5, minor=False)
ax.invert_yaxis()

fig.colorbar(heatmap, fraction=0.046, pad=0.04)

ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(column_labels, minor=False)

plt.xticks(rotation=90)
ax.grid(False)
ax = plt.gca()
for t in ax.xaxis.get_major_ticks():
    t.tick10n = False
    t.tick20n = False
for t in ax.yaxis.get_major_ticks():
    t.tick10n = False
    t.tick20n = False
plt.tight_layout()
fig.savefig('results_debug/corr_'+str(channel.name)+'.pdf', bbox_inches='tight')
plt.close(fig)
