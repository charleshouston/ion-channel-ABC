import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import ast
import numpy as np

import myokit
import channel_setup as cs
import distributions as Dist

plt.style.use('seaborn-colorblind')

# Setup
channel = cs.icat()
v = np.arange(-100, 30, 0.01)
consts_ss = ['dss', 'fss']
consts_time = ['tau_d', 'tau_f']

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

# store mean and sd
dists_ss = [[] for i in range(2)] 
dists_time = [[] for i in range(2)]
for i in range(len(consts_ss)):
    dist = Dist.Arbitrary(ss_ABC[i], weights)
    const_mean = dist.getmean()
    const_sd = np.sqrt(dist.getvar())
    dists_ss[0].append(const_mean)
    dists_ss[1].append(const_sd)
for i in range(len(consts_time)):
    dist = Dist.Arbitrary(time_ABC[i], weights)
    const_mean = dist.getmean()
    const_sd = np.sqrt(dist.getvar())
    dists_time[0].append(const_mean)
    dists_time[1].append(const_sd)

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(9, 2.8))

for i in range(len(consts_ss)):
    ax[0].plot(v, dists_ss[0][i], label=consts_ss[i])
    ax[0].fill_between(v, dists_ss[0][i]-dists_ss[1][i], dists_ss[0][i]+dists_ss[1][i], alpha=0.25, lw=0)
    # ax[0].plot(v, ss_orig[i], ls='--', color='#D55E00')
ax[0].set_ylim([0,None])
ax[0].set_xlabel('Voltage (mV)')
ax[0].legend()

# for i in range(len(consts_time)):
ax[1].plot(v, dists_time[0][0], label=consts_time[0])
ax[1].fill_between(v, dists_time[0][0]-dists_time[1][0], dists_time[0][0]+dists_time[1][0], alpha=0.25, lw=0)
# ax[1].plot(v, time_orig[0], ls='--', color='#D55E00')
ax[1].set_ylim([0,None])
ax[1].set_xlabel('Voltage (mV)')
ax[1].legend()

ax[2].plot(v, dists_time[0][1], label=consts_time[1], color='#009E73')
ax[2].fill_between(v, dists_time[0][1]-dists_time[1][1], dists_time[0][1]+dists_time[1][1], alpha=0.25, lw=0, color='#009E73')
# ax[2].plot(v, time_orig[1], ls='--', color='#D55E00')
ax[2].set_ylim([0,None])
ax[2].set_xlabel('Voltage (mV)')
ax[2].legend()


uppercase_letters = map(chr, range(65,91))
for i,a in enumerate(ax.flatten()):
    a.text(-0.07, -0.06,
            uppercase_letters[i], transform=a.transAxes,
            fontsize=16, fontweight='bold', va='top', ha='right')
plt.tight_layout()
fig.savefig('results/consts_'+str(channel.name)+'.pdf', bbox_inches="tight")
plt.close(fig)

# Plot covariance between parameters
import pandas as pd
from pandas.plotting import scatter_matrix

high_var_params = ['k_ftau1','k_ftau2','k_ftau3','k_ftau4','k_ftau5','k_ftau6']
df = pd.DataFrame(pool, columns = channel.parameter_names)
param_dists = np.swapaxes(pool, 0, 1)

nparams = len(high_var_params)
sm = scatter_matrix(df[high_var_params], alpha=0.2, figsize=(nparams*1.1,nparams*1.1), diagonal='hist', hist_kwds={'bins':8, 'weights':weights})
# Change label rotation
# [s.xaxis.set_visible(False) for s in sm.reshape(-1)]
# [s.xaxis.label.set_rotation(45) for s in sm.reshape(-1)]
# [s.yaxis.label.set_rotation() for s in sm.reshape(-1)]
# [s.get_yaxis().set_label_coords(-1.2,0.4) for s in sm.reshape(-1)]
# [plt.setp(item.xaxis.get_label(), 'size', 14) for item in sm.reshape(-1)]
# [plt.setp(item.yaxis.get_label(), 'size', 14) for item in sm.reshape(-1)]

# Hide all ticks
# [s.set_xticks(()) for s in sm.reshape(-1)]
# [s.set_yticks(()) for s in sm.reshape(-1)]
plt.tight_layout()
plt.savefig('results/cov_'+str(channel.name)+'.pdf', bbox_inches="tight")
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
fig.savefig('results/corr_'+str(channel.name)+'.pdf', bbox_inches='tight')
plt.close(fig)

