import matplotlib.pyplot as plt
import ast
import numpy as np
import scipy.stats as st
import math

import myokit

import channel_setup

def ResetSim(s, params, channel):
    # Reset the model state before evaluating again
    s.reset()
    # Set parameters
    for i, p in enumerate(params):
        s.set_constant(channel.parameters[i], p)

channel = channel_setup.FastSodium()

# Get original channel parameters
m,p,x = myokit.load('models/'+channel.model_name)

# Setup simulation
v = m.get('membrane.V')
v.demote()
v.set_rhs(0)
v.set_binding('pace')
s = myokit.Simulation(m)

res_orig = channel.simulate(s)

# Open results from ABC simulation
f = open('results_' + str(channel.name) + '.txt')
pool = f.readline()
pool = ast.literal_eval(pool)

# Generate output for each parameter in pool
act_peaks_posterior = np.array([])
act_cond_posterior = np.array([])
inact_cond_posterior = np.array([])
rec_posterior = np.array([])
i = 0
for params in pool:
    ResetSim(s, params, channel)
    res = channel.simulate(s)
    if i == 0:
        act_peaks_posterior = np.reshape(res[0], (len(res[0]), 1))
        act_cond_posterior = np.reshape(res[1], (len(res[1]), 1))
        inact_cond_posterior = np.reshape(res[2], (len(res[2]), 1))
        rec_posterior = np.reshape(res[3], (len(res[3]), 1))
    else:
        act_peaks_posterior = np.hstack((act_peaks_posterior, np.reshape(res[0], (len(res[0]), 1))))
        act_cond_posterior = np.hstack((act_cond_posterior, np.reshape(res[1], (len(res[1]), 1))))
        inact_cond_posterior = np.hstack((inact_cond_posterior, np.reshape(res[2], (len(res[2]), 1))))
        rec_posterior = np.hstack((rec_posterior, np.reshape(res[3], (len(res[3]), 1))))
    i += 1

# Calculate summary statistics
act_peaks_mean = np.mean(act_peaks_posterior,axis=1)
act_cond_mean = np.mean(act_cond_posterior,axis=1)
inact_cond_mean = np.mean(inact_cond_posterior,axis=1)
rec_mean = np.mean(rec_posterior,axis=1)

act_peaks_std = np.std(act_peaks_posterior,axis=1)
act_cond_std = np.std(act_cond_posterior,axis=1)
inact_cond_std = np.std(inact_cond_posterior,axis=1)
rec_std = np.std(rec_posterior,axis=1)

# Separators for experimental data
sep1 = len(act_peaks_mean)
sep2 = sep1 + len(act_cond_mean)
sep3 = sep2 + len(inact_cond_mean)
sep4 = sep3 + len(rec_mean)

# Plot results
plt.figure()
plt.suptitle('Voltage clamp simulations for fast sodium channel in HL-1 myocytes')
plt.subplot(1,4,1)
plt.title('I-V curve')
plt.xlabel('Voltage (V)')
plt.ylabel('Peak current (pA/pF)')
plt.errorbar(channel.data_exp[0][0:sep1],act_peaks_mean,yerr=act_peaks_std, marker = 'o', color='b',ls='None')
plt.plot(channel.data_exp[0][0:sep1], channel.data_exp[1][0:sep1], 'rx')
plt.plot(channel.data_exp[0][0:sep1], res_orig[0], 'gs')

plt.subplot(1,4,2)
plt.errorbar(channel.data_exp[0][sep1:sep2],act_cond_mean,yerr=act_cond_std,marker = 'o', color='b',ls='None')
plt.title('Activation')
plt.xlabel('Voltage (V)')
plt.ylabel('Relative current')
plt.plot(channel.data_exp[0][sep1:sep2], channel.data_exp[1][sep1:sep2], 'rx')
plt.plot(channel.data_exp[0][sep1:sep2], res_orig[1], 'gs')


plt.subplot(1,4,3)
plt.errorbar(channel.data_exp[0][sep2:sep3],inact_cond_mean,yerr=inact_cond_std,marker = 'o', color='b',ls='None')
plt.title('Inactivation')
plt.xlabel('Voltage (V)')
plt.ylabel('Relative current')
plt.plot(channel.data_exp[0][sep2:sep3],channel.data_exp[1][sep2:sep3],'rx')
plt.plot(channel.data_exp[0][sep2:sep3], res_orig[2], 'gs')

plt.subplot(1,4,4)
plt.errorbar(channel.data_exp[0][sep3:sep4],rec_mean,yerr=rec_std,marker = 'o', color='b',ls='None',label='ABC Simulations')
plt.title('Recovery')
plt.xlabel('Voltage (V)')
plt.ylabel('Relative current')
plt.plot(channel.data_exp[0][sep3:sep4],channel.data_exp[1][sep3:sep4],'rx',label='Experimental')
plt.plot(channel.data_exp[0][sep3:sep4], res_orig[3], 'gs')
plt.legend(loc='lower right')
plt.show()
