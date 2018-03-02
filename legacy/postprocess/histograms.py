#############################################################################
# histograms.py                                                             #
# Plot histograms of stochastic results and location of experimental value. #
#                                                                           #
# Author: C Houston                                                         #
# Date Last Edit: 3/12/17                                                   #
#############################################################################

import matplotlib.pyplot as plt
import pickle
import numpy as np

results_file = open('final_results.pkl','r')
results = pickle.load(results_file)
results = np.array(results)
results = results.swapaxes(0,1)
results = results[:,results[11,:]!=0.0] # filter out incorrect CaT results

exp_vals = [-67, 105, 42, 397]

plt.style.use('seaborn-colorblind')
fig,ax = plt.subplots(nrows=2, ncols=2, figsize=(6,6))
ax[0][0].hist(np.squeeze(results[0,:]), bins=20)
ax[0][0].set_xlabel('Resting potential (mV)')
ax[0][0].axvline(exp_vals[0], color='r', linestyle='dashed', lw=1)

ax[0][1].hist(np.squeeze(results[1,:]), bins=20)
ax[0][1].set_xlabel('AP amplitude (mV)')
ax[0][1].axvline(exp_vals[1], color='r', linestyle='dashed', lw=1)

ax[1][0].hist(np.squeeze(results[6,:]), bins=20)
ax[1][0].set_xlabel('APD at 90% rep. (ms)')
ax[1][0].axvline(exp_vals[2], color='r', linestyle='dashed', lw=1)

ax[1][1].hist(np.squeeze(results[11,:]), bins=20)
ax[1][1].set_xlabel('Ca2+ decay to 90% rep. (ms)')
ax[1][1].axvline(exp_vals[3], color='r', linestyle='dashed', lw=1)

plt.tight_layout()
fig.savefig('histograms.pdf',bbox_inches='tight',dpi=fig.dpi)
fig.show()
import pdb;pdb.set_trace()
