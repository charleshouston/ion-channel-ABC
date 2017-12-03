#############################################################
# parameter_sensitivity.py                                  #
# Run parameter sensitivity study on stochastic cell model. #
#                                                           #
# Author: C Houston                                         #
# Date Last Edit: 3/12/17                                   #
#############################################################

import myokit
import channel_setup as cs
import distributions as Dist
import numpy as np
import ast # for literal_eval
import matplotlib.pyplot as plt

def set_ABC_channels(s):

    # ABC channel list
    channels = [cs.icat(),
                cs.iha(),
                cs.ikur(),
                cs.ikr(),
                cs.ina2(),
                cs.ito(),
                cs.ical(),
                cs.ik1()]

    # Load results from ABC simulation for each channel
    results = []
    for ch in channels:
        results_file = 'results/updated-model/results_' + ch.name + '.txt'
        f = open(results_file, 'r')
        lines = f.readlines()
        particles = ast.literal_eval(lines[0])
        weights = ast.literal_eval(lines[1])
        dists = Dist.Arbitrary(particles, weights)
        results.append(dists)

    # Update ABC channels to draw from distributions
    for k, c in enumerate(channels):
        params = results[k].draw()
        for j, p in enumerate(c.parameters):
            s.set_constant(c.parameters[j], params[j])

    return s

# Load model
m,_,_ = myokit.load('models/Houston2017b.mmt')

# Create normal simulation to find steady-state
s = myokit.Simulation(m)
s.set_max_step_size(0.001)

# Set ABC channel parameters
s = set_ABC_channels(s)

# Run to steady state (hopefully!)
print('Running to steady state...')
s.pre(10000)

# Get current simulation state for sensitivity start point
#  and update model before feeding to PSimulation.
state = s.state()
m.set_state(state)

# Create simulation for sensitivity study
print('Running parameter sensitivity...')
ps = myokit.PSimulation(m,
                       protocol=None,
                       variables=['membrane.V',
                                  'ca_conc.Ca_i',
                                  'ca_conc_sr.Ca_SRrelease',
                                  'ca_conc_sr.Ca_SRuptake'],
                       parameters=['inak.i_NaK_max',
                                   'incx.k_NCX',
                                   'icab.g_Cab',
                                   'inab.g_Nab',
                                   'ryanodine_receptors.k_RyR',
                                   'serca.V_max',
                                   'leak_flux.k_leak']
                        )

ps = set_ABC_channels(ps)

# Set state to steady-state from before
ps.set_step_size(0.001)

# Run parameter sensitivity
d = ps.run(10)

# Plot derivatives with time
fig_V, ax_V = plt.subplots()
ax_V.plot(d[0]['environment.time'], d[1][:,0,0], label='i_NaK_max')
ax_V.plot(d[0]['environment.time'], d[1][:,0,1], label='k_NCX')
ax_V.plot(d[0]['environment.time'], d[1][:,0,2], label='g_Cab')
ax_V.plot(d[0]['environment.time'], d[1][:,0,3], label='g_Nab')
ax_V.plot(d[0]['environment.time'], d[1][:,0,4], label='k_RyR')
ax_V.plot(d[0]['environment.time'], d[1][:,0,5], label='V_max')
ax_V.plot(d[0]['environment.time'], d[1][:,0,6], label='k_leak')
ax_V.legend()

fig_Cai, ax_Cai = plt.subplots()
ax_Cai.plot(d[0]['environment.time'], d[1][:,1,0], label='i_NaK_max')
ax_Cai.plot(d[0]['environment.time'], d[1][:,1,1], label='k_NCX')
ax_Cai.plot(d[0]['environment.time'], d[1][:,1,2], label='g_Cab')
ax_Cai.plot(d[0]['environment.time'], d[1][:,1,3], label='g_Nab')
ax_Cai.plot(d[0]['environment.time'], d[1][:,1,4], label='k_RyR')
ax_Cai.plot(d[0]['environment.time'], d[1][:,1,5], label='V_max')
ax_Cai.plot(d[0]['environment.time'], d[1][:,1,6], label='k_leak')
ax_Cai.legend()

fig_V.show()
fig_Cai.show()
import pdb;pdb.set_trace()
