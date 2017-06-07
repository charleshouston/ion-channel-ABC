import matplotlib.pyplot as plt
import ast
import numpy as np

import math

import myokit

import .. import simulations

import ..data.icat.data_icat as data_exp

def ResetSim(s, params):
    # Reset the model state before evaluating again
    s.reset()

    # Set d gate and f gate to current parameter values
    s.set_constant('icat_d_gate.dssk1',params[0])
    s.set_constant('icat_d_gate.dssk2',params[1])
    s.set_constant('icat_d_gate.dtauk1',params[2])
    s.set_constant('icat_d_gate.dtauk2',params[3])
    s.set_constant('icat_d_gate.dtauk3',params[4])
    s.set_constant('icat_d_gate.dtauk4',params[5])
    s.set_constant('icat_d_gate.dtauk5',params[6])
    s.set_constant('icat_d_gate.dtauk6',params[7])

    s.set_constant('icat_f_gate.fssk1',params[8])
    s.set_constant('icat_f_gate.fssk2',params[9])
    s.set_constant('icat_f_gate.ftauk1',params[10])
    s.set_constant('icat_f_gate.ftauk2',params[11])
    s.set_constant('icat_f_gate.ftauk3',params[12])
    s.set_constant('icat_f_gate.ftauk4',params[13])
    s.set_constant('icat_f_gate.ftauk5',params[14])
    s.set_constant('icat_f_gate.ftauk6',params[11])


# Load experimental data
# - IV data
vsteps,act_peaks_exp = data_exp.fig1B()
vsteps = np.array(vsteps)
act_peaks_exp = np.array(act_peaks_exp)
# - Activation/Inactivation data
vsteps_act,act_exp = data_exp.fig3Bact()
prepulses,inact_exp = data_exp.fig3Binact()
vsteps_act = np.array(vsteps_act)
act_exp = np.array(act_exp)
prepulses = np.array(prepulses)
inact_exp = np.array(inact_exp)
# - Recovery data
intervals,rec_exp = data_exp.fig4B()
intervals = np.array(intervals)
rec_exp = np.array(rec_exp)

# Open results from ABC
f = open('../results/ABCPredCalciumTType.txt','r')
pool = f.readline()
pool = ast.literal_eval(pool)

# Obtain original parameter point estimates
m,p,x = myokit.load('../models/Takeuchi2013_iCaT.mmt')

params_orig = []
params_orig.append(m.get('icat_d_gate.dssk1').value())
params_orig.append(m.get('icat_d_gate.dssk2').value())
params_orig.append(m.get('icat_d_gate.dtauk1').value())
params_orig.append(m.get('icat_d_gate.dtauk2').value())
params_orig.append(m.get('icat_d_gate.dtauk3').value())
params_orig.append(m.get('icat_d_gate.dtauk4').value())
params_orig.append(m.get('icat_d_gate.dtauk5').value())
params_orig.append(m.get('icat_d_gate.dtauk6').value())

params_orig.append(m.get('icat_f_gate.fssk1').value())
params_orig.append(m.get('icat_f_gate.fssk2').value())
params_orig.append(m.get('icat_f_gate.ftauk1').value())
params_orig.append(m.get('icat_f_gate.ftauk2').value())
params_orig.append(m.get('icat_f_gate.ftauk3').value())
params_orig.append(m.get('icat_f_gate.ftauk4').value())
params_orig.append(m.get('icat_f_gate.ftauk5').value())
params_orig.append(m.get('icat_f_gate.ftauk6').value())


# Simulation setup
reversal_potential = m.get('icat.E_CaT').value()
v = m.get('membrane.V')
v.demote()
v.set_rhs(0)
v.set_binding('pace')
s = myokit.Simulation(m)

# Empty arrays to hold simulation data
act_peaks_posterior = np.array([])
act_cond_posterior = np.array([])
inact_cond_posterior = np.array([])
rec_posterior = np.array([])

# Run simulations for each parameter set from the posterior pool
i = 0
for params in pool:

    ResetSim(s,params)
    act_pred = simulations.activation_sim(s,vsteps,reversal_potential)
    ResetSim(s,params)
    inact_pred = simulations.inactivation_sim(s,prepulses,act_pred[0])
    ResetSim(s,params)
    rec_pred = simulations.recovery_sim(s,intervals)

    if i == 0:
        act_peaks_posterior = np.reshape(act_pred[0],(12,1))
        act_cond_posterior = np.reshape(act_pred[1],(8,1))
        inact_cond_posterior = np.reshape(inact_pred,(7,1))
        rec_posterior = np.reshape(rec_pred,(11,1))
    else:
        act_peaks_posterior = np.hstack((act_peaks_posterior,np.reshape(act_pred[0],(12,1))))
        act_cond_posterior = np.hstack((act_cond_posterior,np.reshape(act_pred[1],(8,1))))
        inact_cond_posterior = np.hstack((inact_cond_posterior,np.reshape(inact_pred,(7,1))))
        rec_posterior = np.hstack((rec_posterior,np.reshape(rec_pred,(11,1))))

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

# Obtain similar results for original point estimates
ResetSim(s,params_orig)
act_pred_orig = simulations.activation_sim(s,vsteps,reversal_potential)
ResetSim(s,params_orig)
inact_pred_orig = simulations.inactivation_sim(s,prepulses,act_pred[0])
ResetSim(s,params_orig)
rec_pred_orig = simulations.recovery_sim(s,intervals)

# Plot results
plt.figure()
plt.suptitle('Voltage clamp simulations for T-type calcium channel in HL-1 myocytes')
plt.subplot(1,4,1)
plt.title('I-V curve')
plt.xlabel('Voltage (V)')
plt.ylabel('Peak current (pA/pF)')
plt.errorbar(vsteps,act_peaks_mean,yerr=act_peaks_std, marker = 'o', color='b',ls='None')
plt.plot(vsteps,act_peaks_exp,'rx')
plt.plot(vsteps,act_pred_orig[0],'gs')

plt.subplot(1,4,2)
plt.errorbar(vsteps_act,act_cond_mean,yerr=act_cond_std,marker = 'o', color='b',ls='None')
plt.title('Activation')
plt.xlabel('Voltage (V)')
plt.ylabel('Relative current')
plt.plot(vsteps_act,act_exp,'rx')
plt.plot(vsteps_act,act_pred_orig[1],'gs')

plt.subplot(1,4,3)
plt.errorbar(prepulses,inact_cond_mean,yerr=inact_cond_std,marker = 'o', color='b',ls='None')
plt.title('Inactivation')
plt.xlabel('Voltage (V)')
plt.ylabel('Relative current')
plt.plot(prepulses,inact_exp,'rx')
plt.plot(prepulses,inact_pred_orig,'gs')

plt.subplot(1,4,4)
plt.errorbar(intervals,rec_mean,yerr=rec_std,marker = 'o', color='b',ls='None',label='ABC Simulations')
plt.title('Recovery')
plt.xlabel('Voltage (V)')
plt.ylabel('Relative current')
plt.plot(intervals,rec_exp,'rx',label='Experimental')
plt.plot(intervals,rec_pred_orig,'gs',label='Takeuchi et al')
plt.legend(loc='lower right')
plt.show()
