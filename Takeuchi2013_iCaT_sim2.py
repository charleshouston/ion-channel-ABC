import matplotlib.pyplot as plt
import numpy as np
import myokit
import simulations
# Experimental data
import Deng2009

# Protocols
import protocols

# Get the model and protocol, create simulation
m, p, x = myokit.load('Takeuchi2013_iCaT.mmt')

# Get membrane potential
v = m.get('membrane.V')
# Demote v from a state to an ordinary variable
v.demote()
v.set_rhs(0)

# Bind v's value to the pacing mechanism
v.set_binding('pace')

# Create simulation
s = myokit.Simulation(m)

'''
I-V CURVE
'''
# Import experimental data from first experiment (voltage steps)
steps,I_exp = Deng2009.fig1B()
steps = np.array(steps)
I_exp = np.array(I_exp)

# Get results from simulation
res = simulations.activation_sim(s,steps,m.get('icat.E_CaT').value())

# Show the results
plt.figure()
plt.subplot(1,4,1)
plt.plot(steps,res[0],'bo',label='Simulated')
plt.plot(steps,I_exp,'rx',label='Experimental')
plt.legend(loc='lower right')


'''
ACTIVATION/INACTIVATION RATES
'''
# Import experimental data
steps_exp,act_exp = Deng2009.fig3Bact()
prepulse,inact_exp = Deng2009.fig3Binact()
steps_exp = np.array(steps_exp)
act_exp = np.array(act_exp)
prepulse = np.array(prepulse)
inact_exp = np.array(inact_exp)

plt.subplot(1,4,2)
plt.plot(steps_exp,res[1],'bo',label='Simulated')
plt.plot(steps_exp,act_exp,'rx',label='Experimental')
plt.legend(loc='lower right')

# Reset simulation and remove previous protocol
s.reset()
s.set_protocol(None)

act_pks = res[0]
res = simulations.inactivation_sim(s,prepulse,act_pks)

plt.subplot(1,4,3)
plt.plot(prepulse, res, 'bo', label='Simulated')
plt.plot(prepulse, inact_exp, 'rx', label='Experimental')
plt.legend(loc='lower right')

'''
RECOVERY CURVES
'''
# Recovery experimental data
interval_times,recovery_exp = Deng2009.fig4B()
interval_times = np.array(interval_times)
recovery_exp = np.array(recovery_exp)

# Reset simulation and remove protocol
s.reset()
s.set_protocol(None)

res = simulations.recovery_sim(s,interval_times)

plt.subplot(1,4,4)
plt.plot(interval_times, res, 'bo', label='Simulated')
plt.plot(interval_times,recovery_exp, 'rx', label='Experimental')
plt.show()
