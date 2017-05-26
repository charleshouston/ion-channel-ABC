import matplotlib.pyplot as plt
import numpy as np
import myokit

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

# Create protocol for step experiment
p1 = protocols.steptrain(
    vsteps = np.array(steps), # Use voltage steps from experiments
    vhold = -80, # Holding potential at -80mV
    tpre = 5000, # 5000ms pre-conditioning before each step
    tstep = 300, # 300ms at step potential
)
s.set_protocol(p1)
t1 = p1.characteristic_time()

# Create data log
d1 = myokit.DataLog()

# Run a simulation
d1 = s.run(t1, log=['environment.time', 'membrane.V', 'icat.i_CaT'],log_interval=.1)

# Split the log into chunks for each step
ds1 = d1.split_periodic(5300, adjust=True)

# Trim each new log to contain only the 100ms of peak current
for d in ds1:
    d.trim_left(4900, adjust=True)
    d.trim_right(200)

act_pks = []
for d in ds1:
    act_pks.append(np.min(d['icat.i_CaT']))

# Show the results
plt.figure()
for k, d in enumerate(ds1):
    plt.subplot(3,3,1)
    plt.plot(d['environment.time'], d['icat.i_CaT'])
    plt.subplot(3,3,2)
    plt.plot(d['environment.time'], d['membrane.V'])

# Plot peak currents against experimental data
plt.subplot(3,3,3)
plt.plot(steps,act_pks,'o',label="Simulated")
plt.plot(steps,I_exp,'x',label="Experimental")
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

# Reset simulation and remove previous protocol
s.reset()
s.set_protocol(None)

# Create protocol
p2 = protocols.steptrain_double(
    vsteps = prepulse, # Use prepulse values from experimental data
    vhold = -80, # Holding potential at -80mV
    vpost = -20, # Second step always -20mV
    tpre = 5000, # Pre-conditioning at -80mV for 5000ms
    tstep = 1000, # Initial step for 1000ms
    tbetween = 5, # Time between steps is 5ms
    tpost = 300, # Final pulse for 300ms
)
t2 = p2.characteristic_time()
s.set_protocol(p2)

# Run the simulation
d2 = s.run(t2, log=['environment.time','membrane.V','icat.i_CaT'],log_interval=.1)

# Trim each new log to contain only the 100ms of peak current
ds2 = d2.split_periodic(6305, adjust=True)
for d in ds2:
    d.trim_left(5900,adjust=True)
    d.trim_right(305)

# Create list to store peak currents in for inactivation experiment
inact_pks = []

for d in ds2:
    plt.subplot(3,3,4)
    plt.plot(d['environment.time'],d['icat.i_CaT'])
    plt.subplot(3,3,5)
    plt.plot(d['environment.time'],d['membrane.V'])
    # Find peak current
    current = d['icat.i_CaT']
    index = np.argmax(np.abs(current))
    peak = current[index]
    inact_pks.append(peak)

# Calculate the activation (normalized condutance) from IV curve
# - Create a numpy array with the peak currents
act_pks = np.array(act_pks)
# Get the reversal potential
reversal_potential = m.get('icat.E_CaT').value()
# - Divide the peak currents by (V-E)
act = act_pks / (steps - reversal_potential)
# - Normalise by dividing by the biggest value
act = act / np.max(act)

# Calculate the inactivation (i.e. availability of current)
# Defined as the current at a given pre-pulse potential
# divided by the maximum current in absence of a pre-pulse.
inact_pks = np.array(inact_pks)
inact = np.abs(inact_pks) / np.max(np.abs(act_pks))

# Display the activation measured in the protocol
plt.subplot(3,3,6)
plt.plot(steps, act, 'go',label="Sim. Act.")
plt.plot(steps_exp,act_exp, 'gx', label="Exp. Act.")
plt.plot(prepulse, inact,'ro',label="Sim. Inact.")
plt.plot(prepulse, inact_exp, 'rx', label="Exp. Inact.")
plt.legend(loc='lower right')
plt.xlim(-100,0)

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

# Create structure to hold data
d3 = myokit.DataLog()

# Create intervaltrain protocol
p3 = protocols.intervaltrain(
    vstep = -20, # Voltage steps are to -20mV
    vhold = -80, # Holding potential is -80mV
    vpost = -20, # Final pulse also at -20mV
    tpre = 5000, # Pre-conditioning each experiment for 5000ms
    tstep = 300, # Initial step for 300ms
    tintervals = interval_times, # Varying interval times
    tpost = 300, # Final pulse for 300ms
)
t3 = p3.characteristic_time()
s.set_protocol(p3)

# Run the simulation
d3 = s.run(t3, log=['environment.time','membrane.V','icat.i_CaT'],log_interval=.1)

# Trim each new log to contain only the 100ms of peak current
ds3 = []
d3 = d3.npview()
for t in interval_times:
    # Split each experiment
    d3_split,d3 = d3.split(t+5600)
    ds3.append(d3_split)

    # Adjust times of remaining data
    if len(d3['environment.time']):
        d3['environment.time'] -= d3['environment.time'][0]

peaks = []

for d in ds3:
    d.trim_left(5200,adjust=True)
    current = d['icat.i_CaT']
    index = np.argmax(np.abs(current))
    peak = current[index]
    peaks.append(peak)

    plt.subplot(3,3,7)
    plt.plot(d['environment.time'],d['icat.i_CaT'])
    plt.subplot(3,3,8)
    plt.plot(d['environment.time'],d['membrane.V'])

peaks = np.array(peaks)
recovery = -1*peaks / np.max(np.abs(peaks))

plt.subplot(3,3,9)
plt.plot(interval_times,recovery,'bo',label="Sim. Recovery")
plt.plot(interval_times,recovery_exp,'bx',label="Exp. Recovery")
plt.ylim([0,1])
plt.legend(loc='lower right')
plt.show()
