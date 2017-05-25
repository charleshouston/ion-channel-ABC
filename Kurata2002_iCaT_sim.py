import matplotlib.pyplot as plt
import numpy as np
import myokit

import pdb

# Get the model and protocol, create simulation
m, p, x = myokit.load('Kurata2002_iCaT.mmt')

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
steps = np.arange(-80, 50, 10)
p1 = myokit.pacing.steptrain(
    vsteps=steps,
    vhold = -80,    # Holding potential -80mV
    tpre  = 5000,   # 5000ms pre-conditioning at Vhold
    tstep = 300,    # 0.3s at step potential
    )
s.set_protocol(p1)
t1 = p1.characteristic_time()

# Create data log
d1 = myokit.DataLog()

# Run a simulation
d1 = s.run(t1, log=['environment.time', 'membrane.V', 'icat.i_CaT'],log_interval=.1)

# Split the log into smaller chunks
ds1 = d1.split_periodic(5300, adjust=True)

# Trim each new log to contain only the 100ms of peak current
for d in ds1:
    d.trim_left(5000, adjust=True)
    d.trim_right(100)

act_pks = []
for d in ds1:
    act_pks.append(np.min(d['icat.i_CaT']))

# Show the results
plt.figure()
plt.subplot(3,2,1)
for k, d in enumerate(ds1):
    label = str(steps[k]) + ' mV'
    plt.plot(d['environment.time'], d['icat.i_CaT'], label=label)

plt.subplot(3,2,2)
plt.plot(steps,act_pks,'o')


'''
ACTIVATION/INACTIVATION RATES
'''
s.reset()
s.set_protocol(None)

# Pre pulse potentials between -100 and 0 mV
holding_potentials = np.arange(-100,-39.9,10)

# Prepare datalogs
ds2 = [myokit.DataLog()]*len(holding_potentials)

for i in range(len(holding_potentials)):

    # Reset the simulation
    s.reset()

    # Prepace for 5s
    s.pre(5000)

    # Generate stimulation protocol
    p2 = myokit.Protocol()
    p2.schedule(holding_potentials[i],0,1000) # 1 second at holding potential
    p2.schedule(-80,1000,5) # Briefly back to -80mV
    p2.schedule(-20,1005,300) # Final pulse to -20mV
    t2 = p2.characteristic_time()
    s.set_protocol(p2)

    # Run the simulation
    ds2[i] = s.run(t2, log=['environment.time', 'membrane.V', 'icat.i_CaT'],log_interval=.1)

# Trim each new log to contain only the 100ms of peak current
for d in ds2:
    d.trim_left(1005, adjust=True)
    d.trim_right(200)

# Create list to store peak currents in for inactivation experiment
inact_pks = []

plt.subplot(3,2,3)
for d in ds2:
    plt.plot(d['environment.time'],d['icat.i_CaT'])
    # Find peak current
    current = d['icat.i_CaT']
    index = np.argmax(np.abs(current))
    peak = current[index]
    inact_pks.append(peak)

# Now, calculate the activation / normalized condutance
# - Create a numpy array with the peak currents
act_pks = np.array(act_pks)
# Get the reversal potential
reversal_potential = m.get('icat.E_CaT').value()
# - Divide the peak currents by (V-E)
act = act_pks / (steps - reversal_potential)
# - Normalise by dividing by the biggest value
act = act / np.max(act)

# Now calculate the inactivation (i.e. availability of current)
inact_pks = np.array(inact_pks)
inact = np.abs(inact_pks) / np.max(np.abs(inact_pks))

# Display the activation measured in the protocol
plt.subplot(3,2,4)
plt.plot(steps, act, 'o',label="Activation")
plt.plot(holding_potentials, inact,'o',label="Inactivation")
plt.xlim(-100,0)

'''
RECOVERY CURVES
'''
s.reset()
s.set_protocol(None)
d3 = myokit.DataLog()

# Intervals between pulses between 30 and 350ms
interval_times = np.arange(30,350,30)
interval_times = np.insert(interval_times, 0, 5)

peaks = []

for interval in interval_times:

    # Reset the simulation
    s.reset()

    s.pre(5000)

    # Generate the stimulation protocol
    p3 = myokit.Protocol()
    p3.schedule(-20,0,300)
    p3.schedule(-80,300,interval)
    p3.schedule(-20,300+interval,300)
    t = p3.characteristic_time()
    s.set_protocol(p3)

    # Run the simulation
    d3 = s.run(t, log=['environment.time', 'membrane.V', 'icat.i_CaT'],log_interval=.1)

    # Change the logged times so all simulated traces overlap
    d3 = d3.npview()
    d3['environment.time'] -= d3['environment.time'][0]

    # Plot i_CaT
    plt.subplot(3,2,5)
    label = str(interval) + ' ms'
    plt.plot(d3['environment.time'], d3['icat.i_CaT'],label=label)

    current = d3['icat.i_CaT']
    index = np.argmax(np.abs(current))
    peak = current[index]
    peaks.append(peak)


peaks = np.array(peaks)
recovery = -1*peaks / np.max(np.abs(peaks))

plt.subplot(3,2,6)
plt.plot(interval_times,recovery,'o')
plt.ylim([0,1])
# plt.legend(loc='lower right')
plt.show()
