'''
Author: Charles Houston
Date: 26/5/17

Helper functions to run simulations for ABC estimation.
'''

import myokit
import protocols
import numpy as np

from collections import namedtuple

ActSimResult = namedtuple('ActSimResult', ['peaks', 'cond'])

def activation_sim(s,steps,reversal_potential):
    '''
    Runs the activation simulation protocol from Deng2009.
    '''

    # Create protocol for step experiment
    p = protocols.steptrain(
        vsteps = np.array(steps), # Use voltage steps from experiments
        vhold = -80, # Holding potential at -80mV
        tpre = 5000, # 5000ms pre-conditioning before each step
        tstep = 300, # 300ms at step potential
    )
    s.set_protocol(p)
    t = p.characteristic_time()

    # Run a simulation
    try:
        d = s.run(t, log=['environment.time', 'membrane.V', 'icat.i_CaT'],log_interval=.1)
    except:
        res = ActSimResult(np.zeros(12), cond=np.zeros(8))
        return res

    # Split the log into chunks for each step
    ds = d.split_periodic(5300, adjust=True)

    # Trim each new log to contain only the 100ms of peak current
    for d in ds:
        d.trim_left(5000, adjust=True)
        d.trim_right(200)

    # Find and return current peaks
    act_peaks = []
    for d in ds:
        act_peaks.append(np.min(d['icat.i_CaT']))
    act_peaks = np.array(act_peaks)

    # Calculate the activation (normalized condutance) from IV curve
    # - Divide the peak currents by (V-E)
    act = act_peaks[:8] / (steps[:8] - reversal_potential)
    # - Normalise by dividing by the biggest value
    act = act / np.max(act)

    res = ActSimResult(act_peaks, cond=act)
    return res

def inactivation_sim(s,prepulse,act_pks):
    '''
    Runs the inactivation stimulation protocol from Deng 2009.
    '''
    # Create protocol
    p = protocols.steptrain_double(
        vsteps = prepulse, # Use prepulse values from experimental data
        vhold = -80, # Holding potential at -80mV
        vpost = -20, # Second step always -20mV
        tpre = 5000, # Pre-conditioning at -80mV for 5000ms
        tstep = 1000, # Initial step for 1000ms
        tbetween = 5, # Time between steps is 5ms
        tpost = 300, # Final pulse for 300ms
    )
    t = p.characteristic_time()
    s.set_protocol(p)

    # Run the simulation
    try:
        d = s.run(t, log=['environment.time','membrane.V','icat.i_CaT'],log_interval=.1)
    except:
        return np.zeros(7)

    # Trim each new log to contain only the 100ms of peak current
    ds = d.split_periodic(6305, adjust=True)
    for d in ds:
        d.trim_left(5900,adjust=True)
        d.trim_right(305)

    inact = []

    # Find peak current
    for d in ds:
        current = d['icat.i_CaT']
        index = np.argmax(np.abs(current))
        peak = current[index]
        inact.append(peak)

    inact = np.array(inact)
    inact = np.abs(inact) / np.max(np.abs(act_pks))

    return inact

def recovery_sim(s,intervals):
    '''
    Runs the recovery simulation from Deng 2009.
    '''
    # Create intervaltrain protocol
    p = protocols.intervaltrain(
        vstep = -20, # Voltage steps are to -20mV
        vhold = -80, # Holding potential is -80mV
        vpost = -20, # Final pulse also at -20mV
        tpre = 5000, # Pre-conditioning each experiment for 5000ms
        tstep = 300, # Initial step for 300ms
        tintervals = intervals, # Varying interval times
        tpost = 300, # Final pulse for 300ms
    )
    t = p.characteristic_time()
    s.set_protocol(p)

    # Run the simulation
    try:
        d = s.run(t, log=['environment.time','membrane.V','icat.i_CaT'],log_interval=.1)
    except:
        return np.zeros(11)

    # Trim each new log to contain only the 100ms of peak current
    ds = []
    d = d.npview()
    for interval in intervals:
        # Split each experiment
        d_split,d = d.split(interval+5600)
        ds.append(d_split)

        # Adjust times of remaining data
        if len(d['environment.time']):
            d['environment.time'] -= d['environment.time'][0]

    rec_peaks = []

    for trace in ds:
        trace.trim_left(5200,adjust=True)
        current = trace['icat.i_CaT']
        index = np.argmax(np.abs(current))
        peak = current[index]
        rec_peaks.append(peak)

    rec_peaks = np.array(rec_peaks)
    rec_peaks = -1*rec_peaks / np.max(np.abs(rec_peaks))

    return rec_peaks
