'''
Author: Charles Houston
Date: 26/5/17

Default experimental simulations for T-type Calcium channel.
Taken from descriptions of how data was generated in Deng et al, 2009.
'''

import myokit
import protocols
import numpy as np

from collections import namedtuple

class AbstractSim(object):
    def run(s):
        raise NotImplementedError

class ActivationSim(AbstractSim):
    '''
    Runs the activation simulation protocol from Deng2009.
    '''
    def __init__(self, vsteps, reversal_potential):
        self.vsteps = np.array(vsteps)
        self.protocol = protocols.steptrain(
            vsteps = self.vsteps,
            vhold = -80,
            tpre = 5000,
            tstep = 300,
        )
        self.t = self.protocol.characteristic_time()
        self.reversal_potential = reversal_potential

    def run(self, s):
        s.reset()
        s.set_protocol(self.protocol)
        try:
            d = s.run(self.t, log=['environment.time','icat.i_CaT'], log_interval=.1)
        except:
            return None

        # Split the log into chunks for each step
        ds = d.split_periodic(5300, adjust=True)

        # Trim each new log to contain only the 100ms of peak current
        act_peaks = []
        for d in ds:
            d.trim_left(5000, adjust=True)
            d.trim_right(200)
            act_peaks.append(max(min(d['icat.i_CaT']), max(d['icat.i_CaT']), key=abs))
        act_peaks = np.array(act_peaks)

        # Calculate the activation (normalized condutance) from IV curve
        # - Divide the peak currents by (V-E)
        act_relative = act_peaks / (self.vsteps - self.reversal_potential)
        # - Normalise by dividing by the biggest value
        act_relative = act_relative / act_relative.max()

        res = np.hstack((act_peaks, act_relative))
        return res

class InactivationSim(AbstractSim):
    '''
    Runs the inactivation stimulation protocol from Deng 2009.
    '''
    def __init__(self, prepulses):
        self.protocol = protocols.steptrain_double(
            vsteps = prepulses, # Use prepulse values from experimental data
            vhold = -80, # Holding potential at -80mV
            vpost = -20, # Second step always -20mV
            tpre = 5000, # Pre-conditioning at -80mV for 5000ms
            tstep = 1000, # Initial step for 1000ms
            tbetween = 5, # Time between steps is 5ms
            tpost = 300, # Final pulse for 300ms
        )
        self.t = self.protocol.characteristic_time()

    def run(self, s, act_peaks):
        s.reset()
        s.set_protocol(self.protocol)

        # Run the simulation
        try:
            d = s.run(self.t, log=['environment.time','icat.i_CaT'], log_interval=.1)
        except:
            return None

        # Trim each new log to contain only the 100ms of peak current
        ds = d.split_periodic(6305, adjust=True)
        inact = []
        for d in ds:
            d.trim_left(5900,adjust=True)
            d.trim_right(305)
            inact.append(max(abs(d['icat.i_CaT'])))

        inact = np.array(inact)
        inact = inact / max(np.abs(act_peaks))
        return inact

class RecoverySim(AbstractSim):
    '''
    Runs the recovery simulation from Deng 2009.
    '''
    def __init__(self, intervals):
        # Create intervaltrain protocol
        self.intervals = intervals
        self.protocol = protocols.intervaltrain(
            vstep = -20, # Voltage steps are to -20mV
            vhold = -80, # Holding potential is -80mV
            vpost = -20, # Final pulse also at -20mV
            tpre = 5000, # Pre-conditioning each experiment for 5000ms
            tstep = 300, # Initial step for 300ms
            tintervals = intervals, # Varying interval times
            tpost = 300, # Final pulse for 300ms
        )
        self.t = self.protocol.characteristic_time()

    def run(self, s):
        s.reset()
        s.set_protocol(self.protocol)

        # Run the simulation
        try:
            d = s.run(self.t, log=['environment.time','icat.i_CaT'],log_interval=.1)
        except:
            return None

        # Trim each new log to contain only the 100ms of peak current
        ds = []
        d = d.npview()
        for interval in self.intervals:
            # Split each experiment
            d_split,d = d.split(interval+5600)
            ds.append(d_split)

            # Adjust times of remaining data
            if len(d['environment.time']):
                d['environment.time'] -= d['environment.time'][0]

        rec = []

        for d in ds:
            d.trim_left(5200,adjust=True)
            rec.append(max(min(d['icat.i_CaT']), max(d['icat.i_CaT']), key=abs))

        rec = np.array(rec)
        rec = -1*rec / np.max(np.abs(rec))
        return rec
