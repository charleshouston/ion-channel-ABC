'''
Author: Charles Houston
Date: 26/5/17

Default experimental simulations.
Originally taken from descriptions of how data was generated
in Deng et al, Pharmacological effects of carvedilol on T-type
calcium current in murine HL-1 cells, 2009.
'''

import myokit
import protocols
import numpy as np

class AbstractSim(object):
    def run(s):
        '''
        General method to run the simulation and export relevant results.
        '''
        raise NotImplementedError

class ActivationSim(AbstractSim):
    '''
    MEASURE: Simple measure of peak current response to
             different voltage pulses
    Runs activation simulation protocol (as described in Deng et al, 2009).
    Voltage at ```vhold``` for ```tpre```.
    Voltage to entry in ```vsteps``` for ```tstep```.
    Repeated for each entry in ```vsteps```.
    '''
    def __init__(self, variable, vsteps, reversal_potential, vhold, tpre, tstep):
        self.variable           = variable # Variable of interest
        self.vsteps             = np.array(vsteps) # Voltage steps in protocol
        self.protocol           = protocols.steptrain(
            vsteps              = self.vsteps,
            vhold               = vhold, # Holding potential
            tpre                = tpre, # Pre-conditioning simulation
            tstep               = tstep, # Time held at ```vsteps```
        )
        self.period             = tpre + tstep # Length of each experiment
        self.pre                = tpre # Time before experiment time of interest
        self.t                  = self.protocol.characteristic_time()
        self.reversal_potential = reversal_potential

    def run(self, s):
        s.reset()
        s.set_protocol(self.protocol)
        try:
            d = s.run(self.t, log=['environment.time', self.variable], log_interval=.1)
        except:
            return None

        # Split the log into chunks for each step
        ds = d.split_periodic(self.period, adjust=True)

        # Trim each new log to contain only the 100ms of peak current
        act_peaks = []
        for d in ds:
            d.trim_left(self.pre, adjust=True)
            d.trim_right(self.period - self.pre)
            act_peaks.append(max(min(d[self.variable]), max(d[self.variable]), key=abs))
        act_peaks = np.array(act_peaks)

        # Calculate the activation (normalized condutance) from IV curve
        # - Divide the peak currents by (V-E)
        act_relative = act_peaks / (self.vsteps - self.reversal_potential)
        # - Normalise by dividing by the biggest value
        act_relative = act_relative / act_relative.max()

        return [act_peaks, act_relative]

class InactivationSim(AbstractSim):
    '''
    MEASURE: Comparing peak current values after being held at a prepulse
             voltage prior to the pulse.
    Runs the inactivation stimulation protocol from Deng 2009.
    Hold potential at ```vhold``` for ```tpre```.
    Hold potential at entry from ```prepulses``` for ```tstep```.
    Return potential to ```vhold``` for ```tbetween```.
    Hold potential at ```vpost``` for ```tpost```.
    Repeat for each entry in ```prepulses```.
    '''
    def __init__(self, variable, prepulses, vhold, vpost, tpre, tstep, tbetween, tpost):
        self.protocol = protocols.steptrain_double(
            vsteps    = prepulses, # Use prepulse values from experimental data
            vhold     = vhold, # Holding potential
            vpost     = vpost, # Second step
            tpre      = tpre, # Pre-conditioning
            tstep     = tstep, # Initial step
            tbetween  = tbetween, # Time between steps
            tpost     = tpost, # Final pulse
        )
        self.t        = self.protocol.characteristic_time()
        self.pre      = tpre + tstep + tbetween
        self.post     = tpost
        self.variable = variable
        self.prepulses = prepulses

    def run(self, s):
        s.reset()
        s.set_protocol(self.protocol)

        # Run the simulation
        log_rate = 10.0 # Recording per ms
        try:
            d = s.run(self.t, log=['environment.time', self.variable], log_interval=1/log_rate)
        except:
            return None
        d.npview()


        # Get maximum current
        max_peak = max(np.abs(d[self.variable]))

        # Get normalised inactivation currents
        inact = []
        for pp in self.prepulses:
            d.trim_left(self.pre, adjust=True)
            inact.append(max(np.abs(d[self.variable][0:int(round(self.post*log_rate))])))
            d.trim_left(self.post, adjust=True)

        inact = np.array(inact)
        inact = inact / abs(max_peak)
        return inact

class RecoverySim(AbstractSim):
    '''
    MEASURE: Comparing difference in magnitude of current in two pulses
             separated by variable time interval.
    Runs the recovery simulation from Deng 2009.
    Hold potential at ```vhold``` for ```tpre```.
    Step potential to ```vstep``` for ```tstep```.
    Return potential to ```vhold``` for entry in ```intervals```.
    Step potential to ```vstep``` for ```tpost```.
    Repeat for each entry in ```intervals```.
    '''
    def __init__(self, variable, intervals, vstep, vhold, vpost, tpre, tstep, tpost):
        # Create intervaltrain protocol
        self.intervals    = intervals
        self.protocol     = protocols.intervaltrain(
            vstep         = vstep, # Voltage steps
            vhold         = vhold, # Holding potential
            vpost         = vpost, # Final pulse
            tpre          = tpre, # Pre-conditioning
            tstep         = tstep, # Initial step
            tintervals    = intervals, # Varying interval times
            tpost         = tpost, # Final pulse
        )
        self.t            = self.protocol.characteristic_time()
        self.period_const = tpre + tstep
        self.tpost = tpost
        self.variable     = variable

    def run(self, s):
        s.reset()
        s.set_protocol(self.protocol)

        # Run the simulation
        log_rate = 10.0
        try:
            d = s.run(self.t, log=['environment.time', self.variable], log_interval=1/log_rate)
        except:
            return None

        # Work through logs to get peak recovery currents
        d = d.npview()
        rec = []
        max_peak = max(np.abs(d[self.variable]))
        for interval in self.intervals:
            d.trim_left(self.period_const, adjust=True)
            rec.append(max(np.abs((d[self.variable][0:int(round((interval+self.tpost)*log_rate))]))))
            d.trim_left(interval+self.tpost, adjust=True)

        rec = np.array(rec)
        rec = rec / max_peak 
        return rec
