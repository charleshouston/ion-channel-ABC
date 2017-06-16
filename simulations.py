'''
Author: Charles Houston
Date: 26/5/17

Wrapper for common experimental protocols from
myokit common library.
'''
import myokit
from myokit.lib.common import Activation,Inactivation,Recovery
import numpy as np

class AbstractSim(object):
    def __init__(self):
        self.p_names = []
        self.p_vals = []
        self.sim = None

    def run(self):
        '''
        General method to run the simulation and export relevant results.
        '''
        raise NotImplementedError

    def set_parameters(self, p_names, p_vals):
        '''
        Store new values for parameters
        '''
        self.p_names = p_names
        self.p_vals = p_vals

class ActivationSim(AbstractSim):
    '''
                 +--- vstep ---+
                 +-------------+
                 +-------------+
                 |             |
    vhold -------+             +-
    t=0          t=thold       t=thold+tstep
    no current   current!
    '''
    def __init__(self, variable, vhold, thold,
                 vmin, vmax, dv, tstep):

        super(ActivationSim, self).__init__()

        self.variable=variable
        self.vhold=vhold
        self.thold=thold
        self.vmin=vmin
        self.vmax=vmax
        self.dv=dv
        self.tstep=tstep

    def _generate(self, model_name):
        '''
        Generate simulation in advance to avoid multiple compiles during loop.
        '''
        m,p,x = myokit.load('models/' + model_name)
        self.sim = Activation(m, var=self.variable, vvar='membrane.V')
        self.sim.set_holding_potential(vhold=self.vhold, thold=self.thold)
        self.sim.set_step_potential(vmin=self.vmin, vmax=self.vmax,
                                    dv=self.dv, tstep=self.tstep)

    def run(self, model_name):
        '''
        Run the activation protocol defined.
        '''
        if self.sim is None:
            self._generate(model_name)

        # Reset parameters to new values
        for i,p in enumerate(self.p_names):
            self.sim.set_constant(p, self.p_vals[i])

        try:
            pks = self.sim.peaks()
        except:
            return None

        return pks[self.variable]


class InactivationSim(AbstractSim):
    '''
    --- vstep ---+
    -------------+
    -------------+
                 |
                 +--- vhold ---+-
    t=0          t=tstep       t=tstep+thold
    '''
    def __init__(self, variable, vhold, thold,
                 vmin, vmax, dv, tstep):

        super(InactivationSim, self).__init__()

        self.variable=variable
        self.vhold=vhold
        self.thold=thold
        self.vmin=vmin
        self.vmax=vmax
        self.dv=dv
        self.tstep=tstep

    def _generate(self, model_name):
        """
        Compile simulation in advance of loop
        """
        m,p,x = myokit.load('models/' + model_name)
        self.sim = Inactivation(m, var=self.variable, vvar='membrane.V')
        self.sim.set_holding_potential(vhold=self.vhold, thold=self.thold)
        self.sim.set_step_potential(vmin=self.vmin, vmax=self.vmax,
                                    dv=self.dv, tstep=self.tstep)

    def run(self, model_name):
        '''
        Run the activation protocol defined.
        '''
        if self.sim is None:
            self._generate(model_name)

        # Reset parameters to new values
        for i,p in enumerate(self.p_names):
            self.sim.set_constant(p, self.p_vals[i])
        try:
            pks = self.sim.peaks(normalize=True)
        except:
            return None

        return pks[self.variable]

class RecoverySim(AbstractSim):
    '''
                  +--- vstep ---+         +- vstep -+
                  |             |  twait  |         |
                  |             | <-----> |         |
                  |             |         |         |
    +--- vhold ---+             +- vhold -+         +---
    t=0           t=thold       t+=tstep1 t+=twait  t+=tstep2
    '''
    def __init__(self, variable, vhold, thold,
                 vstep, tstep1, tstep2,
                 twaits):

        super(RecoverySim, self).__init__()

        self.variable=variable
        self.vhold=vhold
        self.thold=thold
        self.vstep=vstep
        self.tstep1=tstep1
        self.tstep2=tstep2
        self.twaits=twaits

    def _generate(self, model_name):
        """
        Compile simulation in advance to avoid dynamic creation in loop at runtime
        """
        m,p,x = myokit.load('models/' + model_name)
        self.sim = Recovery(m, var=self.variable, vvar='membrane.V')
        self.sim.set_holding_potential(vhold=self.vhold, thold=self.thold)
        self.sim.set_step_potential(vstep=self.vstep, tstep1=self.tstep1,
                               tstep2=self.tstep2)
        self.sim.set_specific_pause_durations(twaits=self.twaits)

    def run(self, model_name):
        '''
        Run the activation protocol defined.
        '''
        if self.sim is None:
            self._generate(model_name)

        # Reset parameters to new values
        for i,p in enumerate(self.p_names):
            self.sim.set_constant(p, self.p_vals[i])

        try:
            ratio = self.sim.ratio()
        except:
            return None

        return ratio[self.variable]
