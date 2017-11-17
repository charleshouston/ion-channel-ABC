'''
Author: Charles Houston
Date: 26/5/17

Wrapper for common experimental protocols from
myokit common library.
'''
import myokit
from myokit.lib.common import Activation,Inactivation,Recovery
import myokit.lib.markov as markov
import numpy as np

class AbstractSim(object):
    def __init__(self, max_step_size=None):
        self.p_names = []
        self.p_vals = []
        self.sim = None
        self.max_step_size = max_step_size

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
                 vmin, vmax, dv, tstep, normalise=False):

        super(ActivationSim, self).__init__()

        self.variable=variable
        self.vhold=vhold
        self.thold=thold
        self.vmin=vmin
        self.vmax=vmax
        self.dv=dv
        self.tstep=tstep
        self.normalise=normalise

    def _generate(self, model_name):
        '''
        Generate simulation in advance to avoid multiple compiles during loop.
        '''
        m,p,x = myokit.load('models/' + model_name)
        self.sim = Activation(m, var=self.variable, vvar='membrane.V')
        self.sim.set_holding_potential(vhold=self.vhold, thold=self.thold)
        self.sim.set_step_potential(vmin=self.vmin, vmax=self.vmax,
                                    dv=self.dv, tstep=self.tstep)
        if self.max_step_size is not None:
            self.sim.set_max_step_size(self.max_step_size)

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

        if self.normalise:
            pks[self.variable] = pks[self.variable]/np.abs(np.max(pks[self.variable]))

        return pks[self.variable]

class TimeIndependentActivationSim(AbstractSim):
    '''
                 +--- vstep ---+
                 +-------------+
                 +-------------+
                 |             |
    vhold -------+             +-
    t=0          t=thold       t=thold+tstep
    no current   current!

    Time independent version of activation simulation.
    '''
    def __init__(self, variable, vsteps):

        super(TimeIndependentActivationSim, self).__init__()

        self.variable=variable
        self.vsteps=vsteps

    def run(self, model_name):
        '''
        Run the activation protocol defined.
        '''
        # Load the model
        m,p,x = myokit.load('models/' + model_name)
        v_m = m.get('membrane.V')
        v_m.demote()

        # Reset parameters to new values
        if len(self.p_names) > 0:
            for i,p in enumerate(self.p_names):
                m.get(p).set_rhs(self.p_vals[i])

        pks = []
        try:
            for v in self.vsteps:
                v_m.set_rhs(v)
                i = m.get(self.variable).value()
                pks.append(i)
        except:
            return None

        return pks

class MarkovActivationSim(AbstractSim):
    '''
                 +--- vstep ---+
                 +-------------+
                 +-------------+
                 |             |
    vhold -------+             +-
    t=0          t=thold       t=thold+tstep
    no current   current!

    Analytical Markov model version of activation simulation.
    '''
    def __init__(self, variable, vhold, thold,
                 vsteps, tstep,
                 name, states, params):

        super(MarkovActivationSim, self).__init__()

        self.variable=variable
        self.vhold=vhold
        self.thold=thold
        self.vsteps=vsteps
        self.tstep=tstep
        self.name=name
        self.states=states
        self.params=params

    def _generate(self, model_name):
        '''
        Generate simulation in advance to avoid multiple compiles during loop.
        '''
        m,_,_ = myokit.load('models/' + model_name)
        m_linear = markov.LinearModel.from_component(m.get(self.name), states=self.states,
                                                     parameters=self.params, current=self.variable)
        self.sim = markov.AnalyticalSimulation(m_linear)

    def run(self, model_name):
        '''
        Run the activation protocol defined.
        '''
        if self.sim is None:
            self._generate(model_name)

        # Reset parameters to new values
        self.sim.set_parameters(self.p_vals)

        peaks = []
        try:
            for v in self.vsteps:
                self.sim.reset()
                self.sim.set_membrane_potential(self.vhold)
                # Otherwise numpy will throw underflow error from exp
                t_sim = 0
                while t_sim < self.thold:
                    self.sim.run(20)
                    t_sim += 20
                self.sim.set_membrane_potential(v)
                d = self.sim.run(20)
                t_sim = 20
                while t_sim < self.tstep:
                    self.sim.run(20, log=d)
                    t_sim += 20
                peaks.append(d[self.variable][np.argmax(np.abs(d[self.variable]))])
        except:
            return None

        return peaks

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
                 vmin, vmax, dv, tstep, normalise=False):

        super(InactivationSim, self).__init__()

        self.variable=variable
        self.vhold=vhold
        self.thold=thold
        self.vmin=vmin
        self.vmax=vmax
        self.dv=dv
        self.tstep=tstep
        self.normalise=normalise

    def _generate(self, model_name):
        """
        Compile simulation in advance of loop
        """
        m,p,x = myokit.load('models/' + model_name)
        self.sim = Inactivation(m, var=self.variable, vvar='membrane.V')
        self.sim.set_holding_potential(vhold=self.vhold, thold=self.thold)
        self.sim.set_step_potential(vmin=self.vmin, vmax=self.vmax,
                                    dv=self.dv, tstep=self.tstep)
        if self.max_step_size is not None:
            self.sim.set_max_step_size(self.max_step_size)

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

        if self.normalise:
            pks[self.variable] = pks[self.variable]/np.min(pks[self.variable])

        return pks[self.variable]

class MarkovInactivationSim(AbstractSim):
    '''
    --- vstep ---+
    -------------+
    -------------+
                 |
                 +--- vhold ---+-
    t=0          t=tstep       t=tstep+thold

    Analytical Markov model version of inactivation simulation.
    '''
    def __init__(self, variable, vhold, thold,
                 vsteps, tstep,
                 name, states, params):

        super(MarkovInactivationSim, self).__init__()

        self.variable=variable
        self.vhold=vhold
        self.thold=thold
        self.vsteps=vsteps
        self.tstep=tstep
        self.name=name
        self.states=states
        self.params=params

    def _generate(self, model_name):
        '''
        Generate simulation in advance to avoid multiple compiles during loop.
        '''
        m,_,_ = myokit.load('models/' + model_name)
        m_linear = markov.LinearModel.from_component(m.get(self.name), states=self.states,
                                                     parameters=self.params, current=self.variable)
        self.sim = markov.AnalyticalSimulation(m_linear)

    def run(self, model_name):
        '''
        Run the activation protocol defined.
        '''
        if self.sim is None:
            self._generate(model_name)

        # Reset parameters to new values
        self.sim.set_parameters(self.p_vals)

        peaks = []
        try:
            for v in self.vsteps:
                self.sim.reset()
                self.sim.set_membrane_potential(v)
                # Otherwise numpy will throw underflow error from exp
                t_sim = 0
                while t_sim < self.tstep:
                    self.sim.run(20)
                    t_sim += 20
                self.sim.set_membrane_potential(self.vhold)
                d = self.sim.run(20)
                t_sim = 20
                while t_sim < self.thold:
                    self.sim.run(20, log=d)
                    t_sim += 20
                peaks.append(d[self.variable][np.argmax(np.abs(d[self.variable]))])
        except:
            return None

        m = np.max(peaks)
        if m > 0:
            peaks = [p / m for p in peaks]

        return peaks

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
        if self.max_step_size is not None:
            self.sim.set_max_step_size(self.max_step_size)

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


class MarkovRecoverySim(AbstractSim):
    '''
                  +--- vstep ---+         +- vstep -+
                  |             |  twait  |         |
                  |             | <-----> |         |
                  |             |         |         |
    +--- vhold ---+             +- vhold -+         +---
    t=0           t=thold       t+=tstep1 t+=twait  t+=tstep2

    Analytical Markov model version of recovery simulation.
    '''
    def __init__(self, variable, vhold, thold,
                 vstep, tstep1, tstep2,
                 twaits, name, states, params):

        super(MarkovRecoverySim, self).__init__()

        self.variable=variable
        self.vhold=vhold
        self.thold=thold
        self.vstep=vstep
        self.tstep1=tstep1
        self.tstep2=tstep2
        self.twaits=twaits
        self.name=name
        self.states=states
        self.params=params

    def _generate(self, model_name):
        '''
        Generate simulation in advance to avoid multiple compiles during loop.
        '''
        m,_,_ = myokit.load('models/' + model_name)
        m_linear = markov.LinearModel.from_component(m.get(self.name), states=self.states,
                                                     parameters=self.params, current=self.variable)
        self.sim = markov.AnalyticalSimulation(m_linear)

    def run(self, model_name):
        '''
        Run the activation protocol defined.
        '''
        if self.sim is None:
            self._generate(model_name)

        # Reset parameters to new values
        self.sim.set_parameters(self.p_vals)

        ratios = []
        try:
            for t in self.twaits:
                self.sim.reset()
                self.sim.set_membrane_potential(self.vhold)
                # Otherwise numpy will throw underflow error from exp
                t_sim = 0
                while t_sim < self.thold:
                    self.sim.run(20)
                    t_sim += 20
                self.sim.set_membrane_potential(self.vstep)
                d1 = self.sim.run(5)
                t_sim = 5
                while t_sim < self.tstep1:
                    self.sim.run(5, log=d1)
                    t_sim += 5
                self.sim.set_membrane_potential(self.vhold)
                t_sim = 0
                while t_sim < t:
                    self.sim.run(10)
                    t_sim += 10
                self.sim.set_membrane_potential(self.vstep)
                d2 = self.sim.run(5)
                t_sim = 5
                while t_sim < self.tstep2:
                    self.sim.run(5, log=d2)
                    t_sim += 5

                ratio = np.max(d1[self.variable])
                ratio = np.nan if ratio == 0 else np.max(d2[self.variable]) / ratio
                ratios.append(ratio)
        except:
            return None

        return ratios
