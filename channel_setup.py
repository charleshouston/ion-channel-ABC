'''
Author: Charles Houston

Specific channel settings for use with approximate Bayesian computation procedure.
'''
import numpy as np

import myokit
import distributions as Dist

# Data imports
import data.icat.data_icat as data_icat

# Experimental simulations import
import simulations as sim

class AbstractChannel(object):
    def __init__(self):
        self.parameters = []
        self.simulations = []
    def simulate(self, s):
        # Run previously defined simulations
        for simulation in self.simulations:
            simulation.run(s)

class TTypeCalcium(AbstractChannel):
    def __init__(self):
        self.name = 'icat'
        self.model_name = 'Takeuchi2013_iCaT.mmt'

        # Parameters involved in ABC process
        self.parameters = ['icat_d_gate.dssk1',
                           'icat_d_gate.dssk2',
                           'icat_d_gate.dtauk1',
                           'icat_d_gate.dtauk2',
                           'icat_d_gate.dtauk3',
                           'icat_d_gate.dtauk4',
                           'icat_d_gate.dtauk5',
                           'icat_d_gate.dtauk6',
                           'icat_f_gate.fssk1',
                           'icat_f_gate.fssk2',
                           'icat_f_gate.ftauk1',
                           'icat_f_gate.ftauk2',
                           'icat_f_gate.ftauk3',
                           'icat_f_gate.ftauk4',
                           'icat_f_gate.ftauk5',
                           'icat_f_gate.ftauk6']

        # Parameter specific prior intervals
        self.prior_intervals = [(0,100),
                                (1,10),
                                (0,10),
                                (0,100),
                                (1,100),
                                (0,10),
                                (0,100),
                                (1,100),
                                (0,100),
                                (1,10),
                                (0,0.1),
                                (0,100),
                                (1,100),
                                (0,0.1),
                                (0,100),
                                (1,100)]

        # Parameter specific distributions for perturbing parameters
        g01 = Dist.Normal(0.0,0.01)
        g10 = Dist.Normal(0.0,1.0)
        g100 = Dist.Normal(0.0,10.0)
        self.kernel = [g100, g10, g10, g100, g100, g10, g100, g100,
                       g100, g10, g01, g100, g100, g01, g100, g100]

        # Loading T-type channel experimental data
        vsteps, act_peaks_exp = data_icat.fig1B()
        vsteps = np.array(vsteps)
        act_peaks_exp = np.array(act_peaks_exp)
        vsteps_act, act_exp = data_icat.fig3Bact()
        vsteps_act = np.array(vsteps_act)
        act_exp = np.array(act_exp)
        prepulses, inact_exp = data_icat.fig3Binact()
        prepulses = np.array(prepulses)
        inact_exp = np.array(inact_exp)
        intervals, rec_exp = data_icat.fig4B()
        intervals = np.array(intervals)
        rec_exp = np.array(rec_exp)

        # Concatenate experimental data
        self.data_exp = np.hstack(([vsteps, act_peaks_exp],
                                   [vsteps_act, act_exp],
                                   [prepulses, inact_exp],
                                   [intervals, rec_exp]))

        # Setup simulations
        sim_act = sim.ActivationSim('icat.i_CaT', vsteps, reversal_potential=45, vhold=-80,
                                           tpre=5000, tstep=300)
        sim_inact = sim.InactivationSim('icat.i_CaT', prepulses, vhold=-80, vpost=-20,
                                        tpre=5000, tstep=1000, tbetween=5, tpost=300)
        sim_rec = sim.RecoverySim('icat.i_CaT', intervals, vstep=-20, vhold=-80, vpost=-20,
                                  tpre=5000, tstep=300, tpost=300)
        self.simulations = [sim_act, sim_inact, sim_rec]

    def simulate(self, s):
        '''
        Run the simulations necessary to generate values to compare with
        experimental results.
        '''
        act_sim = self.simulations[0].run(s)
        if act_sim is None:
            return None

        inact_sim = self.simulations[1].run(s, act_sim[0])
        if inact_sim is None:
            return None

        rec_sim = self.simulations[2].run(s)
        if rec_sim is None:
            return None

        return [act_sim[0], act_sim[1][0:8], inact_sim, rec_sim]


class FastSodium(AbstractChannel):
    def __init__(self):
        self.name = 'ina'
        self.model_name = 'Bondarenko2004_iNa.mmt'

        # Parameters involved in ABC process
        self.parameters = ['ina.k_alpha1',
                           'ina.k_alpha2',
                           'ina.k_alpha3',
                           'ina.k_alpha4',
                           'ina.k_alpha5_11',
                           'ina.k_alpha5_12',
                           'ina.k_alpha5_13',
                           'ina.k_alpha6_11',
                           'ina.k_alpha6_12',
                           'ina.k_alpha6_13',
                           'ina.k_alpha7',
                           'ina.k_alpha8',
                           'ina.k_alpha9',
                           'ina.k_alpha10',
                           'ina.k_beta1',
                           'ina.k_beta2_11',
                           'ina.k_beta2_12',
                           'ina.k_beta2_13',
                           'ina.k_beta3',
                           'ina.k_beta4',
                           'ina.k_beta5',
                           'ina.k_beta6',
                           'ina.k_beta7',
                           'ina.k_beta8']
        # Parameter specific prior intervals
        self.prior_intervals = [(0,10),
                                (0,1.0),
                                (0,10),
                                (1,1000),
                                (1,100),
                                (1,100),
                                (1,100),
                                (0,1.0),
                                (0,1.0),
                                (0,1.0),
                                (0,1.0),
                                (0,10),
                                (1,100),
                                (0,1.0),
                                (0,1e-6),
                                (1,10),
                                (0,1.0),
                                (-50,50),
                                (-50,50),
                                (-50,50),
                                (1,100),
                                (0,1.0),
                                (0,1.0),
                                (0,1e-2),
                                (0,1e-4),
                                (0,10)]

        # Parameter specific distributions for perturbing parameters
        g06 = Dist.Normal(0.0,1e-7)
        g04 = Dist.Normal(0.0,1e-5)
        g02 = Dist.Normal(0.0,1e-3)
        g01 = Dist.Normal(0.0,0.01)
        g1 = Dist.Normal(0.0,0.1)
        g10 = Dist.Normal(0.0,1.0)
        g100 = Dist.Normal(0.0,10.0)
        g1000 = Dist.Normal(0.0,100.0)
        self.kernel = [g10, g1, g10, g1000, g100, g100, g100, g1, g1,
                       g1, g1, g10, g100, g1, g06, g10, g1, g100,
                       g100, g100, g100, g1, g1, g02, g04, g10]
