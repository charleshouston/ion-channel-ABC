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
import simulations.simulations_icat as sim

class AbstractChannel(object):
    def __init__(self):
        self.parameters = []
        self.simulations = []
    def simulate(s):
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

    def simulate(s):
        act_sim = self.simulations[0].run(s)
        if not act_sim: return None
        inact_sim = self.simulations[1].run(s, act_res[0:12])
        if not inact_sim: return None
        rec_sim = self.simulations[2].run(s)
        if not rec_sim: return None

        return np.hstack((act_sim, inact_sim, rec_sim))
