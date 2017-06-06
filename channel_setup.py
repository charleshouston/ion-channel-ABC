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
import simulations.simulations_icat as simulations_icat

class AbstractChannel(object):
    def __init__(self):
        self.parameters = []
    def distance(params,s):
        raise NotImplementedError
    def reset(params,s):
        s.reset()
        for i,p in enumerate(params):
            s.set_constant(self.parameters[i],p)

class TTypeCalcium(AbstractChannel):
    def __init__(self):
        self.name = 'icat'
        self.model_name = 'Takeuchi2013_iCaT.mmt'
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
        g01 = Dist.Normal(0.0,0.01)
        g10 = Dist.Normal(0.0,1.0)
        g100 = Dist.Normal(0.0,10.0)
        self.kernel = [g100, g10, g10, g100, g100, g10, g100, g100,
                       g100, g10, g01, g100, g100, g01, g100, g100]

        # Loading experimental data
        vsteps, act_peaks_exp = data_icat.fig1B()
        act_peaks_exp = np.array(act_peaks_exp)
        vsteps_act, act_exp = data_icat.fig3Bact()
        act_exp = np.array(act_exp)
        prepulses, inact_exp = data_icat.fig3Binact()
        inact_exp = np.array(inact_exp)
        intervals, rec_exp = data_icat.fig4B()
        rec_exp = np.array(rec_exp)
        # Concatenate experimental data
        self.data_exp = np.hstack((act_peaks_exp,
                                   act_exp,
                                   inact_exp,
                                   rec_exp))

    def distance(params,s):

