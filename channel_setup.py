'''
Author: Charles Houston

Specific channel settings for use with approximate Bayesian computation procedure.
'''
import numpy as np

import myokit
import distributions as Dist

# Data imports
import data.icat.data_icat as data_icat
import data.ina.data_ina as data_ina
import data.ikur.data_ikur as data_ikur

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

        # Specifying pertubation kernel
        # - Uniform random walk with width 10% of prior range
        self.kernel = []
        for pr in self.prior_intervals:
            param_range = pr[1]-pr[0]
            self.kernel.append(Dist.Uniform(-1*param_range/20, param_range/20))

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
        sim_act_out = self.simulations[0].run(s)
        if sim_act_out is None:
            return None

        sim_inact_out = self.simulations[1].run(s)
        if sim_inact_out is None:
            return None

        sim_rec_out = self.simulations[2].run(s)
        if sim_rec_out is None:
            return None

        return [sim_act_out[0], sim_act_out[1][0:8], sim_inact_out, sim_rec_out]


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
                           'ina.k_alpha11',
                           'ina.k_alpha12',
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
        self.prior_intervals = [(0,10),  # 0
                                (0,1.0), # 1
                                (0,10),  # 2
                                (1,1000),# 3
                                (1,100), # 4
                                (1,100), # 5
                                (1,100), # 6
                                (0,1.0), # 7
                                (0,1.0), # 8
                                (0,1.0), # 9
                                (0,1.0), # 10
                                (0,10),  # 11
                                (1,100), # 12
                                (0,1.0), # 13
                                (0,1e-6),# 14
                                (1,10),  # 15
                                (0,1.0), # 16
                                (-50,50),# 17
                                (-50,50),# 18
                                (-50,50),# 19
                                (1,100), # 20
                                (0,1.0), # 21
                                (0,1.0), # 22
                                (0,1e-2),# 23
                                (0,1e-4),# 24
                                (0,10)]  # 25

        # Specifying pertubation kernel
        # - Uniform random walk with width 10% of prior range
        self.kernel = []
        for pr in self.prior_intervals:
            param_range = pr[1]-pr[0]
            self.kernel.append(Dist.Uniform(-1*param_range/20, param_range/20))

        # Loading fast Na channel experimental data
        vsteps, act_peaks_exp = data_ina.IV_DiasFig6()
        vsteps = np.array(vsteps)
        act_peaks_exp = np.array(act_peaks_exp)

        vsteps_act, act_exp = data_ina.Act_FukudaFig5B()
        vsteps_act = np.array(vsteps_act)
        act_exp = np.array(act_exp)

        prepulses, inact_exp = data_ina.Inact_FukudaFig5C()
        prepulses = np.array(prepulses)
        inact_exp = np.array(inact_exp)

        intervals, rec_exp = data_ina.Recovery_ZhangFig4B()
        intervals = np.array(intervals)
        rec_exp = np.array(rec_exp)

        self.data_exp = np.hstack(([vsteps, act_peaks_exp],
                                   [vsteps_act, act_exp],
                                   [prepulses, inact_exp],
                                   [intervals, rec_exp]))

        # Setup simulations
        sim_act = sim.ActivationSim('ina.i_Na', vsteps, reversal_potential=23.2, vhold=-80,
                                    tpre=3000, tstep=100)
        sim_act2 = sim.ActivationSim('ina.i_Na', vsteps_act, reversal_potential=23.2, vhold=-120,
                                     tpre=3000, tstep=20)
        sim_inact = sim.InactivationSim('ina.i_Na', prepulses, vhold=-120, vpost=-20,
                                        tpre=3000, tstep=500, tbetween=0, tpost=20)
        sim_rec = sim.RecoverySim('ina.i_Na', intervals, vstep=-30, vhold=-120, vpost=-30,
                                  tpre=3000, tstep=20, tpost=20)
        self.simulations = [sim_act, sim_act2, sim_inact, sim_rec]

    def simulate(self, s):
        '''
        Run the simulations necessary to generate values to compare with
        experimental results.
        '''
        sim_act_out = self.simulations[0].run(s)
        if sim_act_out is None:
            return None

        sim_act2_out = self.simulations[1].run(s)
        if sim_act2_out is None:
            return None

        sim_inact_out = self.simulations[2].run(s)
        if sim_inact_out is None:
            return None

        sim_rec_out = self.simulations[3].run(s)
        if sim_rec_out is None:
            return None

        return [sim_act_out[0], sim_act2_out[1], sim_inact_out, sim_rec_out]

class UltraRapidlyActivatingDelayedPotassium(AbstractChannel):
    def __init__(self):
        self.name = 'ikur'
        self.model_name = 'Bondarenko2004_iKur.mmt'

        # Parameters involved in ABC process
        self.parameters = ['ikur.assk1',
                           'ikur.assk2',
                           'ikur.atauk1',
                           'ikur.atauk2',
                           'ikur.atauk3',
                           'ikur.issk1',
                           'ikur.issk2',
                           'ikur.itauk1',
                           'ikur.itauk2',
                           'ikur.itauk3',
                           'ikur.itauk4']

        # Parameter specific prior intervals
        self.prior_intervals = [(0, 10),
                                (1, 100),
                                (0,100),
                                (0,10),
                                (1,100),
                                (0,10),
                                (1,100),
                                (0,1000),
                                (0,100),
                                (1,100),
                                (0,10000)]

        # Specifying pertubation kernel
        # - Uniform random walk with width 10% of prior range
        self.kernel = []
        for pr in self.prior_intervals:
            param_range = pr[1]-pr[0]
            self.kernel.append(Dist.Uniform(-1*param_range/20, param_range/20))

        # Loading i_Kur channel experimental data
        vsteps, act_peaks_exp = data_ikur.IV_MuharaniFig2B()
        vsteps = np.array(vsteps)
        act_peaks_exp = np.array(act_peaks_exp)

        prepulses, inact_exp = data_ikur.Inact_XuFig9C()
        prepulses = np.array(prepulses)
        inact_exp = np.array(inact_exp)

        intervals, rec_exp = data_ikur.Recovery_XuFig10C()
        intervals = np.array(intervals)
        rec_exp = np.array(rec_exp)

        self.data_exp = np.hstack(([vsteps, act_peaks_exp],
                                   [prepulses, inact_exp],
                                   [intervals, rec_exp]))

        # Setup simulations
        sim_act = sim.ActivationSim('ikur.i_Kur', vsteps, reversal_potential=0,
                                    vhold=-60, tpre=5000, tstep=300)
        sim_inact = sim.InactivationSim('ikur.i_Kur', prepulses, vhold=-60, vpost=50,
                                        tpre=0, tstep=5000, tbetween=0, tpost=5000)
        sim_rec = sim.RecoverySim('ikur.i_Kur', intervals, vstep=50, vhold=-70, vpost=50,
                                  tpre=3000, tstep=300, tpost=300)
        self.simulations = [sim_act, sim_inact, sim_rec]

    def simulate(self, s):
        '''
        Run the simulations necessary to generate values to compare with
        experimental results
        '''
        sim_act_out = self.simulations[0].run(s)
        if sim_act_out is None:
            return None

        sim_inact_out = self.simulations[1].run(s)
        if sim_inact_out is None:
            return None

        sim_rec_out = self.simulations[2].run(s)
        if sim_rec_out is None:
            return None

        return [sim_act_out[0], sim_inact_out, sim_rec_out]
