'''
Author: Charles Houston

Specific channel settings for use with approximate Bayesian computation procedure.
'''
import numpy as np

import myokit
import distributions as Dist

# Data imports
import data.ical.data_ical as data_ical
import data.icat.data_icat as data_icat
import data.ina.data_ina as data_ina
import data.ikur.data_ikur as data_ikur
import data.iha.data_iha as data_iha
import data.ikr.data_ikr as data_ikr
import data.ito.data_ito as data_ito
import data.ikach.data_ikach as data_ikach
import data.ik1.data_ik1 as data_ik1
import data.incx.data_incx as data_incx

# Experimental simulations import
import simulations as sim

class AbstractChannel(object):
    def __init__(self):
        self.parameters = []
        self.simulations = []

    def reset_params(self, new_params):
        '''
        Set parameters of channel to new values in prior draw.
        '''
        if len(self.simulations) is 0:
            print "Need to add simulations to channel first!"
        else:
            for sim in self.simulations:
                sim.set_parameters(self.parameters, new_params)

    def simulate(self):
        '''
        Run the simulations necessary to generate values to compare with
        experimental results.
        '''
        if len(self.simulations) is 0:
            print "Need to add simulations to channel first!"
            return None

        sim_output = []

        for simulation in self.simulations:
            out = simulation.run(self.model_name)
            if out is None:
                return None
            sim_output.append(out)

        return sim_output

class icat(AbstractChannel):
    def __init__(self):
        self.name = 'icat'
        self.model_name = 'Takeuchi2013_iCaT.mmt'

        # Parameters involved in ABC process
        self.parameters = ['icat.dssk1',
                           'icat.dssk2',
                           'icat.dtauk1',
                           'icat.dtauk2',
                           'icat.dtauk3',
                           'icat.dtauk4',
                           'icat.dtauk5',
                           'icat.dtauk6',
                           'icat.fssk1',
                           'icat.fssk2',
                           'icat.ftauk1',
                           'icat.ftauk2',
                           'icat.ftauk3',
                           'icat.ftauk4',
                           'icat.ftauk5',
                           'icat.ftauk6']

        # Parameter specific prior intervals
        # Original values given in comments
        self.prior_intervals = [(0,100), # 30
                                (1,10),  # 6.0
                                (0,10),  # 1.068
                                (0,100), # 26.3
                                (1,100), # 30
                                (0,10),  # 1.068
                                (0,100), # 26.3
                                (1,100), # 30
                                (0,100), # 48
                                (1,10),  # 7.0
                                (0,0.1), # 0.0153
                                (0,100), # 61.7
                                (1,100), # 83.3
                                (0,0.1), # 0.015
                                (0,100), # 61.7
                                (1,100)] # 30

        # Specifying pertubation kernel
        # - Uniform random walk with width 10% of prior range
        self.kernel = []
        for pr in self.prior_intervals:
            param_range = pr[1]-pr[0]
            self.kernel.append(Dist.Uniform(-1*param_range/20, param_range/20))

        # Loading T-type channel experimental data
        vsteps, act_exp = data_icat.IV_DengFig1B()
        prepulses, inact_exp = data_icat.Inact_DengFig3B()
        intervals, rec_exp = data_icat.Recovery_DengFig4B()

        # Concatenate experimental data
        self.data_exp = [[vsteps, act_exp],
                         [prepulses, inact_exp],
                         [intervals, rec_exp]]

        # Setup simulations
        sim_act = sim.ActivationSim('icat.i_CaT', vhold=-80, thold=5000,
                                    vmin=min(vsteps), vmax=max(vsteps),
                                    dv=vsteps[1]-vsteps[0], tstep=300)
        sim_inact = sim.InactivationSim('icat.G_CaT', vhold=-20, thold=300,
                                        vmin=min(prepulses), vmax=max(prepulses),
                                        dv=prepulses[1]-prepulses[0], tstep=1000)
        sim_rec = sim.RecoverySim('icat.G_CaT', vhold=-80, thold=5000,
                                  vstep=-20, tstep1=300, tstep2=300,
                                  twaits=intervals)
        self.simulations = [sim_act, sim_inact, sim_rec]


class ina(AbstractChannel):
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
        self.prior_intervals = [(0,10),  # 3.802
                                (0,1.0), # 0.1027
                                (0,10),  # 2.5
                                (1,1000),# 150
                                (1,100), # 17
                                (1,100), # 15
                                (1,100), # 12
                                (0,1.0), # 0.2
                                (0,1.0), # 0.23
                                (0,1.0), # 0.25
                                (0,1.0), # 0.188495
                                (0,10),  # 7
                                (1,100), # 16.6
                                (0,1.0), # 0.393956
                                (0,1e-6),# 7e-7
                                (1,10),  # 7.7
                                (0,1.0), # 0.1917
                                (-10,10),# 2.5
                                (-10,10),# -2.5
                                (-10,10),# -7.5
                                (1,100), # 20.3
                                (0,1.0), # 0.2
                                (0,1.0), # 0.22
                                (0,1e-2),# 0.0084
                                (0,1e-4),# 2e-5
                                (0,10)]  # 7

        # Specifying pertubation kernel
        # - Uniform random walk with width 10% of prior range
        self.kernel = []
        for pr in self.prior_intervals:
            param_range = pr[1]-pr[0]
            self.kernel.append(Dist.Uniform(-1*param_range/20, param_range/20))

        # Loading fast Na channel experimental data
        vsteps, act_exp = data_ina.IV_DiasFig6()
        prepulses, inact_exp = data_ina.Inact_FukudaFig5C()
        intervals, rec_exp = data_ina.Recovery_ZhangFig4B()

        # Concatenate experimental data
        self.data_exp = [[vsteps, act_exp],
                         [prepulses, inact_exp],
                         [intervals, rec_exp]]

        # Setup simulations
        sim_act = sim.ActivationSim('ina.i_Na', vhold=-80, thold=3000,
                                    vmin=min(vsteps), vmax=max(vsteps),
                                    dv=vsteps[1]-vsteps[0], tstep=100)
        sim_inact = sim.InactivationSim('ina.G_Na', vhold=-20, thold=20,
                                        vmin=min(prepulses), vmax=max(prepulses),
                                        dv=prepulses[1]-prepulses[0], tstep=500)
        sim_rec = sim.RecoverySim('ina.G_Na', vhold=-120, thold=3000,
                                  vstep=-30, tstep1=20, tstep2=20,
                                  twaits=intervals)
        self.simulations = [sim_act, sim_inact, sim_rec]


class ikur(AbstractChannel):
    def __init__(self):
        self.name = 'ikur'
        self.model_name = 'Bondarenko2004_iKur.mmt'

        # Parameters involved in ABC process
        self.parameters = ['ikur.g_Kur',
                           'ikur.assk1',
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
        self.prior_intervals = [(0, 1),     # 0.0975
                                (0, 100),   # 22.5
                                (1, 10),    # 7.5
                                (0, 1),     # 0.493
                                (0, 0.1),   # 0.0629
                                (0, 10),    # 2.058
                                (0, 100),   # 45.2
                                (1, 10),    # 5.7
                                (0, 10000), # 1200
                                (0, 1000),  # 170
                                (0, 100),   # 45.2
                                (1, 10)]    # 5.7

        # Specifying pertubation kernel
        # - Uniform random walk with width 10% of prior range
        self.kernel = []
        for pr in self.prior_intervals:
            param_range = pr[1]-pr[0]
            self.kernel.append(Dist.Uniform(-1*param_range/20, param_range/20))

        # Loading i_Kur channel experimental data
        vsteps, act_peaks_exp = data_ikur.IV_MuharaniFig2B()
        prepulses, inact_exp = data_ikur.Inact_XuFig9C()
        intervals, rec_exp = data_ikur.Recovery_XuFig10C()

        self.data_exp = [[vsteps, act_peaks_exp],
                         [prepulses, inact_exp],
                         [intervals, rec_exp]]

        # Setup simulations
        sim_act = sim.ActivationSim('ikur.i_Kur', vhold=-60, thold=5000,
                                    vmin=min(vsteps), vmax=max(vsteps),
                                    dv=vsteps[1]-vsteps[0], tstep=300)
        sim_inact = sim.InactivationSim('ikur.G_Kur', vhold=50, thold=5000,
                                        vmin=min(prepulses), vmax=max(prepulses),
                                        dv=prepulses[1]-prepulses[0], tstep=5000)
        sim_rec = sim.RecoverySim('ikur.G_Kur', vhold=-70, thold=5000,
                                  vstep=50, tstep1=9.5, tstep2=9.5,
                                  twaits=intervals)
        self.simulations = [sim_act, sim_inact, sim_rec]


class ical(AbstractChannel):
    def __init__(self):
        self.name = 'ical'
        self.model_name = 'Houston2017.mmt'

        # Parameters involved in ABC process
        # self.parameters = ['ical.d_ssk1',
        #                    'ical.d_ssk2',
        #                    'ical.tau_dk1',
        #                    'ical.tau_dk2',
        #                    'ical.tau_dk3',
        #                    'ical.tau_dk4',
        #                    'ical.f_ssk1',
        #                    'ical.f_ssk2',
        #                    'ical.f_ssk3',
        #                    'ical.f_ssk4',
        #                    'ical.f_ssk5',
        #                    'ical.tau_fk1',
        #                    'ical.tau_fk2',
        #                    'ical.tau_fk3',
        #                    'ical.tau_fk4',
        #                    'ical.tau_fk5']

        self.parameters = ['ical.kalpha1',
                           'ical.kalpha2',
                           'ical.kalpha3',
                           'ical.kalpha4',
                           'ical.kalpha5',
                           'ical.kalpha6',
                           'ical.kalpha7',
                           'ical.kalpha1',
                           'ical.kalpha9',
                           'ical.kalpha10',
                           'ical.kalpha11',
                           'ical.kalpha12',
                           'ical.kbeta1',
                           'ical.kbeta2',
                           'ical.kbeta3']
                           # 'ical.kKpcf1',
                           # 'ical.kKpcf2',
                           # 'ical.kKpcf3']

        # Parameter specific prior intervals
        # Original values given in comments
        # self.prior_intervals = [(0, 10),    # 1
        #                         (0, 10),    # 6
        #                         (0, 10),    # 5
        #                         (0, 10),    # 6
        #                         (0, 0.1),   # 0.035
        #                         (0, 10),    # 5
        #                         (0, 100),   # 30
        #                         (0, 10),    # 9
        #                         (0, 1),     # 0.6
        #                         (0, 100),   # 50
        #                         (0, 100),   # 20
        #                         (0, 10),    # 2
        #                         (0, 0.1),   # 0.0197
        #                         (0, 0.1),   # 0.0337
        #                         (0, 100),   # 14.5
        #                         (0, 0.1)]   # 0.02
        self.prior_intervals = [(0,1),   # 0.4
                                (0,100), # 12
                                (1,100), # 10
                                (0,10),  # 1.068
                                (0,1),   # 0.7
                                (0,100), # 40
                                (1,100), # 10
                                (0,1),   # 0.75
                                (0,100), # 20
                                (1,1000),# 400
                                (0,1),   # 0.12
                                (0,100), # 12
                                (1,100), # 10
                                (0,0.1), # 0.05
                                (0,100), # 12
                                (1,100)] # 13
                                # (0,100), # 13
                                # (0,100), # 14.5
                                # (1,1000)]# 100

        # Specifying pertubation kernel
        # - Uniform random walk with width 10% of prior range
        self.kernel = []
        for pr in self.prior_intervals:
            param_range = pr[1]-pr[0]
            self.kernel.append(Dist.Uniform(-1*param_range/20, param_range/20))

        # Loading T-type channel experimental data
        vsteps, act_exp = data_ical.IV_DiasFig7()
        prepulses, inact_exp = data_ical.Inact_RaoFig3C()
        intervals, rec_exp = data_ical.Recovery_RaoFig3D()

        # Concatenate experimental data
        self.data_exp = [[vsteps, act_exp],
                         [prepulses, inact_exp],
                         [intervals, rec_exp]]

        # Setup simulations
        sim_act = sim.ActivationSim('ical.i_CaL', vhold=-40, thold=200,
                                    vmin=min(vsteps), vmax=max(vsteps),
                                    dv=vsteps[1]-vsteps[0], tstep=250)
        sim_inact = sim.InactivationSim('ical.G_CaL', vhold=-80, thold=400,
                                        vmin=min(prepulses), vmax=max(prepulses),
                                        dv=prepulses[1]-prepulses[0], tstep=1000)
        sim_rec = sim.RecoverySim('ical.G_CaL', vhold=-40, thold=5000,
                                  vstep=20, tstep1=250, tstep2=250,
                                  twaits=intervals)
        self.simulations = [sim_act, sim_inact, sim_rec]

class ikr(AbstractChannel):
    def __init__(self):
        self.name = 'ikr'
        self.model_name = 'Takeuchi2013_iKr.mmt'

        # Parameters involved in ABC process
        self.parameters = ['ikr.xkr_ssk1',
                           'ikr.xkr_ssk2',
                           'ikr.tau_xkrk1',
                           'ikr.tau_xkrk2',
                           'ikr.tau_xkrk3',
                           'ikr.tau_xkrk4',
                           'ikr.tau_xkrk5',
                           'ikr.rkrk1',
                           'ikr.rkrk2']

        # Parameter specific prior intervals
        self.prior_intervals = [(0, 100),   # 15
                                (1, 10),    # 6
                                (0, 10),    # 2.5
                                (0, 100),   # 31.18
                                (0, 1000),  # 217.18
                                (0, 100),   # 20.1376
                                (1, 100),   # 22.1996
                                (0, 100),   # 55
                                (1, 100)]   # 24

        # Specifying pertubation kernel
        # - Uniform random walk with width 10% of prior range
        self.kernel = []
        for pr in self.prior_intervals:
            param_range = pr[1]-pr[0]
            self.kernel.append(Dist.Uniform(-1*param_range/20, param_range/20))

        # Loading experimental data
        vsteps, act_peaks_exp = data_ikr.IV_Li7B()
        vsteps2, act_exp = data_ikr.Activation_Li7B()
        # Normalise act_exp
        act_exp = [float(i) / max(act_exp) for i in act_exp]

        self.data_exp = [[vsteps, act_peaks_exp],
                         [vsteps2, act_exp]]

        # Setup simulations
        sim_IV = sim.ActivationSim('ikr.i_Kr', vhold=-50, thold=5000,
                                    vmin=min(vsteps), vmax=max(vsteps),
                                    dv=vsteps[1]-vsteps[0], tstep=1000)
        sim_act = sim.InactivationSim('ikr.G_Kr', vhold=-50, thold=2000,
                                      vmin=min(vsteps2), vmax=max(vsteps2),
                                      dv=vsteps2[1]-vsteps2[0], tstep=1000)
        self.simulations = [sim_IV, sim_act]

class iha(AbstractChannel):
    def __init__(self):
        self.name = 'iha'
        self.model_name = 'Majumder2016_iha.mmt'

        # Parameters involved in ABC process
        self.parameters = ['iha.y_ssk1',
                           'iha.y_ssk2',
                           'iha.tau_yk1',
                           'iha.tau_yk2',
                           'iha.tau_yk3',
                           'iha.tau_yk4',
                           'iha.tau_yk5',
                           'iha.tau_yk6',
                           'iha.i_haNak1',
                           'iha.g_ha']

        # Parameter specific prior intervals
        self.prior_intervals = [(0, 100),   # 78.65
                                (1, 10),    # 6.33
                                (0, 1.0),   # 0.11885
                                (0, 100),   # 75
                                (1, 100),   # 28.37
                                (0, 1.0),   # 0.56236
                                (0, 100),   # 75
                                (0, 100),   # 14.19
                                (0, 1.0),   # 0.2
                                (0, 1.0)]   # 0.021

        # Specifying pertubation kernel
        # - Uniform random walk with width 10% of prior range
        self.kernel = []
        for pr in self.prior_intervals:
            param_range = pr[1]-pr[0]
            self.kernel.append(Dist.Uniform(-1*param_range/20, param_range/20))

        # Loading experimental data
        vsteps, act_peaks_exp = data_iha.IV_Sartiana5B()
        prepulses, inact_exp = data_iha.Act_DengFig3B()

        self.data_exp = [[vsteps, act_peaks_exp],
                         [prepulses, inact_exp]]

        # Setup simulations
        sim_act = sim.ActivationSim('iha.i_ha', vhold=-120, thold=1500,
                                    vmin=min(vsteps), vmax=max(vsteps),
                                    dv=vsteps[1]-vsteps[0], tstep=300)
        sim_inact = sim.InactivationSim('iha.G_ha', vhold=40, thold=2000,
                                        vmin=min(prepulses), vmax=max(prepulses),
                                        dv=prepulses[1]-prepulses[0], tstep=500)
        self.simulations = [sim_act, sim_inact]


class ito(AbstractChannel):
    def __init__(self):
        self.name = 'ito'
        self.model_name = 'Takeuchi2013_ito.mmt'

        # Parameters involved in ABC process
        self.parameters = ['ito.g_to',
                           'ito.xto_ssk1',
                           'ito.xto_ssk2',
                           'ito.tau_xtok1',
                           'ito.tau_xtok2',
                           'ito.tau_xtok3',
                           'ito.yto_ssk1',
                           'ito.yto_ssk2',
                           'ito.tau_ytok1',
                           'ito.tau_ytok2',
                           'ito.tau_ytok3',
                           'ito.tau_ytok4']

        # Parameter specific prior intervals
        self.prior_intervals = [(0, 1),     # 0.12375
                                (0, 10),    # 1
                                (0, 100),   # 11
                                (0, 10),    # 1.5
                                (0, 10),    # 3.5
                                (0, 100),   # 30
                                (0, 100),   # 40.5
                                (0, 100),   # 11.5
                                (0, 100),   # 21.21
                                (0, 100),   # 38.4525
                                (0, 100),   # 52.45
                                (0, 100)]   # 15.8827

        # Specifying pertubation kernel
        # - Uniform random walk with width 10% of prior range
        self.kernel = []
        for pr in self.prior_intervals:
            param_range = pr[1]-pr[0]
            self.kernel.append(Dist.Uniform(-1*param_range/20, param_range/20))

        # Loading experimental data
        vsteps, act_peaks_exp = data_ito.IV_KaoFig6()

        self.data_exp = [[vsteps, act_peaks_exp]]

        # Setup simulations
        sim_act = sim.ActivationSim('ito.i_to', vhold=-80, thold=5000,
                                    vmin=min(vsteps), vmax=max(vsteps),
                                    dv=vsteps[1]-vsteps[0], tstep=300)
        self.simulations = [sim_act]

class ikach(AbstractChannel):
    def __init__(self):
        self.name = 'ikach'
        self.model_name = 'Majumder2016_iKAch.mmt'

        # Parameters involved in ABC process
        self.parameters = ['ikach.k1',
                           'ikach.k2',
                           'ikach.k3',
                           'ikach.k4',
                           'ikach.k5',
                           'ikach.k6',
                           'ikach.k7',
                           'ikach.k8']

        # Parameter specific prior intervals
        self.prior_intervals = [(0, 10),   # 0.37488
                                (0, 10),   # 9.13652
                                (0, 1),    # 0.477811
                                (0, 0.1),  # 0.04
                                (0, 1),    # 0.23
                                (0, 1000), # 102
                                (0, 100),  # 10
                                (0, 100)]  # 10

        # Specifying pertubation kernel
        # - Uniform random walk with width 10% of prior range
        self.kernel = []
        for pr in self.prior_intervals:
            param_range = pr[1]-pr[0]
            self.kernel.append(Dist.Uniform(-1*param_range/20, param_range/20))

        # Loading i_Kur channel experimental data
        vsteps, act_peaks_exp = data_ikach.IV_KaoFig6()

        self.data_exp = [[vsteps, act_peaks_exp]]

        # Setup simulations
        sim_act = sim.TimeIndependentActivationSim('ikach.i_KAch', vsteps=vsteps)
        self.simulations = [sim_act]

class ik1(AbstractChannel):
    def __init__(self):
        self.name = 'ik1'
        self.model_name = 'Takeuchi2013_iK1.mmt'

        # Parameters involved in ABC process
        self.parameters = ['ik1.g_K1',
                           'ik1.a_K1_k1',
                           'ik1.a_K1_k2',
                           'ik1.a_K1_k3',
                           'ik1.b_K1_k1',
                           'ik1.b_K1_k2',
                           'ik1.b_K1_k3',
                           'ik1.b_K1_k4',
                           'ik1.b_K1_k5',
                           'ik1.b_K1_k6',
                           'ik1.b_K1_k7']

        # Parameter specific prior intervals
        self.prior_intervals = [(0, 1),    # 0.15
                                (0, 10),   # 1.02 
                                (0, 1),    # 0.2385 
                                (0, 100),  # 59.215
                                (0, 1),    # 0.49124
                                (0, 0.1),  # 0.08032
                                (0, 10),   # 5.476
                                (0, 0.1),  # 0.06175
                                (0, 1000), # 594.31
                                (0, 1),    # 0.5143
                                (0, 10)]   # 4.753

        # Specifying pertubation kernel
        # - Uniform random walk with width 10% of prior range
        self.kernel = []
        for pr in self.prior_intervals:
            param_range = pr[1]-pr[0]
            self.kernel.append(Dist.Uniform(-1*param_range/20, param_range/20))

        # Loading experimental data
        vsteps, act_peaks_exp = data_ik1.IV_GoldoniFig3D()
        # Convert to current densities using value reported for current
        #  density at -150mV in original paper (-42.2pA/pF)
        max_curr_density = -42.2
        act_peaks_exp = [p * max_curr_density / act_peaks_exp[0] for p in act_peaks_exp]

        self.data_exp = [[vsteps, act_peaks_exp]]

        # Setup simulations
        sim_act = sim.TimeIndependentActivationSim('ik1.i_K1', vsteps=vsteps)
        self.simulations = [sim_act]

class ina2(AbstractChannel):
    def __init__(self):
        self.name = 'ina2'
        self.model_name = 'Takeuchi2013_iNa.mmt'

        # Parameters involved in ABC process
        self.parameters = ['ina.g_Na',
                           'ina.v_split',
                           'ina.m_ssk1',
                           'ina.m_ssk2',
                           'ina.tau_mk1',
                           'ina.tau_mk2',
                           'ina.tau_mk3',
                           'ina.tau_mk4',
                           'ina.tau_mk5',
                           'ina.tau_mk6',
                           'ina.h_ssk1',
                           'ina.h_ssk2',
                           'ina.a_hk1',
                           'ina.a_hk2',
                           'ina.a_hk3',
                           'ina.b_hk1',
                           'ina.b_hk2',
                           'ina.b_hk3',
                           'ina.b_hk4',
                           'ina.b_hk5',
                           'ina.b_hk6',
                           'ina.b_hk7',
                           'ina.b_hk8',
                           'ina.j_ssk1',
                           'ina.j_ssk2',
                           'ina.a_jk1',
                           'ina.a_jk2',
                           'ina.a_jk3',
                           'ina.a_jk4',
                           'ina.a_jk5',
                           'ina.a_jk6',
                           'ina.a_jk7',
                           'ina.b_jk1',
                           'ina.b_jk2',
                           'ina.b_jk3',
                           'ina.b_jk4',
                           'ina.b_jk5',
                           'ina.b_jk6',
                           'ina.b_jk7',
                           'ina.b_jk8']

        # Parameter specific prior intervals
        self.prior_intervals = [(0, 100),   # 23
                                (-100, 0),  # -40
                                (0, 100),   # 56.86
                                (1, 10),    # 9.03
                                (0, 1),     # 0.1292
                                (0, 100),   # 45.79
                                (1, 100),   # 15.54
                                (0, 0.1),   # 0.06487
                                (0, 10),    # 4.823
                                (1, 100),   # 51.12
                                (0, 100),   # 71.55
                                (1, 10),    # 7.43
                                (0, 0.1),   # 0.057
                                (0, 100),   # 80
                                (1, 10),    # 6.8
                                (0, 1),     # 0.77
                                (0, 1),     # 0.13
                                (0, 100),   # 10.66
                                (1, 100),   # 11.1
                                (0, 10),    # 2.7
                                (0, 0.1),   # 0.079
                                (0, 100),   # 31
                                (0, 1),     # 0.3485
                                (0, 100),   # 71.55
                                (1, 10),    # 7.43
                                (0, 100),   # 25.428
                                (0, 1),     # 0.2444
                                (0, 10),    # 6.948
                                (0, 0.1),   # 0.04391
                                (0, 100),   # 37.78
                                (0, 1),     # 0.311
                                (0, 100),   # 79.23
                                (0, 1),     # 0.6
                                (0, 0.1),   # 0.057
                                (0, 1),     # 0.1
                                (0, 100),   # 32
                                (0, 0.1),   # 0.02424
                                (0, 0.1),   # 0.01052
                                (0, 1),     # 0.1378
                                (0, 100)]   # 40.14

        # Specifying pertubation kernel
        # - Uniform random walk with width 10% of prior range
        self.kernel = []
        for pr in self.prior_intervals:
            param_range = pr[1]-pr[0]
            self.kernel.append(Dist.Uniform(-1*param_range/20, param_range/20))

        # Loading fast Na channel experimental data
        vsteps, act_exp = data_ina.IV_DiasFig6()
        prepulses, inact_exp = data_ina.Inact_FukudaFig5C()
        intervals, rec_exp = data_ina.Recovery_ZhangFig4B()

        # Concatenate experimental data
        self.data_exp = [[vsteps, act_exp],
                         [prepulses, inact_exp],
                         [intervals, rec_exp]]

        # Setup simulations
        sim_act = sim.ActivationSim('ina.i_Na', vhold=-80, thold=3000,
                                    vmin=min(vsteps), vmax=max(vsteps),
                                    dv=vsteps[1]-vsteps[0], tstep=100)
        sim_inact = sim.InactivationSim('ina.G_Na', vhold=-20, thold=20,
                                        vmin=min(prepulses), vmax=max(prepulses),
                                        dv=prepulses[1]-prepulses[0], tstep=500)
        sim_rec = sim.RecoverySim('ina.G_Na', vhold=-120, thold=3000,
                                  vstep=-30, tstep1=20, tstep2=20,
                                  twaits=intervals)
        self.simulations = [sim_act, sim_inact, sim_rec]

class incx(AbstractChannel):
    def __init__(self):
        self.name = 'incx'
        self.model_name = 'Houston2017.mmt' # run in full model

        # Parameters involved in ABC process
        self.parameters = ['incx.k_NaCa',
                           'incx.k_sat',
                           'incx.eta']

        # Parameter specific prior intervals
        self.prior_intervals = [(0, 1000), # 292.8
                                (0, 1),    # 0.1
                                (0, 1)]    # 0.35

        # Specifying pertubation kernel
        # - Uniform random walk with width 10% of prior range
        self.kernel = []
        for pr in self.prior_intervals:
            param_range = pr[1]-pr[0]
            self.kernel.append(Dist.Uniform(-1*param_range/20, param_range/20))

        # Loading experimental data
        vsteps, act_exp = data_incx.IV_LuFig2()
        self.data_exp = [[vsteps, act_exp]]

        # Setup simulations
        sim_act = sim.TimeIndependentActivationSim('incx.i_NaCa', vsteps=vsteps)
        self.simulations = [sim_act]
