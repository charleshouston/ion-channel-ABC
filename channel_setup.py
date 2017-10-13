'''
Author: Charles Houston

Specific channel settings for use with approximate Bayesian computation procedure.
'''
import numpy as np
import math

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
        # Generate full parameter names for use in myokit model
        self.parameters = [self.name + '.' + p for p in self.parameter_names]

        # Specifying pertubation kernel
        #  Normal distribution with variance 20% of prior width
        self.kernel = []
        for pr in self.prior_intervals:
            prior_width = pr[1]-pr[0]
            self.kernel.append(Dist.Normal(0.0, 0.2 * prior_width))
            # Uncomment below for uniform distribution
            # self.kernel.append(Dist.Uniform(-math.sqrt(1.2 * prior_width),
            #                                 math.sqrt(1.2 * prior_width)))

        self.setup_simulations()

    def reset_params(self, new_params):
        '''
        Set parameters of channel to new values in prior draw.
        '''
        if len(self.simulations) is 0:
            print "Need to add simulations to channel first!"
        else:
            for sim in self.simulations:
                sim.set_parameters(self.parameters, new_params)

    def setup_simulations(self,continuous=False):
        '''
        Creates simulations from defined experimental conditions.
        '''
        self.simulations = []
        # stores vsteps/intervals for each sim output
        self.simulations_x = [[] for i in range(len(self.setup_exp))]

        for i,se in enumerate(self.setup_exp):
            
            normalised = False
            if 'normalise' in se:
                normalised = se['normalise']

            if se['sim_type'] == 'ActivationSim':
                if continuous:
                    dv = 1
                    self.simulations_x[i] = range(int(math.ceil(min(se['vsteps']))), int(math.ceil(max(se['vsteps']))+1))
                else:
                    dv = se['vsteps'][1] - se['vsteps'][0]
                    self.simulations_x[i] = se['vsteps']

                self.simulations.append(
                    sim.ActivationSim(se['variable'],
                                      vhold=se['vhold'],
                                      thold=se['thold'],
                                      vmin=min(se['vsteps']),
                                      vmax=max(se['vsteps']),
                                      dv=dv,
                                      tstep=se['tstep'],
                                      normalise=normalised))

            elif se['sim_type'] == 'InactivationSim':
                if continuous:
                    dv = 1
                    self.simulations_x[i] = range(int(math.ceil(min(se['vsteps']))), int(math.ceil(max(se['vsteps']))+1))
                else:
                    dv = se['vsteps'][1] - se['vsteps'][0]
                    self.simulations_x[i] = se['vsteps']           

                self.simulations.append(
                    sim.InactivationSim(se['variable'],
                                        vhold=se['vhold'],
                                        thold=se['thold'],
                                        vmin=min(se['vsteps']),
                                        vmax=max(se['vsteps']),
                                        dv=dv,
                                        tstep=se['tstep'],
                                        normalise=normalised))

            elif se['sim_type'] == 'RecoverySim':
                twaits = se['twaits']
                if continuous:
                    twaits = range(int(math.ceil(min(twaits))),
                                  int(math.ceil(max(twaits))+1))
                self.simulations_x[i] = twaits

                self.simulations.append(
                    sim.RecoverySim(se['variable'],
                                    vhold=se['vhold'],
                                    thold=se['thold'],
                                    vstep=se['vstep'],
                                    tstep1=se['tstep1'],
                                    tstep2=se['tstep2'],
                                    twaits=twaits))

            elif se['sim_type'] == 'TimeIndependentActivationSim':
                if continuous:
                    dv = 1
                    vsteps = range(int(math.ceil(min(se['vsteps']))), int(math.ceil(max(se['vsteps']))+1))
                else:
                    dv = se['vsteps'][1] - se['vsteps'][0]
                    vsteps = se['vsteps']
                self.simulations_x[i] = vsteps

                self.simulations.append(sim.TimeIndependentActivationSim(se['variable'],
                                                                         vsteps=vsteps))
            elif se['sim_type'] == 'MarkovActivationSim':
                if continuous:
                    dv = 1
                    vsteps = range(int(math.ceil(min(se['vsteps']))), int(math.ceil(max(se['vsteps']))+1))
                else:
                    dv = se['vsteps'][1] - se['vsteps'][0]
                    vsteps = se['vsteps']
                self.simulations_x[i] = vsteps

                self.simulations.append(
                    sim.MarkovActivationSim(se['variable'],
                                            vhold=se['vhold'],
                                            thold=se['thold'],
                                            vsteps=vsteps,
                                            tstep=se['tstep'],
                                            name=self.name,
                                            states=se['states'],
                                            params=se['params']))
            elif se['sim_type'] == 'MarkovInactivationSim':
                if continuous:
                    dv = 1
                    vsteps = range(int(math.ceil(min(se['vsteps']))), int(math.ceil(max(se['vsteps']))+1))
                else:
                    dv = se['vsteps'][1] - se['vsteps'][0]
                    vsteps = se['vsteps']
                self.simulations_x[i] = vsteps

                self.simulations.append(
                    sim.MarkovInactivationSim(se['variable'],
                                        vhold=se['vhold'],
                                        thold=se['thold'],
                                        vsteps=vsteps,
                                        tstep=se['tstep'],
                                        name=self.name,
                                        states=se['states'],
                                        params=se['params']))
            elif se['sim_type'] == 'MarkovRecoverySim':
                twaits = se['twaits']
                if continuous:
                    twaits = range(int(math.ceil(min(twaits))),
                                  int(math.ceil(max(twaits))+1))
                self.simulations_x[i] = twaits

                self.simulations.append(
                    sim.MarkovRecoverySim(se['variable'],
                                    vhold=se['vhold'],
                                    thold=se['thold'],
                                    vstep=se['vstep'],
                                    tstep1=se['tstep1'],
                                    tstep2=se['tstep2'],
                                    twaits=twaits,
                                    name=self.name,
                                    states=se['states'],
                                    params=se['params']))

            else:
                print "Unknown simulation type!"


    def simulate(self):
        '''
        Run the simulations necessary to generate values to compare with
        experimental results.
        '''
        if len(self.simulations) is 0:
            self.setup_simulations()

        sim_output = [[] for i in range(len(self.simulations))]

        # Run simulations with checks for errors
        #np.seterr(all='raise') # numpy raises exceptions if any numeric problems
        for i,sim in enumerate(self.simulations):
            # catch numpy division, overflow errors, etc
            try:
                out = sim.run(self.model_name)
            except:
                return None

            # If simulation fails, output will be None
            if out is None:
                return None

            # Myokit can output nans, so catch this here
            if np.isnan(out).any():
                return None

            # All good => add steps/intervals and dependent var to output!
            sim_output[i] = [self.simulations_x[i], out]

        return sim_output

class icat(AbstractChannel):
    def __init__(self):
        self.name = 'icat'
        self.model_name = 'Takeuchi2013_iCaT.mmt'
        self.publication = 'Takeuchi et al., 2013'

        # Parameters involved in ABC process
        self.parameter_names = ['g_CaT',
                                'E_CaT',
                                'k_dss1',
                                'k_dss2',
                                'k_dtau1',
                                'k_dtau2',
                                'k_dtau3',
                                'k_dtau4',
                                'k_dtau5',
                                'k_dtau6',
                                'k_fss1',
                                'k_fss2',
                                'k_ftau1',
                                'k_ftau2',
                                'k_ftau3',
                                'k_ftau4',
                                'k_ftau5',
                                'k_ftau6']


        # Parameter specific prior intervals
        # Original values given in comments
        self.prior_intervals = [(0, 1),  # 0.4122
                                (0, 100),# 45
                                (0,100), # 30
                                (1,10),  # 6.0
                                (0,10),  # 1.068
                                (0,100), # 26.3
                                (1,100), # 30
                                (0,10),  # 1.068
                                (0,100), # 26.3
                                (1,100), # 30
                                (0,100), # 48
                                (1,10),  # 7.0
                                (0,10),  # 1.53
                                (0,100), # 61.7
                                (1,100), # 83.3
                                (0, 10), # 1.5
                                (0,100), # 61.7
                                (1,100)] # 30

        # Edit which parameters to vary
        use = [1,1,1,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0]
        #use = [1 for i in range(len(self.parameter_names))]
        self.parameter_names = [p for i,p in enumerate(self.parameter_names) if use[i] == 1]
        self.prior_intervals = [pr for i,pr in enumerate(self.prior_intervals) if use[i] == 1]

        # Load experimental data
        vsteps, act_exp = data_icat.IV_DengFig1B()
        prepulses, inact_exp = data_icat.Inact_DengFig3B()
        intervals, rec_exp = data_icat.Recovery_DengFig4B()
        self.data_exp = [[vsteps, act_exp],
                         [prepulses, inact_exp],
                         [intervals, rec_exp]]

        # Define experimental setup for simulations
        setup_exp_act = {'sim_type': 'ActivationSim',
                         'variable': 'icat.i_CaT', 'vhold': -80, 'thold': 5000,
                         'vsteps': vsteps, 'tstep': 300,
                         'xlabel': 'Membrane potential (mV)',
                         'ylabel': 'Current density (pA/pF)'}
        setup_exp_inact = {'sim_type': 'InactivationSim',
                           'variable': 'icat.G_CaT', 'vhold': -20, 'thold': 300,
                           'vsteps': prepulses, 'tstep': 1000,
                           'xlabel': 'Membrane potential (mV)',
                           'ylabel': 'Normalised conductance'}
        setup_exp_rec = {'sim_type': 'RecoverySim',
                         'variable': 'icat.G_CaT', 'vhold': -80, 'thold': 5000,
                         'vstep': -20, 'tstep1': 300, 'tstep2': 300,
                         'twaits': intervals,
                         'xlabel': 'Interval (ms)',
                         'ylabel': 'Relative recovery'}
        self.setup_exp = [setup_exp_act, setup_exp_inact, setup_exp_rec]

        super(icat, self).__init__()


class ina(AbstractChannel):
    def __init__(self):
        self.name = 'ina'
        self.model_name = 'Bondarenko2004_iNa.mmt'
        self.publication = 'Bondarenko et al., 2004'

        # Parameters involved in ABC process
        self.parameter_names = ['g_Na',
                                'E_Na',
                                'p1',
                                'p2',
                                'p3',
                                'p4',
                                'p5',
                                'p6',
                                'p7',
                                'p8',
                                'p9',
                                'p10',
                                'p11',
                                'p12',
                                'p13',
                                'p14',
                                'p15',
                                'q1',
                                'q2',
                                'q3',
                                'q4',
                                'q5',
                                'q6',
                                'q7',
                                'q8',
                                'q9']
        # Parameter specific prior intervals
        self.prior_intervals = [(0, 100),               # 13
                                (0, 100),               # 58.686
                                (0, 1.0 / 3.802),       # 0.1027 / 3.802
                                (0, 1.0 / 3.802),       # 0.2 / 3.802
                                (0, 1.0 / 3.802),       # 0.23 / 3.802
                                (0, 1.0 / 3.802),       # 0.25 / 3.802
                                (-10, 10),              # 2.5
                                (1, 100),               # 17.0
                                (1, 100),               # 15.0
                                (1, 100),               # 12.0
                                (1, 1000),              # 150
                                (0, 1),                 # 0.188495
                                (-10, 10),              # 7.0
                                (1, 100),               # 16.6
                                (0, 1),                 # 0.393956
                                (0, 10),                # 7
                                (1, 10),                # 7.7
                                (0, 1.0),               # 0.1917
                                (0, 1.0),               # 0.20
                                (0, 1.0),               # 0.22
                                (-10, 10),              # 2.5
                                (-10, 10),              # -2.5
                                (-10, 10),              # -7.5
                                (1, 100),               # 20.3
                                (0, 100),               # 84
                                (0, 100)]               # 20

        # Markov parameters and states
        self.markov_params = [self.name + '.' + p for p in self.parameter_names]
        markov_states = ['C1', 'C2', 'C3', 'I1', 'I2', 'IC2', 'IC3', 'IF', 'O']
        self.markov_states = [self.name + '.' + s for s in markov_states]

        # Loading experimental data
        vsteps, act_exp = data_ina.IV_DiasFig6()
        prepulses, inact_exp = data_ina.Inact_FukudaFig5C()
        intervals, rec_exp = data_ina.Recovery_ZhangFig4B()
        self.data_exp = [[vsteps, act_exp],
                         [prepulses, inact_exp]]

        # Define experimental setup for simulations
        setup_exp_act = {'sim_type': 'ActivationSim',
                         'variable': 'ina.i_Na', 'vhold': -80, 'thold': 500,
                         'vsteps': vsteps, 'tstep': 100,
                         'xlabel': 'Membrane potential (mV)',
                         'ylabel': 'Current density (pA/pF)'}
                        #  'params': self.markov_params,
                        #  'states': self.markov_states}
        setup_exp_inact = {'sim_type': 'InactivationSim',
                           'variable': 'ina.G_Na', 'vhold': -20, 'thold': 20,
                           'vsteps': prepulses, 'tstep': 500,
                           'xlabel': 'Membrane potential (mV)',
                           'ylabel': 'Normalised conductance'}
                        #    'params': self.markov_params,
                        #    'states': self.markov_states}
        setup_exp_rec = {'sim_type': 'MarkovRecoverySim',
                         'variable': 'ina.G_Na', 'vhold': -120, 'thold': 500,
                         'vstep': -40, 'tstep1': 20, 'tstep2': 20,
                         'twaits': intervals,
                         'xlabel': 'Interval (ms)',
                         'ylabel': 'Relative recovery'}
        #                  'params': markov_params,
        #                  'states': markov_states}
        self.setup_exp = [setup_exp_act, setup_exp_inact]

        super(ina, self).__init__()


class ikur(AbstractChannel):
    def __init__(self):
        self.name = 'ikur'
        self.model_name = 'Bondarenko2004_iKur.mmt'
        self.publication = 'Bondarenko et al., 2004'

        # Parameters involved in ABC process
        self.parameter_names = ['g_Kur',
                                'k_ass1',
                                'k_ass2',
                                'k_atau1',
                                'k_atau2',
                                'k_atau3',
                                'k_iss1',
                                'k_iss2',
                                'k_itau1',
                                'k_itau2']

        # Parameter specific prior intervals
        self.prior_intervals = [(0, 1),     # 0.0975
                                (0, 100),   # 22.5
                                (1, 10),    # 7.7
                                (0, 1),     # 0.493
                                (0, 10),    # 6.29
                                (0, 10),    # 2.058
                                (0, 100),   # 45.2
                                (1, 10),    # 5.7
                                (0, 10),    # 1.2
                                (0, 10)]    # 1.7

        # Edit which parameters to vary
        use = [1,1,1,0,0,0,1,1,0,0]
        #use = [1 for i in range(len(self.parameter_names))]
        self.parameter_names = [p for i,p in enumerate(self.parameter_names) if use[i] == 1]
        self.prior_intervals = [pr for i,pr in enumerate(self.prior_intervals) if use[i] == 1]

        # Loading experimental data
        vsteps, act_peaks_exp = data_ikur.IV_MaharaniFig2B()
        prepulses, inact_exp = data_ikur.Inact_BrouilleteFig6B()
        intervals, rec_exp = data_ikur.Recovery_BrouilleteFig6D()
        self.data_exp = [[vsteps, act_peaks_exp],
                         [prepulses, inact_exp],
                         [intervals, rec_exp]]

        # Define experimental setup for simulations
        setup_exp_act = {'sim_type': 'ActivationSim',
                         'variable': 'ikur.i_Kur', 'vhold': -60, 'thold': 6000,
                         'vsteps': vsteps, 'tstep': 300,
                         'xlabel': 'Membrane potential (mV)',
                         'ylabel': 'Current density (pA/pF)'}
        setup_exp_inact = {'sim_type': 'InactivationSim',
                           'variable': 'ikur.G_Kur', 'vhold': 30, 'thold': 2500,
                           'vsteps': prepulses, 'tstep': 5000,
                           'xlabel': 'Membrane potential (mV)',
                           'ylabel': 'Normalised conductance'}
        setup_exp_rec = {'sim_type': 'RecoverySim',
                         'variable': 'ikur.G_Kur', 'vhold': -80, 'thold': 2000,
                         'vstep': 30, 'tstep1': 1500, 'tstep2': 500,
                         'twaits': intervals,
                         'xlabel': 'Interval (ms)',
                         'ylabel': 'Relative recovery'}

        self.setup_exp = [setup_exp_act, setup_exp_inact, setup_exp_rec]

        super(ikur, self).__init__()


class ical(AbstractChannel):
    def __init__(self):
        self.name = 'ical'
        self.model_name = 'Houston2017_iCaL.mmt' # use full model
        self.publication = 'Korhonen et al., 2009'

        # Parameters involved in ABC process
        self.parameter_names = ['G_CaL',
                                'p1',
                                'p2',
                                'p3',
                                'p4',
                                'p5',
                                'p6',
                                'p7',
                                'p8',
                                'q1',
                                'q2',
                                'q3',
                                'q4',
                                'q5',
                                'q6',
                                'q7',
                                'q8',
                                'q9']

        # Parameter specific prior intervals
        # Original values given in comments
        self.prior_intervals = [(0, 0.0001), # G_CaL
                                (-50, 50),  # p1
                                (0, 10),    # p2
                                (-100, 50),  # p3 
                                (-50, 50),   # p4
                                (-50, 50),    # p5
                                (-50, 50),    # p6
                                (0, 200),   # p7
                                (0, 200),   # p8
                                (0, 100),   # q1
                                (0, 10),    # q2
                                (0, 10000), # q3 
                                (0, 100),   # q4
                                (0, 1000),  # q5
                                (0, 1000),  # q6   
                                (0, 100),   # q7
                                (0, 100),   # q8
                                (-500, 500)]  # q9 

        # Loading experimental data
        # vsteps, act_exp = data_ical.IV_DiasFig7()
        vsteps, act_exp = data_ical.IV_RaoFig3B()
        act_exp = [17.1*a for a in act_exp] # maximum current reported by Dias et al (2014)
        prepulses, inact_exp = data_ical.Inact_RaoFig3C()
        self.data_exp = [[vsteps, act_exp],
                         [prepulses, inact_exp]]

        # Define experimental setup for simulations
        setup_exp_act = {'sim_type': 'ActivationSim',
                         'variable': 'ical.i_CaL', 'vhold': -80, 'thold': 2000,
                         'vsteps': vsteps, 'tstep': 250,
                         'normalise': False,
                         'xlabel': 'Membrane potential (mV)',
                         'ylabel': 'Current density (pA/pF)'}
        setup_exp_inact = {'sim_type': 'InactivationSim',
                           'variable': 'ical.i_CaL', 'vhold': -20, 'thold': 400,
                           'vsteps': prepulses, 'tstep': 1000,
                           'normalise': True,
                           'xlabel': 'Membrane potential (mV)',
                           'ylabel': 'Normalised conductance'}
        self.setup_exp = [setup_exp_act, setup_exp_inact]


        super(ical, self).__init__()


class ikr(AbstractChannel):
    def __init__(self):
        self.name = 'ikr'
        self.model_name = 'Takeuchi2013_iKr.mmt'
        self.publication = 'Takeuchi et al., 2013'

        # Parameters involved in ABC process
        self.parameter_names = ['g_Kr',
                                'k_xss1',
                                'k_xss2',
                                'k_xtau1',
                                'k_xtau2',
                                'k_xtau3',
                                'k_xtau4',
                                'k_xtau5',
                                'k_r1',
                                'k_r2']

        # Parameter specific prior intervals
        self.prior_intervals = [(0, 1),     # 0.73
                                (0, 100),   # 15
                                (1, 10),    # 6
                                (0, 10),    # 2.5
                                (0, 100),   # 31.18
                                (0, 1000),  # 217.18
                                (0, 100),   # 20.1376
                                (1, 100),   # 22.1996
                                (0, 100),   # 55
                                (1, 100)]   # 24

        # Edit which parameters to vary
        use = [1,1,1,0,0,0,0,0,1,1]
        #use = [1 for i in range(len(self.parameter_names))]
        self.parameter_names = [p for i,p in enumerate(self.parameter_names) if use[i] == 1]
        self.prior_intervals = [pr for i,pr in enumerate(self.prior_intervals) if use[i] == 1]

        # Loading experimental data
        vsteps, act_peaks_exp = data_ikr.IV_Li7B()
        vsteps2, act_exp = data_ikr.Activation_Li7B()
        # Normalise act_exp
        act_exp = [float(i) / max(act_exp) for i in act_exp]
        self.data_exp = [[vsteps, act_peaks_exp],
                         [vsteps2, act_exp]]

        # Experimental setup
        setup_exp_act = {'sim_type': 'ActivationSim',
                         'variable': 'ikr.i_Kr', 'vhold': -50, 'thold': 5000,
                         'vsteps': vsteps, 'tstep': 1000,
                         'xlabel': 'Membrane potential (mV)',
                         'ylabel': 'Current density (pA/pF)'}
        setup_exp_inact = {'sim_type': 'InactivationSim',
                           'variable': 'ikr.G_Kr', 'vhold': -50, 'thold': 2000,
                           'vsteps': vsteps2, 'tstep': 1000,
                           'xlabel': 'Membrane potential (mV)',
                           'ylabel': 'Normalised conductance'}
        self.setup_exp = [setup_exp_act, setup_exp_inact]

        super(ikr, self).__init__()


class iha(AbstractChannel):
    def __init__(self):
        self.name = 'iha'
        self.model_name = 'Majumder2016_iha.mmt'
        self.publication = 'Majumder et al., 2016'

        # Parameters involved in ABC process
        self.parameter_names = ['k_yss1',
                                'k_yss2',
                                'k_ytau1',
                                'k_ytau2',
                                'k_ytau3',
                                'k_ytau4',
                                'k_ytau5',
                                'k_ytau6',
                                'k_ytau7',
                                'k_i_haNa',
                                'g_ha']

        # Parameter specific prior intervals
        self.prior_intervals = [(0, 100),   # 78.65
                                (1, 10),    # 6.33
                                (0, 10),    # 1
                                (0, 1.0),   # 0.11885
                                (0, 100),   # 75
                                (1, 100),   # 28.37
                                (0, 1.0),   # 0.56236
                                (0, 100),   # 75
                                (0, 100),   # 14.19
                                (0, 1.0),   # 0.2
                                (0, 0.1)]   # 0.021

        # Edit which parameters to vary
        use = [1,1,0,0,0,0,0,0,0,1,1]
        #use = [1 for i in range(len(self.parameter_names))]
        self.parameter_names = [p for i,p in enumerate(self.parameter_names) if use[i] == 1]
        self.prior_intervals = [pr for i,pr in enumerate(self.prior_intervals) if use[i] == 1]

        # Loading experimental data
        vsteps, act_peaks_exp = data_iha.IV_Sartiana5B()
        prepulses, inact_exp = data_iha.Inact_Sartiana4B()
        self.data_exp = [[vsteps, act_peaks_exp],
                         [prepulses, inact_exp]]

        # Experimental setup
        setup_exp_act = {'sim_type': 'ActivationSim',
                         'variable': 'iha.i_ha', 'vhold': -120, 'thold': 1500,
                         'vsteps': vsteps, 'tstep': 300,
                         'xlabel': 'Membrane potential (mV)',
                         'ylabel': 'Current density (pA/pF)'}
        setup_exp_inact = {'sim_type': 'InactivationSim',
                           'variable': 'iha.G_ha', 'vhold': 40, 'thold': 2000,
                           'vsteps': prepulses, 'tstep': 500,
                           'xlabel': 'Membrane potential (mV)',
                           'ylabel': 'Normalised conductance'}
        self.setup_exp = [setup_exp_act, setup_exp_inact]

        super(iha, self).__init__()


class ito(AbstractChannel):
    def __init__(self):
        self.name = 'ito'
        self.model_name = 'Takeuchi2013_ito.mmt'
        self.publication = 'Takeuchi et al., 2013'

        # Parameters involved in ABC process
        self.parameter_names = ['g_to',
                                'k_xss1',
                                'k_xss2',
                                'k_xtau1',
                                'k_xtau2',
                                'k_xtau3',
                                'k_yss1',
                                'k_yss2',
                                'k_ytau1',
                                'k_ytau2',
                                'k_ytau3',
                                'k_ytau4']

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

        # Edit which parameters to vary
        use = [1,1,1,0,0,0,1,1,0,0,0,0]
        #use = [1 for i in range(len(self.parameter_names))]
        self.parameter_names = [p for i,p in enumerate(self.parameter_names) if use[i] == 1]
        self.prior_intervals = [pr for i,pr in enumerate(self.prior_intervals) if use[i] == 1]

        # Loading experimental data
        vsteps, act_peaks_exp = data_ito.IV_KaoFig6()
        prepulses, inact_exp = data_ito.Inact_XuFig9C()
        self.data_exp = [[vsteps, act_peaks_exp],
                         [prepulses, inact_exp]]

        # Define experimental setup for simulations
        setup_exp_act = {'sim_type': 'ActivationSim',
                         'variable': 'ito.i_to', 'vhold': -40, 'thold': 1000,
                         'vsteps': vsteps, 'tstep': 300,
                         'xlabel': 'Membrane potential (mV)',
                         'ylabel': 'Current density (pA/pF)'}
        setup_exp_inact = {'sim_type': 'InactivationSim',
                           'variable': 'ito.G_to', 'vhold': 50, 'thold': 500,
                           'vsteps': prepulses, 'tstep': 500,
                           'xlabel': 'Membrane potential (mV)',
                           'ylabel': 'Normalised conductance'}
        self.setup_exp = [setup_exp_act, setup_exp_inact]

        super(ito, self).__init__()


class ikach(AbstractChannel):
    def __init__(self):
        self.name = 'ikach'
        self.model_name = 'Majumder2016_iKAch.mmt'
        self.publication = 'Majumder et al., 2016'

        # Parameters involved in ABC process
        self.parameter_names = ['g_KAch',
                                'k_1',
                                'k_2',
                                'k_3',
                                'k_4',
                                'k_5',
                                'k_6',
                                'k_7',
                                'k_8']

        # Parameter specific prior intervals
        self.prior_intervals = [(0, 1),   # 0.37488
                                (0, 10),  # 3.5
                                (0, 10),   # 9.13652
                                (0, 1),    # 0.477811
                                (0, 0.1),  # 0.04
                                (0, 1),    # 0.23
                                (0, 1000), # 102
                                (0, 100),  # 10
                                (0, 100)]  # 10

        # Loading experimental data
        vsteps, act_peaks_exp = data_ikach.IV_KaoFig6()
        self.data_exp = [[vsteps, act_peaks_exp]]

        # Define experimental setup for simulations
        setup_exp_act = {'sim_type': 'TimeIndependentActivationSim',
                         'variable': 'ikach.i_KAch',
                         'vsteps': vsteps,
                         'xlabel': 'Membrane potential (mV)',
                         'ylabel': 'Current density (pA/pF)'}
        self.setup_exp = [setup_exp_act]

        super(ikach, self).__init__()


class ik1(AbstractChannel):
    def __init__(self):
        self.name = 'ik1'
        self.model_name = 'Korhonen2009_iK1.mmt'
        self.publication = 'Korhonen et al., 2009'

        self.parameter_names = ['g_K1',
                                #'k_1',
                                'k_2',
                                'k_3',
                                'k_4']

        self.prior_intervals = [(0, 0.2),   # 0.0515
                                #(-500, 500),  # 210
                                (0, 50),  # -6.1373
                                (0, 1),     # 0.1653
                                (0, 0.1)]   # 0.0319


        # Loading experimental data
        vsteps, act_peaks_exp = data_ik1.IV_GoldoniFig3D()
        # Convert to current densities using value reported for current
        #  density at -150mV in original paper (-42.2pA/pF)
        max_curr_density = -42.2
        act_peaks_exp = [p * max_curr_density / act_peaks_exp[0] for p in act_peaks_exp]
        self.data_exp = [[vsteps, act_peaks_exp]]

        # Define experimental setup for simulations
        setup_exp_act = {'sim_type': 'TimeIndependentActivationSim',
                         'variable': 'ik1.i_K1',
                         'vsteps': vsteps,
                         'xlabel': 'Membrane potential (mV)',
                         'ylabel': 'Current density (pA/pF)'}
        self.setup_exp = [setup_exp_act]

        super(ik1, self).__init__()

class full_sim(AbstractChannel):
    def __init__(self):
        self.name = 'fullsim'
        self.model_name = 'Houston2017.mmt'
        self.publication = 'na'

        self.parameter_names =['inak.i_NaK_max','incx.k_NCX','icab.g_Cab','inab.g_Nab']

        self.prior_intervals = [(0, 10),    # 0.88
                                (0, 500),   # 292.8
                                (0, 0.001), # 0.000367
                                (0, 0.01)]  # 0.0026

        self.data_exp = [['membrane.V',
                          'calcium_concentration.Cai',
                          'sodium_concentration.Nai',
                          'potassium_conc.Ki',
                          ],
                          [-67.0,0.115]]

        self.setup_exp = [{'sim_type': 'FullSimulation'}]

class ina2(AbstractChannel):
    def __init__(self):
        self.name = 'ina'
        self.model_name = 'Korhonen2009_iNa.mmt'
        self.publication = 'Korhonen et al., 2009'

        # Parameters involved in ABC process
        self.parameter_names = ['g_Na',
                                'v_offset',
                                'p1',
                                'p2',
                                'q1',
                                'q2']
        # Parameter specific prior intervals
        self.prior_intervals = [(0, 100),       # 35
                                (-50, 50),      # 0
                                (0, 100),       # 45
                                (-10, 0),      # -6.5
                                (0, 100),       # 76.1
                                (0, 10)         # 6.07
                                ]

        # Loading experimental data
        vsteps, act_exp = data_ina.IV_DiasFig6()
        prepulses, inact_exp = data_ina.Inact_FukudaFig5C()
        intervals, rec_exp = data_ina.Recovery_ZhangFig4B()
        self.data_exp = [[vsteps, act_exp],
                         [prepulses, inact_exp],
                         [intervals, rec_exp]]

        # Define experimental setup for simulations
        setup_exp_act = {'sim_type': 'ActivationSim',
                         'variable': 'ina.i_Na', 'vhold': -80, 'thold': 500,
                         'vsteps': vsteps, 'tstep': 100,
                         'xlabel': 'Membrane potential (mV)',
                         'ylabel': 'Current density (pA/pF)'}
        setup_exp_inact = {'sim_type': 'InactivationSim',
                           'variable': 'ina.G_Na', 'vhold': -20, 'thold': 20,
                           'vsteps': prepulses, 'tstep': 500,
                           'xlabel': 'Membrane potential (mV)',
                           'ylabel': 'Normalised conductance'}
        setup_exp_rec = {'sim_type': 'RecoverySim',
                         'variable': 'ina.G_Na', 'vhold': -120, 'thold': 500,
                         'vstep': -40, 'tstep1': 20, 'tstep2': 20,
                         'twaits': intervals,
                         'xlabel': 'Interval (ms)',
                         'ylabel': 'Relative recovery'}
        self.setup_exp = [setup_exp_act, setup_exp_inact, setup_exp_rec]

        super(ina2, self).__init__()
