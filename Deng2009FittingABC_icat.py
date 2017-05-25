'''
Author: Charles Houston, MRes Biomedical Research student.

ABC parameter estimation for the Hodkin-Huxley action potential model.
Following process in Daly et al, 2015. Re-written to use with myokit.
'''

import fitting       # import ABC fitting procedure
import distributions as Dist # prob dist functions
import HodgkinHuxley # Hodgkin-Huxley data from original paper

# try:
#     import unittest2 as unittest
# except ImportError:
#     import unittest

import matplotlib.pyplot as plt
import numpy as np

import myokit

class TestHHProto():

    # Fits the 5 parameters controlling potassium conductance in the full Hodgkin-Huxley model
    def TestGKFitting(self):
        # Get model and protocol, create simulation
        m,p,x = myokit.load('hodgkin_huxley.mmt')

        # Get membrane potential
        v = m.get('membrane.V')
        # Demote v from a state to an ordinary variable
        v.demote()
        v.set_rhs(0)
        # Set voltage to pacing variable
        v.set_binding('pace')

        # Get potassium parameters
        ank1 = m.get('potassium_channel_n_gate.ank1')
        ank2 = m.get('potassium_channel_n_gate.ank2')
        ank3 = m.get('potassium_channel_n_gate.ank3')
        bnk1 = m.get('potassium_channel_n_gate.bnk1')
        bnk2 = m.get('potassium_channel_n_gate.bnk2')

        # Set sodium parameters to zero to reduce computation
        alpha_h = m.get('sodium_channel_h_gate.alpha_h')
        beta_h = m.get('sodium_channel_h_gate.beta_h')
        alpha_m = m.get('sodium_channel_m_gate.alpha_m')
        beta_m = m.get('sodium_channel_m_gate.beta_m')
        alpha_h.set_rhs(0)
        beta_h.set_rhs(0)
        alpha_m.set_rhs(0)
        beta_m.set_rhs(0)

        # Create the simulation
        s = myokit.Simulation(m)
        depols    = [109, 100, 88, 76, 63, 51, 38, 32, 26, 19, 10.001] # 10.001 prevents numerical error
        depols[:] = [d-75 for d in depols]
        p = myokit.pacing.steptrain(
            vsteps = depols,
            vhold  = -75,
            tpre  = 10000,
            tstep = 300,
        )
        s.set_protocol(p)
        t = p.characteristic_time()
        d = myokit.DataLog()

        outfile = open('ABCPredPotassium.txt','w')

        # Initial values and priors
        init = [0.5,50.0,50.0,0.5,50.0]
        priors = [Dist.Uniform(0,1),Dist.Uniform(0,100),Dist.Uniform(1,100),Dist.Uniform(0,1),Dist.Uniform(1,100)]

        # ABC expects this form - sets alpha/beta, runs protocol, then returns sq_err of result
        def distance(params,times,vals):
            # Reset the model state before evaluating again
            s.reset()

            # Set alpha_n and beta_n to current parameter values
            s.set_constant(ank1,params[0])
            s.set_constant(ank2,params[1])
            s.set_constant(ank3,params[2])
            s.set_constant(bnk1,params[3])
            s.set_constant(bnk2,params[4])

            # Run model
            d = s.run(t,log=['environment.time','potassium_channel.G_K'])
            # Trim output to include only 12ms of peak current
            ds = d.split_periodic(10300, adjust=True)
            for d in ds:
                d.trim_left(10000, adjust=True)
                d.trim_right(15)

            # Iterate through voltage clamp experiments and average RMSE
            tot_rmse = 0.0
            for i,d in enumerate(ds):
                tot_rmse = tot_rmse = CheckAgainst(d['environment.time'],
                                                   d['potassium_channel.G_K'],
                                                   times[i],
                                                   vals[i])
            return tot_rmse/11

        def kern(orig,new=None):
            g1 = Dist.Normal(0.0,0.1)
            g10 = Dist.Normal(0.0,1.0)
            g100 = Dist.Normal(0.0,10.0)
            
            if new == None:
                new = []
                perturb = [g1.draw(),g100.draw(),g100.draw(),g1.draw(),g100.draw()]
                for i in range(len(orig)):
					new = new + [orig[i]+perturb[i]]
                return new
            else:
                prob = 1.0
                prob = prob*g1.pdf(new[0]-orig[0])
                prob = prob*g100.pdf(new[1]-orig[1])
                prob = prob*g100.pdf(new[2]-orig[2])
                prob = prob*g1.pdf(new[3]-orig[3])
                prob = prob*g100.pdf(new[4]-orig[4])
                return prob

        # Compose a 2D array of the original Hodgkin-Huxley time and conductance data
        original_times = []
        original_G_K = []
        for i in range(11):
            [times,vals,a,b] = HodgkinHuxley.fig3(i)
            original_times = original_times + [times]
            original_G_K = original_G_K + [vals]
        
        result = fitting.approx_bayes_smc_adaptive(init,priors,original_times,original_G_K,prior_func,kern,distance,100,25000,0.003)

        # Write results to the standard output and ABCPredPotassium.txt
        print result.getmean()
        print result.getvar()
        outfile.write(str(result.pool)+"\n")
        outfile.write(str(result.getmean())+"\n")
        outfile.write(str(result.getvar())+"\n")

    # Fits Sodium conductance gating rates alpha_m, beta_m, alpha_h and beta_h from the
    #   simplified Hodgkin-Huxley model over a range of depolarisation values.
    def TestGNaFitting(self):
        # Get model and protocol, create simulation
        m,p,x = myokit.load('hodgkin_huxley.mmt')

        # Get membrane potential
        v = m.get('membrane.V')
        # Demote v from a state to an ordinary variable
        v.demote()
        v.set_rhs(0)
        # Set voltage to pacing variable
        v.set_binding('pace')

        # Get sodium parameters
        ahk1 = m.get('sodium_channel_h_gate.ahk1')
        ahk2 = m.get('sodium_channel_h_gate.ahk2')
        bhk1 = m.get('sodium_channel_h_gate.bhk1')
        bhk2 = m.get('sodium_channel_h_gate.bhk2')
        amk1 = m.get('sodium_channel_m_gate.amk1')
        amk2 = m.get('sodium_channel_m_gate.amk2')
        amk3 = m.get('sodium_channel_m_gate.amk3')
        bmk1 = m.get('sodium_channel_m_gate.bmk1')
        bmk2 = m.get('sodium_channel_m_gate.bmk2')

        # Disable ODEs of potassium parameters
        ank1 = m.get('potassium_channel_n_gate.ank1')
        ank2 = m.get('potassium_channel_n_gate.ank2')
        ank3 = m.get('potassium_channel_n_gate.ank3')
        bnk1 = m.get('potassium_channel_n_gate.bnk1')
        bnk2 = m.get('potassium_channel_n_gate.bnk2')

        ank1.set_rhs(0)
        ank2.set_rhs(0)
        ank3.set_rhs(0)
        bnk1.set_rhs(0)
        bnk2.set_rhs(0)

        # Create the simulation
        s = myokit.Simulation(m)
        s.set_max_step_size(0.1)

        depols    = [109, 100, 88, 76, 63, 51, 38, 32, 26, 19, 10.001, 6]
        p = myokit.pacing.steptrain(
            vsteps = depols,
            vhold  = 0,
            tpre  = 10000,
            tstep = 300,
        )
        s.set_protocol(p)
        t = p.characteristic_time()
        d = myokit.DataLog()

        outfile = open('ABCPredSodium.txt','w')

        # Initial values and priors
        priors_m = [Dist.Uniform(0,1),Dist.Uniform(0,100),Dist.Uniform(1,100),Dist.Uniform(0,10),Dist.Uniform(1,100)]
        priors_h = [Dist.Uniform(0,1),Dist.Uniform(1,100),Dist.Uniform(0,100),Dist.Uniform(1,100)]
        priors = priors_m + priors_h
        init = [0.5,50.0,50.0,0.5,50.0,0.5,50.0,50.0,50.0]

        def distance(params,times,vals):
            # Reset the model state before evaluating again
            s.reset()

            # Set alpha_n and beta_n to current parameter values
            s.set_constant(amk1,params[0])
            s.set_constant(amk2,params[1])
            s.set_constant(amk3,params[2])
            s.set_constant(bmk1,params[3])
            s.set_constant(bmk2,params[4])
            s.set_constant(ahk1,params[5])
            s.set_constant(ahk2,params[6])
            s.set_constant(bhk1,params[7])
            s.set_constant(bhk2,params[8])

                # Run model
            d = s.run(t,log=['environment.time','sodium_channel.G_Na'])
            # Trim output to include only 12ms of peak current
            ds = d.split_periodic(10300, adjust=True)
            for d in ds:
                d.trim_left(10000, adjust=True)
                d.trim_right(12)

            # Iterate through voltage clamp experiments and average RMSE
            tot_rmse = 0.0
            for i,d in enumerate(ds):
                tot_rmse = tot_rmse = CheckAgainst(d['environment.time'],
                                                   d['sodium_channel.G_Na'],
                                                   times[i],
                                                   vals[i])
            return tot_rmse/12

        def kern(orig,new=None):
            g1 = Dist.Normal(0.0,0.1)
            g10 = Dist.Normal(0.0,1.0)
            g100 = Dist.Normal(0.0,10.0)
            
            if new==None:
                perturb_m = [g1.draw(),g100.draw(),g100.draw(),g1.draw(),g100.draw()]
                perturb_h = [g1.draw(),g100.draw(),g100.draw(),g100.draw()]
                perturb = perturb_m + perturb_h
                new = []
                for i in range(len(orig)):
                    new = new + [orig[i]+perturb[i]]
                return new
            else:
                prob = 1.0
                prob = prob*g1.pdf(new[0]-orig[0])
                prob = prob*g100.pdf(new[1]-orig[1])
                prob = prob*g100.pdf(new[2]-orig[2])
                prob = prob*g1.pdf(new[3]-orig[3])
                prob = prob*g100.pdf(new[4]-orig[4])
                prob = prob*g1.pdf(new[5]-orig[5])
                prob = prob*g100.pdf(new[6]-orig[6])
                prob = prob*g100.pdf(new[7]-orig[7])
                prob = prob*g100.pdf(new[8]-orig[8])
                return prob

        # Compose a 2D array of the original Hodgkin-Huxley time and conductance data
        original_times = []
	original_G_Na = []
	for i in range(12):
	    [times,vals,a,b,c,d] = HodgkinHuxley.fig6(i)
	    original_times = original_times + [times]
	    original_G_Na = original_G_Na + [vals]

        result = fitting.approx_bayes_smc_adaptive(init,priors,original_times,original_G_Na,prior_func,kern,distance,100,25000,0.001)

        # Write results to the standard output and ABCPredPotassium.txt
        print result.getmean()
        print result.getvar()
        outfile.write(str(result.pool)+"\n")
        outfile.write(str(result.getmean())+"\n")
        outfile.write(str(result.getvar())+"\n")


'''
    HELPER METHODS
'''

# Evaluates RMSE between experimental and predicted values
# Uses time points in simulation that are closest to experimental
def CheckAgainst(predTimes, predVals, experTimes, experVals):
    curr = 0
    predValsClosest = []

    # Finds experimental output at times closest to experimental
    for tval in experTimes:
        while tval > predTimes[curr+1]:
            if curr >= len(predTimes)-2:
                break
            curr = curr+1
        if abs(tval-predTimes[curr]) < abs(tval-predTimes[curr+1]):
            predValsClosest = predValsClosest + [predVals[curr]]
        else:
            predValsClosest = predValsClosest + [predVals[curr+1]]
    # Calculate squared error
    sq_err = 0

    for i,val in enumerate(experVals):
        sq_err = sq_err + pow(val-predValsClosest[i],2)
    return pow(sq_err/len(experVals),0.5)

# Simple multiplicative prior for list of independent Distribution objects
def prior_func(priors,params):
    prob = 1.0
    for i,distr in enumerate(priors):
        prob = prob * distr.pdf(params[i])
    return prob

if __name__ == '__main__':
    x = TestHHProto()
#    x.TestGKFitting()
    x.TestGNaFitting()
