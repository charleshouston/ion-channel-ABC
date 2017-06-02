'''
Author: Charles Houston, MRes Biomedical Research student.

ABC parameter estimation for the Takeuchi formulation of the T-type
Calcium channel for HL-1 myocytes.
Following process in Daly et al, 2015. Re-written to use with myokit.
'''

import fitting_mult as fitting       # import ABC fitting procedure
import distributions as Dist # prob dist functions
import Deng2009 # Experimental data from Deng et al, 2009

import matplotlib.pyplot as plt
import numpy as np

import myokit

import simulations

class TestICaTProto():

    # Fits the parameters controlling T-type calcium conductance
    def TestICaTFitting(self):

        # Output file
        outfile = open('ABCPredCalciumTType.txt','w')

        # Initial values and priors
        # - Prior is uniform distribution ({0,1},order of mag larger than proposed value)
        # - Init is mean of prior
        priors = [Dist.Uniform(0,100),
                  Dist.Uniform(1,10),
                  Dist.Uniform(0,10),
                  Dist.Uniform(0,100),
                  Dist.Uniform(1,100),
                  Dist.Uniform(0,10),
                  Dist.Uniform(1,100),
                  Dist.Uniform(0,100),
                  Dist.Uniform(1,10),
                  Dist.Uniform(0,0.1),
                  Dist.Uniform(0,100),
                  Dist.Uniform(1,100),
                  Dist.Uniform(0,0.1),
                  Dist.Uniform(1,100)]
        init = [50, 5, 5, 50, 50, 5, 50, 50, 5, 0.05, 50, 50, 0.05, 50]

        # ABC expects this form - sets alpha/beta, runs protocol, then returns sq_err of result
        def distance(params,vals,s,reversal_potential):

            # Run simulations
            ResetSim(s,params)
            act_pred = simulations.activation_sim(s,vsteps,reversal_potential)
            # If this simulation failed, it will return all zeros
            # No point in running the rest!
            if not act_pred[0].any():
                inact_pred = np.zeros(7)
                # rec_pred = np.zeros(11)
            else:
                ResetSim(s,params)
                inact_pred = simulations.inactivation_sim(s,prepulses,act_pred[0])
                # ResetSim(s,params)
                # rec_pred = simulations.recovery_sim(s,intervals)

            if not inact_pred.any():
                rec_pred = np.zeros(11)
            else:
                ResetSim(s,params)
                rec_pred = simulations.recovery_sim(s,intervals)

            # Return RMSE for all simulations
            #predVals = np.hstack((act_pred,inact_pred,rec_pred))
            pred_vals = [act_pred[0], act_pred[1], inact_pred, rec_pred]

            return LossFunction(pred_vals, vals)


        def kern(orig,new=None):
            g1 = Dist.Normal(0.0,0.01)
            g10 = Dist.Normal(0.0,1.0)
            g100 = Dist.Normal(0.0,10.0)
            
            if new == None:
                new = []
                perturb = [g100.draw(),
                           g10.draw(),
                           g10.draw(),
                           g100.draw(),
                           g100.draw(),
                           g10.draw(),
                           g100.draw(),
                           g100.draw(),
                           g10.draw(),
                           g1.draw(),
                           g100.draw(),
                           g100.draw(),
                           g1.draw(),
                           g100.draw()]
                for i in range(len(orig)):
					new = new + [orig[i]+perturb[i]]
                return new
            else:
                prob = 1.0
                prob = prob*g100.pdf(new[0]-orig[0])
                prob = prob*g10.pdf(new[1]-orig[1])
                prob = prob*g10.pdf(new[2]-orig[2])
                prob = prob*g100.pdf(new[3]-orig[3])
                prob = prob*g100.pdf(new[4]-orig[4])
                prob = prob*g10.pdf(new[5]-orig[5])
                prob = prob*g100.pdf(new[6]-orig[6])
                prob = prob*g100.pdf(new[7]-orig[7])
                prob = prob*g10.pdf(new[8]-orig[8])
                prob = prob*g1.pdf(new[9]-orig[9])
                prob = prob*g100.pdf(new[10]-orig[10])
                prob = prob*g100.pdf(new[11]-orig[11])
                prob = prob*g1.pdf(new[12]-orig[12])
                prob = prob*g100.pdf(new[13]-orig[13])

                return prob


        # Load experimental data
        # - IV data
        vsteps,i_exp = Deng2009.fig1B()
        vsteps = np.array(vsteps)
        i_exp = np.array(i_exp)
        # - Activation/Inactivation data
        vsteps_act,act_exp = Deng2009.fig3Bact()
        prepulses,inact_exp = Deng2009.fig3Binact()
        vsteps_act = np.array(vsteps_act)
        act_exp = np.array(act_exp)
        prepulses = np.array(prepulses)
        inact_exp = np.array(inact_exp)
        # - Recovery data
        intervals,rec_exp = Deng2009.fig4B()
        intervals = np.array(intervals)
        rec_exp = np.array(rec_exp)

        # Concatenate all experimental data
        expVals = np.hstack((i_exp,act_exp,inact_exp,rec_exp))

        # Cell configuration filename
        cell_file = 'Takeuchi2013_iCaT.mmt'

        # Calculate result by approximate Bayesian computation
        result = fitting.approx_bayes_smc_adaptive(cell_file,init,priors,expVals,prior_func,kern,distance,100,1000,0.003)

        # Write results to the standard output and ABCPredPotassium.txt
        print result.getmean()
        print result.getvar()
        outfile.write(str(result.pool)+"\n")
        outfile.write(str(result.getmean())+"\n")
        outfile.write(str(result.getvar())+"\n")

'''
    HELPER METHODS
'''

# Loss function for three voltage clamp experiments from Deng 2009.
# Predicted and experimental values are concatenated before using the function.
def LossFunction(pred_vals, exp_vals):
    # If the simulation failed, the arrays will be filled with zeros
    # We return infinite loss
    if not pred_vals[3].any():
        return float("inf")

    # Calculate normalised RMSE for each experiment
    tot_err = 0
    i = 0
    for p in pred_vals:
        e = exp_vals[i:i+len(p)]
        err = np.sum(np.square(p-e))
        err = pow(err/len(p),0.5)
        err = err/abs(np.mean(e))

        i += len(p)
        tot_err += err

    # Forces overflow to infinity
    if tot_err > 15:
        return float("inf")

    return tot_err

def ResetSim(s, params):
    # Reset the model state before evaluating again
    s.reset()

    # Set d gate and f gate to current parameter values
    s.set_constant('icat_d_gate.dssk1',params[0])
    s.set_constant('icat_d_gate.dssk2',params[1])
    s.set_constant('icat_d_gate.dtauk1',params[2])
    s.set_constant('icat_d_gate.dtauk2',params[3])
    s.set_constant('icat_d_gate.dtauk3',params[4])
    s.set_constant('icat_d_gate.dtauk4',params[5])
    s.set_constant('icat_d_gate.dtauk5',params[6])

    s.set_constant('icat_f_gate.fssk1',params[7])
    s.set_constant('icat_f_gate.fssk2',params[8])
    s.set_constant('icat_f_gate.ftauk1',params[9])
    s.set_constant('icat_f_gate.ftauk2',params[10])
    s.set_constant('icat_f_gate.ftauk3',params[11])
    s.set_constant('icat_f_gate.ftauk4',params[12])
    s.set_constant('icat_f_gate.ftauk5',params[13])


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
    x = TestICaTProto()
    x.TestICaTFitting()
