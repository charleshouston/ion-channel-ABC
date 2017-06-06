'''
Author: Charles Houston, MRes Biomedical Research student.

ABC parameter estimation for ion channel dynamics.
Developed initially from work by Daly et al, 2015.
Re-written to use with myokit, multi-processing and further channels.
'''

import fitting_mult as fitting       # import ABC fitting procedure
import distributions as Dist # prob dist functions
import data.icat.data_icat as data_exp # Import experimental data for t-type calcium channel

import channel_setup.TTypeCalcium as ChannelSetup

import numpy as np

import myokit
import simulations

class ChannelProto():

    # Fits the ion channel parameters
    def fit(self):

        # Bring in specific channel settings
        channel = ChannelSetup()

        # Output file
        outfile = open('results/results_' + channel.name + '+.txt','w')

        # Initial values and priors
        # - Prior is uniform distribution ({0,1},order of mag larger than proposed value)
        # - Init is mean of prior
        priors = []
        init = []
        for pr in channel.prior_intervals:
            priors.append(Dist.Uniform(pr[0],pr[1]))
            init.append(priors[-1].getmean())

        # ABC expects this form - sets alpha/beta, runs protocol, then returns sq_err of result
        def distance(params,vals,s,reversal_potential):

            # Run simulations
            ResetSim(s,params)
            act_pred = simulations.activation_sim(s,vsteps,reversal_potential)
            # If this simulation failed, it will return all zeros
            # No point in running the rest!
            if not act_pred[0].any():
                inact_pred = np.zeros(7)
            else:
                ResetSim(s,params)
                inact_pred = simulations.inactivation_sim(s,prepulses,act_pred[0])

            if not inact_pred.any():
                rec_pred = np.zeros(11)
            else:
                ResetSim(s,params)
                rec_pred = simulations.recovery_sim(s,intervals)

            # Return RMSE for all simulations
            pred_vals = [act_pred[0], act_pred[1], inact_pred, rec_pred]

            return LossFunction(pred_vals, vals)


        def kern(orig,new=None):
            # Get channel specific kernel
            kernel = channel.kernel
            if new == None:
                new = []
                perturb = [g.draw() for g in kernel]
                for i in range(len(orig)):
		            new = new + [orig[i]+perturb[i]]
                return new
            else:
                prob = 1.0
                for i,g in enumerate(kernel):
                    prob = prob*g.pdf(new[i]-orig[i])
                return prob


        # # Load icat experimental data
        # # - IV data
        # vsteps,i_exp = data_exp.fig1B()
        # vsteps = np.array(vsteps)
        # i_exp = np.array(i_exp)
        # # - Activation/Inactivation data
        # vsteps_act,act_exp = data_exp.fig3Bact()
        # prepulses,inact_exp = data_exp.fig3Binact()
        # vsteps_act = np.array(vsteps_act)
        # act_exp = np.array(act_exp)
        # prepulses = np.array(prepulses)
        # inact_exp = np.array(inact_exp)
        # # - Recovery data
        # intervals,rec_exp = data_exp.fig4B()
        # intervals = np.array(intervals)
        # rec_exp = np.array(rec_exp)

        # Concatenate all experimental data
        # exp_vals = np.hstack((i_exp,act_exp,inact_exp,rec_exp))

        # Experimental data
        exp_vals = channel.data_exp

        # Cell configuration filename
        cell_file = 'models/' + channel.model_name

        # Calculate result by approximate Bayesian computation
        result = fitting.approx_bayes_smc_adaptive(cell_file,init,priors,exp_vals,prior_func,kern,distance,20,50,0.003)

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

    # Set parameters
    for i,p in enumerate(params):
        s.set_constant(channel.parameters[i],p)

        # s.set_constant('icat_d_gate.dssk1',params[0])
    # s.set_constant('icat_d_gate.dssk2',params[1])
    # s.set_constant('icat_d_gate.dtauk1',params[2])
    # s.set_constant('icat_d_gate.dtauk2',params[3])
    # s.set_constant('icat_d_gate.dtauk3',params[4])
    # s.set_constant('icat_d_gate.dtauk4',params[5])
    # s.set_constant('icat_d_gate.dtauk5',params[6])
    # s.set_constant('icat_d_gate.dtauk6',params[7])

    # s.set_constant('icat_f_gate.fssk1',params[8])
    # s.set_constant('icat_f_gate.fssk2',params[9])
    # s.set_constant('icat_f_gate.ftauk1',params[10])
    # s.set_constant('icat_f_gate.ftauk2',params[11])
    # s.set_constant('icat_f_gate.ftauk3',params[12])
    # s.set_constant('icat_f_gate.ftauk4',params[13])
    # s.set_constant('icat_f_gate.ftauk5',params[14])
    # s.set_constant('icat_f_gate.ftauk6',params[15])


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
    x = ChannelProto()
    x.fit()
