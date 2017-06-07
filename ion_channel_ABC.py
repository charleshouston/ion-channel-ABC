'''
Author: Charles Houston

ABC parameter estimation for ion channel dynamics.
Developed initially from work by Daly et al, 2015.
Re-written to use with myokit, multi-processing and further channels.
'''

import fitting_mult as fitting       # import ABC fitting procedure
import distributions as Dist # prob dist functions
import channel_setup # contains channel-specific settings
import numpy as np
import myokit

class ChannelProto():

    # Fits the ion channel parameters
    def fit(self):

        # Bring in specific channel settings
        channel = channel_setup.TTypeCalcium()

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
        def distance(params,exp_vals,s,reversal_potential):

            # Reset simulation with new parameters
            ResetSim(s, params, channel)

            # Simulate channel with new parameters
            sim_vals = channel.simulate(s)

            return LossFunction(sim_vals, exp_vals)


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

        # Experimental data
        exp_vals = channel.data_exp[1]

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
    # If the simulation failed, pred_vals should be None
    # We return infinite loss
    if pred_vals is None:
        return float("inf")

    # Calculate normalised (by mean of experimental values) RMSE for each experiment
    tot_err = 0
    i = 0
    for p in pred_vals:
        e = exp_vals[i:i+len(p)]
        err = np.sum(np.square(p-e))
        err = pow(err/len(p),0.5)
        err = err/abs(np.mean(e))

        i += len(p)
        tot_err += err

    # Forces overflow to infinity for unreasonable values
    if tot_err > 15:
        return float("inf")

    return tot_err

def ResetSim(s, params, channel):
    # Reset the model state before evaluating again
    s.reset()

    # Set parameters
    for i, p in enumerate(params):
        s.set_constant(channel.parameters[i], p)

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
