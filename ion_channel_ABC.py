'''
Author: Charles Houston

ABC parameter estimation for ion channel dynamics.
Developed initially from work by Daly et al, 2015.
Re-written to use with myokit, multi-processing and further channels.
'''

import fitting                  # import ABC fitting procedure
import distributions as Dist    # prob dist functions
import channel_setup            # contains channel-specific settings
import numpy as np
import myokit
import warnings

class ChannelProto():

    # Fits the ion channel parameters
    def fit(self, channel):
        # Output file
        outfile = open('results/results_' + channel.name + '.txt','w')

        # Initial values and priors
        # - Prior is uniform distribution ({0,1},order of mag larger than proposed value)
        # - Init is mean of prior
        priors = []
        init = []
        for pr in channel.prior_intervals:
            priors.append(Dist.Uniform(pr[0], pr[1]))
            init.append(priors[-1].getmean())

        # Distance function for ABC process
        def distance(params, exp_vals, channel_iter):

            # Reset simulation with new parameters
            channel_iter.reset_params(params)

            # Simulate channel with new parameters
            sim_vals = channel_iter.simulate()

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
        exp_vals = channel.data_exp

        # Calculate result by approximate Bayesian computation
        result = fitting.approx_bayes_smc_adaptive(channel,init,priors,exp_vals,prior_func,kern,distance,len(priors) * 20,10000,0.003)

        # Write results to the standard output and results log
        print result.getmean()
        print result.getvar()
        outfile.write(str(result.pool)+"\n")
        outfile.write(str(result.weights)+"\n")
        outfile.write(str(result.getmean())+"\n")
        outfile.write(str(result.getvar())+"\n")

'''
    HELPER METHODS
'''

# Evaluates RMSE between experimental and predicted values
def LossFunction(sim_vals, exp_vals):
    # If the simulation failed, pred_vals should be None
    # We return infinite loss
    if sim_vals is None:
        return float("inf")

    # Calculate normalised (by mean of experimental values) RMSE for each experiment
    tot_err = 0
    # Catch runtime overflow warnings from numpy
    warnings.filterwarnings('error')
    for i,p in enumerate(sim_vals):
        p = np.array(p) # predicted
        e = np.array(exp_vals[i][1]) # experimental
        try:
            err = np.sum(np.square(p-e))
        except Warning:
            return float("inf")
        except:
            return float("inf")
        # normalise error
        err = pow(err/len(p),0.5)
        err = err/abs(np.mean(e))
        tot_err += err

    return tot_err

# Simple multiplicative prior for list of independent Distribution objects
def prior_func(priors,params):
    prob = 1.0
    for i,distr in enumerate(priors):
        prob = prob * distr.pdf(params[i])
    return prob

if __name__ == '__main__':
    # Load specific channel settings
    channel = channel_setup.icat()
    x = ChannelProto()
    x.fit(channel)
