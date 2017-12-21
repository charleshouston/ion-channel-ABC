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
        result = fitting.approx_bayes_smc_adaptive(channel,
                                                   init,
                                                   priors,
                                                   exp_vals,
                                                   prior_func,
                                                   kern,
                                                   distance,
                                                   20,
                                                   1000,
                                                   0.01)

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

# Evaluates CV(RMSD) between experimental and predicted values
def LossFunction(sim_vals, exp_vals):
    # if the simulation failed, pred_vals should be None
    #  return infinite loss
    if sim_vals is None:
        return float("inf")

    # Finds sim output at x value closest to experimental
    sim_vals_closest = [[] for i in range(len(sim_vals))]
    for i, e in enumerate(exp_vals):
        curr = 0
        for tval in e[0]:
            sim_times = sim_vals[i][0]
            while tval > sim_times[curr+1]:
                if curr >= len(sim_times)-2:
                    break
                curr = curr+1
            if abs(tval-sim_times[curr]) < abs(tval-sim_times[curr+1]):
                sim_vals_closest[i] = sim_vals_closest[i] + [sim_vals[i][1][curr]]
            else:
                sim_vals_closest[i] = sim_vals_closest[i] + [sim_vals[i][1][curr+1]]
                
    return cvchisq(sim_vals_closest, exp_vals)

# Calculate coefficient of variation of weighted residuals
def cvchisq(model, exper):
    
    warnings.filterwarnings('error')
    
    tot_err = 0
    
    for i, m in enumerate(model):
        m = np.array(m) # model prediction
        e = np.array(exper[i][1]) # experimental mean
        err_bars = np.array(exper[i][2]) # experimental standard deviation
        N = np.array(exper[i][3]) # number of experimental measurements taken for this data mean
        
        # normalise all data points to between -1 and 1 by experimental value
        m = normaliseby(m, e)
        e = normaliseby(e, e)
        err_bars = normaliseby(err_bars, e)
        sd = err_bars * np.sqrt(N) # calculate standard deviations from normalised values of SEM
        
        # if any sd value is zero, set to minimum of other recorded values
        for i,sdi in enumerate(sd):
            if sdi == 0:
                sd[i] = np.min(sd[sd!=0])
                
        # normalise weights for weighted sum of squares loss so that in the limit as sd -> 0 
        # is equivalent to sum of squares loss
        w = [1 / np.square(sdi) for sdi in sd]
        w = w / np.max(w)
        
        try:
            err = np.sum(w * np.square(e - m))
        except Warning:
            return float("inf")
        except:
            return float("inf")
        
        # root error
        err = pow(err / len(m), 0.5)
        # err = err / np.ptp(e) # normalised earlier so shouldn't need to normalise again here
        tot_err += err
  
    warnings.resetwarnings()
    
    return tot_err

# Calculate coefficient of variation of the real mean squared distance
def cvrmsd(model, exper):
    
    warnings.filterwarnings('error')
    
    tot_err = 0
    
    for i, m in enumerate(model):
        m = np.array(m)
        e = np.array(exper[i][1])
        
        # normalise between -1 and 1 by experimental values
        m = normaliseby(m, e)
        e = normaliseby(e, e)
        
        try:
            err = np.sum(np.square(m - e))
        except Warning:
            return float("inf")
        except:
            return float("inf")

        # root and normalise error by range
        err = pow(err / len(m), 0.5)
        # err = err / np.ptp(e)
        tot_err += err

    warnings.resetwarnings()
    
    return tot_err

# Simple multiplicative prior for list of independent Distribution objects
def prior_func(priors,params):
    prob = 1.0
    for i,distr in enumerate(priors):
        prob = prob * distr.pdf(params[i])
    return prob

# Normalise `a` between -1 and 1 using `b` as normalisation data
def normaliseby(a, b):
    temp = a
    temp /= np.max(np.abs(b), axis=0)
    return temp

if __name__ == '__main__':
    # Load specific channel settings
    channel = channel_setup.icat()
    x = ChannelProto()
    x.fit(channel)
