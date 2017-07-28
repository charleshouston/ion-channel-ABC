'''
Author: Charles Houston
Date : 01/06/2017

ABC parameter estimation, adapted from Daly et al, 2015 for use in myokit.
Parallel processing version.
'''
import math
import numpy as np
import distributions
import copy
import myokit

import lhsmdu

import pathos.multiprocessing as mp

'''
    ABC-SMC as Described by Toni et al. (2009), modified with adaptive error shrinking

    INPUTS:
        channel: channel object holding information about experimental protocols and
                 parameters to alter
        params: initial list of parameters (for sizing purposes)
        priors:     list of Distribution objects
        exp_vals:   list of experimental values
        prior_func:     takes priors, param vector and returns probability of vector under prior
            (this allows for dependent prior distributions)
        kern:   a perturbation kernel function to be applied to parameter draws;
            with 1 argument returns perturbation, with 2 returns probability of that perturbation
        dist:   a distance function that takes parameters, an x and y vector, and outputs
        post_size:  number of particles to maintain in posterior population
        maxiter:    number of iterations after which to adjust error cutoff and reattempt sampling
        err_cutoff: minimum decrease in error between rounds; algorithm terminates otherwise

    OUTPUTS:
        distributions.Arbitrary object containing final posterior estimating population
'''
class Engine(object):
    def __init__(self,channel,params,priors,exp_vals,prior_func,kern,dist,thresh_val,post,wts,post_size,maxiter):
        self.channel=copy.deepcopy(channel)
        self.params=params
        self.priors=priors
        self.exp_vals=exp_vals
        self.prior_func=prior_func
        self.kern=kern
        self.dist=dist
        self.thresh_val=thresh_val
        self.post=post
        self.wts=wts
        self.post_size=post_size
        self.maxiter=maxiter

    def __call__(self, i):
        # Seed random number generator for each worker
        np.random.seed()

        draw = None
        iters = 0
        while draw == None:
            # At time 0, draw from prior distribution
            if self.post == None:
                draw = [p.draw() for p in self.priors]
            # Otherwise, draw the posterior distribution
            else:
                sum = 0.0
                r = np.random.random()
                for idx in range(self.post_size):
                    sum = sum + self.wts[idx]
                    if sum >= r:
                        break
                draw = self.post[idx]

            # Apply pertubation kernel, then check if new distribution is valid and
            #   meets distance requirements
            draw = self.kern(draw)
            if self.prior_func(self.priors,draw) == 0:
                draw = None
                continue
            if self.dist(draw, self.exp_vals, self.channel) > self.thresh_val:
                draw = None

            # Check if the maximum allowed iterations have been exceeded.
            # If so, exit with failing condition indicating to adjust error threshold.
            iters = iters + 1
            if iters >= self.maxiter:
                return None

        # Draw now accepted - calculate (non-normalized) weight and add to updated posterior
        next_post = draw
        if self.wts == None:
            next_wt = self.prior_func(self.priors,draw)
        else:
            denom = 0
            for idx,el in enumerate(self.post):
                denom = denom + self.wts[idx]*self.kern(el,draw)
            next_wt = self.prior_func(self.priors,draw)/denom

        return [i, next_post, next_wt, iters]


def approx_bayes_smc_adaptive(channel,params,priors,exp_vals,prior_func,kern,dist,post_size=100,maxiter=10000,err_cutoff=0.0001):

    post, wts = [None]*post_size, [1.0/post_size]*post_size
    total_err, max_err = 0.0, 0.0

    # Create copy of channel to avoid passing generated simulations
    # to parallel workers in abc_inner
    channel_copy = copy.deepcopy(channel)

    # Initialise posterior by drawing from latin hypercube over parameters
    post_lhs = lhsmdu.sample(len(priors), post_size)
    # Generate vector of prior widths
    prior_width = [pr[1] - pr[0] for pr in channel.prior_intervals]
    # Valid post size
    valid_post_size = post_size
    errs = []
    for i in range(post_size):
        # Get vector of parameters for each LHS sample
        post[i] = post_lhs[:, i].flatten().tolist()[0]
        # Convert from draws in U(0,1) to original values
        post[i] = np.array(post[i]) * np.array(prior_width)
        post[i] = post[i] + np.array([pr[0] for pr in channel.prior_intervals])
        # Evaluate error from simulation
        curr_err = dist(post[i], exp_vals, channel_copy)
        errs.append(curr_err)

    # Process errors
    # Upper limit to mean + stddev (exclude inf errors first)
    errs_no_inf = [e for e in errs if e != float("inf")]
    err_limit = np.mean(errs_no_inf) + np.std(errs_no_inf)
    for i in range(post_size):
        if errs[i] > err_limit:
            errs[i] = 0.0 # Remove error value for following calculations
            wts[i] = 0.0
            valid_post_size -= 1

    print "Original post size: " + str(post_size)
    print "Valid results: " + str(valid_post_size)
    max_err = max(errs)
    total_err = np.sum(errs)

    # Update weights to sum to 1 after zeroing invalid simulations
    wts = [1.0/valid_post_size if w > 0.0 else w for w in wts]

    # Initialize K to half the average population error
    K = total_err/(2.0*post_size)
    thresh_val = max_err

    # Log results at intermediary stages
    logfile = open('logs/log_' + channel.name + '.log','w')

    # Repeatedly halve improvement criteria K until threshold is met or minimum cutoff met
    while K > err_cutoff:
        logfile.write("Target = "+str(thresh_val-K)+" (K = "+str(K)+")\n")

        # Force empty buffer to file
        logfile.flush()

        next_post, next_wts = abc_inner(channel,params,priors,exp_vals,prior_func,kern,dist,thresh_val-K,post,wts,post_size,maxiter,logfile)

        if next_post != None and next_wts != None:
            post = next_post
            wts = next_wts

            # Write current output to log
            # in case simulation trips up and we lose results.
            logfile.write("Target met\n")
            logfile.write("Current mean posterior estimate: "+str(np.mean(post,0))+"\n")
            logfile.write("Current posterior variance: "+str(np.var(post,0))+"\n")
            logfile.write(str(post)+"\n")
            logfile.write(str(wts)+"\n")

            # Should K exceed half the current max error, it will be adjusted downwards
            thresh_val = thresh_val - K
            if K >= thresh_val*0.5:
                K = thresh_val*0.5

        else:
            logfile.write("Target not met\n")
            K = K*0.5

    print thresh_val
    logfile.close()
    return distributions.Arbitrary(post,wts)


'''
    Helper function for approx_bayes_smc_adaptive
        Draws a new estimate of posterior given previous estimate plus target threshold

    OUTPUTS:
        If a full posterior is accepted within maxiter, returns [particles, weights] 
        If a full posterior is not accepted withing maxiter, returns [None, None]
'''



def abc_inner(channel,params,priors,exp_vals,prior_func,kern,dist,thresh_val,post,wts,post_size,maxiter,logfile):
    next_post, next_wts = [None]*post_size, [0]*post_size
    total_iters = 0
    engine = Engine(channel,params,priors,exp_vals,prior_func,kern,dist,thresh_val,post,wts,post_size,maxiter)

    # Start parallel pool
    try:
        pool_size = min(mp.cpu_count(), 16) # don't be greedy now
        pool = mp.Pool(processes=pool_size, maxtasksperchild=1)
    except:
        raise Exception('Could not start parallel pool!')

    def failed_sim_callback(r):
        # If failed, stop all other tasks
        if r is None:
            pool.terminate()

    # Async send tasks to pool of parallel workers
    # Use apply instead of map as we want to stop if
    # any one simulation fails the criteria.
    results = []
    for i in range(0, post_size):
        r = pool.apply_async(engine, (i,), callback=failed_sim_callback)
        results.append(r)

    # Now wait for workers to finish
    pool.close()
    pool.join()

    # If the pool was terminated, a result will not be ready
    # Return None to signal failure
    for r in results:
        if not r.ready():
            return None, None

    # Otherwise, process the outputs from the parallel simulation
    total_iters = 0
    for r in results:
        output = r.get()
        if output is None:
            return None, None
        next_post[output[0]] = output[1]
        next_wts[output[0]] = output[2]
        total_iters += output[3] # sum total iterations

    logfile.write("ACCEPTANCE RATE: "+str(float(post_size)/total_iters)+"\n")

    # Normalize weights and update posterior
    total_wt = math.fsum(next_wts)
    next_wts = [next_wts[idx]/total_wt for idx in range(post_size)]

    return next_post, next_wts
