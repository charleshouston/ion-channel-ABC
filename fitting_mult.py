'''
Author: Charles Houston
Date : 01/06/2017

ABC parameter estimation, adapted from Daly et al, 2015 for use in myokit.
Parallel processing version.
'''

import math
from numpy import *
import distributions
#from pympler import tracker
import myokit
import random

from pathos.pools import ProcessPool as Pool

'''
    ABC-SMC as Described by Toni et al. (2009), modified with adaptive error shrinking

    INPUTS:
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
    def __init__(self,cell_file,params,priors,exp_vals,prior_func,kern,dist,thresh_val,post,wts,post_size,maxiter):
        self.cell_file=cell_file
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
        #print "Starting simulation " + str(i) + "..."

        # Get the cell model
        m,p,x = myokit.load(self.cell_file)

        # Get membrane potential
        v = m.get('membrane.V')
        # Demote v from a state to an ordinary variable
        v.demote()
        v.set_rhs(0)
        # Set voltage to pacing variable
        v.set_binding('pace')

        # Create the simulation
        sim = myokit.Simulation(m)
        reversal_potential = m.get('icat.E_CaT').value()

        draw = None
        iters = 0
        while draw == None:
            # At time 0, draw from prior distribution
            if self.post == None:
                draw = [p.draw() for p in self.priors]
            # Otherwise, draw the posterior distribution
            else:
                sum = 0.0
                random.jumpahead(i)
                r = random.random()
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
            if self.dist(draw,self.exp_vals,sim,reversal_potential) > self.thresh_val:
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

        #print "Finished sim " + str(i) + " in " + str(iters) + " iterations"

        return [i, next_post, next_wt, iters]


def approx_bayes_smc_adaptive(cell_file,params,priors,exp_vals,prior_func,kern,dist,post_size=100,maxiter=10000,err_cutoff=0.0001):

    #tr = tracker.SummaryTracker()

    post, wts = [None]*post_size, [1.0/post_size]*post_size
    total_err, max_err = 0.0, 0.0

    # Get the cell model
    m,p,x = myokit.load(cell_file)

    # Get membrane potential
    v = m.get('membrane.V')
    # Demote v from a state to an ordinary variable
    v.demote()
    v.set_rhs(0)
    # Set voltage to pacing variable
    v.set_binding('pace')
    # Create the simulation
    sim = myokit.Simulation(m)
    reversal_potential = m.get('icat.E_CaT').value()

    # Initializes posterior to be a draw of particles from prior
    for i in range(post_size):

        # Distance function returns "inf" in the case of overflow error
        curr_err = float("inf")
        while curr_err == float("inf"):
            post[i] = [p.draw() for p in priors]
            curr_err = dist(post[i],exp_vals,sim,reversal_potential)

        total_err = total_err + curr_err
        max_err = max(curr_err,max_err)

    # Initialize K to half the average population error
    K = total_err/(2.0*post_size)
    thresh_val = max_err

    try:
        pool = Pool()
    except:
        print "Could not start parallel pool."

    # Repeatedly halve improvement criteria K until threshold is met or minimum cutoff met
    while K > err_cutoff:
        #tr.print_diff()
        print "Target = "+str(thresh_val-K)+" (K = "+str(K)+")"

        next_post, next_wts = abc_inner(cell_file,params,priors,exp_vals,prior_func,kern,dist,thresh_val-K,post,wts,post_size,maxiter,pool)

        if next_post != None and next_wts != None:
            post = next_post
            wts = next_wts

            print "Target met"
            print "Current mean posterior estimate: "+str(mean(post,0))
            print "Current posterior variance: "+str(var(post,0))

            # Should K exceed half the current max error, it will be adjusted downwards
            thresh_val = thresh_val - K
            if K >= thresh_val*0.5:
                K = thresh_val*0.5
            
        else:
            print "Target not met"
            K = K*0.5

    print thresh_val
    return distributions.Arbitrary(post,wts)


''' 
    Helper function for approx_bayes_smc_adaptive
        Draws a new estimate of posterior given previous estimate plus target threshold

    OUTPUTS:
        If a full posterior is accepted within maxiter, returns [particles, weights] 
        If a full posterior is not accepted withing maxiter, returns [None, None]
'''



def abc_inner(cell_file,params,priors,exp_vals,prior_func,kern,dist,thresh_val,post,wts,post_size,maxiter,pool):
    next_post, next_wts = [None]*post_size, [0]*post_size
    total_iters = 0
    engine = Engine(cell_file,params,priors,exp_vals,prior_func,kern,dist,thresh_val,post,wts,post_size,maxiter)

    outputs = []

    # Get number of nodes in process pool
    node_num = pool.ncpus

    # Map the simulations to multiple processes in batches.
    # pathos doesn't support the use of callbacks so this method is a compromise
    # to check whether a result was produced after each set of simulations.
    i = 0
    while i < post_size:
        res = pool.map(engine, range(i, min(i+node_num, post_size)))
        # Check no results failed
        for r in res:
            if r == None: return None, None
        outputs = outputs + res
        i += node_num

    #outputs = pool.map(engine, range(post_size), callback=check_output)

    # Process outputs from parallel simulation
    total_iters = 0
    for output in outputs:
        next_post[output[0]] = output[1]
        next_wts[output[0]] = output[2]
        total_iters += output[3] # sum total iterations

    print "ACCEPTANCE RATE: "+str(float(post_size)/total_iters)

    # Normalize weights and update posterior
    total_wt = math.fsum(next_wts)
    next_wts = [next_wts[idx]/total_wt for idx in range(post_size)]

    return next_post, next_wts
