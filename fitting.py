''' 
	Written by Aidan Daly, DPhil candidate, University of Oxford Department of Computer Science
	
	Part of a paper tutorial on ABC parameter estimation for the Hodgkin-Huxley 
	action potential model.
'''

import math
from numpy import *
import distributions
from pympler import tracker

''' 
	ABC-SMC as Described by Toni et al. (2009), modified with adaptive error shrinking

	INPUTS:
		params: initial list of parameters (for sizing purposes)
		priors:	list of Distribution objects
		x,y: 	list of experimental recording times, list of recorded conductances
		prior_func:	takes priors, param vector and returns probability of vector under prior
			(this allows for dependent prior distributions)
		kern: 	a perturbation kernel function to be applied to parameter draws;
			with 1 argument returns perturbation, with 2 returns probability of that perturbation
		dist: 	a distance function that takes parameters, an x and y vector, and outputs
		post_size:	number of particles to maintain in posterior population
		maxiter:	number of iterations after which to adjust error cutoff and reattempt sampling
		err_cutoff: minimum decrease in error between rounds; algorithm terminates otherwise

	OUTPUTS:
		distributions.Arbitrary object containing final posterior estimating population
'''
def approx_bayes_smc_adaptive(params,priors,x,y,prior_func,kern,dist,post_size=100,maxiter=10000,err_cutoff=0.0001):
        
        tr = tracker.SummaryTracker()

	post, wts = [None]*post_size, [1.0/post_size]*post_size
	total_err, max_err = 0.0, 0.0

	# Initializes posterior to be a draw of particles from prior
	for i in range(post_size):
		
		# Distance function returns "inf" in the case of overflow error
		curr_err = float("inf")
		while curr_err == float("inf"):
			post[i] = [p.draw() for p in priors]
			curr_err = dist(post[i],x,y)

		total_err = total_err + curr_err
		max_err = max(curr_err,max_err)

	# Initialize K to half the average population error
	K = total_err/(2.0*post_size)
	thresh_val = max_err

	# Repeatedly halve improvement criteria K until threshold is met or minimum cutoff met
	while K > err_cutoff:
                tr.print_diff()
		print "Target = "+str(thresh_val-K)+" (K = "+str(K)+")"
		next_post, next_wts = abc_inner(params,priors,x,y,prior_func,kern,dist,thresh_val-K,post,wts,post_size,maxiter)

		if next_post != None and next_wts != None:
			post = next_post
			wts = next_wts

			print "Target met"
			print "Current mean posterior estimate: "+str(mean(post,0))
			print "Current posterior variance: "+str(var(post,0))

			# Should K exceed the current max error, it will be adjusted downwards
			thresh_val = thresh_val - K
			if K >= thresh_val:
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
def abc_inner(params,priors,x,y,prior_func,kern,dist,thresh_val,post,wts,post_size,maxiter):
	next_post, next_wts = [None]*post_size, [0]*post_size
	total_iters = 0

	# Update each particle in the current posterior estimation:
	for i in range(post_size):
		draw = None
		iters = 0
		while draw == None:
			# At time 0, draw from prior distribution
			if post == None:
				draw = [p.draw() for p in priors]
			# Otherwise, draw from posterior distribution
			else:
				sum = 0.0
				r = random.rand()
				for idx in range(post_size):
					sum = sum + wts[idx]
					if sum >= r:
						break
				draw = post[idx]

			# Apply perturbation kernel, the check if new distribution is valid and 
			#	meets distance requirements
			draw = kern(draw)
			if prior_func(priors,draw) == 0:
				draw = None
				continue
			if dist(draw,x,y) > thresh_val:
				draw = None

			# Check if the maximum allowed iterations have been exceeded.
			# If so, exit with failing condition indicating to adjust error threshold
			iters = iters + 1
			if iters >= maxiter:
				return None,None

		# Draw now accepted - calculate (non-normalized) weight and add to updated posterior
		next_post[i] = draw
		if wts == None:
			next_wts[i] = prior_func(priors,draw)
		else:
			denom = 0
			for idx,el in enumerate(post):
				denom = denom + wts[idx]*kern(el,draw)
			next_wts[i] = prior_func(priors,draw)/denom

		total_iters = total_iters+iters

	print "ACCEPTANCE RATE: "+str(float(post_size)/total_iters)

	# Normalize weights and update posterior
	total_wt = math.fsum(next_wts)
	next_wts = [next_wts[idx]/total_wt for idx in range(post_size)]
	
	return next_post, next_wts
			
