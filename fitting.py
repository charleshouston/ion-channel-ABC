### Methods for ABC-SMC with adaptive error shrinking on multiple processors.

import math
import numpy as np
import distributions as dist
import copy
import myokit
import logging
import lhsmdu
import pathos.multiprocessing as mp


class ParallelEngine(object):
    """Parallelise the inner ABC algorithm."""

    def __init__(self, channel, priors, prior_fn, kern,
                 loss, thresh_val, post, wts, post_size, maxiter):
        self.channel = channel
        self.priors = priors
        self.prior_fn = prior_fn
        self.kern = kern
        self.loss = loss
        self.thresh_val = thresh_val
        self.post = post
        self.wts = wts
        self.post_size = post_size
        self.maxiter = maxiter

    def __call__(self, i):
        np.random.seed()
        draw = None
        iters = 0
        while draw == None:
            # At time 0, draw from prior distribution.
            if self.post == None:
                draw = [distr.draw() for distr in self.priors]
            # Otherwise, draw the posterior distribution.
            else:
                sum = 0.0
                r = np.random.random()
                for idx in range(self.post_size):
                    sum = sum + self.wts[idx]
                    if sum >= r:
                        break
                draw = self.post[idx]

            # Apply pertubation kernel, then check if new distribution is valid.
            draw = self.kern(draw)
            if (self.prior_fn(self.priors, draw) == 0 or
                self.loss(draw, self.channel) > self.thresh_val):
                draw = None

            # Check if `maxiters` has been reached.
            iters = iters + 1
            if iters >= self.maxiter:
                return None

        # If we get here, draw is accepted.
        next_post = draw
        if self.wts == None:
            next_wt = self.prior_fn(self.priors, draw)
        else:
            denom = 0
            for idx, particle in enumerate(self.post):
                denom = denom + self.wts[idx] * self.kern(particle, draw)
            next_wt = self.prior_fn(self.priors, draw) / denom

        return [i, next_post, next_wt, iters]


def abc_inner(engine, thresh_val, post_size):
    """Helper function for `approx_bayes_smc_adaptive`.

    Args:
        engine (ParallelEngine): Object to pass to process workers.
        thresh_val (float): Acceptable error rate for this iteration.
        post_size (int): Number of particles to maintain in posterior estimate.

    Returns:
        List of new particles and weights if successful otherwise None, None.

    Raises:
        Exception: When a parallel pool fails to start.
    """
    next_post, next_wts = [None]*post_size, [0]*post_size
    total_iters = 0

    # Start parallel pool.
    try:
        pool_size = mp.cpu_count()
        pool = mp.Pool(processes=pool_size, maxtasksperchild=None)
        logging.info("Starting parallel pool size " + str(pool_size))
    except:
        raise Exception('Could not start parallel pool!')

    def failed_sim_callback(r):
        # If failed, stop all other tasks.
        if r is None:
            pool.terminate()

    # Async send tasks to pool of parallel workers.
    # Using `apply_async` instead of `map_async` as we want to stop if
    # any worker fails the current iteration criteria.
    results = []
    for i in range(0, post_size):
        r = pool.apply_async(engine, args=(i,),
                             callback=failed_sim_callback)
        results.append(r)

    # Now wait for workers to finish.
    pool.close()
    pool.join()

    # If the pool was terminated, a result will not be ready so
    # return None for failure.
    for r in results:
        if not r.ready():
            return None, None

    # Otherwise, process the outputs from the parallel simulation.
    total_iters = 0
    for r in results:
        try:
            output = r.get()
        except:
            return None, None
        if output is None:
            return None, None
        next_post[output[0]] = output[1]
        next_wts[output[0]] = output[2]
        total_iters += output[3] # sum total iterations

    logging.info("ACCEPTANCE RATE: "
                 + str(float(post_size) / total_iters))
    total_wt = math.fsum(next_wts)
    next_wts = [next_wts[idx] / total_wt for idx in range(post_size)]
    return next_post, next_wts


def abc_smc_adaptive_error(channel, priors, prior_fn, kern, loss,
                           post_size, maxiter, err_cutoff):
    """ABC-SMC with adaptive error shrinking algorithm.

    Args:
        channel (Channel): Channel class containing experimental protocols
            and data to calculate errors.
        priors (List[Distribution]): Initial prior distributions for each
            ABC parameter in channel.
        prior_fn (Callable): Function to accept list of prior Distributions
            and parameter vector and return probabiliy of parameters under
            the prior (for determining iteration weights).
        kern (Callable): Function for pertubation kernel applied to
            parameter draws.
        loss (Callable): Distance function to calculate error for a given
            parameter draw.
        post_size (int): Number of particles to maintain in posterior
            distribution.
        maxiter (int): Number of iterations after which to adjust error
            reduction and reattempt sampling.
        err_cutoff (float): Minimum decrease in relative error between
            rounds, algorithm terminates after value reached.

    Returns:
        A posterior distribution as a Distribution.Arbitrary class.
    """

    post, wts = [None] * post_size, [1.0/post_size] * post_size
    total_err, max_err = 0.0, 0.0

    # Initialise posterior by drawing from latin hypercube over parameters.
    post_lhs = lhsmdu.sample(len(priors), post_size)
    prior_width = [pr[1] - pr[0] for pr in channel.param_priors]
    prior_lows = [pr[0] for pr in channel.param_priors]
    valid_post_size = post_size
    errs = []
    for i in range(post_size):
        post[i] = post_lhs[:, i].flatten().tolist()[0]
        post[i] = np.array(post[i]) * np.array(prior_width)
        post[i] += prior_lows
        # Evaluate error from simulation
        curr_err = loss(post[i], channel)
        errs.append(curr_err)

    # Process errors and get indices with inf error
    inf_indices = [i for i, e in enumerate(errs) if e == float("inf")]
    # set weights of these particles to zero
    for ii in inf_indices:
        errs[ii] = 0.0
        wts[ii] = 0.0
        valid_post_size -= 1

    logging.info("Original post size: " + str(post_size))
    logging.info("Valid results: " + str(valid_post_size))
    max_err = max(errs)
    total_err = np.sum(errs)

    # Update weights to sum to 1 after eliminating invalid simulations.
    wts = [1.0/valid_post_size if w > 0.0 else w for w in wts]

    # Initialize K to half the average population error.
    K = total_err / (2.0 * post_size)
    thresh_val = max_err

    # Repeatedly halve K until threshold or minimum cutoff is satisfied.
    while K / thresh_val > err_cutoff:
        logging.info("Target = " + str(thresh_val-K) + " (K = "
                      + str(K) + ")")

        # Parallel computation engine.
        channel.reset()
        engine = ParallelEngine(channel, priors, prior_fn, kern, loss,
                                thresh_val, post, wts, post_size, maxiter)
        next_post, next_wts = abc_inner(engine, thresh_val, post_size)

        if next_post != None and next_wts != None:
            post = next_post
            wts = next_wts

            # Write current output to log
            # in case simulation trips up and we lose results.
            logging.info("Target met.")
            logging.info("Current mean posterior estimate: "
                         + str(np.mean(post, 0).tolist()))
            logging.info("Current posterior variance: "
                         + str(np.var(post, 0).tolist()))
            logging.info(str(post))
            logging.info(str(wts))

            thresh_val -= K
            if K >= 0.5 * thresh_val:
                K = 0.5 * thresh_val

        else:
            logging.info("Target not met.")
            K *= 0.5

    logging.info("Final threshold value: " + str(thresh_val))
    return dist.Arbitrary(post, wts)
