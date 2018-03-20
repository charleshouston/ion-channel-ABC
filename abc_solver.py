### Main classes and helper functions for ABC ion channel parameter estimation.

import fitting
import distributions as dist
import myokit
import logging
import numpy as np


def prior_fn(priors, params):
    """Simple prior for list of independent Distribution objects.

    Args:
        priors (List[Distribution]): Independent prior distributions.
        params (List[float]): List of channel model parameters.

    Returns:
        Joint distribution of combined prior functions.
    """
    prob = 1.0
    for i, distr in enumerate(priors):
        prob = prob * distr.pdf(params[i])
    return prob


class ABCSolver():
    """Solver for ABC parameter fitting."""
    def __init__(self, error_fn, post_size=100, maxiter=10000,
                 err_cutoff=0.001, init_max_err=float("inf")):
        """Initialisation.

        Args:
            error_fn (Callable): Loss function to call to calculate error
                for single simulation runs.
            post_size (int): Size of posterior discrete distribution.
            maxiter (int): Maximum number of trial iterations before
                relaxing fitting conditions.
            err_cutoff (float): Stopping conditions after improvement to
                error drops below this value.
            init_max_err (float): Maximum value of error function when
                generating first set of posterior particles.
        """
        self.error_fn = error_fn
        self.post_size = post_size
        self.maxiter = maxiter
        self.err_cutoff = err_cutoff
        self.init_max_err = init_max_err

    def __call__(self, channel, logfile='abc_solve.log'):
        """Run the ABC fit.

        Args:
            channel (Channel): Channel class with experiments and data
                to fit model.

        Returns:
            Final estimate of posterior distribution.
        """
        logging.basicConfig(filename=logfile, level=logging.INFO,
                            format='%(asctime)s:%(message)s')
        logging.info("Starting ABC solver...")

        # Initial values and priors
        priors = []
        init = []
        for pr in channel.param_priors:
            priors.append(dist.Uniform(pr[0], pr[1]))
            init.append(priors[-1].getmean())

        def distance(new_params, channel_iter):
            """Error measure for output of single simulation run."""
            channel_iter.set_abc_params(new_params)
            return channel_iter.eval_error(self.error_fn)

        def kern(orig, new=None, kern_width=0.2):
            """Perturbation kernel."""
            kernel = channel.kernel
            param_ranges = channel.param_ranges
            if new == None:
                new = []
                perturb = [kernel.draw()*r*kern_width for r in param_ranges]
                for o, p in zip(orig, perturb):
                    new.append(o+p)
                return new
            else:
                prob = 1.0
                for i, r in enumerate(param_ranges):
                    prob = prob * kernel.pdf((new[i] - orig[i])
                                              / (kern_width*r))
                return prob

        result = fitting.abc_smc_adaptive_error(channel=channel,
                                                priors=priors,
                                                prior_fn=prior_fn,
                                                kern=kern,
                                                loss=distance,
                                                post_size=self.post_size,
                                                maxiter=self.maxiter,
                                                err_cutoff=self.err_cutoff,
                                                init_max_err=self.init_max_err)

        # Write results to the standard output and results log
        logging.info("Result mean:\n" + str(result.getmean()))
        logging.info("Result var:\n" + str(result.getvar()))
        return result
