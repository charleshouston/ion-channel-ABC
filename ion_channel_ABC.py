### Main classes and helper functions for ABC ion channel parameter estimation.

import fitting
import distributions as dist
import myokit
import logging


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
                 err_cutoff=0.001):
        """Initialisation.

        Args:
            error_fn (Callable): Loss function to call to calculate error
                for single simulation runs.
            post_size (int): Size of posterior discrete distribution.
            maxiter (int): Maximum number of trial iterations before
                relaxing fitting conditions.
            err_cutoff (float): Stopping conditions after improvement to
                error drops below this value.
        """
        self.error_fn = error_fn
        self.post_size = post_size
        self.maxiter = maxiter
        self.err_cutoff = err_cutoff

        logname = channel.name + '.log'
        logging.basicConfig(filename=logname, level=logging.INFO)

    def __call__(self, channel):
        """Run the ABC fit.

        Args:
            channel (Channel): Channel class with experiments and data
                to fit model.
        """

        # Initial values and priors
        priors = []
        init = []
        for pr in channel.abc_params.values():
            priors.append(dist.Uniform(pr[0], pr[1]))
            init.append(priors[-1].getmean())

        def distance(new_params, channel_iter):
            """Error measure for output of single simulation run."""
            channel_iter.set_abc_params(new_params)
            return channel_iter.eval_error(self.error_fn)

        def kern(orig, new=None):
            """Perturbation kernel."""
            kernel = self.channel.kernel
            if new == None:
                new = []
                perturb = [g.draw() for g in kernel]
                for i in range(len(orig)):
                    new = new + [orig[i] + perturb[i]]
                return new
            else:
                prob = 1.0
                for i, g in enumerate(kernel):
                    prob = prob * g.pdf(new[i] - orig[i])
                return prob

        result = fitting.abc_smc_adaptive_error(channel=channel,
                                                params=init,
                                                priors=priors,
                                                exp_vals=self.res_exper,
                                                prior_func=prior_fn,
                                                kern=kern,
                                                dist=distance,
                                                post_size=self.post_size,
                                                maxiter=self.maxiter,
                                                err_cutoff=self.err_cutoff)

        # Write results to the standard output and results log
        logging.info("Result mean:\n" + result.getmean())
        logging.info("Result var:\n" + result.getvar())
        logging.info("Result pool:\n" + result.pool)
        logging.info("Result weights:\n" + result.weights)
