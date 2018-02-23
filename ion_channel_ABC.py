### Main classes and helper functions for ABC ion channel parameter estimation.

import fitting
import distributions as dist
import myokit
import logging


def prior_fn(priors, params):
    '''Simple prior for list of independent Distribution objects.

    Args:
        priors (List[Distribution]): Independent prior distributions.
        params (List[float]): List of channel model parameters.

    Returns:
        Joint distribution of combined prior functions.
    '''
    prob = 1.0
    for i, distr in enumerate(priors):
        prob = prob * distr.pdf(params[i])
    return prob


def loss_fn(error_fn, res_sim, res_exper):
    '''Evaluates the loss between simulation and experimental data.

    Args:
        error_fn (Callable): Specific error function to use.
        res_sim (List[float]): Simulation results data.
        res_exper (List[float]): Experiment results data.

    Returns:
        Loss value as float.
    '''
    if res_sim is None:
        return float("inf")

    # Finds sim output at x value closest to experimental
    sim_vals_closest = [[] for i in range(len(res_sim))]
    for i, e in enumerate(res_exper):
        curr = 0
        for tval in e[0]:
            sim_times = res_sim[i][0]
            while tval > sim_times[curr+1]:
                if curr >= len(sim_times)-2:
                    break
                curr = curr+1
            if abs(tval - sim_times[curr]) < abs(tval - sim_times[curr+1]):
                res_sim_close[i] = res_sim_close[i] + [res_sim[i][1][curr]]
            else:
                res_sim_close[i] = res_sim_close[i] + [res_sim[i][1][curr+1]]

    return error_func(res_sim_close, res_exper)


class ChannelProto():
    '''Wrapper for channel running through ABC parameter fitting.'''
    def __init__(self, channel, error_fn):
        '''Initialisation.

        Args:
            channel (AbstractChannel): The channel to run through the
                ABC parameter fit.
            error_fn (Callable): Loss function to call to calculate error
                for single simulation runs.
        '''
        self.channel = channel
        self.error_fn = error_fn

        self.res_exper = channel.data_exp

        logname = channel.name + '.log'
        logging.basicConfig(filename=logname, level=logging.INFO)

    def __call__(self):
        '''Run the ABC fit.'''

        # Initial values and priors
        priors = []
        init = []
        for pr in channel.prior_intervals:
            priors.append(dist.Uniform(pr[0], pr[1]))
            init.append(priors[-1].getmean())

        def distance(params, res_exp, model):
            '''Error measure for output of single simulation run.'''
            model.reset_params(params)
            res_sim = model.simulate()
            return loss_fn(self.error_fn, res_sim, res_exp)

        def kern(orig, new=None):
            '''Perturbation kernel.'''
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

        result = fitting.approx_bayes_smc_adaptive(channel=self.channel,
                                                   params=init,
                                                   priors=priors,
                                                   exp_vals=self.res_exper,
                                                   prior_func=prior_fn,
                                                   kern=kern,
                                                   dist=distance,
                                                   post_size=20,
                                                   maxiter=100,
                                                   err_cutoff=0.01)

        # Write results to the standard output and results log
        logging.info("Result mean:\n" + result.getmean())
        logging.info("Result var:\n" + result.getvar())
        logging.info("Result pool:\n" + result.pool)
        logging.info("Result weights:\n" + result.weights)
