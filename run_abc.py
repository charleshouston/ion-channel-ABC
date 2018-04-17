### Run script for ABC process.

import abc_solver as abc
from error_functions import cvrmsd, cvchisq
import matplotlib.pyplot as plt
import argparse


parser = argparse.ArgumentParser("Run script for ABC parameterisation.")
parser.add_argument('channel', type=str,
                    help="channel to parameterise")
parser.add_argument('-e', '--errorfn', help="error function")
parser.add_argument('-p', '--postsize', type=int,
                    help="number of posterior particles")
parser.add_argument('-i', '--maxiter', type=int,
                    help="maximum number of iterations per step")
parser.add_argument('-s', '--errcutoff', type=float,
                    help="relative error stopping criterion")
parser.add_argument('-m', '--initmaxerr', type=float,
                    help="initial maximum error")
args = parser.parse_args()

# Import correct channel.
channel_obj = None
if args.channel == 'icat':
    import channels.icat
    channel_obj = channels.icat.icat
elif args.channel == 'ikur':
    import channels.ikur
    channel_obj = channels.ikur.ikur
elif args.channel == 'ikr':
    import channels.ikr
    channel_obj = channels.ikr.ikr
elif args.channel == 'iha':
    import channels.iha
    channel_obj = channels.iha.iha
elif args.channel == 'ina':
    import channels.ina
    channel_obj = channels.ina.ina
else:
    raise ValueError("Unrecognised channel.")

# Process arguments.
if args.errorfn == 'cvrmsd':
    error_fn = cvrmsd
elif args.errorfn == 'cvchisq':
    error_fn = cvchisq
else:
    raise ValueError("Unrecognised error function.")
post_size = int(args.postsize) if args.postsize else 100
maxiter = int(args.maxiter) if args.maxiter else 1000
err_cutoff = float(args.errcutoff) if args.errcutoff else 0.001
init_max_err = float(args.initmaxerr) if args.initmaxerr else 10

# Create ABC engine and run solver.
abc_solver = abc.ABCSolver(error_fn=error_fn, post_size=post_size,
                           maxiter=maxiter, err_cutoff=err_cutoff,
                           init_max_err=init_max_err)
savename = args.channel + '_' + args.errorfn + '_' + str(post_size)
post_dist = abc_solver(channel_obj,
    logfile='logs/' + savename + '.log')

# Process and save results.
channel_obj.save('results/' + savename + '.pkl')
post_dist.save('results/' + savename + '_res.pkl')

fig1 = channel_obj.plot_results(post_dist)
plt.savefig('results/' + savename + '_res_plot.pdf')
plt.close(fig1)

fig2 = channel_obj.plot_final_params(post_dist)
plt.savefig('results/' + savename + '_params_plot.pdf')
plt.close(fig2)
