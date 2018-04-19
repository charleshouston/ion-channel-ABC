### Run parameter sensitivity using multivariable regression.

"""
The parameter sensitivity approximation is based on the work by
Sobie EA. Parameter sensitivity analysis in electrophysiological models using
multivariable regression. Biophys J. 2009 Feb 18;96(4):1264-74.
"""

from sklearn import linear_model
from sklearn.metrics import r2_score
import numpy as np
from error_functions import cvrmsd
import matplotlib.pyplot as plt
import argparse


parser = argparse.ArgumentParser(
        "Parameter sensitivity study for channel model.")
parser.add_argument('channel', type=str,
                    help="channel to study")
parser.add_argument('-n', "--numsamples", type=int,
                    help="number of training simulations")
parser.add_argument('-s', "--sigma", type=float,
                    help="spread of scale variation for training")
args = parser.parse_args()

# Parse arguments.
channel = None
if args.channel == 'icat':
    import channels.icat
    channel = channels.icat.icat
elif args.channel == 'ikur':
    import channels.ikur
    channel = channels.ikur.ikur
elif args.channel == 'ikr':
    import channels.ikr
    channel = channels.ikr.ikr
elif args.channel == 'iha':
    import channels.iha
    channel = channels.iha.iha
elif args.channel == 'ina':
    import channels.ina
    channel = channels.ina.ina
elif args.channel == 'ito':
    import channels.ito
    channel = channels.ito.ito
elif args.channel == 'ical':
    import channels.ical
    channel = channels.ical.ical
elif args.channel == 'ik1':
    import channels.ik1
    channel = channels.ik1.ik1
else:
    raise ValueError("Unrecognised channel.")
n = int(args.numsamples) if args.numsamples else 500
sigma = float(args.sigma) if args.sigma else 0.15

# Generate lognormal distribution for parameters.
scale_dist_ln = np.random.lognormal(mean=0.0, sigma=sigma,
                                    size=(n, len(channel.param_names)))

# Get original variable values.
original_vals = np.asarray(channel.get_original_param_vals())

p = len(original_vals)
m = len(channel.experiments)
X = np.empty((n, p))
Y = np.empty((n, m))

for i in range(n):
    X[i, :] = np.multiply(original_vals, scale_dist_ln[i, :])
    channel.set_abc_params(X[i, :])
    Y[i, :] = channel.eval_error(cvrmsd)

# Mean center and normalise
X = np.log(X)
Y = np.log(Y)
X = np.divide(X - np.mean(X, axis=0), np.std(X, axis=0))
Y = np.divide(Y - np.mean(Y, axis=0), np.std(Y, axis=0))

# OLS Regression.
reg = linear_model.LinearRegression()
reg.fit(X, Y)
beta = reg.coef_.transpose()

# Prediction values.
Ypred = np.dot(X, beta)

# Plot regression model prediction against actual.
plt.style.use('seaborn-colorblind')
fig1, ax1 = plt.subplots(ncols=m, figsize=(3*m, 2.8))
for i, ax in enumerate(ax1):
    ax.plot(Y[:, i], Ypred[:, i], 'o',
            label=("r_2 score: "+format(r2_score(Y[:, i], Ypred[:, i]), '.2f')))
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()]),
    ]
    ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.legend(loc='best')
plt.tight_layout()
plt.savefig("results/" + args.channel + "_regression.pdf")
plt.close(fig1)

# Plot sensitivity from regression coefficients.
params = [p.split('.')[-1] for p in channel.param_names]
fig2, ax2 = plt.subplots(nrows=m, sharex=True)
lims = [-np.amax(np.abs(beta)), np.amax(np.abs(beta))]
for i, ax in enumerate(ax2):
    ax.bar(params, beta[:, i])
    ax.set_ylim(lims)
    ax.axhline(y=0, linewidth=1, color='k')
# Formatting.
fig2.subplots_adjust(hspace=0)
plt.setp([a.get_xticklabels() for a in fig2.axes[:-1]], visible=False)
for tick in fig2.axes[-1].get_xticklabels():
    tick.set_rotation(45)
plt.setp([a.spines['top'] for a in fig2.axes], visible=False)
plt.setp([a.spines['bottom'] for a in fig2.axes], visible=False)
plt.setp([a.spines['right'] for a in fig2.axes], visible=False)
plt.setp([a.spines.values() for a in fig2.axes], linewidth=1)
plt.tick_params(top='off', bottom='off', right='off')
plt.savefig("results/" + args.channel + "_paramsens.pdf")
plt.close(fig2)
