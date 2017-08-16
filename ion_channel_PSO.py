import myokit
import myokit.lib.fit as fit
import channel_setup as cs
import numpy as np
import matplotlib.pyplot as plt
import ast
import distributions as Dist
from functools import partial

# Load full model
full_model = cs.full_sim()
m,_,_ = myokit.load('models/' + full_model.model_name)
i_stim = m.get('membrane.i_stim')
i_stim.set_rhs(0)
i_stim.set_binding('pace')
iha = m.get('iha.i_ha')
iha.set_rhs(0)

s = myokit.Simulation(m)

# Load individual channels
channels = [cs.icat(),
            cs.iha(),
            cs.ikur(),
            cs.ikr(),
            cs.ina(),
            cs.ito()]

# Load results from ABC simulation for each channel
results_ABC = []
for ch in channels:
    results_file = 'final_results/results_' + ch.name + '.txt'
    f = open(results_file, 'r')
    lines = f.readlines()
    particles = ast.literal_eval(lines[0])
    weights = ast.literal_eval(lines[1])
    dists = Dist.Arbitrary(particles, weights)
    results_ABC.append(dists)

# Target values
variables = full_model.data_exp[0]
targets = np.array([full_model.data_exp[1]])
weights = np.array([1,1,1,1,1,1,1])
weights = weights * len(weights) / sum(weights)

# Number of simulations to run for average
N = 50

# Parameters and bounds for PSO search
parameters = full_model.parameter_names
bounds = full_model.prior_intervals

prot = myokit.pacing.blocktrain(1000, 2, limit=0, level=-30)

# Loss function
def score(guess):
    import numpy as np
    try:
        error = 0
        for j, p in enumerate(parameters):
            param = m.get(p)
            s.set_constant(param, guess[j])
        results = []
        # Run for N draws from ABC posteriors
        for i in range(N):
            s.reset()
            # Set simulation constants to results from ABC
            for j, c in enumerate(channels):
                params = results_ABC[j].draw()
                for k, p in enumerate(c.parameters):
                    s.set_constant(p, params[k])
            # s.set_protocol(prot)
            s.run(20000)
            s.set_default_state(s.state())
            output = s.run(10, log=variables).npview()
            r = [output[key][0] for key in output.keys()]
            results.append(r)

        # Average over results
        results = np.mean(results,0)
        # results = np.array(results).swapaxes(0,1)
        error = np.sum(np.sqrt(((targets - results)/targets) ** 2))
        return error / len(targets)
    except myokit._err.SimulationError:
        return float('inf')
    except Exception:
        return float('inf')

original = [0.2938, 0,  1, 0.88, 292.8,  0.000367, 0.0026]
original_err = score(original)
print "Original error: " + str(original_err)

def report_results(pg, fg):
    print "Current optimum position: " + str(pg.tolist())
    print "Current optimum score:    " + str(fg)

# Run optimisation algorithm
print 'Running optimisation...'
with np.errstate(all='ignore'):
    x, f = fit.pso(score,bounds,n=20,callback=report_results)

# Set simulation constants to results from ABC
s.reset()
for j, c in enumerate(channels):
    params = results_ABC[j].draw()
    for k, p in enumerate(c.parameters):
        s.set_constant(c.parameters[k], params[k])
for j, p in enumerate(parameters):
    param = m.get(p)
    s.set_constant(param, x[j])

s.pre(95000)
out = s.run(5000,
            log=['environment.time',
                 'membrane.V',
                 'calcium_concentration.Cai',
                 'potassium_concentration.Ki',
                 'sodium_concentration.Nai',
                 'ik1.i_K1',
                 'ipca.i_pCa',
                 'incx.i_NaCa',
                 'inak.i_NaK',
                 'icab.i_Cab',
                 'inab.i_Nab'])
plt.style.use('seaborn-colorblind')
fig, ax = plt.subplots(nrows=2, ncols=2)
ax[0][0].plot(out[out.keys()[0]], out[out.keys()[1]])
ax[0][1].plot(out[out.keys()[0]], out[out.keys()[2]])
ax[0][0].ticklabel_format(useOffset=False)
ax[0][1].ticklabel_format(useOffset=False)
ax[1][0].ticklabel_format(useOffset=False)
ax[1][1].ticklabel_format(useOffset=False)
for k in out.keys()[3:5]:
    ax[1][0].plot(out[out.keys()[0]], out[k], label=k)
ax[1][0].legend(loc='best')
for k in out.keys()[5:]:
    ax[1][1].plot(out[out.keys()[0]], out[k], label=k)
ax[1][1].legend(loc='best')
fig.show()
import pdb;pdb.set_trace()
