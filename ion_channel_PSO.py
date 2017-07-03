import myokit
import myokit.lib.fit as fit
import channel_setup as cs
import numpy as np
import matplotlib.pyplot as plt

# Get channel details
channel = cs.ik1()
current_name = 'ik1.i_K1'
m,p,x = myokit.load('models/' + channel.model_name)
membrane_potential = m.get('membrane.V')
membrane_potential.demote()

# Get experimental data
vsteps = channel.data_exp[0][0]
act_exp = channel.data_exp[0][1]

# Get original values
original = []
for k, v in enumerate(vsteps):
    membrane_potential.set_rhs(v)
    original.append(m.get(current_name).value())

# Parameters and bounds for PSO search
parameters = channel.parameters
bounds = channel.prior_intervals

# Loss function
def score(guess):
    import numpy as np
    try:
        error = 0
        for j, p in enumerate(parameters):
            param = m.get(p)
            param.set_rhs(guess[j])
        for k, v in enumerate(vsteps):
            membrane_potential.set_rhs(v)
            i = m.get(current_name).value()
            r = act_exp[k]
            error += np.sqrt((i - r) ** 2)
        return error / len(vsteps)
    except Exception:
        return float('inf')

# Run PSO algorithm
print 'Running particle swarm optimisation...'
with np.errstate(all='ignore'):
    x, f = fit.pso(score, bounds, n=100, max_iter=1000)

# Write results in same format as ABC
with open('results/results_'+channel.name+'.txt', 'w') as f:
    f.write(str([x.tolist()])+"\n")
    f.write(str([1])+"\n")
    f.write(str(x.tolist())+"\n")
    f.write(str(np.zeros(len(x)).tolist()))

# Set model to PSO-optimised parameters
for i, res in enumerate(x):
    param = m.get(channel.parameters[i])
    param.set_rhs(res)

# Generate simulation output from PSO parameters
out = []
for k, v in enumerate(vsteps):
    membrane_potential.set_rhs(v)
    out.append(m.get(current_name).value())

# Plot results
plt.style.use('seaborn-colorblind')
fig, ax = plt.subplots(figsize=(5,5))
fig.suptitle('Voltage clamp simulations for ' + channel.name + ' in HL-1')
ax.plot(vsteps, act_exp, 'o', label="Simulation")
ax.plot(vsteps, out, 'x', label="Experiment")
ax.plot(vsteps, original, 's', label="Published model")
ax.legend(loc='lower right')

fig.savefig('results/fig_'+channel.name+'.eps', bbox_inches="tight")
