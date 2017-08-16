import myokit
import myokit.lib.markov as markov

import channel_setup as cs
import distributions as Dist

import matplotlib.pyplot as plt
import ast
import numpy as np

plt.style.use('seaborn-colorblind')

ch = cs.ina()
m,_,_ = myokit.load('models/'+ch.model_name)

with open('../results/final/results_' + ch.name + '.txt') as f:
    pool = f.readline()
    pool = ast.literal_eval(pool)
    weights = f.readline()
    weights = ast.literal_eval(weights)

# Pre-pulse and holding potential
V1 = -140
V2 = -30
t_pre = 2
t_step = 18

logs = []
np.seterr(over='raise')
for j,params in enumerate(pool):
    for i,p in enumerate(params):
        m.set_value(ch.name + '.' + ch.parameter_names[i], p)
    conductance = m.get('ina.g_Na').value()
    rev_potential = m.get('ina.E_Na').value()

    try:
    # Create analytical Markov model
        m_linear = markov.LinearModel.from_component(m.get('ina'), current = 'ina.i_Na', states=ch.markov_states, vm='membrane.V')
        s = markov.AnalyticalSimulation(m_linear)
        s.set_membrane_potential(V1)
        s.pre(30)
        d = s.run(t_pre)
        s.set_membrane_potential(V2)
        d = s.run(t_step, log=d)
        s.set_membrane_potential(V1)
        log_states = []
        append=True
        for state in m.states():
            if any(abs(d[state]) > 1.0 ) or any(np.iscomplex(d[state])):
                print "Failed: " + str(j)
                append = False
                break

        if not append:
            continue

        for state in m.states():
            if state.name() is 'O':
                curr = d[state]*conductance*(V2-rev_potential)
            log_states.append(d[state])
        log_states.append(curr)
        logs.append(log_states)
    except Exception:
        print "Failed: " + str(j)

names = []
plot1 = []
plot2 = []
plot3 = []
for i,state in enumerate(m.states()):
    names.append(state.name())
    if state.name() in ['O','C1','C2','C3']:
        plot1.append(i)
    elif state.name() in ['IF','IC2','IC3']:
        plot2.append(i)

logs = np.array(logs)
logs = logs.swapaxes(0,1)
means = []
stddevs = []
for state in logs:
    dst = Dist.Arbitrary(state, weights)
    means.append(np.array(dst.getmean()))
    stddevs.append(np.array(np.sqrt(dst.getvar())))

linestyles = ['-','--','-.',':']

fig, ax = plt.subplots(nrows=1, ncols=3,figsize = (9, 2.8))
ax[0].plot(d.time(), means[-1])
ax[0].fill_between(d.time(), means[-1]-stddevs[-1], means[-1]+stddevs[-1], alpha=0.2)
ax[0].set_xlabel('Time (ms)')
ax[0].set_ylabel('Current (pA/pF)')

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
j = 0
axins = inset_axes(ax[1], 1.2, 0.8, loc=1)
for i in plot1:
    ax[1].plot(d.time(), means[i], label=names[i],ls=linestyles[j])
    ax[1].fill_between(d.time(), means[i]-stddevs[i], means[i]+stddevs[i], alpha=0.2)
    axins.plot(d.time(), means[i], label=names[i],ls=linestyles[j])
    axins.fill_between(d.time(), means[i]-stddevs[i], means[i]+stddevs[i], alpha=0.2)
    j += 1

ax[1].set_xlabel('Time (ms)')
ax[1].set_ylabel('State proportion')
ax[1].set_ylim([0, 1.1])
x1, x2, y1, y2 = 1.9, 3.5, 0., 0.6
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
# from mpl_toolkits.axes_grid1.inset_locator import mark_inset
# mark_inset(ax[1], axins, loc1=2,loc2=4,fc="none",ec="0.5")
ax[1].legend(loc="lower right")


j = 0
for i in plot2:
    ax[2].plot(d.time(), means[i], label=names[i],ls=linestyles[j])
    j += 1
    ax[2].fill_between(d.time(), means[i]-stddevs[i], means[i]+stddevs[i], alpha=0.2)
ax[2].legend(loc="upper left")
ax[2].set_xlabel('Time (ms)')
ax[2].set_ylim([0,1.1])
ax[2].set_ylabel('State proportion')

uppercase_letters = map(chr, range(65,91))
for i,a in enumerate(ax.flatten()):
    a.text(-0.07, -0.06,
            uppercase_letters[i], transform=a.transAxes,
            fontsize=16, fontweight='bold', va='top', ha='right')
plt.tight_layout()
fig.savefig('final_results/states_leg_'+str(ch.name)+'.pdf', bbox_inches="tight")
plt.close(fig)
