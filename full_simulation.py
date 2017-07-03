import myokit
import numpy as np
import matplotlib.pyplot as plt
import ast # for literal_eval
import channel_setup as cs

# Load full model
m,p,x = myokit.load('models/Houston2017.mmt')
print myokit.step(m)

s = myokit.Simulation(m)
# Channel list
channels = [cs.ina(),
            cs.icat(),
            cs.iha(),
            cs.ikur(),
            cs.ikr(),
            cs.ito()]

# Load results from ABC simulation for each channel
results_files = ['results/results_' + c.name + '.txt' for c in channels]
results = []
for rf in results_files:
    f = open(rf, 'r')
    lines = f.readlines()
    results.append(ast.literal_eval(lines[2]))

# Set simulation constants to results from ABC
for i, c in enumerate(channels):
    for j, p in enumerate(c.parameters):
        s.set_constant(c.parameters[j], results[i][j])

s.pre(10000)
out = s.run(5000,
            log=['environment.time',
                 'membrane.V',
                 'ical.i_CaL',
                 'ipca.i_pCa',
                 'incx.i_NaCa',
                 'ina.i_Na',
                 'inak.i_NaK',
                 'ito.i_to',
                 'ik1.i_K1',
                 'ikur.i_Kur',
                 'ikss.i_Kss',
                 'ikr.i_Kr',
                 'iclca.i_ClCa',
                 'icat.i_CaT',
                 'iha.i_ha'],
            log_interval=.1)

plt.style.use('seaborn-colorblind')
fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)
ax1.plot(out[out.keys()[0]], out[out.keys()[1]])
for k in out.keys()[2:len(out.keys())/2]:
    ax2.plot(out[out.keys()[0]], out[k], label=k)
ax2.legend(loc='lower right')
for k in out.keys()[len(out.keys())/2:]:
    ax3.plot(out[out.keys()[0]], out[k], label=k)
ax3.legend(loc='lower right')
fig.show()
import pdb;pdb.set_trace()
