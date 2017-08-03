import myokit
import numpy as np
import matplotlib.pyplot as plt
import ast # for literal_eval
import channel_setup as cs
import distributions as Dist
from myokit.lib.plots import current_arrows

# Load full model
m,p,x = myokit.load('models/Houston2017.mmt')
# m,p,x = myokit.load('models/Bondarenko2004_septal.mmt')
print myokit.step(m)

i_stim = m.get('membrane.i_stim')
i_stim.set_rhs(0)
i_stim.set_binding('pace')
s = myokit.Simulation(m)

# Channel list
channels = [cs.icat(),
            cs.iha(),
            cs.ikur(),
            cs.ikr(),
            cs.ina(),
            cs.ito(),
            cs.ik1(),
            cs.ikach()]

# Load results from ABC simulation for each channel
results = []
for ch in channels:
    results_file = 'results/results_' + ch.name + '.txt'
    f = open(results_file, 'r')
    lines = f.readlines()
    particles = ast.literal_eval(lines[0])
    weights = ast.literal_eval(lines[1])
    if ch.name == 'ina':
        remove = [10,57,78,92,150,186,218,220,253,279,323,380]
        particles = [i for j,i in enumerate(particles) if j not in remove]
        weights = [i for j,i in enumerate(weights) if j not in remove]
    dists = Dist.Arbitrary(particles, weights)
    results.append(dists)

# Set simulation constants to results from ABC
for i, c in enumerate(channels):
    # params = results[i].getmean()
    params = results[i].draw()
    for j, p in enumerate(c.parameters):
        s.set_constant(c.parameters[j], params[j])

p = myokit.pacing.blocktrain(1000, 0.5, limit=0, level=-60)
s.set_protocol(p)
s.pre(100000)
out = s.run(20000,
            log=['environment.time',
                 'membrane.V',
                 'calcium_concentration.Cai',
                 'ical.i_CaL',
                 'incx.i_NaCa',
                 'icab.i_Cab',
                 'iha.i_ha',
                 'icat.i_CaT',
                 # 'membrane.i_stim',
                 'ina.i_Na',
                 'ito.i_to',
                 'ik1.i_K1',
                 'ikur.i_Kur',
                 'ikr.i_Kr',
                 'ikach.i_KAch'])
                 # 'icap.i_pCa',
                 # 'ipca.i_pCa',
                 # 'inak.i_NaK',
                 # 'ikss.i_Kss',
                 # 'iclca.i_ClCa'],
# fig,ax = plt.subplots(1, 1)
# current_arrows(out, 'membrane.V', ['ina.i_Na', 'ik1.i_K1','iha.i_ha','icat.i_CaT', 'ical.i_CaL', 'ito.i_to',  'ikur.i_Kur', 'ikr.i_Kr'], ax)
# import pdb;pdb.set_trace()

plt.style.use('seaborn-colorblind')
fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)
ax1.plot(out[out.keys()[0]], out[out.keys()[1]])
ax2.plot(out[out.keys()[0]], out[out.keys()[2]])
# for k in out.keys()[2:len(out.keys())/2]:
#     ax2.plot(out[out.keys()[0]], out[k], label=k)
# ax2.legend(loc='lower right')
for k in out.keys()[3:]:
    ax3.plot(out[out.keys()[0]], out[k], label=k)
ax3.legend(loc='lower right')
fig.show()
import pdb;pdb.set_trace()
