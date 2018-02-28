import sys
sys.path.append("..")

from experiment import ExperimentStimProtocol
import myokit
import matplotlib.pyplot as plt

vsteps = range(-75, 40, 5)
stim_times = [500, 300, 100]
stim_levels = [-75, vsteps, -75]

def measure_fn(data):
    return min(data[0]['icat.i_CaT'])

measure_index = 1

prot_iv = ExperimentStimProtocol(stim_times, stim_levels,
                                 measure_index, measure_fn)

m, _, _ = myokit.load('../models/Korhonen2009_iCaT.mmt')
v = m.get('membrane.V')
v.demote()
v.set_rhs(0)
v.set_binding(None)

s = myokit.Simulation(m)

vvar = 'membrane.V'
logvars = myokit.LOG_ALL

results = prot_iv(s, vvar, logvars)

fig_iv  = plt.figure()
ax = fig_iv.subplots()
ax.plot(vsteps, results)

intervals = [32, 64, 96, 128, 160, 192, 224, 256, 288, 320]
stim_times = [500, 300, intervals, 300, 500]
stim_levels = [-80, -20, -80, -20, -80]
measure_index = [1, 3]

def measure_fn(data):
    return max(data[1]['icat.G_CaT'])/max(data[0]['icat.G_CaT'])
prot_rec = ExperimentStimProtocol(stim_times, stim_levels,
                                  measure_index, measure_fn)

results_rec = prot_rec(s, vvar, logvars)

fig_rec = plt.figure()
ax = fig_rec.subplots()
ax.plot(intervals, results_rec)
plt.show()
