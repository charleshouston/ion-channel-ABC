import myokit
import myokit.lib.fit as fit
import channel_setup as cs
import numpy as np
import matplotlib.pyplot as plt
import ast
import distributions as Dist
import warnings

def LossFunction(sim_vals, exp_vals):
    # if the simulation failed, pred_vals should be None
    #  return infinite loss
    if sim_vals is None:
        return float("inf")

    # Calculate CV(RMSD) for each experiment
    tot_err = 0
    # Catch runtime overflow warnings from numpy
    warnings.filterwarnings('error')
    sim_vals = np.array(sim_vals) # predicted
    exp_vals = np.array(exp_vals[0]) # experimental
    for i,p in enumerate(sim_vals):
        e = exp_vals[i]
        try:
            err = np.square(p - e)
        except Warning:
            return float("inf")
        except:
            return float("inf")
        # normalise error
        err = pow(err,0.5)
        err = err/abs(np.mean(e))
        tot_err += err

    return tot_err


# Load full model
full_model = cs.full_sim()
m,_,_ = myokit.load('models/' + full_model.model_name)

i_stim = m.get('membrane.i_stim')
i_stim.set_rhs(0)
i_stim.set_binding('pace')

s = myokit.Simulation(m)

# Load individual channels
channels = [cs.icat(),
            cs.iha(),
            cs.ikur(),
            cs.ikr(),
            cs.ina2(),
            cs.ito(),
            cs.ik1(),
            cs.ical()]

# Load results from ABC simulation for each channel
results_ABC = []
for ch in channels:
    results_file = 'results/updated-model/results_' + ch.name + '.txt'
    f = open(results_file, 'r')
    lines = f.readlines()
    particles = ast.literal_eval(lines[0])
    weights = ast.literal_eval(lines[1])
    dists = Dist.Arbitrary(particles, weights)
    results_ABC.append(dists)

# Target values
variables = full_model.data_exp[0]
targets = np.array([full_model.data_exp[1]])
# weights = np.array([1,1,1,1])
# weights = weights * len(weights) / sum(weights)

# Number of simulations to run for average
N = 1

# Parameters and bounds for PSO search
parameters = full_model.parameter_names
bounds = full_model.prior_intervals

prot = myokit.pacing.blocktrain(1000, 0.5, limit=0, level=-80)

# Loss function
def score(guess):
    import numpy as np
    try:
        error = 0
        # for j, p in enumerate(parameters):
        #     param = m.get(p)
        #     s.set_constant(param, guess[j])
        results = []
        # Run for N draws from ABC posteriors
        for i in range(N):
            # Reset simulation to no stimulation
            # s.reset()
            # s.set_protocol(None)

            # # Set simulation constants to results from ABC
            # for j, c in enumerate(channels):
            #     params = results_ABC[j].draw()
            #     for k, p in enumerate(c.parameters):
            #         s.set_constant(p, params[k])

            # # Run no stimulation results
            # s.pre(20000)
            # output = s.run(1000, log=variables).npview()
            # r = [output[key][-1] for key in output.keys()]

            # Paced simulations
            s.reset()
            s.set_protocol(prot)

            # Set simulation constants to results from ABC
            for j, c in enumerate(channels):
                params = results_ABC[j].draw()
                for k, p in enumerate(c.parameters):
                    s.set_constant(p, params[k])

            output = s.run(101000,
                           log=['environment.time',
                                'membrane.V',
                                'ca_conc.Ca_i',
                                'ca_conc_sr.Ca_SR'])
            output.trim_left(99800, adjust=True)

            # Find resting potential before stim
            stim_time = 200.0
            stim_index = output.find(stim_time)
            v_rp = output['membrane.V'][stim_index]

            # Find voltage peak
            peak_index = output['membrane.V'].index(max(output['membrane.V']))
            peak_time = output['environment.time'][peak_index]
            v_max = output['membrane.V'][peak_index]
            APA = v_max - v_rp

            # Find APDs
            rep90 = APA * 0.1 + v_rp
            v_curr = v_max
            index = peak_index
            while v_curr > rep90:
                index += 1
                v_curr = output['membrane.V'][index]

            APD90 = output['environment.time'][index] - stim_time

            # Calcium measurements
            # Diastole
            ca_i_diastole = output['ca_conc.Ca_i'][stim_index]
            ca_sr_diastole = output['ca_conc_sr.Ca_SR'][stim_index]

            # Systole
            ca_i_systole = max(output['ca_conc.Ca_i'])
            peak_index_ca = output['ca_conc.Ca_i'].index(ca_i_systole)
            peak_time_ca = output['environment.time'][peak_index_ca]
            ca_sr_systole = min(output['ca_conc_sr.Ca_SR'])
            ca_time_to_peak  = peak_time_ca - stim_time

            # Decay measurements
            ca_amplitude = ca_i_systole - ca_i_diastole
            decay50 = ca_amplitude * 0.5 + ca_i_diastole
            decay90 = ca_amplitude * 0.1 + ca_i_diastole
            ca_curr = ca_i_systole
            index = peak_index_ca

            while ca_curr > decay50:
                index += 1
                ca_curr = output['ca_conc.Ca_i'][index]
            CaT50 = output['environment.time'][index] - stim_time

            while ca_curr > decay90:
                index += 1
                ca_curr = output['ca_conc.Ca_i'][index]
            CaT90 = output['environment.time'][index] - stim_time

            r = []
            r.append(v_rp)
            r.append(APD90)
            r.append(APA)
            r.append(ca_i_diastole)
            r.append(ca_time_to_peak)
            # r.append(ca_sr_diastole)
            r.append(ca_i_systole)
            # r.append(ca_sr_systole)
            r.append(CaT50)
            r.append(CaT90)
            results.append(r)

        # Average over results
        results = np.mean(results,0)
        # results = np.array(results).swapaxes(0,1)
        # error = np.sum(np.sqrt(((targets - results)/targets) ** 2))
        error = LossFunction(results, targets)
        return error / len(targets)
    except myokit._err.SimulationError:
        return float('inf')
    except Exception:
        return float('inf')

original = [2.7, 2.268e-16, 0.0008, 0.0026, 0.01, 0.9996]
original_err = score(original)
print "Original error: " + str(original_err)

def report_results(pg, fg):
    print "Current optimum position: " + str(pg.tolist())
    print "Current optimum score:    " + str(fg)

# Run optimisation algorithm
print 'Running optimisation...'
with np.errstate(all='ignore'):
    x, f = fit.pso(score, 
                   bounds, 
                   n=100, 
                   parallel=True,
                   callback=report_results)

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
                 'ca_conc.Ca_subSL',
                 'k_conc.K_i',
                 'na_conc.Na_i',
                 'ik1.i_K1',
                 'incx.i_NCX',
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
