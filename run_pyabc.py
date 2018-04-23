### Script to run ABCSMC using pyabc library.

from pyabc import (ABCSMC, Distribution, RV)
from pyabc.populationstrategy import AdaptivePopulationSize
import matplotlib.pyplot as plt
import os
import tempfile
import pandas as pd
import scipy as sp
import numpy as np
import subprocess
from io import BytesIO
db_path = ("sqlite:///" +
           os.path.join(tempfile.gettempdir(), "test.db"))

def simulate(**pars):
    """Wrapper to simulate the myokit model.

    Simulates in a subprocess running python2 by passing
    parameters as arguments to (another) wrapper script.
    """
    myokit_python = ("/home/cph211/miniconda3/envs" +
                     "/ion_channel_ABC/bin/python")
    script = "run_channel.py"
    args = [myokit_python, script]
    for p in pars:
        try:
            args.append("-" + str(p))
            args.append(str(pars[p]))
        except:
            print("Error: " +
                  "args is " + str(args))
    re = subprocess.run(args, stdout=subprocess.PIPE)
    d = pd.read_table(BytesIO(re.stdout),
                      delim_whitespace=True,
                      header=0, index_col=0)
    return d

# Get experimental measurements.
myokit_python = ("/home/cph211/miniconda3/envs" +
                 "/ion_channel_ABC/bin/python")
args = [myokit_python, "get_measurements.py"]
re = subprocess.run(args, stdout=subprocess.PIPE)
measurements = pd.read_table(BytesIO(re.stdout),
                             delim_whitespace=True,
                             header=0, index_col=0)

limits = dict(g_Na=(0, 100),
              E_Na=(0, 100),
              p1=(0, 100),
              p2=(-10, 0),
              p3=(0, 1),
              p4=(0, 100),
              p5=(-1, 0),
              p6=(0, 1),
              p7=(0, 100),
              q1=(0, 100),
              q2=(0, 10))

prior = Distribution(**{key: RV("uniform", a, b - a)
                        for key, (a, b) in limits.items()})

def distance(sim, obs):
    sim = pd.DataFrame(sim)
    obs = pd.DataFrame(obs)
    dist = 0
    for i in obs.exp.unique():
        try:
            err = np.sum(np.square(obs[obs.exp == i].y - sim[sim.exp == i].y))
        except:
            return float("inf")
        err = pow(err / len(obs[obs.exp == i].y), 0.5)
        err /= np.ptp(obs[obs.exp == i].y)
        dist += err
    return dist

def simulate_pyabc(parameter):
    res = simulate(**parameter)
    return res

abc = ABCSMC(simulate_pyabc, prior, distance,
             population_size=AdaptivePopulationSize(100, 0.3,
                                                    max_population_size=5000))
abc_id = abc.new(db_path, measurements)
history = abc.run(max_nr_populations=10, minimum_epsilon=0.5)
