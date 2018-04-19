import pandas as pd

import channels.icat
ch = channels.icat.icat

# Collect parameters from input arguments.
import argparse
parser = argparse.ArgumentParser()
for p in ch.pars:
    parser.add_argument("-" + p, type=float)
args = parser.parse_args()
sim = ch(vars(args))

with pd.option_context('display.max_rows', -1, 'display.max_columns', 5):
    print sim.to_string()
