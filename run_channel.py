import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('channel', type=str)
args = parser.parse_known_args()

# Import correct channel.
ch = None
if args[0].channel == 'icat':
    import channels.icat
    ch = channels.icat.icat
elif args[0].channel == 'ikur':
    import channels.ikur
    ch = channels.ikur.ikur
elif args[0].channel == 'ikr':
    import channels.ikr
    ch = channels.ikr.ikr
elif args[0].channel == 'iha':
    import channels.iha
    ch = channels.iha.iha
elif args[0].channel == 'ina':
    import channels.ina
    ch = channels.ina.ina
elif args[0].channel == 'ito':
    import channels.ito
    ch = channels.ito.ito
elif args[0].channel == 'ik1':
    import channels.ik1
    ch = channels.ik1.ik1
elif args[0].channel == 'ical':
    import channels.ical
    ch = channels.ical.ical
else:
    raise ValueError("Unrecognised channel.")

# Collect parameters from input arguments.
for p in ch.pars:
    parser.add_argument("-" + p, type=float)
args = parser.parse_args()
args_d = vars(args)
del args_d['channel']
sim = ch(args_d)

with pd.option_context('display.max_rows', -1, 'display.max_columns', 5):
    print sim.to_string()
