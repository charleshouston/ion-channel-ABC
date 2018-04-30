import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('channel', type=str)
args = parser.parse_args()

# Import correct channel.
ch = None
if args.channel == 'icat':
    import channels.icat
    ch = channels.icat.icat
elif args.channel == 'ikur':
    import channels.ikur
    ch = channels.ikur.ikur
elif args.channel == 'ikr':
    import channels.ikr
    ch = channels.ikr.ikr
elif args.channel == 'iha':
    import channels.iha
    ch = channels.iha.iha
elif args.channel == 'ina':
    import channels.ina
    ch = channels.ina.ina
elif args.channel == 'ito':
    import channels.ito
    ch = channels.ito.ito
elif args.channel == 'ik1':
    import channels.ik1
    ch = channels.ik1.ik1
elif args.channel == 'ical':
    import channels.ical
    ch = channels.ical.ical
else:
    raise ValueError("Unrecognised channel.")

measurements = pd.DataFrame(columns = ['exp', 'x', 'y'])

i = 0
for exp in ch.experiments:
    data = exp.data.df
    data['exp'] = i
    i += 1
    measurements = measurements.append(data)

with pd.option_context('display.max_rows', -1, 'display.max_columns', 5):
    print measurements.to_string(index=False)
