import pandas as pd

import channels.icat
ch = channels.icat.icat

measurements = pd.DataFrame(columns = ['exp', 'x', 'y'])

i = 0
for exp in ch.experiments:
    data = exp.data.df
    data['exp'] = i
    i += 1
    measurements = measurements.append(data)

with pd.option_context('display.max_rows', -1, 'display.max_columns', 5):
    print measurements.to_string()
