import numpy as np

METRICS = {'single_mean': {'call': lambda x: np.mean(x.values), 'is_singlecol': True},
           'multicol_mean': {'call': lambda x: np.mean(x.values), 'is_singlecol': False}}
