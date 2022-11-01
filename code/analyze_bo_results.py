from ray.tune import ExperimentAnalysis
from joblib import dump, load
import pandas as pd
from pathlib import Path
from sys import argv
try:
    s = Path(argv[1])
except:
    s = Path('/N/project/phyloML/deep_ils/results/torch/asha_classify')
try:
    metric = argv[2]
except:
    metric = 'loss'
mode = 'min' if metric == 'loss' else 'max'

analysis = ExperimentAnalysis(s)
analysis.set_filetype('json')
a = analysis.dataframe()
a = a.join(
    pd.DataFrame(a['config/layers'].tolist(),
                 index=a.index).add_prefix('layer_').fillna(0))


def get_num_params(l): return sum(a*b for a, b in zip(l[:-1], l[1:]))


a['n_params'] = a['config/layers'].map(
    get_num_params)
a.to_csv(s.with_suffix('.analysis.csv.gz'))

print(a.corr()[metric].dropna())

best_n = a.loc[((1-2*(mode == 'min'))*a[metric]).nlargest(10).index]
print(
    best_n[['config/layers', 'config/lr', metric]].sort_values(metric, 0)
)
print('number of trials:', len(a))

if mode == 'min':
    top = a[a[metric] < a[metric].quantile(.01)]
else:
    top = a[a[metric] > a[metric].quantile(.99)]

ld = top.iloc[top.n_params.argmin()].logdir
config = analysis.get_all_configs()[ld]
config_str = s/'best_model.config'
# config = analysis.get_best_config(metric=metric, mode=mode)
dump(config, config_str)
