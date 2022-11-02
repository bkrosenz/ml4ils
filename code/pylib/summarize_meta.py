from functools import partial, reduce
from itertools import *
from typing import Callable

import dask.dataframe as dd
import pandas as pd

import parallel_utils3 as par
import utils as u

column_mapper = {'qCF': 'top_1', 'qDF1': 'top_2',
                 'qDF2': 'top_3', 'qN': 'nsites'}

cols = ['qCF', 'qDF1', 'qDF2', 'qN', 'pdist', 'tid',
        'seq_length']


def summarize_dataset(p: u.Path, by='taxa', min_samps=25):
    fns = [pp.stem.split('.')[0] for pp in p.glob('*.pd.gz')]

    x = (pd
         .concat(
             [pd.read_pickle(fname)[cols] for fname in p.glob('*.pd.gz')],
             keys=fns,
             names=['gene'])
         .rename(columns=column_mapper))

    labels = u.leaves2labels(list('1234'), diag=False)

    mapper = {
        "pdist": partial(
            u.triu, labels=[l for l in labels],
            diag=False
        )
    }
    x = par.apply_mapping(x, mapper=mapper)

    def summarize_genome(df: pd.DataFrame) -> pd.DataFrame:
        if len(df) < min_samps:
            return pd.DataFrame()
        return par.summarize_chunk(df, group_cols=['tid'])

    x.index = x.index.droplevel('gene')
    taxa = x.index.names
    g = x.groupby(taxa)  # .apply(summarize)
    # elif by == 'genes':
    #     g = x.groupby('gene')  # .sample(1).reindex()
    #     n_replicates = g.top_1.count().min()  # bootstrap with replacement?
    # g = g.sample(n_replicates).reset_index(drop=True)
    # TODO: parallelize summarize func across samples; need per-sample index to group by

    res = par.map_parallel(summarize_genome, g, 8)
    res.index = res.index.droplevel(-1)

    return res
