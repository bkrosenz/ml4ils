from scipy import stats
from statsmodels.stats.multitest import multipletests
from itertools import repeat
import re
from typing import List
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import seaborn as sns
import numpy as np
from xarray import corr
from itertools import repeat, chain

data_matrices = ['Borowiec2015_Best108',
                 'Borowiec2015_Total1080',
                 'Nosenko2013_ribo_11057',
                 'Ryan2013_est',
                 'Ryan2013_est_only_choanozoa',
                 'Ryan2013_est_only_holozoa',
                 'Whelan2015_D10',
                 'Whelan2015_D10_only_choanozoa',
                 'Whelan2015_D1_only_choanozoa',
                 'Whelan2015_D1_only_holozoa',
                 'Whelan2015_D20_only_choanozoa',
                 'Whelan2017_full',
                 'Whelan2017_full_only_choanozoa']


def revert_dict(d): return dict(
    chain(*[zip(val, repeat(key)) for key, val in d.items()]))


DATA_DIM = 168


def grouped_shap(shap_values, features, groups):
    groupmap = revert_dict(groups)
    shap_Tdf = pd.DataFrame(
        shap_values,
        columns=pd.Index(features, name='features')).T
    shap_Tdf['group'] = shap_Tdf.reset_index().features.map(groupmap).values
    shap_grouped = shap_Tdf.groupby('group').sum().T
    return shap_grouped


def plot_gene_counts(dd):
    c = pd.DataFrame(
        {'Number of Genes': df.counts.sum(1), 'Data Matrix': df.matrix})
    sns.displot(data=c.reset_index(), hue='Data Matrix',
                x='Number of Genes', fill=True, bins=200, element='step')


def plot_topo(dd, outfile, col='topo'):
    g = sns.displot(
        data=dd,
        y='matrix',
        hue=col,
        stat='frequency',
        hue_order=['C', 'P', 'Pa'],
        # legend=False,
        common_norm=False,
        multiple='dodge',)
    g.set(ylabel=r'Data Matrix')
    # g.set(xscale='log')
    g._legend.set_title(r'Topology')
    g.set(xlabel=r'Supporting Quartets')

    # plt.tight_layout()
    plt.savefig(outfile)

    plt.close()


def plot_p(dd, outfile, col='preds'):
    g = sns.stripplot(
        data=dd,
        y='matrix',
        x=col,
        hue='result',
        jitter=.2,
        size=2.5)
    g.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    g.set(ylabel=r'Data Matrix')
    g.set(xlabel=r'$\hat{p}$')
    # plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()


def make_keys(s): return [p.name.replace('.genes.preds.pd.gz', '')
                          for p in s.glob('*.genes.preds.pd.gz')]


def avg_predictor(df: pd.DataFrame,
                  cols: List = ['matrix', 'ebl', 'ibl', 'dset_no']):
    """Average pred value across non-specified columns

    Args:
        df (dataframe): pred df
        cols (list, optional): columns to group by. Defaults to ['matrix', 'ebl', 'ibl', 'dset_no'].

    Returns:
        df: new preds
    """
    new_pred = df.groupby(cols).agg(np.mean).preds
    new_pred.name = 'avg_p'
    return df.merge(new_pred.to_frame(),
                    right_index=True,
                    left_on=cols)


def avg_topo_predictor(df: pd.DataFrame,
                       cols: List = ['matrix', 'ebl', 'ibl', 'dset_no']):
    """group by cols (e.g. label permutations), average the log-probabilities,
    and return the corresponding max.

    Args:
        df (pd.DataFrame): df with preds0,1,2 columns
        cols (List, optional): columns to aggregate over.
            Defaults to ['matrix', 'ebl', 'ibl', 'dset_no'].

    Returns:
        DataFrame: index of maximum.
    """
    topo = (
        df.groupby(
            cols
        )[['preds0', 'preds1', 'preds2']]
        .agg(lambda x: np.mean(np.exp(x)))
        .idxmax(1)
    )
    topo.name = 'avg_topo'
    return df.merge(
        topo,
        right_index=True,
        left_on=cols
    )


def get_z_scores(p: pd.DataFrame,
                 value: pd.Series,
                 tail: float = None,
                 cutoff: float = None):
    """Get the variables with the most extreme values - quantiles or z scores.

    Args:
        p (pd.DataFrame): _description_
        value (pd.Series): _description_
        tail (float, optional): _description_. Defaults to .05.
        cutoff (float, optional): _description_. Defaults to .5.

    Returns:
        _type_: _description_
    """
    p_mean = p.mean()
    p_std = p.std()
    if tail is not None:
        p_small = (p[value < value.quantile(tail)]-p_mean)/p_std
        p_large = (p[value >= value.quantile(1-tail)]-p_mean)/p_std
        return p_small, p_large
    else:
        p_small_z = (p_small.mean()-p_mean)/p_std
        p_large_z = (p_large.mean()-p_mean)/p_std
        # d = pd.concat([p_small_z[p_small_z.abs() > cutoff],
        #                p_large_z[p_large_z.abs() > cutoff]])
        return p_small_z[p_small_z.abs() > cutoff], p_large_z[p_large_z.abs() > cutoff]


def plot_log_odds(p: pd.DataFrame, tail=.05):
    print({'preds0': 'P', 'preds1': 'C', 'preds2': 'Pa'})
    likelihood_ratio = p.preds0-p.preds1
    return get_z_scores(p, likelihood_ratio, tail)


ppred_file = Path(
    '/N/project/phyloML/deep_ils/results/bo_final_2/metazoa2/metazoa.preds.pd.gz'
)
cpred_file = Path(
    '/N/project/phyloML/deep_ils/results/bo_final_topology/metazoa_classify2/metazoa.preds.pd.gz'
)

full_data_file = ppred_file.with_suffix('.full_data.pd.gz')

if __name__ == '__main__':
    try:
        dd = pd.read_pickle(
            full_data_file
        )
    except:
        m = pd.read_csv(
            '/N/project/phyloML/deep_ils/data/metazoa/li2021/animal_tree_root/manuscript/Supplementary_tables/Supplementary_Table_2.csv'
        )
        m['result'] = m.result.map(
            {'Ctenophora-sister': 'C', 'Porifera-sister': 'P', 'Unresolved': '*'}
        )
        m = (m.groupby('matrix')
             .result
             .agg(pd.Series.mode)
             .map(lambda s: sorted(s, key=lambda t: 'Z' if t == '*' else t))
             .apply(lambda s: '/'.join(s))
             .to_frame()
             )

        df = pd.read_pickle(ppred_file)
        index_names = df.index.names
        dfp = pd.read_pickle(cpred_file)

        # , on=df.columns.intersection(dfp.columns).to_list())
        dd = (df
              .reset_index()
              .merge(dfp.reset_index())
              .dropna()
              .merge(
                  m,
                  left_on='matrix',
                  right_index=True,
                  how='left',)
              )
        # dd.columns = dd.columns.map(lambda s: ''.join(map(str, s)))
        topo_cols = ['preds0', 'preds1', 'preds2']
        # list(
        #     zip(('preds0', 'preds1', 'preds2'), repeat(''), repeat('')))
        pred_col = 'preds'
        dd.columns = [c if not isinstance(c, tuple) or not (
            c[1] == '' and c[2] == '') else ''.join(c) for c in dd.columns]
        dd['topo'] = (
            dd[topo_cols]
            .idxmax(1)
            .map(dict(zip(topo_cols, ('P', 'C', 'Pa'))))
        )
        dd.loc[dd[pred_col] < .3, pred_col] = .3
        dd.loc[dd.result.isna(), 'result'] = '*'
        grouping_cols = ['ParaHoxozoa', 'Ctenophora',
                         'Porifera', 'Outgroup', 'matrix']
        dd = avg_predictor(
            dd, cols=grouping_cols)

        dd = (avg_topo_predictor(dd, cols=grouping_cols)
              .sort_values(['matrix', 'result']))
        dd['avg_topo'] = dd.avg_topo.map(
            {'preds0': 'P', 'preds1': 'C', 'preds2': 'Pa'})

        dd.to_pickle(full_data_file)

    dd = (dd
          .groupby('matrix')
          .filter(lambda x: len(x) >= 100))

    # DNN-Top
    for i in dd.Permutation.unique():
        plot_topo(
            dd.query(f'Permutation=={i}'),
            cpred_file.parent/f'metazoa_classify_preds_{i}.png')
        plot_p(
            dd.query(f'Permutation=={i}'),
            ppred_file.parent/f'metazoa_preds_{i}.png')

    plot_topo(dd,
              cpred_file.parent / f'metazoa_classify_preds_avg.png',
              col='avg_topo')

    plot_p(dd,
           ppred_file.parent/f'metazoa_preds_avg.png',
           col='avg_p')

    # DNN-Class
    # gmeanOpt = 0.9281
    gmeanOpt = 0.9511

    mcol = 'Data Matrix'
    dd.columns = dd.columns.map(lambda s: s if isinstance(
        s, tuple) else s.replace('matrix', mcol))
    sns.displot(data=dd, x='avg_p', hue=mcol)
    plt.xlabel(r'$\hat{p}$')
    plt.ylabel('Number of Quartets')
    plt.savefig(ppred_file.parent/'metazoa_DNN-Pred.png')
    plt.clf()

    # DNN-Class
    p = dd
    p = p[~p[mcol].str.startswith('Hejnol2009')]

    p['Concordant'] = p.preds > gmeanOpt

    g = p.groupby(mcol).Concordant

    c = g.mean().sort_index()

    c.name = '$p>0.9$'

    labels = (g.sum().map(str)+' / ' + g.count().map(str)).sort_index()
    ax = c.plot.barh(stacked=True, legend=False,)
    ax.bar_label(ax.containers[0],
                 labels=labels,
                 padding=8,  # color='b',
                 fontsize=10)
    ax.invert_yaxis()
    plt.xlim(0, 1)
    plt.xlabel('Positive Predictions')
    plt.ylabel('Matrix')
    plt.tight_layout()
    plt.savefig(ppred_file.parent/'metazoa_DNN-Class.png')
    plt.clf()

    # DNN-Top

    # porifera_low, porifera_hi = get_z_scores(p, p.preds0, cutoff=.1)
    por_minus_cte_low, por_minus_cte_hi = plot_log_odds(p, tail=.05)
    # # more options can be specified also
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print((por_minus_cte_low.mean() - por_minus_cte_hi.mean()))

    features = dd.columns.drop(
        [c for c in dd.columns if not isinstance(c, tuple)])

    # feature_levels = dict(zip(features.names, features.levels))
    # groups_by_feature = features.groupby(feature_levels['feature'])
    labels = {'1_2': r'$d(1,2)$', '1_3': r'$d(1,3)$', '1_4': r'$d(1,4)$',
              '2_3': r'$d(2,3)$', '2_4': r'$d(2,4)$', '3_4': r'$d(3,4)$',
              'counts': 'Topology Counts',
              'nsites': 'Number of Informative Sites',
              'seq_length': 'Sequence Length',
              'top_1': 'sCF-P', 'top_2': 'sCF-C',
              'top_3': 'sCF-Pa'}
    from collections import defaultdict
    groups = defaultdict(list)
    for f in features:
        groups[labels[f[0]]].append(f)

    def label_mapper(s): return (labels.get(s[0], s[0]),
                                 s[1].replace('nan', '').capitalize(),
                                 f'({int(s[2])})')
    columns = features.map(lambda tup: ' '.join(label_mapper(tup)))
    column_mapper = dict(zip(features, columns))
    phm = por_minus_cte_hi.mean()
    plm = por_minus_cte_low.mean()
    phm.index = phm.index.map(column_mapper)
    plm.index = plm.index.map(column_mapper)
    phm = phm[~phm.index.isna()]
    plm = plm[~plm.index.isna()]
    plm = plm[plm.abs() > .5].sort_values()
    phm = phm[phm.abs() > .5].sort_values()

    plm.plot.barh()
    plt.xlabel('Z-score')
    plt.tight_layout()
    plt.savefig(ppred_file.parent/'metazoa_low_porifera_score.png')
    plt.clf()

    phm.plot.barh()
    plt.xlabel('Z-score')
    plt.tight_layout()
    plt.savefig(ppred_file.parent/'metazoa_hi_porifera_score.png')
    plt.clf()

    # # Check out the Nosenko dataset
    # dn = dd[dd['Data Matrix'].str.contains('Nosenko2013')]
    # dnn = dd[~dd['Data Matrix'].str.contains('Nosenko2013')]
    # dnn = dnn[dnn.columns.dropna()]
    # dn = dn[dn.columns.dropna()]

    # utests = [(c, *stats.mannwhitneyu(dn[c], dnn[c])) for c in dn.columns]
    # ut = pd.DataFrame(utests, columns=['variable', 'u', 'pvalue'])

    # reject, pvalues, * \
    #     _ = multipletests(ut.pvalue, alpha=.05, method='holm-sidak')
    # zscore_nos = (dd[dd['Data Matrix'].str.contains(
    #     'Nosenko2013')].mean()-dd.mean())/dd.std()

    # zscore_nos.index = zscore_nos.index.map(column_mapper)


# p = pd.read_pickle(
#     full_data_file
# )
    predictors = [c for c in p.columns.map(lambda s: ''.join(
        map(str, s))) if c.endswith('0') and 'pred' not in c]

    intervals = np.array([
        [.4, .5],
        [.85, .95],
        [.95, 1.]
    ])

    pred_zscores = dict()
    dstd = p.std()
    dmean = p.mean()
    for i, (l, r) in enumerate(intervals):
        subset = p.query(f'{l}<avg_p<{r}')
        pred_zscores[i] = (subset.mean()-dmean) / dstd
    pred_zscores = pd.DataFrame(pred_zscores)
    n = 20
    low_minus_high = pred_zscores[0]-pred_zscores[1]
    print('low p vs high p:',
          low_minus_high.iloc[np.argpartition(-low_minus_high.abs(), n)[
              :n]].sort_values()
          )

    # print(
    #     pred_zscores.iloc[np.argpartition(-(pred_zscores[0]-pred_zscores[2]).abs(), n)[:n]])

    # class_zscores = dict()
    # dstd = p.std()
    # dmean = p.mean()
    # for i, (l, r) in enumerate(intervals):
    #     subset = p.query(f'Concordant=={l}')
    #     class_zscores[i] = (subset.mean()-dmean) / dstd
    # class_zscores = pd.DataFrame(class_zscores)
    # n = 15
    # print('class z scores',
    #       class_zscores.iloc[np.argpartition(-(class_zscores[1]-class_zscores[0]).abs(), n)[:n]])
