from itertools import permutations
from scipy.stats import kruskal, wilcoxon
from statsmodels.stats.multitest import multipletests

import pandas as pd


def show_IQR(df: pd.DataFrame):
    """return 50%

    Args:
        df (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    quantiles = df.describe()
    IQR = quantiles.loc['75%']-quantiles.loc['25%']
    print('\tQuantiles:\n', quantiles, '\nIQR:', IQR)
    return IQR


def get_significant(w: pd.Series):
    reject, pvals, *alphas = multipletests(
        w,
        alpha=.01,
        method='holm')
    output = pd.DataFrame.from_dict(
        {'rejected': w.index[reject], 'pvals': pvals[reject]})
    print(alphas)
    return output


def rank_sum_test(df: pd.DataFrame):
    return {
        '>'.join((k1, k2)): wilcoxon(df[k1], df[k2], alternative="greater")[1] for k1, k2 in permutations(df.columns, 2)
    }
