from pathlib import Path
from sys import argv
from typing import Tuple

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Polygon
from stat_utils import *
matplotlib.use("Agg")
plt.ion()


out_dir = Path('/N/project/phyloML/deep_ils/results/all_algs')

pred_name = 'preds'  # prediction column in dnn pred file
EBL_MAX = 1
# filename to newick:
# rx=re.compile(r"t_(\d+)_([\d\.]+)_([\d\.]+)")

conds = [
    # 'ibl<=0.01 and ebl<=0.5',
    '0.001<ibl<=0.01 and ebl<=0.5',
    '0.01<ibl and ebl<=0.5',

    # '0.01<ibl<=0.1 and ebl<=0.3',
    # 'ibl<=0.01 and ebl<=0.3',

    '0.01<ibl and 0.5<ebl<=1.25',
    '0.001<ibl<=0.01 and 0.5<ebl<=1.25',
    '0.5<=ebl<=1.5',
    f'ebl<={EBL_MAX}',

]


def sigmoid(y: pd.DataFrame, eps=1e-10) -> pd.DataFrame:
    return 1 / (1 + np.exp(-y))


def plot_res(df: pd.DataFrame, query: str, outfile: Path = None):
    """Plot box + strip plots of by Algorithm.

    Args:
        df (pd.DataFrame): error values
        query (str): filter to apply to df
        outfile (Path, optional): outfile path.  if None, does not save fig. Defaults to None.
    """
    data = df.query(query)
    g = sns.boxplot(x='Algorithm',
                    y=pred_name,
                    data=data,
                    whis=[5, 95],
                    width=.6,
                    showfliers=False,
                    palette="vlag")
    sns.stripplot(x='Algorithm',
                  y=pred_name,
                  data=data,
                  size=2,
                  color=".3",
                  linewidth=0)
    g.set_ylabel(r'$\hat{p}-p$')
    g.set_ylim((-2./3, 2./3))
    if outfile is not None:
        plt.savefig(outfile.with_suffix('.'+query+'.png'))
    plt.clf()


def summarize():
    c_metrics = {'Acc': met.accuracy_score,
                 'F1': met.f1_score,
                 'AUC': met.roc_auc_score,
                 'Prec': met.precision_score,
                 'Recall': met.recall_score,
                 'MCC': met.matthews_corrcoef
                 }
    for r, v in c_metrics.items():
        print(r)
        try:
            d[r] = (clas.apply(lambda x: v(y_true_b, x)))
        except:
            d[r] = (b.apply(lambda x: v(y_true_b, x)))


def combine_results(s: str, sd: str) -> Tuple[pd.DataFrame, pd.Series]:
    """read predictions and return the errors.

    Args:
        s (str): path to sklearn results csv
        sd (str): path to dnn results csv

    Returns:
        pd.DataFrame,pd.Series: errors p_hat-p
    """
    d = sigmoid(pd.read_csv(s, index_col=[0, 1, 2]))
    dnn = pd.read_csv(sd, index_col=[0, 1, 2])
    dnn['DNN-Pred'] = dnn[pred_name]
    m = d.merge(dnn.drop(columns='y_true'),
                left_index=True,
                right_index=True)
    ix = m.index.to_frame()
    ix[['ebl', 'ibl']] /= 100
    m.columns.name = 'Algorithm'
    m = (m
         .set_index(pd.MultiIndex.from_frame(ix))
         .query(f'ebl<{EBL_MAX}'))

    y_true = m.y_true

    m = m[['Mean', 'ElasticNet', 'RF', 'AdaBoost',
           'GradBoost', 'DNN-Pred']].subtract(m.y_true, 0)
    return m, y_true


def reshape_df(df):
    return (df.groupby(['ebl', 'ibl'])
            .mean().stack()
            .reset_index()
            .rename(columns={0: pred_name})
            .reset_index())


if __name__ == '__main__':
    try:
        s, sd, outfile = argv[1:]
    except Exception as e:
        print(e)
        s = '/N/project/phyloML/deep_ils/results/test_data/g500_l500_20-skl/regress.preds.csv.gz'
        sd = '/N/project/phyloML/deep_ils/results/bo_final_2/test/g500_l500_f0.0_20/preds.csv.gz'
        outfile = 'bar'
    outfile = out_dir/outfile
    print(s, sd, outfile)

    m, y_true = combine_results(s, sd)
    m_absolute = m.abs()

    m_rel = (m
             .drop(columns='Mean')
             .divide(y_true, 0))
    print('error:')
    show_IQR(m)
    print('\n-----\nabsolute error:')
    show_IQR(m_absolute)
    print('\n-----\nrelative error:')
    show_IQR(m_rel)

    m_long_form = reshape_df(m)
    m_rel = reshape_df(m_rel)
    wtests = {}
    wtests_all = {}
    kwtests = {}
    for cond in conds:
        plot_res(m_long_form,
                 cond,
                 outfile=outfile)
        plot_res(m_rel,
                 cond,
                 outfile=outfile.with_name(outfile.name+'_rel_err'))
        df = m.query(cond)
        kwtests[cond] = kruskal(*df.values.T)
        wtests[cond] = {
            k: wilcoxon(df[k], df.Mean, alternative="two-sided")[1] for k in m.columns[1:]
        }
        wtests_all[cond] = rank_sum_test(df)

    wtests = pd.DataFrame.from_dict(wtests)
    wtests_all = pd.DataFrame.from_dict(wtests_all)
    kwtests = pd.DataFrame(kwtests, index=['stat', 'pval'])
    np.set_printoptions(precision=4, suppress=False)
    print('Kruskal-Wallis:', kwtests.T)
    print(get_significant(wtests[conds[-1]]))
    wtest_results = get_significant(wtests_all[conds[-1]])
    print(wtest_results)
    edges = [edge.split('>') for edge in wtest_results.rejected]
    graph = nx.from_edgelist(edges, create_using=nx.DiGraph)

    plt.figure(figsize=(12, 12), dpi=150)
    nx.draw_circular(
        graph,
        arrowsize=12,
        with_labels=True,
        node_size=8000,
        node_color="#ffff8f",
        linewidths=2.0,
        width=1.5,
        font_size=14,
    )
    plt.savefig(
        out_dir/'highest_error_wilcoxon.png'
    )
    # try:
    # TODO: fix 'UnicodeDecodeError: 'utf-8' codec can't decode byte 0x89 in position 0: invalid start byte' error
    #     df = pd.read_csv(filepath, index_col=(0, 1))
    # except:
    #     print("syntax: python plot_ml_results.py <results_file.csv>")
    #     raise
    # finally:
    #     print("called with", argv)

    # metrics = df.index.get_level_values("metrics").unique()

    # for metric in metrics:
    #     data = df.xs(metric, level=1, drop_level=True)
    #     baseline = sorted(
    #         data.index.intersection(
    #             ["Random", "Trivial", "Mean", "Median"]).tolist()
    #     )
    #     learners = sorted(data.index.difference(baseline).tolist()) + baseline

    #     print(learners)
    #     data = data.transpose()
    #     ax1, box_dict = data.boxplot(
    #         column=learners, rot=45, fontsize=9,
    #         return_type="both", patch_artist=True
    #     )

    #     # Hide these grid behind plot objects
    #     # ax1.set_axisbelow(True)
    #     ax1.set_title("Performance")
    #     ax1.set_xlabel("Algorithm")
    #     ax1.set_ylabel(metric)

    #     if len(argv) > 2 and argv[2] == "log":
    #         ax1.set_yscale("log")

    #     # Now fill the boxes with desired colors
    #     boxColors = ["darkkhaki", "royalblue"]

    #     for patch, name in zip(box_dict["boxes"], learners):
    #         if name in baseline:
    #             patch.set(facecolor="pink", color="red")
    #         else:
    #             patch.set(facecolor="lightblue")

    #     plt.tight_layout()

    #     plt.savefig(path.splitext(filepath)[0] + ".%s.png" % metric)
    #     plt.clf()
