from os import path
from pathlib import Path
from sys import argv
import pandas as pd
from matplotlib.patches import Polygon
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

matplotlib.use("Agg")
plt.ion()


# filename to newick:
# rx=re.compile(r"t_(\d+)_([\d\.]+)_([\d\.]+)")

# import matplotlib
# matplotlib.interactive(True)

def plot_res(query, outfile: Path = None):
    data = perf.query(query)
    g = sns.boxplot(x='Algorithm',
                    y=pred_name,
                    data=data,
                    whis=[0, 100],
                    width=.6,
                    palette="vlag")
    sns.stripplot(x='Algorithm',
                  y=pred_name,
                  data=data,
                  size=4,
                  color=".3",
                  linewidth=0)
    g.set_ylabel(r'$\hat{p}-p$')
    if outfile is not None:
        plt.savefig(outfile/(query+'.png'))
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


filepath = (
    len(argv) > 1 and argv[1] or "/Users/ben/deep_ils/results/results.classify.csv"
)

pred_name = 'preds'

d = u.sigmoid(pd.read_csv(s, index_col=[0, 1, 2]))
dnn = pd.read_csv(sd, index_col=[0, 1, 2])
dnn['DNN-Pred'] = dnn.preds
m = d.merge(dnn.drop(columns='y_true'),
            left_index=True,
            right_index=True)
m = m[['Mean', 'ElasticNet', 'RF', 'AdaBoost',
       'GradBoost', 'DNN-Pred']].subtract(m.y_true, 0)
m_rel = m.drop(columns='Mean').divide(m.Mean, 0)

m.columns.name = 'Algorithm'

perf = (m.groupby(['ebl', 'ibl'])
        .mean().stack()
        .reset_index()
        .rename(columns={0: pred_name})
        .reset_index())


try:
    # TODO: fix 'UnicodeDecodeError: 'utf-8' codec can't decode byte 0x89 in position 0: invalid start byte' error
    df = pd.read_csv(filepath, index_col=(0, 1))
except:
    print("syntax: python plot_ml_results.py <results_file.csv>")
    raise
finally:
    print("called with", argv)

metrics = df.index.get_level_values("metrics").unique()

for metric in metrics:
    data = df.xs(metric, level=1, drop_level=True)
    baseline = sorted(
        data.index.intersection(
            ["Random", "Trivial", "Mean", "Median"]).tolist()
    )
    learners = sorted(data.index.difference(baseline).tolist()) + baseline

    print(learners)
    data = data.transpose()
    ax1, box_dict = data.boxplot(
        column=learners, rot=45, fontsize=9,
        return_type="both", patch_artist=True
    )

    # Hide these grid behind plot objects
    # ax1.set_axisbelow(True)
    ax1.set_title("Performance")
    ax1.set_xlabel("Algorithm")
    ax1.set_ylabel(metric)

    if len(argv) > 2 and argv[2] == "log":
        ax1.set_yscale("log")

    # Now fill the boxes with desired colors
    boxColors = ["darkkhaki", "royalblue"]

    for patch, name in zip(box_dict["boxes"], learners):
        if name in baseline:
            patch.set(facecolor="pink", color="red")
        else:
            patch.set(facecolor="lightblue")

    plt.tight_layout()

    plt.savefig(path.splitext(filepath)[0] + ".%s.png" % metric)
    plt.clf()
