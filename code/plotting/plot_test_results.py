import argparse
import re
from functools import partial, reduce
# from joblib import Parallel, delayed, dump, load
from pathlib import Path

import numpy as np
from arviz import hdi
from pandas.core.frame import DataFrame
from pandas.io.sql import DatabaseError
from scipy.special import rel_entr

try:
    from .plotting_utils import *
    from .stat_utils import *
except:
    from plotting_utils import *
    from stat_utils import *

MAX_EBL = 120


def confidence(p,
               groups: DataFrame,
               interval: float = .8):
    z = {}
    for k, v in groups:
        if interval == 1:
            h = (v.preds.min(), v.preds.max())
        else:
            h = hdi(v.preds.values, hdi_prob=interval)
        z[k] = min(h[0], 1) <= p.loc[k] <= h[1]
    return pd.Series(z)*2-1


def d_kl(p, p_hat):
    return rel_entr(p, p_hat) + rel_entr(1 - p, 1 - p_hat)


def d_H(p, p_hat):
    sqrt_p = np.sqrt(np.array([p, 1 - p]))
    sqrt_phat = np.sqrt(np.array([p_hat, 1 - p_hat]))
    return np.sqrt(np.sum((sqrt_p - sqrt_phat) ** 2, axis=0) / 2)


def d_TV(p, p_hat):
    return np.abs(p-p_hat)


def summarize_binary(d: DataFrame):
    from sklearn import metrics


def summarize_class(d: DataFrame, threshold=0.5) -> DataFrame:
    """Compute performance metrics on binary classifications

    Args:
        d (DataFrame): df with pred and y_true columns
        threshold (float, optional): unused. Defaults to 0.5.

    Raises:
        ValueError: _description_

    Returns:
        DataFrame: _description_
    """
    from sklearn.metrics import f1_score, log_loss
    labels = list(range(3))
    d = d.drop(columns='dset_no', errors='ignore')
    preds = pd.DataFrame()
    try:

        preds['pred'] = (d[['0', '1']]
                         .idxmax(1)
                         .astype(int))
        preds['pred_proba'] = d['1']
    except KeyError:
        preds['pred_proba'] = d['preds']
        preds['pred'] = preds['pred_proba'] >= threshold
        if preds['pred'].min() < 0 or preds['pred'].max() > 1:
            raise ValueError
    preds['y_true'] = d['y_true']
    g = preds.groupby(['ebl', 'ibl'])
    return g.apply(
        lambda x: pd.Series(
            {'fnr': ((x.pred == 0) & (x.y_true == 1)).mean(),
                'fpr': ((x.pred == 1) & (x.y_true == 0)).mean(),
                'accuracy': (x.pred == x.y_true).mean(),
             'f1': f1_score(x.y_true, x.pred, pos_label=1),
             'cross_entropy': log_loss(x.y_true, x.pred_proba, labels=[0, 1])}
        )
    )


def summarize_top(d: DataFrame) -> DataFrame:
    """compute performance metrics for topology classifier.

    Args:
        d (DataFrame): predictions

    Returns:
        DataFrame: accuracy,cross_entropy
    """
    from sklearn.metrics import log_loss
    labels = list(range(3))

    preds = np.exp(d.drop(columns='dset_no', errors='ignore').iloc[:, :3])
    preds['pred'] = preds.idxmax(1).astype(int)
    preds['y_true'] = d['y_true']

    g = preds.groupby(['ebl', 'ibl'])
    return g.apply(
        lambda x: pd.Series({'accuracy': (x.pred == x.y_true).mean(),
                             'cross_entropy': log_loss(x.y_true, x.iloc[:, :3], labels=labels)}
                            )
    )


def summarize(x: DataFrame,
              metric=["kl", "hellinger", "hpd"],
              interval=.8):
    """returns kl divergence,hellinger,hpd, pred-true for predicted p."""
    groups = x.groupby(['ebl', 'ibl'])
    p, preds = groups.y_true.mean(), groups.preds.mean()
    d = dict(
        dH=d_H(p, preds),
        dKL=d_kl(p, preds),
        dTV=d_TV(p, preds),
        ERROR=preds-p,
        REL_ERR=(preds-p)/p,
        HDI=confidence(p, groups, interval=interval),
        ACC=accuracy(p, preds))
    d = DataFrame(d)
    d.index.names = p.index.names
    return d


def accuracy(ptrue, ppred):
    """calculate accuracy for .5 threshold binary classifier"""
    return (ptrue.round() == ppred.round()).astype(int)


def calc_hdi(p,
             preds: DataFrame,
             levels=["ebl", "ibl"],
             interval=0.8):
    x = {}
    for c in preds:
        g = preds[c].groupby(level=levels)
        z = {}
        for k, v in g:
            h = hdi(v, hdi_prob=interval)
            z[k] = h[0] < p[k].mean() < h[1]
        x[c] = z

    return pd.DataFrame(x)


def plot_recomb_set(outdir, file_path, models, f):
    pref = "f{:.1f}".format(f)
    file_path = file_path.format(pref)
    if not path.exists(file_path):
        return
    else:
        print("processing ", file_path, models.keys())
    X = pd.read_hdf(file_path, "x")  # .dropna()
    y_true, p = map(np.squeeze, load_y(X, odds=False))
    try:
        preds = pd.DataFrame(
            {m: sigmoid(models[m].predict(X)) for m in models}, index=y_true.index
        )
    except ValueError:
        print("couldn't run models: {}".format(models))
        return
    if "dset_no" in preds:
        preds.drop("dset_no", 1, inplace=True)
    divergence, hellinger, hdi_hits = summarize(p, preds)

    for params in (
        {
            "X": preds,
            "prefix": pref + ".std",
            "agg": "std",
            "title": r"$\sigma(\hat{p})$",
            "vmin": 0,
            "vmax": 0.5,
            "outdir": outdir,
        },
        {
            "X": preds,
            "true_p": p,
            "prefix": pref + ".raw",
            "agg": "mean",
            "title": r"Mean predicted $\hat{p}$",
            "outdir": outdir,
        },
        {
            "X": preds.apply(lambda x: (x - p) / p),
            "prefix": pref + ".diff",
            "agg": "mean",
            "title": r"Relative Error $(\hat{p}-p)/p$",
            "center": 0,
            "outdir": outdir,
        },
        {
            "X": divergence,
            "prefix": pref + ".kld",
            "title": r"$D_{KL}$",
            "vmin": 0,
            "outdir": outdir,
        },
        {
            "X": hellinger,
            "prefix": pref + ".hellinger",
            "title": r"$d_{H}$",
            "vmin": 0,
            "vmax": 1,
            "outdir": outdir,
        },
        {
            "X": divergence,
            "prefix": pref + ".max-kld",
            "title": r"max $D_{KL}$",
            "agg": "max",
            "vmin": 0,
            "outdir": outdir,
        },
        {
            "X": hellinger,
            "prefix": pref + ".max-hellinger",
            "title": r"max $d_{H}$",
            "agg": "max",
            "vmin": 0,
            "vmax": 1,
            "outdir": outdir,
        },
        {
            "X": hdi_hits,
            "true_p": p,
            "outdir": outdir,
            "prefix": pref + ".hdi",
            "agg": "mean",
            "title": r"$p\in$ 80% HDI",
        },
    ):
        make_hex_plots(**params)

    (divergence
     .rename(index=np.log10, level=1)
     .to_csv(outdir / pref + ".divergence.csv.gz")
     )

    (hellinger
     .rename(index=np.log10, level=1)
     .to_csv(outdir / pref + ".hellinger.csv.gz")
     )


def intersect(group):
    ix = None
    for f, fgroup in group.groupby('Fraction'):
        if ix is None:
            ix = fgroup.index
        else:
            ix = ix.intersection(fgroup.index)
    return ix


def plot_rec(rectype: str,
             s: Path = Path('/N/project/phyloML/deep_ils/results/test_data/'),
             metric=r'$d_H$',
             length: int = 500,
             ngenes: int = None):
    z = []
    if rectype == 'Fraction':
        search_path = '*_f*/preds.csv.gz'
        rx = re.compile('^g(\d+)_l(\d+)_f([\d\.]+)_(?!b).*')
    elif rectype == 'Blocks':
        search_path = '*_f*5_b*/preds.csv.gz'
        rx = re.compile('^g(\d+)_l(\d+)_f0+.5_b([\d\.]+)_.*')
    else:
        raise ValueError(f'{rectype} unknown')
    for p in s.glob(search_path):
        dirname = p.parent.name
        match = rx.findall(dirname)
        if not match:
            continue
        if rectype == 'Blocks':
            number_of_genes, a_length, rec = map(int, match[0])
        elif rectype == 'Fraction':
            number_of_genes, a_length = map(int, match[0][:-1])
            rec = float(match[0][-1])
        if ngenes and number_of_genes != ngenes:
            continue
        if length and a_length != length:
            continue
        # if length <= 50:
        #     continue
        x = pd.read_csv(p, index_col=[0, 1])
        x[r'$d_H$'] = d_H(x.y_true, x.preds)
        x['ERROR'] = x.preds-x.y_true
        x['abs_err'] = x.ERROR.abs()
        x['accuracy'] = accuracy(x.y_true, x.preds)
        x['Length'] = a_length
        x['Genes'] = number_of_genes
        x[rectype] = rec
        z.append(x)
    # if rectype=='Blocks':
    #     ix = reduce(
    #         lambda x, y: x.intersection(y.index),
    #         [zz for zz in z if not zz.query(
    #             '1000>Genes>50 & 50<Length<1000 & Blocks>2').empty],
    #         z[0].index)
    # else:
    #     ix = reduce(
    #         lambda x, y: x.intersection(y.index),
    #         [zz for zz in z if not zz.query(
    #             '1000>Genes>50 & 50<Length<1000').empty],
    #         z[0].index)

    z = pd.concat(z).query('10>ibl>.1 & 110>=ebl>50')  # .query(q)
    if rectype == 'Blocks':
        # g = z.query('7>ibl>.05 & 200>ebl>50').groupby(
        #     ['Length', 'Genes', 'Blocks'])
        # min_samps = g.count().min()[0]
        # z=g.sample(min_samps)
        plot_blocks()
        return z
        z = z.query('7>ibl>.05 & 110>=ebl>50 & Blocks<4 & Genes>50')

    elif rectype == 'Fraction':
        if length:
            z_new = z.query(f'Length=={length}')
        else:
            try:
                g = z.groupby('Length')
                mix = g.apply(intersect)
                z_new = pd.concat(
                    [z.query(f'Length=={idx}').loc[mix[idx]]
                     for idx in mix.index]
                )

                if len(z_new) == 0:
                    print('no shared params')
                    raise ValueError
            except:
                z_new = z
        z = z_new  # z.loc[ix]

    # TODO: group by index, make sure each g/l combo has same ebl/ibl distribution

    #     (rectype == 'Blocks' and f'& {rectype} > 1' or f'& {rectype}>0')
    data = z.reset_index(drop=True)[['Length', 'Genes', rectype, metric]]
    if ngenes:
        g = sns.lineplot(data=data,
                         y=metric,
                         #  col="genes",
                         hue="Length",
                         x=rectype,
                         markers=True,
                         style='Length')
        if rectype == 'Blocks':
            g.set_xticks([1, 2, 3, 4])
        else:
            g.set_xlabel('Fraction Recombinant')
        if metric == 'abs_err':
            g.set_ylabel(r'$|\hat{p}-p|$')
    else:
        #     g = sns.relplot(data=data,
        #   ...:                         y=metric,
        #   ...:                         col=rectype,
        #   ...:                         hue="Length",
        #   ...:                         x="Genes",
        #   ...:                         markers=True,
        #   ...:                         style='Length',
        #   ...:                         kind="line")

        if rectype == 'Blocks':
            g = sns.relplot(data=data,
                            y=metric,
                            col='Length',
                            x=rectype,
                            markers=True,
                            hue="Genes",
                            # style="Genes",
                            kind="line")
            # g.set(xticks=data.Genes.unique())
        else:
            g = sns.relplot(data=data,
                            y=metric,
                            col="Length",
                            hue="Genes",
                            x=rectype,
                            markers=True,
                            style='Genes',
                            kind="line")
            g.set_xlabels('Fraction Recombinant')
        if metric == 'abs_err':
            g.set_ylabels(r'$|\hat{p}-p|$')
    plt.tight_layout()

    return z


def plot_blocks():
    data = z.query('Genes==500 & Length==500 & 8>ibl>.05 & 110>=ebl>30')
    # data = z.query('Genes==500 & Length==1000 & 8>ibl>.05 & 125>=ebl>60')
    g = sns.boxplot(x='Blocks',
                    y='ERROR',
                    data=data,
                    whis=[5, 95],
                    width=.6,
                    palette="vlag")
    sns.stripplot(x='Blocks',
                  y='ERROR',
                  data=data,
                  size=4,
                  color=".3",
                    linewidth=0)
    g.set_ylabel(r'$\hat{p}-p$')


def plot_model_misspec(s, outdir: Path = None):
    # TODO: subset common ibl/ebl ix
    from itertools import product
    if outdir is None:
        outdir = s
    m = ('LG', 'WAG')
    x = []
    for p in product(m, m):
        try:
            df = pd.read_csv(
                next(s.glob('{}_{}_*20/preds.csv.gz'.format(*p))),
                index_col=[0, 1]).query('ebl<=110')
        except StopIteration:
            return None
        df = summarize(df)
        df['Model'] = '{}/{}'.format(*p)
        x.append(df)
    x = pd.concat(x)
    conds = ['LG/LG', 'LG/WAG', 'WAG/WAG', 'WAG/LG']
    x.sort_values(by='Model', key=lambda s: s.map(
        lambda c: conds.index(c)), inplace=True)

    # x.ERROR = np.log(np.abs(x.ERROR))*np.sign(x.ERROR)
    # from IPython import embed
    # embed()

    for err, title in zip(('ERROR', 'dTV', 'dH'), (r'$\hat{p}-p$', r'$d_{TV}$', r'$d_H$')):
        g = sns.boxplot(data=x.reset_index(),
                        x='Model',
                        y=err,
                        # kind='box',
                        whis=[5, 95], width=.6, palette="vlag")

        sns.stripplot(data=x.reset_index(),
                      x='Model',
                      y=err,
                      # kind='strip',
                      size=4, color=".3", linewidth=0
                      )
        plt.ylabel(title)
        if err != 'ERROR':
            plt.ylim(0, .7)
        # else:
        #     plt.ylabel(r"$\sigma(\hat{p})\cdot\log\left(|\hat{p}-p|\right)$")
        result_file = outdir/f'{err}-model_misspec.png'
        plt.tight_layout()
        plt.savefig(result_file)
        plt.close()
    return x


def load_preds(s, condition):
    df = pd.read_csv(s, index_col=[0, 1]).query('ebl<=100')
    df = summarize(df)
    df['Condition'] = condition
    return df


def plot_heterotachy(s, outdir: Path = None):
    from itertools import product
    if outdir is None:
        outdir = s
    res = []
    for p in s.glob('*hetero/preds.csv.gz'):
        df_hetero = load_preds(
            p,
            'Site+Lineage')
        df_no_cats = load_preds(
            str(p).replace('_d1000_hetero', '_d1000_one_rate'),
            '-')
        df_no_hetero = load_preds(
            str(p).replace('_d1000_hetero', '_f0_b0_20'),
            'Site')

        ix = (df_hetero
              .index
              .intersection(df_no_hetero.index)
              .intersection(df_no_cats.index))

        x = (pd
             .concat(
                 [df_no_cats.loc[ix], df_no_hetero.loc[ix], df_hetero.loc[ix], ])
             )
        #  .sort_values(by='Condition', ascending=False))

        model = p.parent.name
        x['model'] = model

        for err, title in zip(('ERROR', 'dTV', 'dH'), (r'$\hat{p}-p$', r'$|\hat{p}-p|$', r'$d_H$')):
            #
            g = sns.boxplot(
                data=x.reset_index(),
                x='Condition',
                y=err,
                # kind='box',
                whis=[5, 95],
                width=.6,
                showfliers=False,
                palette="vlag")
            sns.stripplot(
                data=x.reset_index(),
                x='Condition',
                y=err,
                # kind='strip',
                size=3,
                color=".3",
                linewidth=0
            )
            plt.ylabel(title)
            plt.xlabel('Heterogeneity Condition')
            if err != 'ERROR':
                plt.ylim(0, .7)
            else:
                g.set_ylim((-2./3, 2./3))
            result_file = outdir/f'{model}_{err}.png'
            plt.tight_layout()
            plt.savefig(result_file)
            plt.close()
        res.append(x)
    return pd.concat(res)


def summarize_kw_test(df: pd.DataFrame):
    kw_res = kruskal(*df.values.T)
    print('\nKruskal-Wallis test', kw_res, '\n')
    # if kw_res.pvalue < .05:
    res = rank_sum_test(df)
    rs_results = pd.Series(res.values(), res.keys())
    wtest_results = get_significant(rs_results)
    if wtest_results.empty:
        print('no significant p-values', rs_results)
    print(wtest_results)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="""plot recomb, site/lineage heterogeneity""")
    parser.add_argument(
        "--procs",
        "-p",
        type=int,
        default=4)
    parser.add_argument(
        "--ngenes",
        "-g",
        type=int,
        help="num genes in each dataset",
    )
    parser.add_argument(
        "--length",
        type=int,
        help="gene length",
    )
    parser.add_argument(
        "--suffix",
        help="filename suffix",
        default="recomb")

    parser.add_argument(
        "--predfile",
        help="predfile",
        type=Path,
        default="")
    parser.add_argument(
        "--outdir",
        help="directory to store results files",
        type=Path,
        default="/N/dc2/projects/bkrosenz/deep_ils/results/ms_learned",
    )
    parser.add_argument(
        "--recdir",
        help="""path to test data csv.
                must have format: ngenes, length, nblocks,
                dirname or ngenes, length, frac, dirname""",
        type=Path,
        default=Path('/N/project/phyloML/deep_ils/results/test_data'),
    )

    args = parser.parse_args()
    print("args", args)

    outdir = args.outdir
    outdir.mkdir(exist_ok=True)
    s = args.recdir

    # plot_rec(rectype='Blocks', s=s, metric='abs_err')
    # plt.savefig(outdir/'blocks.png')
    # exit()
    # if 'preds' not in x:
    #     m = {'0': 0, '1': 1}
    #     x.columns = x.columns.map(lambda cname: m.get(cname, cname))
    #     x['preds'] = x[[0, 1]].idxmax(1)

    x = plot_heterotachy(s, outdir)
    df = x.pivot_table(columns='Condition', values='ERROR', index=x.index)
    summarize_kw_test(df)
    # else:
    #     print('not significant')
    x = plot_model_misspec(s, outdir)
    df = x.pivot_table(columns='Model', values='ERROR', index=x.index)
    summarize_kw_test(df.dropna())
    # exit()
    plt.close()
    plot_rec(rectype='Fraction',
             s=s,
             length=args.length,
             ngenes=args.ngenes,
             metric='abs_err')
    plt.savefig(outdir/f'g{args.ngenes}_frac.png')
    plt.close()

    for p in s.glob('*/preds.csv.gz'):
        x = pd.read_csv(p, index_col=[0, 1])
        z = summarize(x, interval=.5)
        fig, axes = plt.subplots(2, 2)
        for ax, col in zip(axes.ravel(), z):
            sns.heatmap(z[col].unstack(), ax=ax)
            ax.invert_yaxis()
            ax.set_title(col)
        plt.tight_layout()
        plt.savefig(outdir/f'{p.parent.name}-heatmap.png')
        plt.close()

    exit()

    d = pd.read_csv(args.recfiles)

    d = d[d.ngenes == args.ngenes]

    processor = partial(read_results,
                        parentdir=args.parentdir)
    mf = pd.concat(
        Parallel(n_jobs=args.procs)(
            delayed(processor)(row) for row in d.itertuples(index=False)
        )
    ).dropna()
    mf["length"] = pd.Categorical(mf.length, ordered=True)
    print("mf", mf.columns)
    if "nblocks" in mf:
        mf["nblocks"] = pd.Categorical(mf.nblocks, ordered=True)

    algs = ["RF", "MLP", "Median"]
    ycol = "y" if "y" in mf else "y_true"

    for a in algs:
        try:
            mf[a] = summarize(mf[ycol], mf[a])
        except Exception as e:
            print(mf, ycol, a)
            raise e
    var = "frac" if "frac" in mf else "nblocks"

    z = mf.reset_index().melt(
        id_vars=["length", var, "ebl", "ibl"],
        value_vars=algs,
        var_name="alg",
        value_name="d_H",
    )

    sns.catplot(data=z,
                y="d_H",
                col="alg",
                hue="length",
                x=var,
                kind="violin")
    plt.savefig(
        path.join(args.outdir, args.suffix + ".catplot.png"))
    plt.clf()
    sns.relplot(data=z,
                y="d_H",
                col="alg",
                hue="length",
                x=var,
                kind="line")
    plt.savefig(
        path.join(args.outdir, args.suffix + ".relplot.png"))
    plt.clf()

    for target, name in zip(
        ["Median", "RF", "MLP"], [
            "Median (baseline) Regressor", "Random Forest", "MLP"]
    ):
        if "frac" in mf:
            sns.lineplot(data=mf, x="frac", y=target, hue="length")
            plt.xlabel("fraction recombinant")
            # suffix = args.suffix + ".frac.png"
        elif "nblocks" in mf:
            sns.lineplot(data=mf, x="nblocks", y=target, hue="length")
            plt.xlabel("number of blocks")
            # suffix = args.suffix + ".blocks.png"
        plt.ylabel(r"$d_H$")
        plt.ylim((0, 0.2))
        plt.title(name)
        plt.tight_layout()
        outfile = path.join(args.outdir, target + "." + args.suffix + ".png")
        print("saved as", outfile)
        plt.savefig(outfile)
        plt.clf()

    print("finished writing to", args.outdir)
