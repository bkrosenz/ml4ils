from logging import error
from pathlib import Path
from typing import Union
import pandas as pd
import argparse
import numpy as np
from scipy import interpolate
import matplotlib as mpl
import matplotlib.pyplot as plt
from pylib import utils as u
from plotting import plot_test_results as pu

vmax = .6
# norm = mpl.colors.Normalize(vmin=-vmax, vmax=vmax)
centered_norm = mpl.colors.CenteredNorm(0, vmax)
THETA = .01
EBL_MAX = 110


def plot_error(err: pd.Series,
               scaler=None,
               norm=None,
               cmap='coolwarm',
               s=3,
               k=3,
               sigma=.5,
               method='spline',
               limits=(20, 300, .01, 20,),
               logscale=False,
               title=None):
    # TODO: rewrite for multiindex ebl/ibl
    binary = len(np.unique(np.array(err))) == 2
    # if not norm and (err.min() < 0).squeeze() and not binary:
    if not norm and (err.min() < 0) and not binary:
        norm = centered_norm
    if isinstance(err, pd.Series):
        err = err.to_frame()
    # if scaler:
    #     err = err.query(
    #         '{} < ebl < {} & {} < ibl < {}'.format(*limits)
    #     )
    x, y, z = err.index.get_level_values(
        'ebl'), err.index.get_level_values('ibl'), err.values
    if limits is None:
        limits = x.min(), x.max(), y.min(), y.max()
    if scaler:
        xmin, xmax, ymin, ymax = map(scaler, limits)
        x, y = map(scaler, (x, y))
    else:
        xmin, xmax, ymin, ymax = [s*THETA for s in limits]
        x *= THETA
        y *= THETA

    X = (np.linspace(xmin, xmax, 100)),
    Y = (np.linspace(ymin, ymax, 100))
    X, Y = np.meshgrid(X, Y)

    if method == 'spline':
        interp = interpolate.SmoothBivariateSpline(
            x, y, z, s=s, kx=k, ky=k)
        Z = interp(X, Y, grid=False).squeeze()
    elif method == 'linear':
        interp = interpolate.LinearNDInterpolator(
            list(zip(x, y)), z, )
        Z = interp(X, Y).squeeze()
    elif method == 'nearest':
        interp = interpolate.NearestNDInterpolator(list(zip(x, y)), z,)
        Z = interp(X, Y).squeeze()
    elif method == 'clough':
        interp = interpolate.CloughTocher2DInterpolator(list(zip(x, y)), z,)
        Z = interp(X, Y).squeeze()

    fig, ax = plt.subplots()
    if scaler is None and logscale:
        # plt.yscale('log')
        ax.set_xscale('symlog')
    if method is None:
        import seaborn as sns
        if isinstance(err, pd.DataFrame):
            err = err.query(
                '{}<=ebl<={} & {}<=ibl<={}'.format(*limits)
            )[err.columns[0]]
        err = err.unstack().T
        err.columns /= 100
        err.index /= 100
        X, Y, Z = err.index, err.columns, err.values
        g = sns.heatmap(err, ax=ax)

        g.set_facecolor('grey')
        g.invert_yaxis()
    else:
        if sigma is not None:
            from scipy.ndimage import gaussian_filter
            Z = gaussian_filter(Z, sigma)

        im = ax.pcolormesh(X, Y, Z,
                           cmap=cmap,
                           norm=norm,
                           )
        if not binary:
            fig.colorbar(im, ax=ax)

    plt.ylabel('IBL')
    plt.xlabel('EBL')
    plt.tight_layout()
    # else:
    #     ax.invert_yaxis()
    plt.title(title)


def plot_counts(filepath):
    c = pd.read_hdf(filepath)['counts']
    s = c.sum(1)
    err = (c[('', 1.0)]/s).to_frame().query('20<ebl<300')[0]
    err.index = err.index.map(lambda s: tuple(scaler(s)))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="""plot heatmaps""")

    parser.add_argument("--resultdir",
                        help="dir containing pred file",
                        type=Path,
                        default="")
    parser.add_argument("--classify",
                        action='store_true')
    parser.add_argument("--topology",
                        action='store_true')

    args = parser.parse_args()
    print("args", args)
    try:
        predictions = pd.read_csv(
            args.resultdir/'preds.csv.gz',
            index_col=[0, 1])
    except FileNotFoundError:
        print(f'file not found: {args.resultdir/"preds.csv.gz"}')
        exit()

    seqdir = Path('/N/project/phyloML/deep_ils/results/seq_div')
    try:
        lengths = pd.read_csv(seqdir/'lengths.csv.gz', index_col=0)
    except:
        lengths = u.coal2div2(seqdir, model='_LG')
        lengths.to_csv(seqdir/'lengths.csv.gz')
    lengths = lengths.groupby(lengths.index).mean()

    scaler = interpolate.UnivariateSpline(
        lengths.index,
        lengths.seq_div,
        s=1, k=3)

    if args.classify:
        error_metrics = pu.summarize_class(predictions)
        titles = ('FNR', 'FPR', 'Accuracy', 'F1', 'Cross-Entropy Loss')
        cmaps = ('Reds', 'Reds', 'Reds_r', 'Reds_r', 'Reds',)
        print(error_metrics.describe())
    elif args.topology:
        error_metrics = pu.summarize_top(predictions)
        titles = ('Accuracy', 'Cross-Entropy Loss')
        cmaps = ('Reds_r', 'Reds',)
    else:
        error_metrics = pu.summarize(predictions)

        titles = (r'$d_H$', r'$d_{KL}$', r'$|\hat{p}-p|$', r'$\hat{p}-p$',
                  r'$\frac{\hat{p}-p}{p}$', 'HDI', 'Accuracy')
        cmaps = ('Reds', 'Reds', 'Reds', 'PuOr', 'PuOr', 'Greens_r', 'Reds_r')

    error_metrics.to_csv(args.resultdir/'error_metrics.csv.gz')

    for err, title, cmap in zip(error_metrics, titles, cmaps):
        d = error_metrics[err]
        for method in (None, 'linear', 'nearest', 'clough'):
            for logscale in ('', '-log'):
                plot_error(d, scaler,
                           cmap=cmap,
                           title=title,
                           limits=(25, EBL_MAX, .05, 16,),
                           method=method,
                           sigma=.1,
                           logscale=logscale == '-log'
                           )
                result_file = args.resultdir / \
                    f'heatmap-{err}{logscale}-seqdiv.png'
                plt.savefig(result_file)
                plt.close()
                plot_error(d, scaler=None,
                           cmap=cmap,
                           title=title,
                           limits=(20, EBL_MAX, .001, 20,),
                           method=method,  # 'clough',
                           sigma=.1,
                           logscale=logscale == '-log'
                           )
                result_file = args.resultdir / \
                    f'heatmap-{err}-{method}{logscale}-coal.png'
                plt.tight_layout()
                plt.savefig(result_file)
                plt.close()

        print(result_file, seqdir)
