from logging import error
from operator import mod
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


def plot_error(err: pd.Series,
               scaler=None,
               norm=None,
               cmap='coolwarm',
               s=3,
               k=3,
               limits=(.01, 20, 20, 300,),
               title=None):
    # TODO: rewrite for multiindex ebl/ibl
    binary = len(err.unique()) == 2
    if not norm and err.min() < 0 and not binary:
        norm = centered_norm
    if isinstance(err, pd.Series):
        err = err.to_frame()
    if scaler:
        err = err.query(
            '{} < ibl < {} & {} < ebl < {}'.format(*limits)
        )
    x, y, z = err.index.get_level_values(
        'ebl'), err.index.get_level_values('ibl'), err.values
    if scaler:
        xmin, xmax, ymin, ymax = map(scaler, limits)
        x, y = map(scaler, (x, y))
    else:
        xmin, xmax, ymin, ymax = limits
    # interp = interpolate.SmoothBivariateSpline(x, y, z, s=s, kx=k, ky=k)
    X = (np.linspace(xmin, xmax, 200)),
    Y = (np.linspace(ymin, ymax, 200))
    X, Y = np.meshgrid(X, Y)
    # interp = interpolate.LinearNDInterpolator(list(zip(x, y)), z,)
    interp = interpolate.NearestNDInterpolator(list(zip(x, y)), z,)
    Z = interp(X, Y).squeeze()
    from scipy.ndimage import gaussian_filter
    Z = gaussian_filter(Z, sigma=1)
    fig, ax = plt.subplots()
    im = ax.pcolormesh(X, Y, Z,
                       cmap=cmap,
                       norm=norm,
                       )
    if not binary:
        fig.colorbar(im, ax=ax)
    plt.ylabel('IBL')
    plt.xlabel('EBL')
    if scaler is None:
        plt.yscale('log')
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

    args = parser.parse_args()
    print("args", args)
    predictions = pd.read_csv(
        args.resultdir/'preds.csv.gz',
        index_col=[0, 1])

    error_metrics = pu.summarize(predictions)

    seqdir = Path('/N/project/phyloML/deep_ils/results/seq_div')
    try:
        lengths = pd.read_csv(seqdir/'lengths.csv.gz', index_col=0)
    except:
        lengths = u.coal2div2(seqdir, model='_LG')
        lengths.to_csv(seqdir/'lengths.csv.gz')
    lengths = lengths.groupby(lengths.index).mean()

    scaler = interpolate.UnivariateSpline(lengths.index,
                                          lengths.seq_div,
                                          s=1, k=3)
    titles = (r'$d_H$', r'$d_{KL}$', r'$|\hat{p}-p|$', r'$\hat{p}-p$',
              r'$\frac{\hat{p}-p}{p}$', 'HDI')
    cmaps = ('Reds', 'Reds', 'Reds', 'PuOr', 'PuOr', 'Greens_r')
    for err, title, cmap in zip(error_metrics, titles, cmaps):
        d = error_metrics[err]
        plot_error(d, scaler, cmap=cmap, title=title)
        result_file = args.resultdir/f'heatmap-{err}-seqdiv.png'
        plt.savefig(result_file)
        plt.close()
        plot_error(d, scaler=None, cmap=cmap, title=title)
        result_file = args.resultdir/f'heatmap-{err}-coal.png'
        plt.savefig(result_file)
        plt.close()

        print(result_file, seqdir)
