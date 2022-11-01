import argparse
from logging import error
from pathlib import Path
from sys import argv
from typing import Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
try:
    from pylib import utils as u
except:
    from ..pylib import utils as u

from scipy import interpolate
# from plot_heatmap import
from plotting import plot_test_results as pu

vmax = .6
# norm = mpl.colors.Normalize(vmin=-vmax, vmax=vmax)
centered_norm = mpl.colors.CenteredNorm(0, vmax)
THETA = .01
EBL_MAX = 120


def plot_error(err: pd.Series,
               scaler=None,
               norm=None,
               cmap='coolwarm',
               s=2,
               k=2,
               sigma=1,
               method='spline',
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
            x, y, z,
            s=1,
            kx=k,
            ky=k)
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
    if sigma is not None:
        from scipy.ndimage import gaussian_filter
        Z = gaussian_filter(Z, sigma)

    im = ax.pcolormesh(X, Y, Z,
                       cmap=cmap,
                       norm=norm,
                       vmin=0,
                       vmax=1
                       )
    if not binary:
        fig.colorbar(im, ax=ax)

    plt.ylabel('IBL')
    plt.xlabel('EBL')
    plt.tight_layout()
    # else:
    #     ax.invert_yaxis()
    plt.title(title)


if __name__ == '__main__':
    if len(argv) > 2:
        gtop, itop = (
            pd.read_csv(fn, index_col=['ebl', 'ibl']).query(f'ebl<={EBL_MAX}') for fn in argv[1:3])
    else:
        exit("must supply gtop/itop files")
    if len(argv) == 4:
        method = argv[3]
    else:
        method = 'linear'
    mindex = gtop.index.intersection(itop.index)
    prefix = Path(argv[1]).parent/f"sim_heatmap"
    if 'seq_length' in itop.columns:
        seq_length = itop.groupby("seq_length").count().idxmax()[0]
        m = itop.query(f"seq_length=={seq_length}")
        prefix += ".l{seq_length}"
    else:
        m = itop

    for name, c, title in zip(('true', 'inferred'),
                              (gtop, m),
                              ('Concordant Gene Trees', 'Concordant Gene Trees (Inferred)')):
        c.sort_index(inplace=True)
        fig, ax = plt.subplots(1, 1)
        c = (c
             .query('topology==1')['c']
             .divide(c.c.groupby(c.index).sum(1), 0))
        x, y = c.index.levels
        X, Y = np.meshgrid(x, y)
        if method == 'None':
            pc = sns.heatmap(
                c.unstack().interpolate("linear", limit_direction="both").T,
                cmap="viridis",
                vmin=0.33,
                vmax=1,
                ax=ax
            )
        else:
            plot_error(c,
                       method=method,
                       cmap='viridis',
                       title=title)
            # ax.set_yscale("log")
            ax.set_ylabel(r"IBL")
            ax.set_xlabel(r"EBL")
            # fig.colorbar(pc)
        plt.savefig(
            f"{prefix}_{name}.png"
        )
        plt.clf()
