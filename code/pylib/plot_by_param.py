import math
from sklearn import metrics
import matplotlib,json,h5py
#matplotlib.use('TkAgg') # uncomment for gui
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sys import argv
from os import path
plt.ion()

def add_common_labels(fig,xlab,ylab):
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axes
    plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    plt.grid(False)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    

respath = len(argv)>1 and argv[1] or None
respath = '/N/dc2/projects/bkrosenz/deep_ils/results/ms_learned/theta0.01'

with open(path.join(respath,'config.json')) as confile:
    config = json.load(confile)


with h5py.File(config['data'], mode='r', libver='latest') as h5f:
    algnames = list(h5f.keys())
    
for alg in algnames:
    y_frac_mask = np.load(path.join(respath,alg,'yfmask.npy'))
    y_bin_mask = np.load(path.join(respath,alg,'ybmask.npy'))

    res = pd.read_csv(path.join(respath,alg,'results.'+alg+'.regress.preds.csv.gz'),index_col=0)
    learners = res.columns.drop('y_true')
    with h5py.File(config['data'], mode='r',libver='latest') as h5f:
        ds = h5f[alg]
        y_attrs = ds['y'].attrs['column_names'].astype(str)
        print('dataset',ds['y'],ds['y'].shape, y_frac_mask.shape)
        y_full = pd.DataFrame(
            np.nan_to_num( ds['y'][y_frac_mask,:] ), columns = y_attrs
        ).join( res )
    print(y_full.groupby(['ebl_mean', 'ibl_mean'])[learners].corrwith(y_full.y_true))#,method='spearman')
    y_full.plot.hexbin('ebl_mean', 'ibl_mean' , gridsize=10)
    deviations = y_full[learners].apply(lambda x:np.abs(x-y_full.y_true))
    deviations[['ebl_mean', 'ibl_mean']] = y_full[['ebl_mean', 'ibl_mean']]

    ncols=3
    nrows=math.ceil(len(learners)/ncols)
    vmin,vmax = 0,.5
    cmap=plt.cm.Blues
    norm = matplotlib.colors.Normalize(vmin=vmin,vmax=vmax)
    fig, axs = plt.subplots(ncols=ncols,
                            nrows=nrows,
                            sharex=True,
                            sharey=True,
                            figsize=(6,6))
    cbar_ax = fig.add_axes([.91, .3, .03, .4])
    ebl,ibl = deviations['ebl_mean'],deviations['ibl_mean']
    n = len(learners)
    for i,ax in axs.ravel():
        if i<n:
            learner = learners[i]
            im = ax.hexbin(ebl,ibl, deviations[learner],
                           gridsize=10, vmin=0, vmax=vmax,
            cmap=cmap)
            ax.set_title(learner)
            ax.set_facecolor("lightslategray")
            ax.xaxis.label.set_visible(False)
            ax.yaxis.label.set_visible(False)
        else:
            fig.delaxes(ax)
    fig.colorbar( im, cax=cbar_ax)
    add_common_labels(fig,r'External Branch Length','Internal Branch Length')
    fig.suptitle('Mean Absolute Deviation',size=16)
    plt.tight_layout(rect=[0,.03,.9,.95])
    plt.savefig(path.join(respath,alg,'hexplot.%s.png'%alg))
    plt.clf()
