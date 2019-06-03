import math
from sklearn import metrics
import matplotlib,json,h5py
#matplotlib.use('TkAgg') # uncomment for gui
matplotlib.use('Qt5Agg')
import re
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

with open(path.join(respath,'config.json')) as confile:
    config = json.load(confile)


with h5py.File(config['data'], mode='r', libver='latest') as h5f:
    algnames = list(h5f.keys())

sp_tree="_(4,(3,(1,2)));"
rx=re.compile(re.escape(sp_tree))
    
for alg in algnames:
    try:
        y_frac_mask = np.load(path.join(respath,alg,'yfmask.npy'))
        res = pd.read_csv(path.join(respath,alg,'results.'+alg+'.regress.preds.csv.gz'),index_col=0)
    except FileNotFoundError:
        continue
    learners = res.columns.drop('y_true')
    with h5py.File(config['data'], mode='r',libver='latest') as h5f:
        ds = h5f[alg]
        y_attrs = ds['y'].attrs['column_names'].astype(str)
        if sum(y_frac_mask)==len(y_frac_mask): # aka config['balanced']==True
            y_full = pd.DataFrame(
                np.nan_to_num( ds['y'] ), 
                columns = y_attrs
                )\
                .drop(columns=[c for c in y_attrs if not sp_tree in c])\
                .join( res )
        else:
            y_full = pd.DataFrame(
                np.nan_to_num( ds['y'][y_frac_mask,:] ), 
                columns = y_attrs
                )\
                .drop(columns=[c for c in y_attrs if not sp_tree in c])\
                .join( res )
        y_full.columns=[rx.sub('',s) for s in y_full.columns]
    if y_full.empty:
        print('no data found for ',alg,argv[1])
        continue
        
    print(alg, 'correlations', y_full.groupby(['ebl_mean', 'ibl_mean'])[learners].corrwith(y_full.y_true))
    #y_full.plot.hexbin('ebl_mean', 'ibl_mean' , gridsize=10)
    deviations = y_full[learners].apply(lambda x:np.abs(x-y_full.y_true)) # TODO: try w/o abs (can't do logscale, however)
    deviations[['ebl_mean', 'ibl_mean']] = y_full[['ebl_mean', 'ibl_mean']]
    ebl = deviations['ebl_mean']
    ibl = deviations['ibl_mean']
    deviations.to_csv(path.join(respath,alg,'deviations.regress.%s.csv.gz'%alg))

    n = len(learners)
    ncols=3
    nrows=math.ceil(len(learners)/ncols)
    vmin,vmax = 1e-6,.7 # true and pred should always be in [1/3,1]
    cmap=plt.cm.Blues
    norm = matplotlib.colors.Normalize(vmin=vmin,vmax=vmax)
    fig, axs = plt.subplots(ncols=ncols,
                            nrows=nrows,
                            sharex=True,
                            sharey=True,
                            figsize=(6,6))
    cbar_ax = fig.add_axes([.91, .3, .03, .4])

    for i,ax in enumerate(axs.ravel()):
        if i<n:
            learner = learners[i]
            try:
                im = deviations.plot.hexbin(x='ebl_mean', y='ibl_mean', C=learner, ax=ax,
                               gridsize=40, vmin=vmin, vmax=vmax, 
                               cmap=cmap)
                # unresolved bug in matplotlib w/ noninteractive plots: https://github.com/matplotlib/matplotlib/issues/5541/
                # im = ax.hexbin(x=ebl, y=ibl, C=deviations[learner],
                #                gridsize=40, vmin=vmin, vmax=vmax, 
                #                cmap=cmap)
                ax.set_yscale("log")
            except ValueError as e:
                print(learner,alg,argv[1],deviations[learner])
                raise e
            ax.set_title(learner)
            ax.set_facecolor("lightslategray")
            ax.xaxis.label.set_visible(False)
            ax.yaxis.label.set_visible(False)
        else:
            fig.delaxes(ax)
    fig.colorbar( im, cax=cbar_ax)
    add_common_labels(fig,xlab=r'External Branch Length', ylab=r'Internal Branch Length')
    fig.suptitle('Mean Absolute Deviation',size=16)
    plt.tight_layout(rect=[0,.03,.9,.95])
    plt.savefig(path.join(respath,alg,'hexplot.regress.%s.png'%alg))
    plt.clf()
