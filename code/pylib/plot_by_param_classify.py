from sklearn import metrics
import matplotlib,json,h5py,re,math
#matplotlib.use('TkAgg') # uncomment for gui
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
#from sys import argv
from os import path
plt.ion()

def add_common_labels(fig, xlab, ylab):
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axes
    plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    plt.grid(False)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    

respath = len(argv)>1 and argv[1] or None

with open(path.join(respath,'config.json')) as confile:
    config = json.load(confile)

if 'ils' not in config:
     config['ils'] = 1

print('\nsettings',respath,config,'\n-------\n')

with h5py.File(config['data'], mode='r', libver='latest') as h5f:
    algnames = list(h5f.keys())

sp_tree="_(4,(3,(1,2)));"
rx=re.compile(re.escape(sp_tree))
count_cols=[ 'g_(4,(1,(2,3)));',
         'g_(4,(3,(1,2)));',
         'g_(4,(2,(1,3)));',]

for alg in algnames: #['lg_wag'] :#
    try:
        index_path = path.join(respath,alg,'ybmask.npz')
        y_bin_inds = np.load(index_path)
        y_bin_mask = np.hstack( (y_bin_inds['ils'], y_bin_inds['no_ils']) )
        res = pd.read_csv(path.join(respath,alg,'results.'+alg+'.classify.preds.csv.gz'),index_col=0)
    except FileNotFoundError:
        continue
    learners = res.columns.drop('y_true')
    with h5py.File(config['data'], mode='r',libver='latest') as h5f:
        ds = h5f[alg]
        y_attrs = ds['y'].attrs['column_names'].astype(str)
        # WARNING: assumes y_bin_mask is in format (ils_inds, no_ils_inds)
        y_full = pd.DataFrame(
            np.nan_to_num( ds['y'] ), columns = y_attrs
            )
        if config['balanced']:
            y_full = y_full.iloc[y_bin_mask.ravel()].reset_index()
        y_full = y_full.drop(columns=[c for c in y_attrs if sp_tree not in c and c not in count_cols ])\
                       .join( res )
    if y_full.empty:
        print('no data found for ',alg,argv[1])
        continue
    # print('corr',
    #       y_full.groupby(['ebl_mean', 'ibl_mean'])[learners].corrwith(y_full.y_true)) #,method='spearman')

    print(y_attrs,'\ncols old',y_full.columns)
    y_full.columns=[s if s in count_cols else rx.sub('',s) for s in y_full.columns]
    print('cols new',y_full.columns)

    y_full['frac'] = y_full[count_cols[1]]/y_full[count_cols].sum(1)
    y_full['bin']=y_full.frac < config['ils']
    misclass=pd.DataFrame({c : (y_full[c].round()!=y_full.bin)  for c in learners}) 
    #    deviations = y_full[learners].apply(lambda x:np.abs(x-y_full.y_true))

    misclass[['ebl', 'ibl']] = y_full[['ebl_median', 'ibl_median']]

    ncols=3
    nrows=math.ceil(len(learners)/ncols)
    vmin,vmax = 0, .5
    cmap=plt.cm.Blues
    norm = matplotlib.colors.Normalize(vmin=vmin,vmax=vmax)
    fig, axs = plt.subplots(ncols=ncols,
                            nrows=nrows,
                            sharex=True,
                            sharey=True,
                            figsize=(6,6))
    cbar_ax = fig.add_axes([.91, .3, .03, .4])
    ebl = misclass['ebl']
    ibl = misclass['ibl']
    n = len(learners)
    for i,ax in enumerate(axs.ravel()):
        if i<n:
            learner = learners[i]
            im = ax.hexbin(x=ebl, y=ibl, C=misclass[learner],
                           gridsize=40, vmin=vmin, vmax=vmax,
                           yscale='log',
                           cmap=cmap)
            ax.set_title(learner)
            ax.set_facecolor("lightslategray")
            ax.xaxis.label.set_visible(False)
            ax.yaxis.label.set_visible(False)
        else:
            fig.delaxes(ax)
    fig.colorbar( im, cax=cbar_ax)
    add_common_labels(fig,
                      xlab=r'External Branch Length',
                      ylab=r'Internal Branch Length')
    fig.suptitle('Mean Misclassification Error',size=15)
    plt.tight_layout(rect=[0,.03,.9,.95])
    plt.savefig(path.join(respath,alg,'hexplot.classify.%s.png'%alg))
    plt.clf()
