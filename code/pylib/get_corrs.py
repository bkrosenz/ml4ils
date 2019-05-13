import pandas as pd
import numpy as np
from os import path
from ete3 import Tree
import os
import re
from joblib import Parallel,delayed
from  itertools import islice

outdir='/N/dc2/projects/bkrosenz/ml4ils/results/stats'
simdir='/N/dc2/projects/bkrosenz/deep_ils/sims/seqgen-aa-1000bp-theta0.01'
gtdir='trees'
itdir='inferred_trees'
ifiles = [fn for fn in os.listdir(path.join(simdir,itdir)) if 'LG_PROTCATLG.raxml.rooted' in fn]

rx = re.compile('(t_.*)_LG_PROTCATLG.*')
gfiles = [rx.sub(r'\1.rooted.trees',s) for s in ifiles]
d = lambda t1:t1.get_distance('2','1')


def get_corr(ifn,gfn):
    with open(path.join(simdir,itdir,ifn)) as ifile, open(path.join(simdir,gtdir,gfn)) as gfile, path.join:
        ti=[Tree(s) for s in ifile]
        tg=list( islice( (Tree(s) for s in gfile if s.startswith('(')), len(ti) ) )
        print(len(ti),len(tg))
        di,dg = (*map(d,ti),), (*map(d,tg),)
        with open(path.join(outdir,'di.ji.csv'),'a') as f:
            f.write('%.5f\t%.5f\n'%(np.mean(di), np.mean(dg)))
    return ( gfn, np.corrcoef(di,dg)[0,1] )
            
r = Parallel(n_jobs=4)(delayed(get_corr)(ifn,gfn) for ifn,gfn in zip(ifiles,gfiles))
rx2=re.compile('t_(\d+)_([\.\d]+)_([\.\d]+)\.root.*')

df = pd.DataFrame(r)
df.columns = ['idx','corr']
df = df.set_index('idx').rename( index = lambda q:tuple( map(float,rx2.match(q).groups()) ) )\
                        .reindex(pd.MultiIndex.from_tuples(df.index,names=['ab','bc','cd']))\
                        .reset_index()
df['ibl']=df['bc']-df['ab']
df['ebl']=df['ab']
df.drop(columns=['ab','bc','cd'])
df.to_csv(path.join(outdir,'corr_d12.csv'),header=True)


sns.heatmap(df.pivot(index='ibl',columns='ebl',values='corr'))
plt.title(r'$Corr(d_{ab},\hat{d}_{ab})$')
