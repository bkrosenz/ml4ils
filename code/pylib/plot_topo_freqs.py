import pandas as pd
from os import path
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import h5py
from sys import argv

fnames = len(argv)>1 and (argv[1],) \
    or (
        '/N/dc2/projects/bkrosenz/deep_ils/results/ms1000aa-theta0.01-n{}.hdf5'.format(s) for s in (50,100,250,500,1000,2000)
    )
    

for s in fnames:
    print('processing',s)
    with h5py.File(s, mode='r',libver='latest') as h5f:
        for alg in h5f:
            ds=h5f[alg]
            res={}
            for c in ['x','y']:
                res[c]=pd.DataFrame(ds[c],columns=ds[c].attrs['column_names'])
            x2=res['x'].join(res['y'])
            x2.columns=x2.columns.map(lambda t: t.decode() if type(t)!=str else t)
            x2['ebl']=x2['ebl_median_(4,(3,(1,2)));']
            x2['ibl']=np.round(x2['ibl_median_(4,(3,(1,2)));'],3)
            for name,col in (('true','g_(4,(3,(1,2)));'),
                             ('inferred','(4,(3,(1,2)));')
            ):
                gts = x2.groupby(['ebl','ibl'])[col].mean().reset_index()
                ax=sns.heatmap(data=gts.pivot('ebl','ibl',col))
                plt.savefig(
                    '/N/dc2/projects/bkrosenz/deep_ils/results/'+'.'.join(
                        (path.basename(s),alg,name,'png')
                    )
                )
                plt.clf()
            
                                                
