from glob import glob

# plotting
from sqlalchemy.orm import load_only,sessionmaker
from sqlalchemy import Table, Column, Integer, String, MetaData, ForeignKey, Float, Sequence, create_engine
import sqlalchemy as sa
from sqlalchemy.sql.expression import func
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.schema import PrimaryKeyConstraint,UniqueConstraint

import matplotlib
matplotlib.use('Agg')
from itertools import chain,islice
import matplotlib.pyplot as plt
from collections import defaultdict,Counter
from io import StringIO
from functools import partial,reduce
from operator import add
import argparse,re,gc,pickle
from ete3 import Tree
from scipy.sparse import csr_matrix, triu
from os import path,walk,remove
from scipy import stats
from numba import njit,vectorize
from numba.types import *
import pandas as pd
import numpy as np
import utils
import h5py
from joblib import dump, load, Parallel, delayed
import dask
from dask.diagnostics import ProgressBar
import dask.bag as db
from dask.distributed import Client, LocalCluster, fire_and_forget, progress
# use with distributed to take care of this error https://github.com/dask/distributed/issues/1467
import pandas as pd
pd.options.mode.chained_assignment = None
import dask.array as da
import dask.dataframe as dd
from sqlalchemy.orm import load_only

dask.config.config

def get_topo_freqs(session, table, col):
    '''get topology freqs by species tree.  (Can't subsample)'''
    q=session.query(
        func.count(col).label('count'), col,table.c.sid
    ).group_by(col, table.c.sid) # redundant
    gtops = pd.read_sql(
        q.statement, q.session.bind
    ).pivot(index='sid',columns=col)
    gtops.columns = gtops.columns.droplevel()
    return gtops

def process_df(df, xcols, ycols, xtop_col, ytop_col, topnames, flatten=True, split_size=None):
    res = []
    if split_size is None or split_size <= 1:
        split_size = len(df)
    for i in range(len(df)//split_size):
        res.append(
            process_df_helper( df.iloc[i*split_size:((i+1)*split_size)],
                               xcols,
                               ycols,
                               xtop_col,
                               ytop_col,
                               topnames,
                               flatten)
        )
    return res

def default_counter(keys,counts):
    return [counts[k] if k in counts else 0 for k in keys]
        
def process_df_helper(df, xcols, ycols, xtop_col=None, ytop_col=None, topnames=None, flatten=True):
#    print (df.columns)
    x = np.empty((len(utils.summaries), len(xcols)))
    utils.summarize_into(df[xcols], x)
    y = np.empty((len(utils.summaries), len(ycols)))
    utils.summarize_into(df[ycols], y)
    if flatten:
        x, y=x.flatten(), y.flatten()
        if xtop_col is not None and ytop_col is not None and topnames is not None:
            xtops=default_counter(topnames,Counter(df[xtop_col])) #dict.fromkeys(topnames).update(Counter(df[xtop_col]) #df[xtop_col].value_counts().values.flatten()
            ytops=default_counter(topnames,Counter(df[ytop_col]))
            return ( np.concatenate((x,xtops)), np.concatenate((y,ytops)) )
    return (x,y)

def make_hdf5_file(outpath,algnames,xcols,ycols,overwrite=False):
    if path.isfile(outpath):
        if overwrite:
            remove( outpath )
        else:
            raise IOError('file exists:',outpath)
        
    n_algs = len(algnames)
    return utils.HDF5Store(
        outpath,
        datasets = ['/%s/x'%a for a in algnames] + ['/%s/y'%a for a in algnames],
        shapes = [(len(xcols),)]*n_algs + [(len(ycols),)]*n_algs,
        dtypes = ['f8']*n_algs*2,
        attributes = ('column_names',
                      [xcols]*n_algs + [ycols]*n_algs)
    )

def main(args):
    tree_config=utils.TreeConfig(leafnames=range(1,5),
                                 outgroup=4,
                                 include_outgroup=True,
                                 subtree_sizes=[4])
              
    #### start processing
    algnames =  [a1+'_'+a2 for a1, a2 in utils.combr(('wag','lg'),2)]
    
    # col names for sql db
    cov_cols = next(tree_config.cov_iter(range(1,tree_config.subtree_sizes[0]+1)))
    numeric_cols = cov_cols + ['vmr','length']
    true_numeric_cols = ['g_'+c for c in numeric_cols] 
    sp_tree = '(4,(3,(1,2)));' # todo : dont hardcode
    
    tops = list( tree_config.nw_iter() ) # note: MUST SORT, ow will not match order of pd.Series.value_counts

    summary_strs = lambda pref: [pref+'_'+summary for summary in utils.summaries]

    summaries = reduce(add,map(summary_strs,cov_cols)) + summary_strs('vmr') + summary_strs('length')
    
    #colnames for hdf output
    xcols = summaries + tops
    ycols = ['g_'+s for s in summaries] + \
            reduce(add, map(summary_strs, ('ebl','ibl','s_length'))) + \
            ['g_'+s for s in tops] + \
             ['sp_tree_ind', 'concordant']
    print( 'xc',xcols,'yc',ycols)

    outpath=path.join(args.outdir,args.outfile)    
    hdf5_store = make_hdf5_file(outpath,algnames,xcols,ycols,args.overwrite)
    
    print('hdf5 shapes:',hdf5_store.shapes,tree_config.subtree_sizes, hdf5_store.dtypes)

    engine = create_engine('postgresql://bkrosenz@localhost/sim4') #sqlite:///%s'%args.out)
    metadata = MetaData(bind=engine)
    metadata.reflect()
      
    Session = sessionmaker(bind=engine)
    
    session=Session()

    for alg in algnames:
        
        try:
            table = Table(alg, metadata, autoload=True, autoload_with=engine)        
            gtops = get_topo_freqs(session,table,'gtop')
            itops = get_topo_freqs(session,table,'itop')
            cmin,cmax,csum = itops.min(1),itops.max(1),itops.sum(1)

            #TODO: this only ignores @ level of strees . should we ignore @ lev of subsamples?
            keep_idx = itops.index[(cmax-cmin)/csum < args.tol] # TODO: ignore polytomies
                        
            gtops=gtops.loc[keep_idx]
            itops=itops.loc[keep_idx]

            stree_cols =['s_1:2','s_1:3','s_length']
            # q = session.query(*stree_cols)
            # s_stats = pd.read_sql(
            #     q.statement, q.session.bind, index_col='sid'
            # ).loc[keep_idx].distinct('sid').values
            # print(s_stats.shape,s_stats)
            # get other stats
            summaries = Parallel(n_jobs=args.procs)(
                delayed( process_df )( pd.read_sql(q.statement,q.session.bind),
                                       xcols=numeric_cols,
                                       ycols=true_numeric_cols+stree_cols,
                                       xtop_col='itop',
                                       ytop_col='gtop',
                                       topnames=tops,
                                       split_size=args.splitsize)
                for q in (
                        session.query(table).filter(table.c.sid==sid) for sid in keep_idx
                )
            )

            x,y = zip(*chain.from_iterable(summaries)) # assume summaries is list of lists
            x=np.array(x)

#            print([yy.shape for yy in y],x.shape,x[0].shape)
#            exit()
            y=np.vstack(y)
            print('x',x.shape,'y',y.shape)

            #            x = np.hstack( [x,itops]  )

            # todo: modify for splits

            st_ind = xcols.index(sp_tree)
            
            #            concordant = (itops.idxmax(axis=1)==sp_tree).values
            xcol_dict = dict((k,v) for v,k in enumerate(xcols))
            top_idx = [xcol_dict[k] for k in tops]
            concordant = y[:,top_idx].argmax(1)==st_ind
#            print ('\nconc',y[:,top_idx],concordant.shape)
            n_strees=len(concordant)
            n_samples,n_features = x.shape
            sp_tree_ind = np.empty( (n_samples, 1) )
            sp_tree_ind.fill(tops.index(sp_tree))

            y = np.hstack( [ y, sp_tree_ind, np.reshape(concordant, (-1,1)) ]  )
            print('yshape',y.shape)
            hdf5_store.extend(x,'/%s/x'%alg)
            hdf5_store.extend(y,'/%s/y'%alg)        
            del x,y
            gc.collect()
            
        except Exception as e:
            pickle.dump(summaries,open('res.pkl','wb'))
            itops.to_pickle('itops.gz')
#            print('x',x,'itop',itops)
#            print('xcols',xcols,'\nycols\n',ycols)
            print('conc',concordant.values.shape,
                  'shapes:',
                  [ s.shape
                    for s in [y, itops, sp_tree_ind, np.reshape(concordant.values,(-1,1))] ],
                  'x',x.shape,
            )

            raise e
        finally:
            pass
    print('finished!')


if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('--procs',
                        '-p',
                        type=int,
                        help='num procs',
                        default=4)
    parser.add_argument('--splitsize',
                        '-s',
                        type=int,
                        help='num genes in each dataset',
                        default=1000)
    parser.add_argument('--verbose',action='store_true',help='debug')
    parser.add_argument('--overwrite',action='store_true',help='overwrite')
    parser.add_argument('--threads','-t',type=int,help='num threads per proc',default=4)
    parser.add_argument('--mem','-m',type=float,help='memory (in bytes)',default=4e9)
    parser.add_argument('--tol', default=.3,
                        help='''observed frequencies must be within "tol" of each other
                        to be considered ils''')
    
    #TODO: make these 2 into a mutually exclusive required group
    parser.add_argument('--outdir',
                        help='directory to store results files',
                        default='/N/dc2/projects/bkrosenz/deep_ils/results')
    parser.add_argument('--outfile',
                        help='output hdf5 name',
                        default='covs_trio.hdf5')
    parser.add_argument('--folds','-f',type=int,help='CV folds',default=10)
    #TODO group
    parser.add_argument('--ntaxa',
                        type=int,
                        help='total number of taxa in each tree',
                        default=4)
    parser.add_argument('--subtree',
                        type=int,
                        help='number of taxa in each subtree to predict',
                        default=4)
    parser.add_argument('--leafnames',
                        nargs='*',
                        help='taxa names - list or filename')
    parser.add_argument('--outgroup',
                        help='taxa names - list or filename')

    args = parser.parse_args()
    print( 'Arguments: ',args)
    main(args)
    
