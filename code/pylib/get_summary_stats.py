from glob import glob

# plotting
import matplotlib
matplotlib.use('Agg')
from itertools import combinations as combs
from itertools import chain,islice
import matplotlib.pyplot as plt
from collections import defaultdict
from io import StringIO
from functools import partial
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
    
dask.config.config

def process_dir(d,alg,tree_config, tol, verbose=False):
    if verbose: print('processing dir:',d,'....\n')

    s_top =     utils.read_top_file(path.join(d,'s_tree.trees.topos.parquet/'))
    g_top =     utils.read_top_file(path.join(d,'g_tree.trees.topos.parquet/'))

    dirpath = path.join(d,'dataset.fastTree-%s.covs.parquet'%alg)
    if not path.isdir(dirpath):
        if verbose: print('no inferences for dir %s, alg %s'%(d,alg))
        return [[],[]]
    res = []
    
    covs = dd.read_parquet(dirpath, engine='pyarrow')
    tops = utils.read_top_file(path.join(d,'dataset.fastTree-%s.topos.parquet'%alg))

    g_covs = dd.read_parquet(
        path.join(d,'g_tree.trees.covs.parquet/'),
        engine='pyarrow'
    )

    # if covs.isna().values.any().compute():
    #     print(dirpath,'has nas')
    #     exit()
        
    for taxa in tops.taxa.unique().compute():
        taxa_list = taxa.split(',')
        counts = tops[tops.taxa == taxa]['count']
        cmin,cmax,csum = dask.compute( counts.min(),counts.max(),counts.sum() )
        # TODO: ensure that gen'd tops are merged with empty df to get NaNs for topos not observed
        #        if dask.compute((counts.min()-counts.max())/counts.sum())[0] > tol: # TODO: ignore polytomies
        if (cmax-cmin)/csum > tol: # TODO: ignore polytomies
            del counts        
            continue

        # TODO make this faster by only generating once per tree_config
        empty_tops = pd.DataFrame(index = tree_config.nw_iter(taxa_list)) # unsorted, but join will sort index
        counts_full = tops[tops.taxa == taxa].join(empty_tops, how='right')['count']
        
        cov_inds = utils.leafset2pairs(taxa)
        tips = utils.tips(taxa)
        
        c=covs[cov_inds].dropna()
#        print('\n....\ncov',dask.compute( c[tips] ))
        #     da.concatenate(
        #         utils.cov_summaries(c) # TODO: do we need a mask on c? .mask(c==0,np.nan))#.to_dask_array())
        #     ),
        #     'summaries',
        #     utils.cov_summaries(c))
        # ) # TODO: do we need a mask on c? .mask(c==0,np.nan))#.to_dask_array())

         # TODO: do we need a mask on c? .mask(c==0,np.nan))#.to_dask_array())
        #cov_res, count_res, vmr, length
        x_results = dask.compute(
            da.concatenate( utils.cov_summaries(c) ),
            counts_full.values,
            utils.cov_summaries( utils.VMR(c[tips]) ),
            utils.cov_summaries( utils.total_length(c[tips]) )            
        )
        
        x = tuple(chain.from_iterable(x_results)) #(*cov_res,*count_res, *vmr, *length)
        
        # summaries = da.concatenate(
        #     (utils.cov_summaries(c.to_dask_array()), counts_full.values)
        # )

        g_top_full = g_top[g_top.taxa==taxa].join(empty_tops, how='right')['count']
        sp_tree = s_top[s_top.taxa==taxa]['count'].idxmax()
        try:
            g_c = g_covs[cov_inds].dropna()
            
            gcounts, sp_tree_ind, concordant, y_summaries, y_vmr, y_length = dask.compute(
                g_top_full.values,
                np.where(g_top_full.index==sp_tree),
                sp_tree == g_top_full.idxmax(),
                da.concatenate( utils.cov_summaries(g_c) ),
                utils.cov_summaries( utils.VMR(g_c[tips]) ),
                utils.cov_summaries( utils.total_length(g_c[tips]) )
            )
        except Exception as e:
            print(g_c.compute(),
                  g_covs[tips].compute(),
                  '\nci',cov_inds,
                  '\ntips',tips,
                  path.join(d,'g_tree.trees.covs.parquet/'),
                  'taxa', taxa)
            raise e
            
            
        sp_tree_ind = next(chain.from_iterable(sp_tree_ind))
        y = (*gcounts, sp_tree_ind, concordant, *y_summaries, *y_vmr, *y_length)
            
        del counts_full,c

        res.append((x,y))
        
    del s_top,g_top,covs,tops
    gc.collect()
    
    return res

def main(args):

    # nested funcs, all of which use params from args
    
    @vectorize([int16(int32),
                int16(int64),
                int16(float32),
                int16(float64)])
    def extremize(z):
        """0=no ILS, 1=ILS"""
        if z < args.tol+1./3:
            return 1
        if z > 1-args.tol:
            return 0
        return -1
                
    # configuration for this dataset - can only use 1 subtree size 
    if args.leafnames:
        tree_config = utils.TreeConfig(leafnames=args.leafnames,
                                       outgroup=0,
                                       subtree_sizes=[args.subtree])
    else:
        tree_config = utils.TreeConfig(ntaxa=args.ntaxa,
                                       outgroup=0,
                                       subtree_sizes=[args.subtree])
    if args.dirlist:
        with open(args.dirlist) as f:
            dirs = [s.strip() for s in f]
    elif args.indir:
        dirs = [path.join(args.indir,x) for x in next(walk(args.indir))[1]]

    #### start processing
    algnames =  ('jtt','wag','lg')

    ### for testing
    # algnames =  ('lg',)#'wag','lg')
    # dirs = dirs[:-10]

    n_algs = len(algnames)
    tops = list( tree_config.nw_iter(range(1,args.subtree+1)) )
    vmrs = ['vmr_' +summary for summary in utils.summaries]
    lengths = ['length_' +summary for summary in utils.summaries]
    summaries = [
        pair+'_'+summary
        for summary in utils.summaries
        for pair in next(tree_config.cov_iter(range(1,args.subtree+1)))
    ]
    
    print(list( tree_config.cov_iter(range(1,args.subtree+1))), utils.summaries)

    xcols = summaries + tops + vmrs + lengths
    print(xcols)
    xcols = [a.encode('utf8') for a in xcols]
    ycols = tops + ['sp_tree_ind', 'concordant'] + summaries + vmrs + lengths
    ycols = [a.encode('utf8') for a in ycols]
    print(('column_names',
                      [[xcols]]*n_algs + [[ycols]]*n_algs))
    outpath=path.join(args.outdir,args.outfile)
    
    if args.overwrite and path.isfile(outpath):
        remove( outpath )

    hdf5_store = utils.HDF5Store(
        outpath,
        datasets = ['/%s/x'%a for a in algnames] + ['/%s/y'%a for a in algnames],
        shapes = [(len(xcols),)]*n_algs + [(len(ycols),)]*n_algs,
        dtypes = ['f8']*n_algs + ['u4']*n_algs,
        attributes = ('column_names',
                      [xcols]*n_algs + [ycols]*n_algs)
    )
    print( hdf5_store.shapes,tree_config.subtree_sizes, hdf5_store.dtypes)
    for alg in algnames:
        print('running alg',alg)
        results = Parallel(n_jobs=args.procs, verbose=1)(
            delayed(process_dir)(d,
                                 alg,
                                 tree_config,
                                 args.tol,
                                 args.verbose) for d in dirs)
        try:
            x,y = zip(*filter(None,chain.from_iterable(results))) # ignore empty 
        except Exception as e:
            print('couldnt zip x,y')
            pickle.dump(results,open('res.pkl','wb'))
            print(results[:10])
            raise e
        finally:
            #pickle.dump(results,open('res.pkl','wb'))
            pass
        cols = sorted(tree_config.nw_iter())

        hdf5_store.extend(x,'/%s/x'%alg)
        hdf5_store.extend(y,'/%s/y'%alg)        
        del x,y
        gc.collect()

if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('--procs',
                        '-p',
                        type=int,
                        help='num procs',
                        default=4)
    parser.add_argument('--verbose',action='store_true',help='debug')
    parser.add_argument('--overwrite',action='store_true',help='overwrite')
    parser.add_argument('--threads','-t',type=int,help='num threads per proc',default=4)
    parser.add_argument('--mem','-m',type=float,help='memory (in bytes)',default=4e9)
    parser.add_argument('--tol', default=.3,
                        help='observed frequencies must be within tol of each other')
    #TODO: make these 2 into a mutually exclusive required group
    pq_file_group = parser.add_mutually_exclusive_group(required=True)

    pq_file_group.add_argument('--indir',
                        help='directory to search for files')
    pq_file_group.add_argument('--dirlist',type=str,help='file with list of dirs to process')
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
                        default=3)
    parser.add_argument('--leafnames',
                        nargs='*',
                        help='taxa names - list or filename')
    parser.add_argument('--outgroup',
                        help='taxa names - list or filename')

    args = parser.parse_args()
    print( 'Arguments: ',args)
    main(args)
    
