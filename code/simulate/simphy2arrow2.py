from itertools import chain
from collections import Counter,defaultdict
from itertools import combinations as combs
from itertools import combinations_with_replacement as combr
import traceback,sys,tracemalloc,linecache,glob
from io import StringIO
from functools import partial
import argparse,re
from numba import njit,vectorize
from numba.types import *
import pandas as pd
import numpy as np
import os,gc,re
import pyarrow as pa
import pyarrow.parquet as pq

#from multiprocess.pool import Pool

from dask.distributed import Client, LocalCluster, fire_and_forget, progress
# use with distributed to take care of this error https://github.com/dask/distributed/issues/1467
import pandas as pd
pd.options.mode.chained_assignment = None

from ete3 import Tree
from gzip import open as gzopen

import dask #from dask import compute, delayed
import dask.bag as db

trailing_str = re.compile('_[\d]+_[\d]+')
gene_copy_0 = re.compile('_0_0')
m = re.compile('([\d]+)_')
OUTGROUP_NAME = '0'

nC3 = partial(combs,r=3)
nC2 = partial(combs,r=2)
leaves2cov = lambda leaf1,leaf2: ':'.join((leaf1,leaf2))

#GLOBAL VARS
NTAXA = 39
LEAF_NAMES = [str(i) for i in range(NTAXA)]
COV_SCHEMA = {leaves2cov(leaf1,leaf2):np.float32 for leaf1,leaf2 in combr(LEAF_NAMES,2)}
TOP_SCHEMA = [('topo',np.str),
              ('count',np.int16)] # must be list so we know which is which
#{'(%s,%s)%s'%(*sorted((a,b)),c):np.int16 for a,b,c in nC3(LEAF_NAMES)}
#columns=['topo','count']

# one func: calc triplet topologies (gt,wag,jtt,...) -> 1*ntaxa^3 parquet
# one func: calc dmat (gt,wag,jtt,...) -> nloci*ntaxa^2 parquet

#open = lambda x: gzopen(x,'rt') if x.endswith('.gz') else open(x)

def make_tree(nwstr):
    """makes ete3 tree from newck string, sets outgroup if it is present"""
    t = Tree(nwstr) #trailing_str.sub('',nwstr))

    # keep the highest-index gene from each family
    gene_copies = {m.search(k.name).group(1):k for k in t.get_leaves()}
        
    t.prune(gene_copies.values()) # keep 1st copy
    
    for leaf in t: # now remove trailing id
        leaf.name=trailing_str.sub('',leaf.name)
        
    try:
        t.set_outgroup(OUTGROUP_NAME)
    except Exception as e:
        #print(t.write(),e)
        pass 
        
    return t

def display_top(snapshot, key_type='lineno', limit=3):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print("#%s: %s:%s: %.1f KiB"
              % (index, filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))

#############################

def find_cherry(t,taxa):
    """returns triplet in order ((a,b),c);"""
    if not all(x in t for x in taxa):
        return None # not in this tree
    a,b,c = taxa
    a1 = t.get_common_ancestor( a,c )
    a2 =  t.get_common_ancestor( b,c )
    a3 =  t.get_common_ancestor( a,b )
    if a1==a2:
        return '(%s,%s)%s'%(*sorted((a,b)),c)
    elif a2==a3:
        return '(%s,%s)%s'%(*sorted((a,c)),b)
    elif a1==a3:
        return '(%s,%s)%s'%(*sorted((b,c)),a)

def get_cov(t, leaf_names = None):
    """return phylogenetic covariance dict; cov"""
    m = {}
    for i,leaf1 in enumerate(leaf_names): 
        for j,leaf2 in enumerate(leaf_names[i:]):
            key = ':'.join((leaf1,leaf2))
            leaf1 = t.get_leaves_by_name(leaf1)[0]
            leaf2 = t.get_leaves_by_name(leaf2)[0]
            m[key] = t.get_distance(leaf1.get_common_ancestor(leaf2))
    return m

def get_cov_all(t):
    """return phylogenetic covariance dict; cov over all tips, including duplicate genes"""
    m = {k:np.nan for k in COV_SCHEMA}
    leaves = t.get_leaves()
    for i,leaf1 in enumerate(leaves): 
        for j,leaf2 in enumerate(leaves[i:]):
            key = leaves2cov(leaf1.name,leaf2.name)
            m[key] = t.get_distance(leaf1.get_common_ancestor(leaf2))
    return m

def get_trio_topologies(tree):
    leaf_names = sorted(tree.get_leaf_names())
    cherries = nC3(leaf_names)
    return [find_cherry(tree,taxa) for taxa in cherries]

#@dask.delayed
def process(dname,args,compute_now=True):
    # filePrefs = ['g_tree']+['dataset']*3
    # fileSuffs = ['trees']+['fastTree-jtt','fastTree-wag','fastTree-lg']
    prefixes = ['s_tree','g_tree']+['dataset']*3
    tops = covs = -1
    results = []
    print('processing',dname)
    try:
        for pref,suf in zip(prefixes,args.suffixes):
            globstr = os.path.join(dname,'%s*%s'%(pref,suf) )
            filenames= glob.glob(globstr)
            if not filenames:
                continue
            print('loading',globstr)
            x = db.read_text(
                globstr,
                files_per_partition=max(1, len(filenames) // (args.procs*args.threads))
            ).map(make_tree)

            covs = x.map(get_cov_all).to_dataframe(meta=COV_SCHEMA).to_parquet(
                os.path.join(args.outdir, os.path.basename(dname),'%s.%s.covs.parquet'%(pref,suf)),
                compute=compute_now,
                engine='pyarrow'
            )#.to_delayed()

            # NB: df.set_index() is expensive, don't do it
            tops = x.map(get_trio_topologies).flatten().frequencies().to_dataframe(meta=TOP_SCHEMA).to_parquet(
                os.path.join(args.outdir, os.path.basename(dname),'%s.%s.topos.parquet'%(pref,suf)),
                compute=compute_now,
                engine='pyarrow'
            )#.to_delayed()

            if not compute_now:
                results += [covs,tops]
                
            print('finished',globstr,results)
        
    except Exception as e:
        print("\n----------\nerror: %s" % e)
        print('partitions:',len(filenames)// (args.procs*2))
        print('pid',os.getpid(),os.path.join(args.outdir, os.path.basename(dname),'%s.%s.covs.parquet'%(pref,suf)))
        print('tops',x.map(get_trio_topologies).flatten().frequencies().compute(),'covs',covs)
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_tb(exc_traceback, limit=1, file=sys.stdout)
        print( "*** print_exception:" )
        traceback.print_exception(exc_type, exc_value, exc_traceback,
                                                                limit=10, file=sys.stdout)
        snapshot = tracemalloc.take_snapshot()
        display_top(snapshot)
        print("\n----------\n")
        raise
    finally:
        return results

def process_all(dnames,args,compute_now= False):
    prefixes = ['s_tree','g_tree']+['dataset']*3
    tops = covs = -1
    results = []
    for dname in dnames:
    #    print('processing',dname)
        try:
            for pref,suf in zip(prefixes,args.suffixes):
                globstr = os.path.join(dname,'%s*%s'%(pref,suf) )
                filenames= glob.glob(globstr)
                if not filenames:
                    continue
     #           print('loading',globstr)
                x = db.read_text(
                    globstr,
                    files_per_partition=len(filenames) // (args.procs*2) + 1
                ).map(make_tree)
                
                covs = x.map(get_cov_all).to_dataframe(meta=COV_SCHEMA).to_parquet(
                    os.path.join(args.outdir, os.path.basename(dname),'%s.%s.covs.parquet'%(pref,suf)),
                    compute=compute_now,
                    engine='pyarrow'
                )#.to_delayed()
                
                # NB: df.set_index() is expensive, don't do it
                tops = x.map(get_trio_topologies).flatten().frequencies().to_dataframe(meta=TOP_SCHEMA).to_parquet(
                os.path.join(args.outdir, os.path.basename(dname),'%s.%s.topos.parquet'%(pref,suf)),
                    compute=compute_now,
                    engine='pyarrow'
            )#.to_delayed()
        except Exception as e:
            print("\n----------\nerror: %s" % e)
            print('partitions:',len(filenames)// (args.procs*2))
            print('pid',os.getpid(),os.path.join(args.outdir, os.path.basename(dname),'%s.%s.covs.parquet'%(pref,suf)))
            print('tops',x.map(get_trio_topologies).flatten().frequencies().compute(),'covs',covs)
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_tb(exc_traceback, limit=1, file=sys.stdout)
            print( "*** print_exception:" )
            traceback.print_exception(exc_type, exc_value, exc_traceback,
                                      limit=10, file=sys.stdout)
            snapshot = tracemalloc.take_snapshot()
            display_top(snapshot)
            print("\n----------\n")
            raise
        finally:
            return results



def main(args):

    tracemalloc.start()
    cluster = LocalCluster(n_workers=args.procs, 
                           threads_per_worker=args.threads,
                           memory_limit=args.mem//args.procs, #worker args here
                           processes=True
#                           , ncores=1
    )
    client = Client(cluster,
                    silence_logs=False)
    print('client:',client,'ports:', client.scheduler_info())

    if args.dirlist:
        with open(args.dirlist) as f:
            dirs = [s.strip() for s in f]
    elif args.indir:
        dirs = [os.path.join(args.indir,x) for x in next(os.walk(args.indir))[1]]

    print('processing %d dirs: \n%s'%(len(dirs),dirs),'\n-----------------\n')

    ######## do this
    # futures = client.map( partial(process,args=args,compute_now=False),
    #                     dirs )
    # print(client.gather(*chain.from_iterable(futures)) )

    # process(dirs[0],args,compute_now=True)
    # print('finished one!',dirs[0])

    ######## or this
    # for d in dirs:
    #     task = client.submit(process,d,args,compute_now=True)
    #     fire_and_forget(task)

    ######## or this
    # for d in dirs:
    #     task = client.submit(process,d,args,compute_now=False)
    #     fire_and_forget(task)

    ######## or this
    # step=args.procs
    # for i in range(0,len(dirs),step):
    #     print('computing %d dirs:'%step, dirs[i:(i+step)])
    #     results = client.compute(
    #         [ process(x,args) for x in dirs[i:(i+step)] ]
    #     )
    #     print('results:',results)
    #     del results
    #     gc.collect()

    ####### or this
    writes = [p for x in dirs for p in process(x,args,compute_now=False)]
#    writes = [process(x,args,compute_now=False) for x in dirs]
    print('computing...')
    print('client:',client,'ports:', client.scheduler_info())
    dask.visualize(filename=os.path.join(args.outdir,'dask_graph.svg'))
    #    dask.compute(*writes)
    print(writes)
    results = client.persist(writes,traverse=True)
    progress(results)
    # print (results)

    print('finished!')
    cluster.close(timeout=20)
    
if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('--outdir', help='directory to store results files',default='/N/dc2/projects/bkrosenz/deep_ils/results')
    parser.add_argument('--indir', help='directory to search for files',default='/N/dc2/projects/bkrosenz/deep_ils/sims/simphy/SimPhy38')
    parser.add_argument('--procs','-p',type=int,help='num procs',default=4)
    parser.add_argument('--threads','-t',type=int,help='num threads per proc',default=4)
    parser.add_argument('--mem','-m',type=float,help='memory (in bytes)',default=4e9)
    parser.add_argument('--dirlist',type=str,help='file with list of dirs to process')
    parser.add_argument('--suffixes',
                        nargs='*',
                        default=['trees']*2+['fastTree-jtt','fastTree-wag','fastTree-lg'],
                        help='suffices of files to process')

    args = parser.parse_args()

    print('called with arguments:',args)
    main(args)
