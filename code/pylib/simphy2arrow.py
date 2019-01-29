from itertools import chain
from collections import Counter,defaultdict
from itertools import combinations as combs
from itertools import combinations_with_replacement as combr
import traceback,sys,tracemalloc,linecache,glob,argparse,re
from io import StringIO
from functools import partial
from numba import njit,vectorize
from numba.types import *
import pandas as pd
import numpy as np
import os,gc,re
import time
from ete3 import Tree
from toolz import partition_all

import dask
from dask.diagnostics import ProgressBar
import dask.bag as db
from dask.distributed import Client, LocalCluster, fire_and_forget, progress
# use with distributed to take care of this error https://github.com/dask/distributed/issues/1467
import pandas as pd
pd.options.mode.chained_assignment = None

# local imports
import utils,diagnostics

#############################

#@dask.delayed
def process(dname,args,tree_config,compute_now=True):
    try:
        outdir = os.path.join(args.outdir, re.findall('(\d+)\/?$',dname)[0]) # only works for simphy output
    except:
        outdir = os.path.join(args.outdir, os.path.basename(dname)) # only works if there's no trailing slash after dname
    tops = covs = -1
    results = []
    try:
        for pref,suf in zip(args.prefixes,args.suffixes):
            globstr = os.path.join(dname,'%s*%s'%(pref,suf) )
            filenames= glob.glob(globstr)
            if not filenames:
                continue
#            print(globstr)
            x = db.read_text(
                globstr,
                files_per_partition=max(1, len(filenames) // (args.procs*args.threads))
            ).map(tree_config.make_tree).remove(lambda x: x is None)

            if args.covs:
                covs = x.map(tree_config.get_cov).to_dataframe(meta=tree_config.cov_schema).to_parquet(
                    os.path.join(outdir,'%s.%s.covs.parquet'%(pref,suf)),
                    compute=compute_now,
                    engine='pyarrow'
                )

            if args.tops:
                # NB: df.set_index() is expensive, don't do it
                tops = x.map(tree_config.get_topologies).flatten().repartition(args.procs*args.threads).frequencies().to_dataframe(meta=tree_config.top_schema)
                tops['taxa']=tops.topo.map(utils.nw2leafstr)
                tops = tops.set_index('topo').to_parquet(
                    os.path.join(outdir,'%s.%s.topos.parquet'%(pref,suf)),
                    compute=compute_now,
                    engine='pyarrow'
                )
            
            results += [covs,tops]
                
    except Exception as e:
        print("\n----------\nerror: %s" % e)
        print('partitions:',len(filenames)// (args.procs*2))
        print('pid',os.getpid(),os.path.join(args.outdir, os.path.basename(dname),'%s.%s.covs.parquet'%(pref,suf)))
        print('x',x)
        print('tops',x.map(tree_config.get_topologies).flatten().repartition(args.procs*args.threads).frequencies().compute(),'covs',covs)
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_tb(exc_traceback, limit=1, file=sys.stdout)
        print( "*** print_exception:" )
        traceback.print_exception(exc_type, exc_value, exc_traceback,
                                  limit=10, file=sys.stdout)
        snapshot = tracemalloc.take_snapshot()
        diagnostics.display_top(snapshot)
        print("\n----------\n")
        raise e
    finally:
        if  compute_now:
            del results
            results = None
            gc.collect()
        return results


def main(args):

    tracemalloc.start()

    # configuration for this dataset
    if args.leafnames:
        if len(args.leafnames)==1:
            with open(args.leafnames[0]) as f:
                leafnames = [s.strip() for s in f.readlines()]
        else:
            leafnames = args.leafnames
        tree_config = utils.TreeConfig(leafnames=leafnames,
                                       outgroup=args.outgroup,
                                       subtree_sizes=args.subtrees)#,4])
    else:
        tree_config = utils.TreeConfig(ntaxa=args.nleaves,
                                       outgroup=args.outgroup,
                                       subtree_sizes=args.subtrees)#,4])
    if args.dirlist:
        with open(args.dirlist) as f:
            dirs = [s.strip() for s in f]
    elif args.indir:
        dirs = [os.path.join(args.indir,x) for x in next(os.walk(args.indir))[1]]

    # to turn off multiprocessing altogether
    if args.procs == 1:
        for x in dirs:
            res = process(x,args,tree_config,compute_now=True)
            print('got result:',x,res)
            del res
            gc.collect()
        exit()

        
    # t0=time.time()
    # print('testing',dirs[0])
    # with ProgressBar():
    #     res=dask.compute( process(dirs[0],args,tree_config,compute_now=False) )
    # print('test took',time.time()-t0)
    # del res
    # gc.collect()

    from multiprocess.pool import Pool
    pool=Pool(args.procs)
    # with dask.config.set(pool=pool):
    #     for dname in dirs:
    #         print('processing dir',dname)
    #         t0=time.time()
    #         process(dname,args,tree_config,compute_now=True) 
    #         print('time',time.time()-t0)

    #task = partial(process,args=args,tree_config=tree_config,compute_now=True)
    results=[]
    process(dirs[0],args,tree_config,True) # for testing

    for x in dirs:
        results.append(
            pool.apply_async(process, args = (x,args,tree_config,True) )
        )
    for x,res in zip(dirs,results):
        print (x,res.get())
        gc.collect()


    exit()
    #########################
    
    print('processing %d dirs: \n%s'%(len(dirs),dirs),'\n-----------------\n')
    
    memory_per_worker = args.mem//args.procs
    cluster = LocalCluster(n_workers=args.procs, 
                           threads_per_worker=args.threads,
                           processes=True,
#                           ncores=1,
                           diagnostics_port=0,
                           #worker args here
                           memory_limit=memory_per_worker,
                           resources={'memory':memory_per_worker}
    )
    client = Client(cluster)
    
    print('config:',dask.config.config)
    #dask.config.set(scheduler='single-threaded')
    for d in dirs:
        task = client.submit( process,d,args,tree_config,compute_now=True,
                              fifo_timeout='0ms',
                              resources = {'memory':memory_per_worker}
                       )
        fire_and_forget( task )
        gc.collect()
    exit()
    #     task = client.submit(process,d,args,tree_config,compute_now=True)
    #     fire_and_forget(task)
    # for d in dirs:
    #     print('processing dir',d)
    #     t0=time.time()
    #     task = client.submit(process,d,args,tree_config,compute_now=False)
    #     fire_and_forget(task)
    #     #process(dname,args,tree_config,compute_now=True) 
    #     print('time',time.time()-t0)
        
    # exit()

    writes = [p for x in dirs for p in process(x,args,tree_config,compute_now=False)]
    print(writes)
    
    print('computing...')
    print('client:',client,'ports:', client.scheduler_info())
    dask.visualize(filename=os.path.join(args.outdir,'dask_graph.svg'))
    
    for task in partition_all(args.procs, writes):
        t0=time.time()
        #res =
        fire_and_forget( client.persist(task,
                                        traverse=True,
                                        optimize_graph=True,
                                        fifo_timeout='0ms',
                                        resources = {'memory':memory_per_worker}
        ) )

        # progress(res)
        # del res
        gc.collect()

        # time.sleep(0.5)
        # progress(res)
        # print(res,'took time:',time.time()-t0)
        # client.recreate_error_locally(res)
        # del res
    gc.collect()

    exit()
    results = client.persist(writes,traverse=True)
    progress(results)

    #dask.set_options(pool=Pool(args.procs))
    
    # print (results)

    print('finished!')
    cluster.close(timeout=20)
    
if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('--tops',action='store_true',
                        help='calculate topologies')
    parser.add_argument('--covs',action='store_true',
                        help='calculate covariances')

    parser.add_argument('--outdir',
                        help='directory to store results files.',
                        default='/N/dc2/projects/bkrosenz/deep_ils/results')
    parser.add_argument('--dirlist',
                        type=str,
                        help='file with list of dirs to process.  WARNING: dirnames must not contain a trailing slash')
    parser.add_argument('--indir',
                        help='directory to search for files',
                        default='/N/dc2/projects/bkrosenz/deep_ils/sims/simphy/SimPhy38')
    parser.add_argument('--procs','-p',
                        type=int,
                        help='num procs',default=4)
    parser.add_argument('--threads','-t',
                        type=int,
                        help='num threads per proc',default=4)
    parser.add_argument('--mem','-m',
                        type=float,
                        help='memory (in bytes)',default=4e9)
    parser.add_argument('--suffixes',
                        nargs='*',
                        default=['trees']*2+['fastTree-jtt','fastTree-wag','fastTree-lg'],
                        help='suffices of files to process')
    parser.add_argument('--prefixes',
                        nargs='*',
                        default = ['s_tree','g_tree']+['dataset']*3,
                        help='prefixes of files to process')
    taxa_group = parser.add_mutually_exclusive_group(required=True)
    taxa_group.add_argument('--leafnames',
                        nargs='*',
                        help='taxa names - list or filename')
    taxa_group.add_argument('--nleaves',
                            type=int,
                            help='number of taxa, 0,...n-1')
    parser.add_argument('--outgroup',
                        default=0,
                        help='taxa names - list or filename')
    parser.add_argument('--subtrees',
                        nargs='*',
                        default=[3],
                        help='subtree sizes to process (e.g. 3, 4, 5)')

    
    args = parser.parse_args()

    print('called with arguments:',args)

    main(args)
