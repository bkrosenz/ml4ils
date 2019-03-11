import resource
import objgraph
import random
import inspect

from contextlib import closing
from multiprocess import Pool, Manager
from joblib import Parallel, delayed
from Bio import Phylo, AlignIO
import numpy as np
from functools import partial,reduce
from io import BytesIO, StringIO
from itertools import *
from glob import glob
import utils,argparse,re
from os import path
from ete3 import Tree

import pandas as pd

njobs = 4

VMR = lambda x: np.nanvar(x) / np.nanmean(x) # aka index of dispersion
total_length = lambda x: sum(x)

def summarize(t):
    covs = tree_config.get_cov(t)
    leaf_lengths = {l.name:l.dist for l in t}
    return {'id':utils.nwstr(t),
            **covs,
            'topology':utils.nwstr(t,format=9),
            'vmr':VMR(list(covs.values())),
            'length':total_length(covs[tip] for tip in utils.tips(t.get_leaf_names())),
            **leaf_lengths
    }


outgroup = 4
leafnames = range(1,5)
tree_config = utils.TreeConfig(leafnames=leafnames,
                               outgroup=outgroup,
                               subtree_sizes=[4])

def df_max(df,df1):
    return df.where(df > df1, df1).fillna(df)
def df_min(df,df1):
    return df.where(df < df1, df1).fillna(df)
    
def trees2df(itree_values):
    itr=pd.DataFrame(map(summarize, itree_values))
    ab,ac,bc=itr['1:2'],itr['1:3'],itr['2:3']
    itr['ibl']=df_max(bc,ac)-df_min(ab,ac)
    return itr

def get_trees(filepath):
    try:
        with open(filepath) as f:
            itree_values = [ Tree(tree) for tree in f if '(' in tree ]
        return itree_values
    except FileNotFoundError:
        print('file not found',filepath)
        return None

#treemaker = lambda s: tuple(map(trees2df, get_trees(simpath=simpath,sim=s)))

def get_covs(gtree_file, itree_file):
    trees = map(get_trees, (gtree_file, itree_file))
    x,y = map(trees2df, trees)
    return x.corrwith(y, method='spearman')


if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Process some data.')
    parser.add_argument('--simpath',
                        default='/N/dc2/projects/bkrosenz/deep_ils/sims/seqgen-aa-1000bp',
                        help='sim directory')
    args = parser.parse_args()

    print('called with',args)

    rx = re.compile(r"(t_.*?)_[A-Z].*") 

    itree_files = glob(path.join(args.simpath, 'inferred_trees/*raxml.rooted.trees'))
    sim_prefixes = [ rx.findall(x)[0] for x in itree_files ] 
    gtree_files_all = set( path.splitext(path.basename(x))[0] for x in glob(path.join(args.simpath,'trees/*.rooted.trees')) )
    print('gtree',gtree_files_all, 'sim', sim_prefixes)
    gtree_files = [ path.join(args.simpath, 'trees/'+pref+'.trees') for pref in sim_prefixes if pref in gtree_files_all] # handles repeats

    print(  *zip( itree_files, gtree_files ), )
    
    with closing( Pool(njobs) ) as p:
        #    covs = p.imap_unordered(get_covs, sim_prefixes[:100],chunksize=4)
        covs = p.starmap(get_covs, zip( itree_files, gtree_files ) )

    print(covs)
    res = pd.concat((c for c in covs if c is not None), axis=1)
    res.columns = [sim_prefixes[i] for i,c in enumerate(covs) if c is not None]
    results_file = path.join(args.simpath,'correlations.csv.gz')
    res.to_csv(results_file)
    print('wrote results to',results_file)
    
    import gc
    gc.collect()
