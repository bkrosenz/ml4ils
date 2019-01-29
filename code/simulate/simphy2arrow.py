# from sklearn.gaussian_process import GaussianProcessClassifier,GaussianProcessRegressor
# from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor,ExtraTreeRegressor,ExtraTreeClassifier
# from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,RandomForestClassifier,AdaBoostRegressor,RandomForestRegressor,ExtraTreesClassifier,ExtraTreesRegressor,GradientBoostingRegressor,GradientBoostingClassifier
# from sklearn.neural_network import MLPClassifier,MLPRegressor
# from sklearn.linear_model import ElasticNetCV, LogisticRegressionCV
# from sklearn.svm import SVC
# from sklearn.dummy import DummyClassifier,DummyRegressor
# from sklearn import metrics as met
# from sklearn.model_selection import cross_validate,StratifiedKFold
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.feature_selection import SelectFromModel
# from sklearn.base import clone
# scaler = StandardScaler() # should we use this?


# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt

from collections import Counter,defaultdict
from itertools import combinations as combs

from io import StringIO
from functools import partial
import argparse,re
from numba import njit,vectorize
from numba.types import *
import pandas as pd
import numpy as np
from os import path,walk
import dask.dataframe as dd
from ete3 import Tree
from multiprocess import pool
from gzip import open as gzopen

# one func: calc triplet topologies (gt,wag,jtt,...) -> 1*ntaxa^3 parquet
# one func: calc dmat (gt,wag,jtt,...) -> nloci*ntaxa^2 parquet

open = lambda x: gzopen(x,'rt') if x.endswith('.gz') else open(x)

@njit
def itl(k):
    """convert row,col indices to lower triangular 1-d array"""
    return j+(i-1)*i/2

@njit
def lti(i,j):
    i,j = sorted(i,j)
    """convert row,col indices to lower triangular 1-d array"""
    return j+(i-1)*i/2

class dictOfLists(defaultdict):
    def __init__(self,*args):
        super().__init__(list, args)
        
    def update(self,kv_list):
        """update w/ list of k,v pairs"""
        try:
            for k,v in kv_list:
                self[k].append(v)
        except TypeError: # just a tuple
            self[kv_list[0]].append(kv_list[1])
        
    def merge(self,other):
        """merge with another dictOfLists"""
        for k,v in other.items():
            self[k].extend(v)
            
#############################3        
# # outline:
# for each sp tree:
#     calc cov mat
#     for each dataset:
#         for each gt:
#             calc inferred cov mat
#             summarize inferred  gt cov mats: mean,med,min,max
#             write inferred summaries + freqs
#             for each trio:
#                 if gt topologies are roughly equal:
#                     write index

# ILS
#df = dd.read_parquet("data/indiv-*.parquet", engine='pyarrow', columns=['occupation'])

def find_cherry(t,taxa):
    """returns triplet in order ((a,b),c);"""
    a,b,c = taxa
    a1 = t.get_common_ancestor( a,c )
    a2 =  t.get_common_ancestor( b,c )
    a3 =  t.get_common_ancestor( a,b )
    if a1==a2:
        return (*sorted(a,b),c)
    elif a2==a3:
        return (*sorted(a,c),b)
    elif a1==a3:
        return (*sorted(b,c),a)


Tree=partial(Tree,format=3)
nCr = partial(combs,r=3)

locus_id = re.compile('_[\d_]*$')

def get_distances(t,dists):
    for mpair in dists:
        try:
            dists[mpair] = t.get_distance(*mpair[1])
        except ValueError: # this pair aint in the tree
            dists[mpair] = np.nan

def get_cov(t):
    """return phylogenetic covariance matrix; cov"""
    n = len(t)-1 # ignore outgroup
    m = np.ndarray((n-1)*n/2)
    leaves = t.get_leaf_names()
    for i,leaf1 in enumerate(leaves): # 
        for j,leaf2 in enumerate(leaves[i:]):
            m[lti(j,i)] = t.get_distance(leaf1.get_common_ancestor(leaf2))            
    return zip(leaves,m)


# use only the <x% and >1-x% ILS percentiles
tol = .2
@vectorize([int16(int32,int32),
            int16(int64,int64),
            int16(float32,float32),
            int16(float64,float64)])
def extremize(x,y):
    """0=no ILS, 1=ILS"""
    z=x-y
    if z < tol:
        return 1
    if z > 1-tol:
        return 0
    return -1

# assume 3 topos
@vectorize([int16(int32),
            int16(int64),
            int16(float32),
            int16(float64)])
def extremize(z):
    """0=no ILS, 1=ILS"""
    if z < tol+1./3:
        return 1
    if z > 1-tol:
        return 0
    return -1

    
def main(args):

    ### load dataset
    outgroup = '0'

#    p = pool(args.procs)

    topology_list = []
    cov_list = []
    for root, dirs, files in walk(args.indir):
        print('processing dir: ',root)
        #p.imap(process_files,files)
        covs = dictOfLists()
        for fname in files:
            prefix = path.splitext(fname)[0]
            topo_counts=Counter()
            if fname.endswith('s_tree.trees'):
                tree = Tree(path.join(root,fname),format=3) # names with branch lengths
                tree.set_outgroup('0')
                leaf_names = map(str,range(1,len(t)+1)) # exclude outgroup
                cherries = nCr(leaf_names)

                covs.update( get_cov(tree) )
                topo_counts.update(find_cherry(tree,taxa) for taxa in cherries)
                topo_counts['gene']=prefix
                covs['gene']=[prefix]*len(trees)

            elif fname.endswith('.list.gz'):
                with gzopen(path.join(root,fname)) as f:
                    for i,tree_strs in enumerate(f):
                        trees = map(Tree,tree_strs)
                        g_tree.set_outgroup('0')
                        for tree in trees:
                            covs.update( get_cov(tree) )
                            topo_counts.update(find_cherry(tree,taxa) for taxa in cherries)
                topo_counts['gene']=prefix
                covs['gene']=[prefix]*len(trees)
        print(covs)
        pd.DataFrame(covs).to_parquet(path.join(root,prefix)+'.parquet',engine='pyarrow')
        topology_list.append(topo_counts)
        pd.DataFrame(topo_counts).to_csv(path.join(root,prefix)+'.counts.csv')
                
        # covs.clear()       
        # topo_counts.clear()


    exit()

    #############
    if args.use_counts:
        X = full_dataset[x_counts+BRANCHES].copy()
        X[x_counts] /= n_loci_mode #normalize - or we can do this later
    else:
        X = full_dataset[BRANCHES].copy()
    
    y = full_dataset[y_counts].copy().div(full_dataset[y_counts].sum(axis=1), axis=0) # normalize, drop all but counts
    
    sort_y = np.sort(
        y.values,
         axis=1
    ) # sort each row

    # for regressing on % of non-ILS trees
    y_frac = y.values[:,0] #y[:,1:-1].sum(axis=0)
    y_bin = extremize(y_frac)
    
    keep_rows = y_bin > -1
    
    y_bin=y_bin[keep_rows]

    X_bin=X[keep_rows]
    
    X_bin.to_csv( path.join(args.outdir, 'data.Xbin.csv') )
    pd.DataFrame(y_bin).to_csv( path.join(args.outdir, 'data.ybin.csv') )
    X.to_csv( path.join(args.outdir, 'data.X.csv') )
    pd.DataFrame(y_frac).to_csv( path.join(args.outdir, 'data.y_frac.csv') )
    pd.DataFrame(y).to_csv( path.join(args.outdir, 'data.y.csv') )
                
if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('--outdir', help='directory to store results files',default='/N/dc2/projects/bkrosenz/deep_ils/results')
    parser.add_argument('--indir', help='directory to search for files',default='/N/dc2/projects/bkrosenz/deep_ils/sims/simphy/SimPhy38')
    parser.add_argument('--procs','-p',type=int,help='num procs',default=14)

    args = parser.parse_args()
    main(args)
    
