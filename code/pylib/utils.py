import pandas as pd
from os import path
import h5py
import numpy as np
import numba
from ete3 import Tree
from collections import Counter,defaultdict
from itertools import chain,product,permutations
from itertools import combinations as combs
from itertools import combinations_with_replacement as combr
import traceback,sys,tracemalloc,linecache,glob
from scipy.special import comb as nchoosek
from io import StringIO
from functools import partial
import re
import dask.dataframe as dd
import dask.array as da
from enumerate_trees import enum_unordered
import numba

trailing_str = re.compile('_[\d]+_[\d]+')
m = re.compile('([\d]+)_')
nwchars = re.compile('[();]')

nw2leaves = lambda x : tuple(sorted(nwchars.sub('',x).split(',')))
nw2leafstr = lambda x : ','.join(sorted(nwchars.sub('',x).split(',')))

nC4 = partial(combs,r=4)
nC3 = partial(combs,r=3)
nC2 = partial(combs,r=2)
leaves2cov = lambda leaf1,leaf2: ':'.join((leaf1,leaf2))
pair2cov = lambda pair: ':'.join(map(str,pair))
triplet2nw = lambda a,b,c: '(%s,(%s,%s);'%(*sorted((a,b)),c)

def tips(leaves):
    if type(leaves)==str:
        return [pair2cov((c,c)) for c in leaves.split(',')]
    else:
        return [pair2cov((c,c)) for c in leaves]

    
# assumes this pair is already sorted, maps '1,2,3' -> ['1:1','1:2',...]
def leafset2pairs(leaves):
    if type(leaves)==str:
        return [pair2cov(c) for c in combr(leaves.split(','),2)] 
    else:
        return [pair2cov(c) for c in combr(leaves,2)]


# operates along rows of df
VMR = lambda x: np.nanvar(x,1) / np.nanmean(x,1) # aka index of dispersion
total_length = lambda x: np.sum(x,1)

# TODO: make this a class
summaries = ('mean','std','min','max','median')
dask_median = partial(da.percentile,q=50)
# TODO: doesnt handle nans

@numba.guvectorize([(numba.float32[:,:], numba.float32[:,:]),(numba.float64[:,:], numba.float64[:,:])], '(n,k),(m,k)',nopython=True,target='parallel')
def summarize_into(x,res):
    for i in range(x.shape[1]):
        res[0,i] = np.nanmean(x[:,i])
        res[1,i] = np.nanstd(x[:,i])
        res[2,i] = np.nanmin(x[:,i])
        res[3,i] = np.nanmin(x[:,i])
        res[4,i] = np.nanmedian(x[:,i])
        
        
cov_summaries = lambda x: (np.nanmean(x,0),
                           np.nanstd(x,0),
                           np.nanmin(x,0),
                           np.nanmax(x,0),
                           np.nanmedian(x,0)
                           )

# regex to extract taxa names
triotaxa = re.compile('\((\d+),\(?(\d+),(\d+)\)?\);') # ? to handle polytomies

def num_rooted_trees(n):
    return int(np.math.factorial(2*n-3)/(2**(n-2)*np.math.factorial(n-2)))

def standardize(t):
    t.sort_descendants()
    t.ladderize()
    return t

def nwstr(t,format=1):
    """FORMAT DESCRIPTION SAMPLE
    0 flexible with support values ((D:0.723274,F:0.567784)1.000000:0.067192,(B:0.279326,H:0.756049)1.000000:0.807788);
    1 flexible with internal node names ((D:0.723274,F:0.567784)E:0.067192,(B:0.279326,H:0.756049)B:0.807788);
    2 all branches + leaf names + internal supports ((D:0.723274,F:0.567784)1.000000:0.067192,(B:0.279326,H:0.756049)1.000000:0.807788);
    3 all branches + all names ((D:0.723274,F:0.567784)E:0.067192,(B:0.279326,H:0.756049)B:0.807788);
    4 leaf branches + leaf names ((D:0.723274,F:0.567784),(B:0.279326,H:0.756049));
    5 internal and leaf branches + leaf names ((D:0.723274,F:0.567784):0.067192,(B:0.279326,H:0.756049):0.807788);
    6 internal branches + leaf names ((D,F):0.067192,(B,H):0.807788);
    7 leaf branches + all names ((D:0.723274,F:0.567784)E,(B:0.279326,H:0.756049)B);
    8 all names ((D,F)E,(B,H)B);
    9 leaf names ((D,F),(B,H));
    100 topology only ((,),(,));"""
    return standardize(t).write(format=format)

def subtree(parent_tree, taxa):
    """returns n-taxon topologically invariant subtree.  sorted in lexicographic->ladderized order: ((a,b),c);"""    
    if not all(x in parent_tree for x in taxa):
        return None # not in this tree
    t = parent_tree.copy()
    t.prune(taxa)
    return standardize(t).write(format=9)

class TreeConfig:
    def __init__(self, leafnames=None,
                 ntaxa = None,
                 outgroup = None, 
                 subtree_sizes = [3],
                 include_outgroup = False):
        self.include_outgroup = include_outgroup # return outgroup w/ every nw topo
        if ntaxa is None and leafnames is None:
            raise('must specify one of: ntaxa,leafnames')
        elif ntaxa:
            self.ntaxa=ntaxa
            self.leafnames = {str(i) for i in range(ntaxa)}
        elif leafnames:
            self.leafnames = {str(i) for i in leafnames}
            self.ntaxa=len(leafnames)
        self.outgroup_name = str(outgroup)
        self.subtree_sizes = [int(s) for s in subtree_sizes]
        self.ntops = sum(num_rooted_trees(k) for k in self.subtree_sizes) # TODO: make this a list
        self.npairs = sum(k*(k+1) // 2 for k in self.subtree_sizes)
        self.top_schema = [('topo',np.str),
                           ('count',np.int16)] # must be list so we know which is which
        self.cov_schema = {leaves2cov(leaf1,leaf2):np.float32
                           for leaf1,leaf2 in combr(sorted(self.leafnames),2)} # have to sort if its a set

        self.trio_pair_inds = np.empty( shape=(nchoosek(self.ntaxa,3,exact=True), 3),
                                        dtype=np.int8)
        get_trio_inds( self.trio_pair_inds, self.ntaxa)

    def make_tree(self, nwstr, single_copy=True):
        """makes ete3 tree from newick string, sets outgroup if it is present. 
        If single_copy=False, keep the highest-index gene from each family,
        otherwise ignore families with duplicates"""
        try:
            t = Tree(nwstr) #trailing_str.sub('',nwstr))
        except Exception as e:
            print('couldn not create tree from',nwstr)
            raise e

        if not single_copy:
            gene_copies = {m.search(k.name).group(1):k for k in t.get_leaves()}
            t.prune(gene_copies.values()) # keep 1st copy
        elif len(set(trailing_str.sub('',leaf.name) for leaf in t)) < len(t):
            # ignore multi-copy genes
            return None
        
        for leaf in t: # now remove trailing id
            leaf.name=trailing_str.sub('',leaf.name)
            
        try:
            t.prune(self.leafnames)
            t.set_outgroup(self.outgroup_name)
        except Exception as e: # usually its ValueError("Node names not found: "...
            # TODO: should we ignore these trees?
            pass 
        
        return t

    def get_cov(self,t):
        """return phylogenetic covariance dict; cov over all tips, including duplicate genes"""
        m = {k:np.nan for k in self.cov_schema}
        leaves = [l for l in t if l.name in self.leafnames]
        for leaf1,leaf2 in combr(sorted(leaves, key = lambda x:x.name), 2):
            key = leaves2cov(leaf1.name,leaf2.name)
            m[key] = t.get_distance(leaf1.get_common_ancestor(leaf2))
        if np.any(np.isnan(list(m.values()))):
            print(m,nwstr(t))
            raise ValueError
        return m

    def get_topologies(self,tree):
        leaf_names = sorted(l.name for l in tree if l.name!=self.outgroup_name)
        taxa_subsets = chain.from_iterable([combs(leaf_names,r) for r in self.subtree_sizes])
        return [subtree(tree,taxa) for taxa in taxa_subsets]
    
    def nw_iter(self,leaf_names=None):
        '''generates all nw trees for a,b in leafnames in sort-ladderize order'''
        if leaf_names is None:
            leaf_names = sorted( self.leafnames )
            leaf_names.remove(self.outgroup_name) #[ln for ln in self.leafnames if self.outgrou
        for tup in enum_unordered(leaf_names):
            if self.include_outgroup:
                tup=(self.outgroup_name,tup)
            yield standardize( Tree( re.sub('[\"\' ]','',str(tup))+';') ).write(format=9)
    
    def nw_cov_iter(self,leaf_names=None):
        '''generates a:b strings for a,b in leafnames, exclude variances a:a and outgroup'''
        return zip(self.nw_iter(leaf_names),
                   self.cov_iter(leaf_names)
                   )
    
    def cov_top_iter(self,leaf_names=None):
        '''generates a:b strings for a,b in leafnames, exclude variances a:a and outgroup'''
        if leaf_names is None:
            leaf_names = sorted( self.leafnames )
            leaf_names.remove(self.outgroup_name) #[ln for ln in self.leafnames if self.outgroup_name!=ln]
        taxa_subsets = chain.from_iterable([combs(leaf_names,r) for r in self.subtree_sizes])
        for c in taxa_subsets:
            yield ([ pair2cov(leaves) for leaves in combs(c,2)],
                   [pair2cov(leaves) for leaves in combr(c,2)])

    def top_iter(self,leaf_names=None):
        '''generates a:b strings for a,b in leafnames, exclude variances a:a and outgroup'''
        if leaf_names is None:
            leaf_names = self.leafnames #[ln for ln in self.leafnames if self.outgroup_name!=ln]
        taxa_subsets = chain.from_iterable([combs(leaf_names,r) for r in self.subtree_sizes])
        for c in taxa_subsets:
            yield [pair2cov(leaves) for leaves in combs(c,2)]

    def cov_iter(self,leaf_names=None):
        '''generates a:b strings for a,b in leafnames, include variances a:a and outgroup'''
        if leaf_names is None:
            leaf_names = self.leafnames #[ln for ln in self.leafnames if self.outgroup_name!=ln]
        leaf_names = sorted(leaf_names)
        taxa_subsets = chain.from_iterable([combs(leaf_names,r) for r in self.subtree_sizes])
        for c in taxa_subsets:
            yield [pair2cov(leaves) for leaves in combr(c,2)]


##### covariance -> top funcs
#@numba.njit(parallel=True) 
def covs2top(arr):
    '''assume input is in form ab,ac,bc or aa,ab,ac,bb,bc,cc
    for 3 taxa, returns the label of the cherry;  0=(ab)c, 1=(ac)b, 2=(bc)a, None=polytomy'''
    if len(arr)==3:
        s = np.sort(arr)
        if s[1]!=s[0]: # polytomy
            return np.nan
        return arr.idxmax()
    
    # elif len(arr)==6: # TODO
    #     s = np.sort(arr)
    #     if s[3]==s[4] and s[2]=s[1]=s[0]: # ladder
    #         return np.argsort(x)[[5,3,1]] # return ab,ac/bc,ad/bd/cd
    #     elif 
    #     return 

@numba.njit(parallel=True)     
def get_trio_inds(x,n):
    '''x is (n-choose-3)-by-3'''
    r,l=x.shape
    row = 0
    for i in range(n-2):
        x[row,0]=i
        for j in range(i+1,n-1):
            x[row,1]=i*n+j
            for k in range(j+2,n):
                x[row,2]=k
                row+=1
                
    
###### data join and reduction funcs ########

#dd.Aggregation('custom_sum',)

#@numba.njit(parallel=True)
def sort_tops(row):
    if row.count_t.isna():
        cols = [leaves2cov(*leaves) for leaves in combr(row.index.split(','),2)]
        

class Reductions:
    def __init__(self,covs):
        self.covs=covs
        self.stat_names = ('mean','std','min','max')
        self.reducers = lambda x: (x.mean(0),x.std(0),x.min(0),x.max(0))
        self.col_names=covs.columns
        
    def reduce(self,col_names=None):
        if col_names is None: col_names=self.col_names
        x = self.covs[col_names].dropna().to_dask_array()
        stats = da.concatenate(
            self.reducers(x),
            axis=0
        )#.reshape((1,-1))
        return stats #.to_dask_dataframe(columns=meta.keys())

    def get_metadata(self,col_names):
        return {'_'.join(k) : np.float32
                for k in product(col_names,self.stat_names)
        }


    
######## HDF

class HDF5Store(object):
    """
    Simple class to append value to a hdf5 file on disc (usefull for building keras datasets)
    
    Params:
        datapath: filepath of h5 file
        dataset: dataset name within the file
        shape: dataset shape (not counting main/batch axis)
        dtype: numpy dtype
    
    Usage:
        hdf5_store = HDF5Store('/tmp/hdf5_store.h5',['X'], shape=[(20,20,3)])
        x = np.random.random(hdf5_store.shape)
        hdf5_store.append(x)
        hdf5_store.append(x)
        
    From https://gist.github.com/wassname/a0a75f133831eed1113d052c67cf8633
    """
    def __init__(self,
                 datapath,
                 datasets,
                 shapes,
                 dtypes,
                 attributes=('',''),
                 compression="gzip",
                 chunk_len=1,
                 nan_to_num=False):
        self.datapath = datapath
        self.datasets = datasets
        self.shapes = dict(zip(datasets,shapes))
        self.chunk_len=chunk_len
        self.dtypes = dtypes
        self.noNans = nan_to_num
        attribute_list = (np.array(a, dtype='S20') for a in attributes[1])  # utf8
            
        if not path.isfile(datapath):
            with h5py.File(self.datapath, mode='w',libver='latest') as h5f:
                for dtype, (dataset, shape), attr  in zip(self.dtypes,
                                                          self.shapes.items(),
                                                          attribute_list):
                    dset = h5f.create_dataset(
                        dataset,
                        shape = (0, ) + shape,
                        maxshape = (None, ) + shape,
                        dtype = dtype,
                        compression = compression,
                        chunks = True) # since chunks depend on length of avi
                    dset.attrs[attributes[0]] = attr
        else:
            print('file exists: %s\nappending...'%datapath)
                    
                    
    def append(self, values,dataset):
        shape = self.shapes[dataset]
        with h5py.File(self.datapath, mode='a',libver='latest') as h5f:
            dset = h5f[dataset]
            last = dset.shape[0]
            dset.resize((last + 1, ) + shape)
            dset[last] = [np.reshape(values,shape)]
            h5f.flush()

    def extend(self, value_list, dataset):
        shape = self.shapes[dataset]
        n = len(value_list)
        try:
            vals = np.reshape(value_list,(-1,*shape))        
            if self.noNans:
                vals = np.nan_to_num(vals, copy=False)
            with h5py.File(self.datapath, mode='a', libver='latest') as h5f:
                dset = h5f[dataset]
                last = dset.shape[0]
                dset.resize((last + n, ) + shape)
                dset[last:(last+n)] = vals
                h5f.flush()
        except Exception as e:
            print(#'value lens',[len(v) for v in value_list],
                                self.shapes,self.datasets,
                                'shape',shape,dataset,'value len',n, (-1,*shape)
            )
                  
            print('shape:',shape,'first value:',value_list[0])
            raise(e)


###### I/O

def read_top_file(fpath):
    ds = dd.read_parquet(fpath,engine='pyarrow')#.compute() # since its small
    if 'taxa' not in ds.columns:
        ds['taxa'] = ds.topo.map(utils.nw2leaves)
    if 'topo' in ds.columns:
        ds = ds.set_index('topo')
    return ds

####### learning
def get_feature_str(learner):
    return str(get_features(learner))

def get_features(learner):
    if hasattr(learner,'feature_importances_'):
        return learner.feature_importances_
    elif hasattr(learner,'coef_'):
        return learner.coef_.ravel()
    return None

def multidict_to_df(user_dict,names=('ix1','ix2')):
    """convert dict-of-dicts or dict-of-lists to multilevel dataframe. 
    2nd arg is the 2 names of the index variables."""
    keys = list(user_dict.keys())
    # print('type(user_dict[next(iter(user_dict))])','\nkey\n',
    #       keys,'type',user_dict[keys[0]],
    #       type(user_dict[keys[0]]))
    try:
        if type(user_dict[keys[0]]) == list: # dict of lists
            df = pd.DataFrame.from_dict({(i,j): user_dict[i][j] 
                                         for i in keys
                                         for j in range(len(user_dict[i]))},
                                        orient='index')
        else:
            df = pd.DataFrame.from_dict({(i,j): user_dict[i][j] 
                                         for i in keys
                                         for j in user_dict[i].keys()},
                                        orient='index')
    except Exception as e:
        print (user_dict[keys[0]])
        raise(e)
        
    df.index = pd.MultiIndex.from_tuples(df.index)
    df.index.names = names
    return df

#### general
from itertools import zip_longest

def grouper(iterable, n, fillvalue=None):
        args = [iter(iterable)] * n
        return zip_longest(*args, fillvalue=fillvalue)

lmap=lambda func,*iterables:list(map(func,*iterables))

def idem(f, x, *args, **kwargs):
    """makes conversions idempotent"""
    try:
        return f(x, *args, **kwargs)
    except:
        return x

#### descriptive stats

def read_trees(tree_file, outgroup='4'):
    with open(tree_file,'r') as f:
        trees=[Tree(s) for s in f if s.endswith(';\n')]
        for t in trees:
            t.set_outgroup(outgroup)
    return trees

def get_tree_dist(gtrees, itrees):
    comps = (t1.compare(t2) for t1,t2 in zip(gtrees,itrees))
    return np.mean(
        np.array([s['rf'] for s in comps])!=0
    )
