from sklearn.gaussian_process import GaussianProcessClassifier,GaussianProcessRegressor
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor,ExtraTreeRegressor,ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,RandomForestClassifier,AdaBoostRegressor,RandomForestRegressor,ExtraTreesClassifier,ExtraTreesRegressor,GradientBoostingRegressor,GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier,MLPRegressor
from sklearn.linear_model import ElasticNetCV, LogisticRegressionCV
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier,DummyRegressor
from sklearn import metrics as met
from sklearn.model_selection import cross_validate,StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.base import clone
scaler = StandardScaler() # should we use this?

from glob import glob

import matplotlib
matplotlib.use('Agg')
from itertools import combinations as combs
import matplotlib.pyplot as plt
from io import StringIO
from functools import partial
import argparse,re
from scipy import stats
from numba import njit,vectorize
from numba.types import *
import pandas as pd
import numpy as np
import utils

from ete3 import Tree
from scipy.sparse import csr_matrix, triu
from os import path

import dask
import dask.array as da
import dask.dataframe as dd
import dask.bag as db
# from sklearn.grid_search import GridSearchCV
from dklearn.grid_search import GridSearchCV
# from sklearn.pipeline import Pipeline
from dklearn.pipeline import Pipeline
    
dask.config.config

# one func: calc triplet topologies (gt,wag,jtt,...) -> 1*ntaxa^3 parquet
# one func: calc dmat (gt,wag,jtt,...) -> nloci*ntaxa^2 parquet

@njit
def itl(k):
    """convert row,col indices to lower triangular 1-d array"""
    return j+(i-1)*i/2

@njit
def lti(i,j):
    i,j = sorted(i,j)
    """convert row,col indices to lower triangular 1-d array"""
    return j+(i-1)*i/2

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

def read_topos(fname):
    """assumes we have a header"""
    return np.loadtxt(fname,dtype='int16')[1,:] 

from dask.array import stats 
def summary_stats(x):    
    return [stats.moment(x,1,nan_policy='omit'),
            stats.moment(x,2,nan_policy='omit'),
            stats.skew(x,1,nan_policy='omit'),
            ]

def train_and_test(X,y,learners,metrics,outfile,nfolds=10,nprocs=1):
    """main loop"""
    results = dict.fromkeys( learners )
    feature_weights = { l : [] for l in learners }

    with open(outfile+'.txt','w') as f:
        npos=sum(y)
        f.write('npos (ILS): %d, nneg %d, nfolds: %d\n' %(npos,len(y)-npos,nfolds))
        f.write('metric\tmean\tstd\n')
        
        for learner_name, learner in learners.iteritems():
            f.write('\n----\n%s\n'%learner_name)
            f.write('\nparams:\t%s\n'%learner.get_params())
            res = []
            true = []

            folds = StratifiedKFold(n_splits=nfolds,
                                       shuffle=True, random_state=12345)
            clf = make_pipeline(StandardScaler(),learner) #SelectFromModel(learner,threshold='mean'))
            results[learner_name] = cross_validate(clf,X,y,scoring=metrics,cv=folds,n_jobs=nprocs)
            
            for k,v in results[learner_name].items():
                f.write('%s\t%f\t%f\t'%(k,np.mean(v),np.std(v)))

    df = u.multidict_to_df(
        results,
        names = ['learners','metrics']
        )
    df.to_csv(outfile+'.csv')

    # weights trained on WHOLE dataset
    d = {learner_name:u.get_features(clone(learner).fit(X,y)) for learner_name, learner in learners.iteritems() }
    for learner_name, learner in learners.iteritems():
        clf = clone(learner)
        clf.fit(X,y)
        ftrs=u.get_features(clf)
        if not ftrs is None: print(len(ftrs),X.shape,y.shape)
        
    for k,v in d.items():
        print( k,v )
    feature_weights = pd.DataFrame.from_dict(
        d
    )
    feature_weights.index = X.columns
    feature_weights.to_csv(outfile+'.features.csv')
    print( 'feature_weights',feature_weights)
    
def main(args):

    tracemalloc.start()

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

    # configuration for this dataset
    if args.leafnames:
        tree_config = utils.TreeConfig(leafnames=args.leafnames,
                                       outgroup=0,
                                       subtree_sizes=[3])#,4])
    else:
        tree_config = utils.TreeConfig(ntaxa=39,
                                       outgroup=0,
                                       subtree_sizes=[3])#,4])        
    if args.dirlist:
        with open(args.dirlist) as f:
            dirs = [s.strip() for s in f]
    elif args.indir:
        dirs = [os.path.join(args.indir,x) for x in next(os.walk(args.indir))[1]]

    n_procs = 4

    # should include this as another column
    #    f=lambda s: re.sub('[();]','',s).split(',') # don't assume these are ints, return list of strs

    # want this to be sorted so we can convert to cov easily
    f = lambda s:  tuple(sorted(int(ss) for ss in re.sub('[();]','',s).split(','))) 
    #    f = lambda s: re.sub('[();]','',s)
    find_polytomies = lambda s: re.search('\d+,\d+,\d+',s) is not None 
    pq2df = lambda x: dd.read_parquet(x,engine='pyarrow').repartition(npartitions=args.procs)

    for d in dirs:
        s_cov = pq2df(path.join(d,'s_tree.trees.covs.parquet/'))
        s_top = pq2df(path.join(d,'s_tree.trees.topos.parquet/')).rename(columns={'count':'count_t'})
        #g_cov = pq2df(path.join(d,'g_tree.all.trees.covs.parquet/'))
        g_top = pq2df(path.join(d,'g_tree.all.trees.topos.parquet/'))
        y = g_top.merge(s_top,how='right',on='topo',suffixes=('_i','_t'))['topo','count_i'] # keep only true sp tree freq
        for tops,covs in zip(
                map(pq2df,sorted(glob(path.join(d,'dataset.*topos*')))),
                map(pq2df,sorted(glob(path.join(d,'dataset.*covs*'))))
        ):
            
            # TODO: for arbitrary n, need to have topological ordering

            # need sp tree to get maj/minor topos. TODO: for quartets, need to distinguish between symmetric and asymmetric as well.
            tops = tops.merge(s_top,how='left',on='topo')
            
            # TODO: make sure all the df->bag->array operations are order-preserving
            to_drop = tops.topo.apply(find_polytomies)
            tops = tops[~to_drop]
            trio_inds = tops.topo.apply(f,meta=('trios','str'))#.to_dask_array().compute()
            trios = trio_inds.drop_duplicates()
            freq_grouped = tops.groupby(trio_inds)['count']
            fmin = freq_grouped.min()
            fmax = freq_grouped.max()
            fsum = freq_grouped.sum()
            fmask = freq_grouped.count()>=min(tree_config.subtree_sizes)
            x = (fmax[fmask]-fmin[fmask]) / fsum[fmask]
            x = x[x< args.tol]#.index.compute()
            tops=tops.assign(tid=trio_inds).merge(x.to_frame(),
                                                  left_on='tid',
                                                  right_index=True,
                                                  suffixes=('_i','_diff'),
                                                  how='inner')
            
            # tops.compute().groupby(trio_inds).apply(lambda x:x['count_i'].values)
            # tops.compute().groupby(trios)#.apply(lambda x: (x.count_i.max()-x.count_i.min())/x.
            cov_summaries = utils.Reductions(covs)

            t_c = tops.tid.apply(utils.leafset2subset_cov,meta='str').to_bag()
            covariance_mat = da.stack([s for s in t_c.map(cov_summaries.reduce)],axis=0)
            
            cov_summaries.get_metadata()
            
            # each row in X consists of the tree topo counts and some summary stats derived from the 500+ inferred gene trees
                                
            concordant_topo_counts = tops[~tops.count_t.isna()][['tid','count_i']].rename(columns={'count_i':'concordant'})
            ils = tops[tops.count_t.isna()]
             # annoyingly, dask groupby only works on scalar outputs.
             # TODO: find another workaround for > 2 ILS trees
            ils_topo_counts = ils.groupby(ils.tid).count_i
            counts_mat = concordant_topo_counts.merge(ils_topo_counts.first().to_frame().rename(columns={'count_i':'ils1'}),
                                                     left_on='tid',
                                                      right_index=True
            ).merge(ils_topo_counts.last().to_frame().rename(columns={'count_i':'ils2'}),
                   left_on='tid',
                   right_index=True
            )
            try:
                print('fraction of runs for which the true tree is the dominant topology:',
                      (counts_mat.iloc[:,1:].apply(np.argmax,axis=1)=='concordant').mean().compute())
            except:
                pass

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
    

    if args.true_dmat:
        true_dmat = u.read_data_frames(args.true_dmat)
        true_dmat['join_col'] = true_dmat.filename.map(clean_filenames)
        X_true_gene_trees = pd.merge(X,true_dmat,on=join_cols,how='right')

    # classify based on presence/absence of ILS
    #y_bin = np.apply_along_axis(np.all,arr=y,axis=1)

    # regress on % of all trees

    decision_tree_params = {
                            'max_depth':3,
                            'min_samples_leaf':1}
    #'criterion':'gini', # by default

    learners = {'Random':DummyClassifier('stratified'),
                'Trivial':DummyClassifier('most_frequent'),
                'RBF-SVM':SVC(kernel='rbf'),
                'RF':RandomForestClassifier(bootstrap=True,n_estimators=20, **decision_tree_params),
                'ExtraTrees':ExtraTreesClassifier(bootstrap=True,n_estimators=20, **decision_tree_params),
                'AdaBoost':AdaBoostClassifier(n_estimators=20, base_estimator=DecisionTreeClassifier(**decision_tree_params)),
                'GradBoost':GradientBoostingClassifier(n_estimators=20, criterion='friedman_mse', **decision_tree_params),
                'GP':GaussianProcessClassifier(copy_X_train=False),
                'LogisticReg':LogisticRegressionCV(penalty='l1',class_weight = 'balanced', solver='liblinear',cv=10),
                'MLP':MLPClassifier(solver='sgd',batch_size=50,learning_rate='adaptive',learning_rate_init=0.01,momentum=0.9,nesterovs_momentum=True,
                                    hidden_layer_sizes=(10,10,10), max_iter=500, shuffle=True)
    }

    metrics = {'Acc':met.accuracy_score,
               'F1':met.f1_score,
               'Prec':met.precision_score,
               'Recall':met.recall_score,
               'MCC':met.matthews_corrcoef}

    # cv requires scoring fn
    for m in metrics: metrics[m] = met.make_scorer(metrics[m])
    
    #met.roc_auc_score requires y_score



    results_raxml = train_and_test(X_bin, y_bin, learners, metrics,
                                   outfile=path.join(args.outdir,'results.classify'),
                                   nfolds=args.folds,
                                   nprocs=n_procs)

    if args.true_dmat:
        results_true = train_and_test(X_true_gene_trees, y_bin, learners, metrics,
                                      outfile=path.join(args.outdir,'results.true_trees.classify'),
                                      nfolds=args.folds,
                                   nprocs=n_procs)


    ###### REGRESSION #######
    decision_tree_params = {
                            'max_depth':3,
                            'min_samples_leaf':1}
    # 'criterion':'mse', # by default

    learners = {'Mean':DummyRegressor('mean'),
                'Median':DummyRegressor('median'),
                'RF':RandomForestRegressor(bootstrap=True,n_estimators=20, **decision_tree_params),
                'ExtraTrees':ExtraTreesRegressor(bootstrap=True,n_estimators=20, **decision_tree_params),
                'AdaBoost':AdaBoostRegressor(base_estimator=DecisionTreeRegressor(**decision_tree_params)),
                'GradBoost':GradientBoostingRegressor(n_estimators=20, criterion='friedman_mse', **decision_tree_params),
                'GP':GaussianProcessRegressor(copy_X_train=False),
                'ElasticNet':ElasticNetCV(cv=10),
                'MLP':MLPRegressor(solver='sgd',batch_size=50,learning_rate='adaptive',learning_rate_init=0.01,momentum=0.9,nesterovs_momentum=True,
                                    hidden_layer_sizes=(20,20,20), max_iter=500, shuffle=True)
    }

    metrics = {'MSE':met.mean_squared_error,
               'MAE':met.mean_absolute_error,
               'EV':met.explained_variance_score
    }
    for m in metrics: metrics[m] = met.make_scorer(metrics[m])
    
    results_raxml = train_and_test(X, y_frac, learners, metrics,
                                   outfile=path.join(args.outdir,'results.regress'),
                                    nfolds=args.folds,
                                   nprocs=n_procs)


                
if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('--procs','-p',type=int,help='num procs',default=4)
    parser.add_argument('--threads','-t',type=int,help='num threads per proc',default=4)
    parser.add_argument('--mem','-m',type=float,help='memory (in bytes)',default=4e9)
    parser.add_argument('--tol', default=.3, help='observed frequencies must be within tol of each other')
    parser.add_argument('--indir', help='directory to search for files',default='/N/dc2/projects/bkrosenz/deep_ils/sims/simphy/SimPhy38')
    parser.add_argument('--dirlist',type=str,help='file with list of dirs to process')
    parser.add_argument('--outdir', help='directory to store results files',default='/N/dc2/projects/bkrosenz/deep_ils/results')
    parser.add_argument('--folds','-f',type=int,help='CV folds',default=10)
    parser.add_argument('--use_counts',action='store_true',help='use topology frequencies of inferred trees')


                        # dest='accumulate', action='store_const',
                        #                     const=sum, default=max,
                        #                     help='sum the integers (default: find the max)')

    args = parser.parse_args()
    print( 'Arguments: ',args)
    main(args)
    
