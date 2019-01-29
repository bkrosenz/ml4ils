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

from os import mkdir
from glob import glob
import utils as u
import argparse
from os import path,walk
from scipy import stats
import pandas as pd
import numpy as np
import h5py
from joblib import dump, load, Parallel, delayed

import pandas as pd
pd.options.mode.chained_assignment = None
    
def summary_stats(x):
    
    return [stats.moment(x,1,nan_policy='omit'),
            stats.moment(x,2,nan_policy='omit'),
            stats.skew(x,1,nan_policy='omit'),
            ]

def train_and_test(X,
                   y,
                   learners,
                   metrics,
                   outfile,
                   nfolds=10,
                   nprocs=1):
    """main loop"""
    results = dict.fromkeys( learners )
    feature_weights = { l : [] for l in learners }

    with open(outfile+'.txt','w') as f:
        npos=sum(y)
        f.write('npos (ILS): %d, nneg %d, nfolds: %d\n' %(npos,len(y)-npos,nfolds))
        f.write('metric\tmean\tstd\n')
        
        for learner_name, learner in learners.items():
            f.write('\n----\n%s\n'%learner_name)
            f.write('\nparams:\t%s\n'%learner.get_params())
            res = []
            true = []

            clf = make_pipeline(StandardScaler(),
                                learner) #SelectFromModel(learner,threshold='mean'))
            print('CV: ',learner_name,y,learner)
            results[learner_name] = cross_validate(clf,
                                                   X,
                                                   y,
                                                   scoring = metrics,
                                                   cv = nfolds, # Kfold or StratifiedKFold depending on type of learner
                                                   n_jobs = nprocs,
                                                   return_estimator=True)

            
            for k,v in results[learner_name].items():
                f.write('%s\t%f\t%f\t'%(k,np.mean(v),np.std(v)))

    print(results)
    #    exit()
    df = u.multidict_to_df(
        results,
        names = ['learners','metrics']
    )
    df.to_csv(outfile+'.csv')

    # weights trained on WHOLE dataset
    d = {
        learner_name:u.get_features(clone(learner).fit(X,y))
        for learner_name, learner in learners.items()
    }
    
    for learner_name, learner in learners.items():
        clf = clone(learner)
        clf.fit(X,y)

        dump(clf,path.join(args.outdir,learner_name+'.joblib')) # pickle it
        
        ftrs = u.get_features(clf)
        if ftrs is not None:
            print(len(ftrs),X.shape,y.shape)
               
    feature_weights = pd.DataFrame.from_dict( d )
    #    feature_weights.index = X.columns
    feature_weights.to_csv(outfile+'.features.csv')

def main(args):

    # use for both class and regress
    decision_tree_params = {
        'max_depth':3,
        'min_samples_leaf':1
    }
    
    classification_learners = {
        'Random':DummyClassifier('stratified'),
        'Trivial':DummyClassifier('most_frequent'),
        'RBF-SVM':SVC(kernel='rbf'),
        'RF':RandomForestClassifier(bootstrap=True,
                                    n_estimators=20,
                                    **decision_tree_params),
        'ExtraTrees':ExtraTreesClassifier(bootstrap=True,
                                          n_estimators=20,
                                          **decision_tree_params),
        'AdaBoost':AdaBoostClassifier(n_estimators=20,
                                      base_estimator=DecisionTreeClassifier(**decision_tree_params)),
        'GradBoost':GradientBoostingClassifier(n_estimators=20,
                                               criterion='friedman_mse',
                                               **decision_tree_params),
        'GP':GaussianProcessClassifier(copy_X_train=False),
        'LogisticReg':LogisticRegressionCV(penalty='l1',
                                           class_weight = 'balanced',
                                           solver='liblinear',
                                           max_iter=300,
                                           cv=10),
        'MLP':MLPClassifier(solver='sgd',
                            batch_size=50,
                            learning_rate='adaptive',
                            learning_rate_init=0.01,
                            momentum=0.9,
                            nesterovs_momentum=True,
                            hidden_layer_sizes=(10,10,10),
                            max_iter=600,
                            shuffle=True)
    }

    ###### REGRESSION #######
    # 'criterion':'mse', # by default
    
    regression_learners = {
        'Mean':DummyRegressor('mean'),
        'Median':DummyRegressor('median'),
        'RF':RandomForestRegressor(bootstrap=True,
                                   n_estimators=20,
                                   **decision_tree_params),
        'ExtraTrees':ExtraTreesRegressor(bootstrap=True,
                                         n_estimators=20,
                                         **decision_tree_params),
        'GradBoost':GradientBoostingRegressor(n_estimators=20,
                                              criterion='friedman_mse',
                                              **decision_tree_params),
        'AdaBoost':AdaBoostRegressor(base_estimator=DecisionTreeRegressor(**decision_tree_params)),
        'GP':GaussianProcessRegressor(copy_X_train=False),
        'ElasticNet':ElasticNetCV(cv=10),
        'MLP':MLPRegressor(solver='sgd',
                           batch_size=50,
                           learning_rate='adaptive',
                           learning_rate_init=0.01,
                           momentum=0.9,
                           nesterovs_momentum=True,
                           hidden_layer_sizes=(10,10,10),
                           max_iter=600, shuffle=True)
    }
    
    
    c_metrics = {'Acc':met.accuracy_score,
                 'F1':met.f1_score,
                 'Prec':met.precision_score,
                 'Recall':met.recall_score,
                 'MCC':met.matthews_corrcoef
    }

    # cv requires scoring fn
    for m in c_metrics: c_metrics[m] = met.make_scorer(c_metrics[m])

    r_metrics = {'MSE':met.mean_squared_error,
                 'MAE':met.mean_absolute_error,
                 'EV':met.explained_variance_score
    }
    for m in r_metrics: r_metrics[m] = met.make_scorer(r_metrics[m])

    algnames =  [a1+'_'+a2 for a1, a2 in u.combr(('wag','lg'),2)]
    #    algnames =  ('jtt','wag','lg')
    for alg in algnames:
        try:
            outdir = path.join(args.outdir,alg)
            if not path.exists(outdir): mkdir(outdir)            
        
            with h5py.File(args.data, mode='r',libver='latest') as h5f:
                ds = h5f[alg]
                X=np.nan_to_num( ds['x'] ) # TODO: fix this in the dataset itself
                
                y_attrs = ds['y'].attrs['column_names']
                d=dict((k,v) for v,k in enumerate(y_attrs))
                print(d)
                print([d[k] for k in d if b';' in k ] ) # tops only

                # todo: pass arbitrary restrictions to the program
#                short_trees = np._nan_to_num( ds['y'][:,y_attrs==b'g_length_median']<150 ) 
                y=np.nan_to_num( ds['y'][:, [d[k] for k in d if b';' in k] ] ) # tops only
                
            keep = y.sum(1)>args.mintrees #np.logical_and( y.sum(1)>args.mintrees, short_trees )
            y = y[keep,:]
            X = X[keep,:]
            
            n = y.shape[0]
            
            # dont assume max is the sp tree
            sp_tree_ind = np.ndarray.astype(ds['y'][keep, d[b'sp_tree_ind']], np.int)
            print(n,'y',y.shape,sp_tree_ind.shape,'attribs',d,sp_tree_ind)
            y_frac = y[range(n), sp_tree_ind] / np.sum(y,1)
            
            np.save('Xtrain', X)
            np.save('ytrain', y_frac)
            # classification
            
            if args.balanced:
                ind = np.argsort(y_frac)
                
                ils = np.where( y_frac[ind] < args.ils )[0]
                no_ils = ind[-len(ils):] # numerical index
                keep = np.sort( np.concatenate((ils,no_ils)) ) # must be sorted to index h5py dataset
                X_bin = X[keep,:]
                y_bin = np.concatenate(
                    (np.ones(len(ils)), np.zeros(len(no_ils)))
                )
            else:
                X_bin = X
                y_bin = y_frac < args.ils # ils = 1, no_ils = 0
            print(X_bin.shape,y_bin.shape)
            np.save('Xbin',X_bin)
            np.save('ybin',y_bin)
            
            print('classifiers....\n')
            results_c = train_and_test(X_bin,
                                       y_bin,
                                       classification_learners,
                                       c_metrics,
                                       outfile=path.join(outdir,
                                                         'results.%s.classify'%alg),
                                       nfolds=args.folds,
                                       nprocs=args.procs)
            
            # compute results
            print('regressors....\n')
            results_r = train_and_test(X,
                                       y_frac,
                                       regression_learners,
                                       r_metrics,
                                       outfile=path.join(outdir,
                                                         'results.%s.regress'%alg),
                                       nfolds=args.folds,
                                       nprocs=args.procs)
        except  Exception as e:
            raise e
            continue
    print('finished')

                
if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('--procs',
                        '-p',
                        type=int,
                        help='num procs',
                        default=4)
    parser.add_argument('--mintrees',
                        '-m',
                        type=int,
                        help='min num of gene trees in a sample',
                        default=30)
    parser.add_argument('--balanced',
                        action='store_true',
                        help='use balanced ILS/NoILS classes; o.w. use all the data')
    parser.add_argument('--tol',
                        default=.3,
                        help='observed frequencies must be within tol of each other')
    parser.add_argument('--ils',
                        default=.8,
                        type=float,
                        help='species tree frequency must be at most this value to be considered ils.')
    parser.add_argument('--outdir',
                        help='directory to store results files',
                        default='/N/dc2/projects/bkrosenz/deep_ils/results')
    parser.add_argument('--data',
                        '-i',
                        help='input hdf5 file')
    parser.add_argument('--folds',
                        '-f',
                        type=int,
                        help='CV folds',
                        default=10)

    args = parser.parse_args()
    print( 'Arguments: ',args)
    main(args)
    
