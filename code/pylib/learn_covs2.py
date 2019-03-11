from sklearn.gaussian_process import GaussianProcessClassifier,GaussianProcessRegressor
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor,ExtraTreeRegressor,ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,RandomForestClassifier,AdaBoostRegressor,RandomForestRegressor,ExtraTreesClassifier,ExtraTreesRegressor,GradientBoostingRegressor,GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier,MLPRegressor
from sklearn.linear_model import ElasticNetCV, LogisticRegressionCV
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier,DummyRegressor
from sklearn import metrics as met
from sklearn.model_selection import cross_validate,StratifiedKFold, KFold, cross_val_predict
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

import numpy as np
import h5py
from joblib import dump, load, Parallel, delayed
import json

import pandas as pd
pd.options.mode.chained_assignment = None

SEED = 123456

def train_and_test(X,
                   y,
                   learners,
                   metrics,
                   outfile,
                   outdir,
                   fold_splitter=10,
                   nprocs=1,
                   predict=False):
    """main loop"""
    results = dict.fromkeys( learners )
    feature_weights = { l : [] for l in learners }

    if predict:
        preds = dict.fromkeys( learners )
    with open(outfile+'.txt','w') as f:
        npos=sum(y)
        f.write('npos (ILS): %d, nneg %d, nfolds: %d\n' %(npos,len(y)-npos,fold_splitter.n_splits))
        f.write('metric\tmean\tstd\n')
        
        for learner_name, learner in learners.items():
            f.write('\n----\n%s\n'%learner_name)
            f.write('\nparams:\t%s\n'%learner.get_params())
            res = []
            true = []

            clf = make_pipeline(StandardScaler(),
                                learner) #SelectFromModel(learner,threshold='mean'))
#            print('CV: ',learner_name,y,learner)
            results[learner_name] = cross_validate(clf,
                                                   X,
                                                   y,
                                                   scoring = metrics,
                                                   cv = fold_splitter, # Kfold or StratifiedKFold depending on type of learner
                                                   return_train_score=True,
#                                                   return_estimator = predict,
                                                   n_jobs = nprocs)

            if predict:
                try:
                    preds[learner_name] = cross_val_predict(clf,
                                                            X,
                                                            y,
                                                            cv = fold_splitter, # Kfold or StratifiedKFold depending on type of learner
                                                            n_jobs = nprocs,
                                                            method='predict_proba')[:,1]
                except AttributeError: # most likely has no predict proba feature
                    preds[learner_name] = cross_val_predict(clf,
                                                            X,
                                                            y,
                                                            cv = fold_splitter, # Kfold or StratifiedKFold depending on type of learner
                                                            n_jobs = nprocs)

            for k,v in results[learner_name].items():
                f.write('%s\t%f\t%f\t'%(k,np.mean(v),np.std(v)))

    df = u.multidict_to_df(
        results,
        names = ['learners','metrics']
    )
    df.to_csv(outfile+'.csv.gz',compression='gzip')
    if predict:
        pred_df = pd.DataFrame({**preds,'y_true':y})
        pred_df.to_csv(outfile+'.preds.csv.gz',compression='gzip')

    # weights trained on WHOLE dataset
    d = {
        learner_name:u.get_features(clone(learner).fit(X,y))
        for learner_name, learner in learners.items()
    }
    
    for learner_name, learner in learners.items():
        clf = clone(learner)
        clf.fit(X,y)
        
        dump(clf,path.join(outdir,'models',learner_name+'.pkl.gz')) # pickle it
        
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
        'RBF-SVM':SVC(kernel='rbf',
                      gamma='auto',
                      max_iter=1000,
                      probability=args.predict # if we need the predictions; o.w. its faster
        ),
        'RF':RandomForestClassifier(bootstrap=True,
                                    n_estimators=40,
                                    **decision_tree_params),
        'ExtraTrees':ExtraTreesClassifier(bootstrap=True,
                                          n_estimators=40,
                                          **decision_tree_params),
        'AdaBoost':AdaBoostClassifier(n_estimators=40,
                                      base_estimator=DecisionTreeClassifier(**decision_tree_params)),
        'GradBoost':GradientBoostingClassifier(n_estimators=40,
                                               criterion='friedman_mse',
                                               **decision_tree_params),
        'GP':GaussianProcessClassifier(copy_X_train=False),
        'LogisticReg':LogisticRegressionCV(penalty='l1',
                                           class_weight = 'balanced',
                                           solver='liblinear',
                                           max_iter=1000,
                                           cv=10),
        'MLP':MLPClassifier(solver='sgd',
                            batch_size=50,
                            learning_rate='adaptive',
                            learning_rate_init=0.01,
                            momentum=0.9,
                            nesterovs_momentum=True,
                            hidden_layer_sizes=(10,10,10),
                            max_iter=1000,
                            shuffle=True),
        'MLP-big':MLPClassifier(solver='sgd',
                            batch_size=50,
                            learning_rate='adaptive',
                            learning_rate_init=0.01,
                            momentum=0.9,
                            nesterovs_momentum=True,
                            hidden_layer_sizes=(20,20,20,20),
                            max_iter=1000,
                            shuffle=True)
    }

    ###### REGRESSION #######
    # 'criterion':'mse', # by default
    
    regression_learners = {
        'Mean':DummyRegressor('mean'),
        'Median':DummyRegressor('median'),
        'RF':RandomForestRegressor(bootstrap=True,
                                   n_estimators=40,
                                   **decision_tree_params),
        'ExtraTrees':ExtraTreesRegressor(bootstrap=True,
                                         n_estimators=40,
                                         **decision_tree_params),
        'GradBoost':GradientBoostingRegressor(n_estimators=40,
                                              criterion='friedman_mse',
                                              **decision_tree_params),
        'AdaBoost':AdaBoostRegressor(
            n_estimators=40,
            base_estimator=DecisionTreeRegressor(**decision_tree_params)),
        'GP':GaussianProcessRegressor(copy_X_train=False),
        'ElasticNet':ElasticNetCV(cv=10),
        'MLP':MLPRegressor(solver='sgd',
                           batch_size=50,
                           learning_rate='adaptive',
                           learning_rate_init=0.01,
                           momentum=0.9,
                           nesterovs_momentum=True,
                           hidden_layer_sizes=(10,10,10),
                           max_iter=1000, shuffle=True),
        'MLP_big':MLPRegressor(solver='sgd',
                           batch_size=50,
                           learning_rate='adaptive',
                           learning_rate_init=0.01,
                           momentum=0.9,
                           nesterovs_momentum=True,
                            hidden_layer_sizes=(20,20,20,20),
                           max_iter=1000, shuffle=True)
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

    dump( [args,str(regression_learners), str(classification_learners)],
         path.join(args.outdir,'arguments.pkl.gz') )


    with h5py.File(args.data, mode='r',libver='latest') as h5f:
        algnames = list(h5f.keys())

    print('running algs',algnames)
    
    for alg in algnames:
        try:
            outdir = path.join(args.outdir,alg)
            if not path.exists(outdir): mkdir(outdir)
            modeldir=path.join(outdir,'models')
            if not path.exists(modeldir): mkdir(modeldir)            

            with h5py.File(args.data, mode='r',libver='latest') as h5f:
                ds = h5f[alg]
                y_attrs = ds['y'].attrs['column_names'].astype(str)
                x_attrs = ds['x'].attrs['column_names'].astype(str)
                y_full = pd.DataFrame(np.nan_to_num( ds['y'] ), columns = y_attrs)
                x_full = pd.DataFrame(np.nan_to_num( ds['x'] ), columns = x_attrs)
                
            ycount_attrs = [c for c in y_full.columns if ';' in c]
            y_frac_mask = y_full[ycount_attrs].sum(1)>args.mintrees
            print('-----\nalg',alg,'\ny_frac_mask',y_frac_mask.shape,'y_full',y_full.shape)
            
            # dict of the form: 'conditions':{'feature_name':'>2',...}
            if 'conditions' in args: 
                for attr,cond in args.conditions.items():
                    code = compile("ds['y'][:,{}]{}".format(
                        y_attr2ind[attr], cond
                    ), '', 'eval')
                    print('setting',attr,cond)
                    y_frac_mask = np.logical_and(y_frac_mask,eval(code))

            if 'features' in args:
                x_attrs = [a for a in x_attrs if a in args.features]
            X = x_full[x_attrs][y_frac_mask]
            y = y_full[ycount_attrs][y_frac_mask] # drop all features

            sp_tree_cols = [ycount_attrs[s] for s in y_full['sp_tree_ind'].astype(int)]

            y_frac = y_full.lookup( np.squeeze(y_frac_mask.nonzero()),
                                    sp_tree_cols) / y_full[ycount_attrs].sum(1)

            print('yfmask', y_frac_mask.shape)
            np.save(path.join(args.outdir,alg,'yfmask'), y_frac_mask)
            # classification
            

            if args.balanced:
                ind = np.argsort(y_frac)
                ils = np.squeeze(np.where( y_frac < args.ils ))
                n_ils = len(ils)
                if n_ils > y_frac.size/2:
                    print('not enough samples for non-overlapping binarization with ils threshold', args.ils,'; using 50-50 split')
                    n_ils = ind.size//2
                    ils = ind[:n_ils]
                no_ils = ind[-n_ils:] 
                y_bin_ind = np.concatenate((ils,no_ils))
                X_bin = X.iloc[y_bin_ind,:]
                y_bin = np.concatenate(
                    (np.ones(n_ils), np.zeros(n_ils) )
                )
                y_bin_mask = np.vstack([ils,no_ils])
            else:
                X_bin = X
                y_bin = y_frac < args.ils # ils = 1, no_ils = 0
                y_bin_mask = np.vstack([np.where(y_bin),np.where(~y_bin)])
                
            np.save(path.join(args.outdir,alg,'ybmask'), y_bin_mask)
            
            print('\nclassifiers....\n')
            results_c = train_and_test(X_bin,
                                       y_bin,
                                       classification_learners,
                                       c_metrics,
                                       outfile=path.join(outdir,
                                                         'results.%s.classify'%alg),
                                       outdir=outdir,
                                       fold_splitter=StratifiedKFold(args.folds,shuffle=True,random_state=SEED),
                                       nprocs=args.procs,
                                       predict=args.predict)
            
            # compute results
            print('\nregressors....\n')
            results_r = train_and_test(X,
                                       y_frac,
                                       regression_learners,
                                       r_metrics,
                                       outfile=path.join(outdir,
                                                         'results.%s.regress'%alg),
                                       outdir=outdir,
                                       fold_splitter=KFold(args.folds,shuffle=True,random_state=SEED),
                                       nprocs=args.procs,
                                       predict=args.predict)
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
    parser.add_argument('--predict',
                        action='store_true',
                        help='write cross-val predictions to file')
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
    parser.add_argument('--config',
                        '-c',
                        help='input json config file.  All flags will overwrite command line args.')
    parser.add_argument('--folds',
                        '-f',
                        type=int,
                        help='CV folds',
                        default=10)

    args = parser.parse_args()
    if args.config:
        arg_dict = vars(args)
        config = json.load(open(args.config))
        for k in config:
            arg_dict[k] = config[k]
    print( 'Arguments: ',args)
    main(args)
    