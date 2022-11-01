import argparse
import json
import os
from glob import glob
from tempfile import TemporaryDirectory
from time import time
from functools import partial, reduce
from collections import namedtuple

import h5py
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, dump, load
from sklearn import metrics as met
from sklearn.base import clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (AdaBoostClassifier, AdaBoostRegressor,
                              ExtraTreesClassifier, ExtraTreesRegressor,
                              GradientBoostingClassifier,
                              GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.feature_selection import SelectFromModel
from sklearn.gaussian_process import (GaussianProcessClassifier,
                                      GaussianProcessRegressor)
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.linear_model import ElasticNetCV, LogisticRegressionCV
from sklearn.model_selection import (KFold, StratifiedKFold, cross_val_predict,
                                     cross_validate)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.tree import (DecisionTreeClassifier, DecisionTreeRegressor,
                          ExtraTreeClassifier, ExtraTreeRegressor)
from torch._C import dtype

import utils as u

pd.options.mode.chained_assignment = None

SEED = 123456

clipper = FunctionTransformer(partial(np.clip, a_min=0, a_max=1))


def train_and_test(X,
                   y,
                   learners: dict,
                   outfile: u.Path,
                   metrics: dict = None,
                   nprocs: int = 4,
                   fold_splitter: int = None,
                   impute=True,
                   predict: bool = False):
    """train all learners and write performance metrics.
    NOTE: does not clip regression output.

    Args:
        X (df or np array): predictors
        y (_type_): target
        learners (dict): ML algs
        outfile (u.Path): out path
        metrics (dict, optional): _description_. Defaults to None.
        nprocs (int, optional): number of processes for cross val split. Defaults to 4.
        fold_splitter (int, optional): If None, does not give CV metric summaries. Defaults to None.
        impute (bool, optional): impute missing. Defaults to True.
        predict (bool, optional): give CV estimates for each datapoint. Defaults to False.
    """
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.DataFrame):
        index = y.index
        y = y.values
    else:
        index = None
    results = dict.fromkeys(learners)

    scaler = StandardScaler()
    imputer = SimpleImputer(missing_values=np.nan,
                            strategy='median',
                            fill_value=0)
    if fold_splitter is not None:
        with TemporaryDirectory() as tmpdir:
            for learner_name, learner in learners.items():
                if impute:
                    clf = make_pipeline(
                        make_union(imputer, MissingIndicator(
                            features='all'), n_jobs=2),
                        scaler,
                        learner,
                        memory=tmpdir)  # SelectFromModel(learner,threshold='mean'))
                else:
                    clf = make_pipeline(
                        scaler,
                        learner,
                        memory=tmpdir)  # SelectFromModel(learner,threshold='mean'))

            if predict:
                preds = dict.fromkeys(learners)
                try:
                    preds[learner_name] = cross_val_predict(
                        clf,
                        X,
                        y,
                        cv=fold_splitter,  # Kfold or StratifiedKFold depending on type of learner
                        n_jobs=nprocs,
                        pre_dispatch=2 * nprocs,
                        method='predict_proba')[:, 1]
                except AttributeError:  # the learner has no predict proba feature
                    preds[learner_name] = cross_val_predict(
                        clf,
                        X,
                        y,
                        pre_dispatch=2 * nprocs,
                        cv=fold_splitter,  # Kfold or StratifiedKFold depending on type of learner
                        n_jobs=nprocs)
                pred_df = pd.DataFrame(
                    {**preds, 'y_true': y},
                    index=index)
                pred_df.to_csv(
                    outfile.with_suffix('.preds.csv.gz'),
                    compression='gzip')

    modeldir = outfile.parent / 'models'
    modeldir.mkdir(exist_ok=True)

    # weights trained on WHOLE dataset
    for learner_name, learner in learners.items():
        if impute:
            clf = make_pipeline(
                make_union(imputer, MissingIndicator(
                    features='all'), n_jobs=2),
                scaler,
                clone(learner))
        else:
            clf = make_pipeline(
                make_union(imputer, MissingIndicator(
                    features='all'), n_jobs=2),
                scaler,
                clone(learner))
        clf.fit(X, y)

        dump(clf,
             modeldir / f'{outfile.name}_{learner_name}.pkl.gz')
        # we can pickle it


def main(args: argparse.Namespace):
    # use for both class and regress
    decision_tree_params = {
        'max_depth': 4,
        'min_samples_leaf': 1,
    }
    params = [args]
    if args.classify:
        classification_learners = {
            'Random': DummyClassifier(strategy='stratified'),
            # same as most_frequent, except predict_proba returns the class prior.
            'Trivial': DummyClassifier(strategy='prior'),
            'RF': RandomForestClassifier(bootstrap=True,
                                         n_estimators=50,
                                         criterion='gini',
                                         n_jobs=2,
                                         class_weight="balanced",
                                         **decision_tree_params),
            'AdaBoost': AdaBoostClassifier(n_estimators=50,
                                           base_estimator=DecisionTreeClassifier(
                                               criterion='gini',
                                               class_weight="balanced",
                                               **decision_tree_params)),
            'GradBoost': GradientBoostingClassifier(n_estimators=50,
                                                    criterion='friedman_mse',
                                                    **decision_tree_params),
            'LogisticReg': LogisticRegressionCV(Cs=5,
                                                penalty='l1',
                                                class_weight='balanced',
                                                solver='saga',  # NOTE: saga is fast only if data has been scaled
                                                max_iter=1000,
                                                n_jobs=2,
                                                tol=1e-4,
                                                cv=3),
            #     'MLP': MLPClassifier(solver='sgd',
            #                          batch_size=100,
            #                          activation='relu',
            #                          learning_rate='adaptive',
            #                          learning_rate_init=0.01,
            #                          momentum=0.8,
            #                          nesterovs_momentum=True,
            #                          hidden_layer_sizes=(40, 10, 10, 10),
            #                          tol=1e-4,
            #                          max_iter=1000,
            #                          shuffle=True)
        }

        c_metrics = {'Acc': met.accuracy_score,
                     'F1': met.f1_score,
                     'AUC': met.roc_auc_score,
                     'Prec': met.precision_score,
                     'Recall': met.recall_score,
                     'MCC': met.matthews_corrcoef
                     }
        params.append(str(classification_learners))
        # cv requires scoring fn
        for m in c_metrics:
            c_metrics[m] = met.make_scorer(c_metrics[m])

    ###### REGRESSION #######
    # 'criterion':'mse', # by default

    if args.regress:
        regression_learners = {
            'Mean': DummyRegressor(strategy='mean'),
            'Median': DummyRegressor(strategy='median'),
            'RF': RandomForestRegressor(bootstrap=True,
                                        n_jobs=2,
                                        n_estimators=50,
                                        **decision_tree_params),
            'GradBoost': GradientBoostingRegressor(n_estimators=50,
                                                   criterion='friedman_mse',
                                                   **decision_tree_params),
            'AdaBoost': AdaBoostRegressor(
                n_estimators=50,
                base_estimator=DecisionTreeRegressor(**decision_tree_params)),
            'ElasticNet': ElasticNetCV(cv=3,
                                       precompute='auto',
                                       n_alphas=50,
                                       normalize=False,
                                       selection='random',
                                       n_jobs=2,
                                       max_iter=1000),
            #      'MLP': MLPRegressor(solver='sgd',
            #                          batch_size=100,
            #                          activation='relu',
            #                          learning_rate='adaptive',
            #                          learning_rate_init=0.01,
            #                          momentum=0.8,
            #                          nesterovs_momentum=True,
            #                          hidden_layer_sizes=(40, 10, 10, 10),
            #                          tol=1e-4,
            #                          early_stopping=False,
            #                          max_iter=1000,
            #                          shuffle=True)
        }
        params.append(str(regression_learners))
    dump(params, args.outdir / 'arguments.pkl.gz')

    r_metrics = {'MSE': met.mean_squared_error,
                 'MAE': met.mean_absolute_error,
                 'EV': met.explained_variance_score
                 }
    for m in r_metrics:
        r_metrics[m] = met.make_scorer(r_metrics[m])

    try:
        X = u.load_hdf_files(args.data, args.procs).query('ebl<=200')
        if args.dropna:
            X.dropna(inplace=True)
        y = X['y_prob']
        X.drop(columns='y_prob', inplace=True)

        if 'features' in args:
            # allow substring matches
            x_attrs = [a for a in X.columns if any(a.startswith(feature)
                                                   for feature in args.features)]
            X = X[x_attrs]

        if args.classify:
            # TODO: don't reorder the input data!
            y_bin = (u.sigmoid(y) > args.ils)  # ils = 0, no_ils = 1
            n_positive = y_bin.sum()
            if not n_positive or n_positive == y_bin.size:
                print('need 2 classes in training data')
                args.classify = False
            else:
                print(f'pos: {n_positive}, neg: {y_bin.size-n_positive}')

            if args.cnoise > 0:
                y_bin = y_bin.logical_xor(
                    np.random.uniform(size=y_bin.shape) > args.cnoise)
        if args.rnoise > 0:
            y += np.random.normal(scale=args.rnoise,
                                  size=y.shape)

        if args.regress:
            print('\nregressors....\n')
            now = time()
            results_r = train_and_test(X,
                                       y,
                                       regression_learners,
                                       #               metrics=r_metrics,
                                       outfile=args.outdir/'results_regress',
                                       #    fold_splitter=KFold(args.folds,
                                       #                        shuffle=True,
                                       #                        random_state=SEED),
                                       nprocs=args.procs,
                                       predict=args.predict,
                                       impute=not args.dropna)
            print('time:', time()-now)

        if args.classify:
            print('\nclassifiers....\n')
            now = time()
            results_c = train_and_test(X,
                                       y_bin,
                                       classification_learners,
                                       # metrics=c_metrics,
                                       outfile=args.outdir/'results_classify',
                                       # fold_splitter=StratifiedKFold(args.folds,
                                       #                               shuffle=True,
                                       #                             random_state=SEED),
                                       nprocs=args.procs,
                                       predict=args.predict,
                                       impute=not args.dropna)
            print('time:', time()-now)

    except Exception as e:
        raise e
        print(e)
    print('finished')


if __name__ == "__main__":
    # TODO: make this match the dataloading format of the torch scripts.  ignore config.json
    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('--cnoise',
                        type=float, default=0,
                        help='''p for Bernoulli noise to flip classification labels.''')
    parser.add_argument('--rnoise',
                        type=float, default=0,
                        help='''sigma for Gaussian noise added to regression label.''')
    parser.add_argument('--procs',
                        '-p',
                        type=int,
                        help='num procs',
                        default=4)
    parser.add_argument('--dropna',
                        action='store_true',
                        help='drop all rows with null values')
    parser.add_argument('--predict',
                        action='store_true',
                        help='write cross-val predictions to file')
    parser.add_argument('--regress',
                        action='store_true',
                        help='regression')
    parser.add_argument('--classify',
                        action='store_true',
                        help='regression')
    parser.add_argument('--balanced',
                        action='store_true',
                        help='''use balanced ILS/NoILS classes;
                         default is true.''')
    parser.add_argument('--ils',
                        default=.9,
                        type=float,
                        help='''species tree topology frequency can 
                        be at most this value to be considered discordant.''')
    parser.add_argument('--data',
                        '-i',
                        nargs="+",
                        type=u.Path,
                        help='input hdf5 file')
    parser.add_argument('--outdir',
                        type=u.Path,
                        help='directory to store results files')
    parser.add_argument('--config',
                        '-c',
                        help='''input json config file.  
                        All flags will overwrite command line args.''')
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
    print('Arguments: ', args)
    for fname in args.data:
        if not fname.exists():
            raise OSError("file not found:", fname)

    if not args.outdir:
        args.outdir = args.data[0].with_suffix('')
    args.outdir.mkdir(exist_ok=True)
    main(args)
