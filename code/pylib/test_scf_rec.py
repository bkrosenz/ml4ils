import argparse
import json
from glob import glob
from os import mkdir, path, walk
from tempfile import TemporaryDirectory, mkdtemp
from time import time

import numpy as np
import pandas as pd
import seaborn as sns
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
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.tree import (DecisionTreeClassifier, DecisionTreeRegressor,
                          ExtraTreeClassifier, ExtraTreeRegressor)

import utils as u

pd.options.mode.chained_assignment = None

SEED = 123456


def train_and_test(X,
                   y,
                   learners,
                   metrics,
                   outfile,
                   outdir,
                   n_jobs,
                   fold_splitter=10,  # Kfold or StratifiedKFold depending on type of learner
                   predict=False):
    """main loop"""
    results = dict.fromkeys(learners)

    scaler = StandardScaler()
    imputer = SimpleImputer(missing_values=np.nan,
                            strategy='median',
                            add_indicator=True)

    if predict:
        preds = dict.fromkeys(learners)
    with open(outfile + '.txt', 'w') as f, TemporaryDirectory() as tmpdir:
        f.write('metric\tmean\tstd\n')
        for learner_name, learner in learners.items():
            learner = clone(learner)
            f.write('\n----\n%s\n' % learner_name)
            f.write('\nparams:\t%s\n' % learner.get_params())

            clf = make_pipeline(
                imputer,
                scaler,
                learner,
                memory=tmpdir)  # SelectFromModel(learner,threshold='mean'))

            results[learner_name] = cross_validate(clf,
                                                   X,
                                                   y,
                                                   return_estimator=True,
                                                   scoring=metrics,
                                                   cv=fold_splitter,
                                                   return_train_score=True,
                                                   n_jobs=n_jobs)

            if predict:
                try:
                    preds[learner_name] = cross_val_predict(clf,
                                                            X,
                                                            y,
                                                            cv=fold_splitter,
                                                            n_jobs=n_jobs,
                                                            method='predict_proba')[:, 1]
                except AttributeError:  # most likely has no predict proba feature
                    preds[learner_name] = cross_val_predict(clf,
                                                            X,
                                                            y,
                                                            cv=fold_splitter,  # Kfold or StratifiedKFold depending on type of learner
                                                            n_jobs=n_jobs)

            estimator = results[learner_name].pop('estimator')[-1]
            f.write('\nestimator:\t{}\n'.format(estimator))
            for k, v in results[learner_name].items():
                f.write('%s\t%f\t%f\t' % (k, np.mean(v), np.std(v)))

    print(results)
    df = u.multidict_to_df(
        results,
        names=['learners', 'metrics']
    )
    print('preds', preds, 'y', y)
    df.to_csv(outfile+'.csv.gz', compression='gzip')
    if predict:
        pred_df = pd.DataFrame({
            **preds,
            'y_true': y.values.ravel()}
        ).set_index(y.index)
        print(pred_df, pred_df)
        pred_df.to_csv(outfile+'.preds.csv.gz', compression='gzip')

    # weights trained on WHOLE dataset
    for learner_name, learner in learners.items():
        clf = make_pipeline(
            imputer,  # MissingIndicator(features='all'), n_jobs=2),
            scaler,
            clone(learner),
            memory=tmpdir)  # SelectFromModel(learner,threshold='mean'))

        clf.fit(X, y)

        dump(clf,
             path.join(outdir, 'models', path.basename(
                 outfile) + '_' + learner_name + '.pkl.gz')
             )  # pickle it


def main(args):
    # TODO: for each seq length, load trained model from folder.
    # For each rec frac, load hdf file and make preds.
    # calculate rel_err/d_H across ENTIRE ibl for ebl=200.
    # boxplot-line graphs for each condition:
    x = pd.read_hdf(fn, 'x')
    s = x.xs(['counts', 1], 1, ['feature', 'tid']).div(
        x.xs('counts', 1, 'feature').sum(1), axis=0)
    g = s.groupby(level=[0, 1]).agg(['mean']).reset_index()
    sns.lineplot(x='ibl', y='c', hue='nblocks', data=g.query(
        'tid==1 & ebl==130'), legend='full')
    sns.lineplot(x='length', y=r'$d_H$', hue='r', data=D,
                 estimator='mean', ci=96, n_boot=200)

    # use for both class and regress
    decision_tree_params = {
        'max_depth': 4,
        'min_samples_leaf': 1
    }

    ###### REGRESSION #######

    regression_learners = {
        'Mean': DummyRegressor('mean'),
        'Median': DummyRegressor('median'),
        'RF': RandomForestRegressor(bootstrap=True,
                                    n_estimators=50,
                                    **decision_tree_params),
        'GradBoost': GradientBoostingRegressor(n_estimators=50,
                                               criterion='friedman_mse',
                                               **decision_tree_params),
        'AdaBoost': AdaBoostRegressor(
            n_estimators=50,
            base_estimator=DecisionTreeRegressor(**decision_tree_params)),
        'ElasticNet': ElasticNetCV(cv=5,
                                   precompute='auto',
                                   n_alphas=50,
                                   normalize=False,
                                   selection='random',
                                   max_iter=1000),
        'MLP': MLPRegressor(solver='adam',
                            batch_size=400,
                            n_iter_no_change=25,
                            hidden_layer_sizes=(128, 64, 32, 16),
                            tol=1e-9,
                            early_stopping=False,
                            max_iter=1000,
                            shuffle=True)
    }

    r_metrics = {'MSE': met.mean_squared_error,
                 'MAE': met.mean_absolute_error,
                 'EV': met.explained_variance_score
                 }

    for m in r_metrics:
        r_metrics[m] = met.make_scorer(r_metrics[m])

    if not path.exists(args.outdir):
        mkdir(args.outdir)
    modeldir = path.join(args.outdir, 'models')
    if not path.exists(modeldir):
        mkdir(modeldir)

    X = pd.read_hdf(args.data, 'x')

    g = pd.read_csv(
        '/N/project/phyloML/deep_ils/results/train_data/gene_tree_counts.csv',
        index_col=['ebl', 'ibl', 'tid']
    )
    y_true = u.log_odds(
        (g / g.sum(level=('ebl', 'ibl'))).xs(1, 0, 'tid')
    ).reindex(X.index.droplevel('dset_no'))
    y_true.columns = ['f']
    y_true_mask = ~y_true.isna().values
    ibl = X.index.get_level_values('ibl').to_frame(False).set_index(X.index)
    y_prob = u.log_odds(1 - 2 * np.exp(-2 * ibl) / 3, 1e-11)
    y_prob.columns = ['p']
    y_true[y_true_mask].to_csv(
        path.join(args.outdir, 'y_frac.csv.gz'), compression='gzip')
    y_prob.to_csv(path.join(args.outdir, 'y_prob.csv.gz'), compression='gzip')

    # compute results
    print('\n....regress on true top freqs....\n')
    now = time()
    n_jobs = min(args.folds, args.procs)
    train_and_test(X[y_true_mask],
                   y_true[y_true_mask],
                   regression_learners,
                   r_metrics,
                   outfile=path.join(args.outdir,
                                     'results.frac'),
                   outdir=args.outdir,
                   fold_splitter=KFold(
        args.folds, shuffle=True, random_state=SEED),
        n_jobs=n_jobs,
        predict=args.predict)

    print('\n....regress on expected top freqs - P(concordant)....\n')
    train_and_test(X,
                   y_prob,
                   regression_learners,
                   r_metrics,
                   outfile=path.join(args.outdir,
                                     'results.prob'),
                   outdir=args.outdir,
                   fold_splitter=KFold(
                       args.folds, shuffle=True, random_state=SEED),
                   n_jobs=n_jobs,
                   predict=args.predict)
    print('time:', time()-now)
    print('finished')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train and test.')

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
    parser.add_argument('--ils',
                        default=.99,
                        type=float,
                        help='''species tree frequency must be at 
                                most this value to be considered ils''')
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
                        default=5)

    args = parser.parse_args()
    print('Arguments: ', args)
    main(args)
