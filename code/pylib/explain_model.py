
import pandas as pd
import gc

from joblib import dump, load
import numpy as np
import shap
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import preprocessing
from torch.nn.utils.weight_norm import weight_norm
from train_config import *
import matplotlib.pyplot as plt


import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import shap
import matplotlib.pyplot as plt

from itertools import repeat, chain


def revert_dict(d): return dict(
    chain(*[zip(val, repeat(key)) for key, val in d.items()]))


DATA_DIM = 168


def gradboost_features():
    from joblib import Parallel, delayed, dump, load

    alg = load('/N/project/phyloML/deep_ils/results/train_data/nonrec/allLengths_dropna/models/results_regress_GradBoost.pkl.gz')
    gb = alg[-1]
    ftrs = pd.Series(gb.feature_importances_[
                     :168], index=columns).sort_values()
    ftrs[ftrs > .005].plot.barh()


def grouped_shap(shap_values, features, groups):
    groupmap = revert_dict(groups)
    shap_Tdf = pd.DataFrame(
        shap_values,
        columns=pd.Index(features, name='features')).T
    shap_Tdf['group'] = shap_Tdf.reset_index().features.map(groupmap).values
    shap_grouped = shap_Tdf.groupby('group').sum().T
    return shap_grouped


def main(args):
    print("cpus:", os.sched_getaffinity(0))
    preprocessor = load(args.preprocessor)

    if args.config is None:
        args.config = args.outdir/'model.config'
    device = models.get_device()
    config = load(args.config)

    model = models.load_model(
        config,
        DATA_DIM,
        checkpoint_dir=args.outdir)

    trainset, _ = u.load_data(
        data_dir=args.data_dir,
        data_files=args.data_files[1:],
        train_test_file=args.outdir / "train_test_split.npz",
        from_scratch=args.overwrite,
        test_size=0,
        topology=args.topology,
        dropna=True,
        conditions='ebl<=120',
        preprocessor=args.preprocessor,
        log_proba='train')

    x_test = u.load_and_filter(args.data_files[0], topologies=False).dropna()
    y_test = x_test.y_prob
    x_test.drop(columns='y_prob', inplace=True)

    features = x_test.columns.drop('y_prob', 0, errors='ignore')

    x_train = trainset.tensors[0].to(device)

    try:
        e = torch.load(args.outdir/'deep_explainer.torch')
    except:

        e = shap.DeepExplainer(
            model,
            x_train[np.random.choice(
                len(x_train), size=(100000,), replace=False)]
        )
        torch.save(e, args.outdir/'deep_explainer.torch')

    feature_levels = dict(zip(features.names, features.levels))
    groups_by_feature = features.groupby(feature_levels['feature'])
    labels = {'1_2': r'$d(1,2)$', '1_3': r'$d(1,3)$', '1_4': r'$d(1,4)$',
              '2_3': r'$d(2,3)$', '2_4': r'$d(2,4)$', '3_4': r'$d(3,4)$',
              'counts': 'Topology Counts',
              'nsites': 'Number of Informative Sites',
              'seq_length': 'Sequence Length',
              'top_1': 'SCF', 'top_2': 'SCF',
              'top_3': 'SCF'}
    from collections import defaultdict
    groups = defaultdict(list)
    for f in features:
        groups[labels[f[0]]].append(f)
    try:
        clust = load(args.outdir / 'clustering.joblib')
    except:
        clust = shap.utils.hclust(x_test, y_test, linkage="complete")
        dump(clust, args.outdir / 'clustering.joblib')

    for test_condition in ('ebl<120', 'ebl<50 and ibl<.5',
                           'ebl<120 and 20>ibl>3', 'ebl<120 and ibl<.5',
                           'ebl<50 and 20>ibl>3', 'ebl<50',):
        try:
            shap_values = np.load(
                args.outdir/f'{test_condition}_shapley.npy')
            samp = pd.read_pickle(
                args.outdir / f'{test_condition}_sample.pd.gz')
        except:
            samp = x_test.query(test_condition).sample(500)
            samp.to_pickle(args.outdir / f'{test_condition}_sample.pd.gz')
            shap_values = e.shap_values(
                (torch
                    .from_numpy(samp.values)
                    .to(model.device))
            )
            np.save(
                args.outdir/f'{test_condition}_shapley.npy',
                shap_values,
            )

        df = pd.DataFrame(
            {
                "mean_abs_shap": np.mean(np.abs(shap_values), axis=0),
                "stdev_abs_shap": np.std(np.abs(shap_values), axis=0),
            },
            index=features)
        df.to_pickle(args.outdir / f'{test_condition}_shapley.pd.gz')

        shap_features = grouped_shap(shap_values, features, groups)
        print(df.sort_values("mean_abs_shap", ascending=False)[:10])
        plt.clf()
        shap.summary_plot(
            shap_features.values,
            features=shap_features.columns,
        )
        plt.tight_layout()
        plt.savefig(args.outdir/f'{test_condition}_shapley.png')

        plt.close()
        
        try:
            quartiles = pd.qcut(
                u.sigmoid(y_test).loc[samp.index],
                np.linspace(0, 1, 5),
                labels=np.arange(4)
            )

            fig, ax = plt.subplots(1, 5, figsize=(36, 6))
            for q in range(4):
                plt.sca(ax[q])
                quartile_mask = quartiles == q
                quartile_values = x_test.loc[quartiles.index[quartile_mask]]
                quartile_values.columns = (quartile_values
                                           .columns
                                           .map(lambda s: '-'.join(map(str, s)))
                                           .str.replace('nan', '')
                                           .str.replace('.0', ''))

                shap.summary_plot(shap_values[quartile_mask],
                                  quartile_values,
                                  show=False,
                                  plot_size=None,
                                  color_bar=False,
                                  max_display=6)
                plt.title(f"Quintile {q} of predictions")
            plt.tight_layout()
            plt.savefig(args.outdir/f'{test_condition}_quartiles.png')

        except Exception as err:
            print(err)
        finally:
            plt.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train and test.")

    parser.add_argument(
        "--config",
        type=Path,
        help="path to config dict (joblib pickle)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="number of training epochs",
        default=500
    )

    parser.add_argument(
        "--patience",
        type=int,
        help="number of epochs with no DECREASE in val loss",
        default=None
    )
    parser.add_argument(
        "--topology",
        action="store_true",
        help="predict topology",
        # help="binarize y (p>.999 -> 1)",
    )
    parser.add_argument(
        "--outdir",
        help="directory to store results files",
        type=Path,
        default="/N/project/phyloML/deep_ils/results",
    )
    parser.add_argument(
        "--preprocessor",
        type=Path,
        default=None,
        help="path to sklearn preprocessor"
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="regenerate train/test splits from db AND train the model from scratch",
    )

    parser.add_argument(
        "--data_files",
        "-i",
        type=Path,
        nargs="+",
        default=None,
        help="input hdf5 file(s) "
    )

    parser.add_argument(
        "--data_dir",
        type=Path,
        default=None,
        help="dir containing input files"
    )

    args = parser.parse_args()
    main(args)
