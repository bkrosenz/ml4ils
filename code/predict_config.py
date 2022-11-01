import torch.utils.data as data_utils
from torch import Tensor
from train_config import *
import torch.nn.functional as F

DATA_DIM = 168


def predict(data, model, preprocessor=None):
    dfs = []
    p = u.Permuter()
    for i in range(3):
        p.ix = np.ones_like(data.index)*i
        x = p.permute(data).replace(np.inf, 0)
        tensor = Tensor(preprocessor.transform(x.values)
                        ) if preprocessor is not None else Tensor(x.values)
        preds = (
            model(tensor.to(model.device))
            .detach()
            .squeeze()
            .cpu()
            .numpy())
        shape = preds.shape
        if len(shape) == 1:
            cols = 'preds'
        else:
            cols = [f'preds{i}' for i in range(shape[1])]
        x[cols] = preds

        x['Permutation'] = i
        dfs.append(x)
    return pd.concat(dfs)


def main(args):
    print("Arguments: ", args)
    print("cpus:", os.sched_getaffinity(0))
    preprocessor = load(args.preprocessor)

    if args.config is None:
        args.config = args.model_dir/'model.config'

    config = load(args.config)

    best_trained_model = models.load_model(
        config,
        DATA_DIM,
        checkpoint_dir=args.model_dir)

    best_trained_model.eval()
    dfs = []

    for data_file in args.data_files:
        with pd.HDFStore(data_file) as hdf:
            keys = hdf.keys()
        if len(keys) > 1:
            for key in keys:
                data = (pd
                        .read_hdf(data_file, key)
                        .drop(columns='randomcolumn', errors='ignore'))
                if data.empty:
                    print('no quartets found')
                    continue
                x = predict(data, best_trained_model, preprocessor)
                x['matrix'] = key.replace('/', '')
                dfs.append(x)
        else:
            data = (pd
                    .read_hdf(data_file)
                    .drop(columns='randomcolumn', errors='ignore'))
            if data.empty:
                print('no quartets found')
                continue
            x = predict(data, best_trained_model, preprocessor)
            x['matrix'] = data_file.name.replace('.genes.hdf5', '')
            dfs.append(x)
    outfile = args.outdir/args.outfile
    pd.concat(dfs).to_pickle(outfile)
    print(f'wrote to {outfile}')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Test on metazoa data.")

    parser.add_argument(
        "--config",
        type=Path,
        help="path to config dict (joblib pickle)",
    )
    parser.add_argument(
        "--classify",
        action="store_true",
        help="predict topology",
        # help="binarize y (p>.999 -> 1)",
    )
    parser.add_argument(
        "--model_dir",
        help="directory of trained model",
        type=Path,
        default=Path("/N/project/phyloML/deep_ils/results/final_trained/"),
    )
    parser.add_argument(
        "--outdir",
        help="dir to store data matrix and prediction file",
        type=Path,
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
        "--outfile",
        type=Path,
        default='metazoa.preds.pd.gz',
        help="path to sklearn preprocessor"
    )
    parser.add_argument(
        "--preprocessor",
        type=Path,
        default=None,
        help="path to sklearn preprocessor"
    )
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    main(args)
