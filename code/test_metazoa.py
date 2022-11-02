from models import get_device
from train_config import *

import glob
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from joblib.parallel import Parallel, delayed
from sklearn.impute import KNNImputer, MissingIndicator, SimpleImputer
from sklearn.pipeline import make_pipeline, make_union
import torch.utils.data as data_utils
from torch import Tensor


def load_data(datafile, k, preprocessor, retrain_preprocessor, device):
    testset = pd.read_hdf(datafile, k)
    index = testset.index
    if retrain_preprocessor:
        preprocessor.fit(testset)
    testset = Tensor(preprocessor.transform(testset)).to(device)
    testset = data_utils.TensorDataset(testset)
    return testset, index


def main(args):
    print("Arguments: ", args)
    print("cpus:", os.sched_getaffinity(0))
    device = models.get_device()

    try:
        preprocessor = load(args.preprocessor)
    except:
        scaler = StandardScaler()
        imputer = SimpleImputer(missing_values=np.nan,
                                strategy='median',
                                fill_value=0)
        preprocessor = make_pipeline(
            make_union(imputer, MissingIndicator(
                features='all'), n_jobs=4),
            scaler)
        retrain_preprocessor = True

    config = load(args.model_dir/'model.config')

    with pd.HDFStore(args.data_file, 'r') as hdf:
        keys = list(hdf.keys())

    testset, index = load_data(
        args.data_file, keys[0], preprocessor, retrain_preprocessor,
        device)
    best_trained_model = models.load_model(config,
                                           testset,
                                           checkpoint_dir=args.model_dir)

    best_trained_model.eval()
    retrain_preprocessor = False

    all_preds = []
    for k in keys:
        testset, index = load_data(
            args.data_file, k,
            preprocessor,
            retrain_preprocessor,
            device)

        with torch.no_grad():
            preds = best_trained_model(testset.tensors[0]).squeeze().cpu()
        preds = pd.DataFrame(
            {'preds': preds, 'matrix': k[1:]}, index=index)
        all_preds.append(preds)
    all_preds = pd.concat(all_preds)
    outfile = args.outdir/args.data_file.with_suffix('.preds.csv.gz').name
    all_preds.to_csv(
        outfile)
    print(f'wrote {len(preds)} preds to {outfile}')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train and test.")
    # parser.add_argument(
    #     "--config",
    #     type=Path,
    #     help="path to config dict (joblib pickle)",
    # )
    parser.add_argument(
        "--model_dir",
        help="directory of trained model",
        type=Path,
    )
    parser.add_argument(
        "--outdir",
        help="dir to store data matrix and prediction file",
        type=Path,
    )
    parser.add_argument(
        "--data_file",
        "-i",
        type=Path,
        default=None,
        help="input hdf5 file(s) "
    )
    parser.add_argument(
        "--preprocessor",
        type=Path,
        default=None,
        help="joblib dump of sklearn preprocessor"
    )
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    main(args)
