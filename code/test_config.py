from train_config import *
import torch.nn.functional as F


def main(args):
    print("Arguments: ", args)
    print("cpus:", os.sched_getaffinity(0))

    _, testset, _, index = u.load_data(
        data_dir=args.data_dir,
        data_files=args.data_files,
        test_size=1.,
        classify=args.classify,
        rlim=.9 if args.classify else None,
        llim=.4 if args.classify else None,

        topology=args.topology,
        return_index=True,
        from_scratch=True,
        dropna=True,
        preprocessor=args.preprocessor,
        train_test_file=args.outdir/'test_data.npz')
    if args.config is None:
        args.config = args.model_dir/'model.config'
    config = load(args.config)
    best_trained_model = models.load_model(
        config,
        testset,
        checkpoint_dir=args.model_dir)

    best_trained_model.eval()
    test_acc, preds = models.test_error(
        best_trained_model,
        testset,
        loss=F.nll_loss if args.topology or args.classify else F.l1_loss,
        predict=True)
    # y = testset
    print("Best trial test error: {}".format(test_acc))
    preds = preds.cpu()
    if len(preds.shape) > 1 and preds.shape[1] > 1:
        res = pd.DataFrame(dict(enumerate(preds.T)), index=index)
        res['y_true'] = testset.tensors[1]
    else:
        res = pd.DataFrame(
            {'preds': preds.squeeze().cpu(),
             'y_true': testset.tensors[1]},
            index=index)

    if args.classify:
        res['pred_class'] = res.preds > args.threshold

    res.to_csv(
        args.outdir/'preds.csv.gz',
        compression='gzip')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Test.")

    parser.add_argument(
        "--config",
        type=Path,
        help="path to config dict (joblib pickle)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=.9511,
        help="threshold for DNN-Class.  Ignored if args.classify is False.",
    )
    parser.add_argument(
        "--topology",
        action="store_true",
        help="predict topology",
    )
    parser.add_argument(
        "--classify",
        action="store_true",
        help="predict binary target.",
    )
    parser.add_argument(
        "--model_dir",
        help="directory of trained model",
        type=Path
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
        "--data_dir",
        type=Path,
        default=None,
        help="dir containing input files"
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
