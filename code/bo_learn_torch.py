from sklearn import impute
from torch._C import device
import torch.utils.data as data_utils
import argparse
from itertools import repeat
import json
import os
import tempfile
from torch.nn.utils import clip_grad

from functools import partial
from glob import glob
from pathlib import Path, PosixPath
from tempfile import TemporaryDirectory, mkdtemp
from time import time

import h5py
# import hpsklearn as hps
import numpy as np
import pandas as pd
import ray
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest import bohb
from sklearn.impute import KNNImputer, MissingIndicator, SimpleImputer

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import random_split
import models
from pylib import utils as u
SEED = 123454
# from sqlalchemy.sql.functions import mode
val_batch_size = 256

N_EPOCHS = 8
CLIPPING = 500


def train(config: dict,
          checkpoint_dir: PosixPath = None,
          data_files: list = None,
          data_dir: PosixPath = None):
    # ray.util.pdb.set_trace()
    train_subset, val_subset = u.load_data(
        data_dir=data_dir,
        data_files=data_files,
        log_proba='train',
        test_size=.2,
        random_seed=SEED,
        dropna=True,
        from_scratch=False)
    n_features = train_subset[0][0].shape[0]
    net = models.instantiate_model(n_features, config["layers"])

    criterion = nn.SmoothL1Loss(reduction='mean')
    # val criterion uses mse of raw probabilities
    val_criterion = nn.MSELoss(reduction='mean')

    optimizer = optim.Adam(net.parameters(), lr=config["lr"])
    # optimizer = optim.SGD(net.parameters(), lr=config["lr"], momentum=0.9)

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            checkpoint_dir / "checkpoint")
        net.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    trainloader = data_utils.DataLoader(
        train_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=1)
    valloader = data_utils.DataLoader(
        val_subset,
        batch_size=val_batch_size,
        shuffle=True,
        num_workers=1)

    for epoch in range(1, N_EPOCHS+1):  # loop over the dataset multiple times
        running_loss = 0.0
        net.train()
        for epoch_steps, data in enumerate(trainloader, 1):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(net.device), labels.to(net.device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs).log().squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            clip_grad.clip_grad_value_(
                net.parameters(), CLIPPING)
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if not epoch*epoch_steps % 4000:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch, epoch_steps,
                                                running_loss / epoch_steps))
                running_loss = 0.0

        # Validation loss
        val_loss = 0.0
        net.eval()
        for val_steps, data in enumerate(valloader, 1):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(net.device), labels.to(net.device)
                outputs = net(inputs).squeeze()
                # don't need to exponentiate
                loss = val_criterion(outputs, labels.to(outputs.device))
                val_loss += loss.cpu().numpy()
        val_loss /= val_steps

        with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((net.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=val_loss)
    print("Finished Training")


def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')  # don't have GPU
    return device


def df_to_tensor(df):
    """convert a df to tensor to be used in pytorch"""
    device = get_device()
    return torch.from_numpy(df.values).float().to(device)


def sample_layer_config():
    layers = [2 ** np.random.randint(5, 11)
              for _ in range(np.random.randint(3, 12))]
    layers.append(1)
    return layers


def main(args, max_num_epochs=1, gpus_per_trial=1):
    ray.init(include_dashboard=False, ignore_reinit_error=True)
    # ray.init(local_mode=True)
    print("Arguments: ", args)
    print("cpus:", os.sched_getaffinity(0))

    modeldir = u.make_if_not_exists(args.outdir / "models")

    config = {
        "layers": tune.choice([sample_layer_config() for _ in range(10000)]),  # tune.sample_from(
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([64, 128, 256, 512, 1024]),
        # "clip": tune.loguniform(.1, 10)
    }

    trainset, _ = u.load_data(data_dir=args.data_dir,
                              data_files=args.data_files,
                              log_proba='train',
                              preprocessor=args.preprocessor,
                              from_scratch=False)
    n_features = trainset[0][0].shape[0]
    del trainset
    # TODO: replace config with ConfigSpace, ASHA -> BOHB
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=3)
    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["loss", "training_iteration"])
    trainable = partial(train,
                        checkpoint_dir=args.outdir,
                        data_dir=args.data_dir,
                        data_files=args.data_files)
    result = tune.run(
        trainable,
        resources_per_trial={"cpu": 8, "gpu": gpus_per_trial},
        config=config,
        local_dir=args.outdir,
        num_samples=args.trials,
        keep_checkpoints_num=5,
        checkpoint_score_attr='min-loss',
        # checkpoint_at_end=True,
        #    checkpoint_freq=5, # can't use these with functional api checkpointing
        scheduler=scheduler,
        progress_reporter=reporter)

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    # print("Best trial final validation mse: {}".format(
    #     best_trial.last_result["mse"]))
    best_trained_model = models.instantiate_model(
        n_features, best_trial.config['layers'])

    best_checkpoint_dir = best_trial.checkpoint.value
    model_state, optimizer_state = torch.load(
        os.path.join(
            best_checkpoint_dir, "checkpoint")
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train and test.")

    parser.add_argument(
        "--procs",
        "-p",
        type=int,
        help="num procs to run concurrencly",
        default=4
    )
    parser.add_argument(
        "--threads",
        type=int,
        help="num threads for ensemble methods",
        default=4
    )
    parser.add_argument(
        "--trials",
        "-t",
        type=int,
        help="max number trials (hyperparameter settings)",
        default=500
    )
    parser.add_argument(
        "--balanced",
        action="store_true",
        help="use balanced ILS/NoILS classes; o.w. use all the data",
    )
    parser.add_argument(
        "--outdir",
        help="directory to store results files",
        type=Path,
        default="/N/project/phyloML/deep_ils/results",
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
        "--preprocessor",
        type=Path,
        default=None,
        help="path to sklearn preprocessor"
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=None,
        help="dir containing input files"
    )
    parser.add_argument(
        "--mongodb",
        action="store_true",
        help="use mongo db for parallel search"
    )
    args = parser.parse_args()
    main(args)
