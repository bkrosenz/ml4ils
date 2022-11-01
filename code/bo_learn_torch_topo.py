import argparse
import json
import os
import tempfile
from functools import partial
from glob import glob
from itertools import repeat
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
import torch.utils.data as data_utils
from joblib import dump
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest import bohb
from torch import nn as nn
from torch._C import device
from torch.nn.utils import clip_grad

import models

try:
    from pylib import utils as u
except:
    from .pylib import utils as u

SEED = 123454
# from sqlalchemy.sql.functions import mode
val_batch_size = 256

N_EPOCHS = 8
CLIPPING = 500


# @ray.remote(num_gpus=0.25, max_calls=1)
def train(config: dict,
          checkpoint_dir: PosixPath = None,
          data_files: list = None,
          data_dir: PosixPath = None):
    print('\n---\ncwd', os.getcwd())
    import sys
    sys.path.append(config['driver_cwd'])
    # from pylib import utils as u
    # import models

    import models
    try:
        from pylib import utils as u
    except:
        from .pylib import utils as u

    # ray.util.pdb.set_trace()
    print("ray.get_gpu_ids(): {}".format(ray.get_gpu_ids()))
    print("CUDA_VISIBLE_DEVICES: {}".format(
        os.environ["CUDA_VISIBLE_DEVICES"]))

    train_subset, val_subset = u.load_data(
        data_dir=data_dir,
        data_files=data_files,
        test_size=.2,
        random_seed=SEED,
        dropna=True,
        topology=True,
        from_scratch=False)
    n_features = train_subset[0][0].shape[0]
    net = models.instantiate_model(n_features, config["layers"])

    criterion = nn.NLLLoss(reduction='mean')
    # val criterion uses mse of raw probabilities
    val_criterion = models.calculate_accuracy

    optimizer = optim.Adam(net.parameters(), lr=config["lr"])
    # optimizer = optim.SGD(net.parameters(), lr=config["lr"], momentum=0.9)

    if checkpoint_dir:
        checkpoint_file = checkpoint_dir / "checkpoint"
        if checkpoint_file.exists():
            model_state, optimizer_state = torch.load(
                checkpoint_file)
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
        drop_last=True,
        num_workers=1)

    for epoch in range(1, N_EPOCHS+1):  # loop over the dataset multiple times
        train_loss = running_loss = 0.0
        net.train()
        for epoch_steps, data in enumerate(trainloader, 1):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(net.device), labels.to(net.device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
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
                train_loss += running_loss
                running_loss = 0.0

        # Validation loss
        val_accuracy = 0.0
        net.eval()
        for val_steps, data in enumerate(valloader, 1):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(net.device), labels.to(net.device)
                outputs = net(inputs).squeeze()
                # don't need to exponentiate
                val_accuracy += val_criterion(outputs, labels).item()

        with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((net.state_dict(), optimizer.state_dict()), path)

        tune.report(accuracy=val_accuracy/val_steps,
                    loss=train_loss/N_EPOCHS)
    print("Finished Training")


def sample_layer_config():
    """include dim-3 outputs for the 3 topos"""
    layers = 2 ** np.random.randint(5, 11, size=np.random.randint(3, 13))
    layers[-1] = 3
    return layers


def main(args, max_num_epochs=2, gpus=1):
    print("Arguments: ", args)
    cpus = os.sched_getaffinity(0)
    num_gpus = models.get_device() != 'cpu'
    # num_gpus = models.get_device().type == 'gpu'
    num_cpus = len(cpus)
    print("cpus:", cpus, "gpus:", gpus)

    ray.init(include_dashboard=False,
             ignore_reinit_error=True,
             #  num_cpus=num_cpus,
             #  num_gpus=num_gpus,
             address='auto')
    # ray.init(local_mode=True)

    config = {
        "layers": tune.choice([sample_layer_config() for _ in range(2000)]),  # tune.sample_from(
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([64, 128, 256, 512]),
        # "clip": tune.loguniform(.1, 10)
        "driver_cwd": os.getcwd()
    }

    trainset, _ = u.load_data(data_dir=args.data_dir,
                              data_files=args.data_files,
                              topology=True,
                              preprocessor=args.preprocessor,
                              from_scratch=False)
    n_features = trainset[0][0].shape[0]
    del trainset
    # TODO: replace config with ConfigSpace, ASHA -> BOHB
    scheduler = ASHAScheduler(
        metric="accuracy",
        mode="max",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=3)
    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["accuracy", "training_iteration"])
    trainable = partial(train,
                        checkpoint_dir=args.outdir,
                        data_dir=args.data_dir,
                        data_files=args.data_files)
    if args.debug:
        for k in config:
            config[k] = config[k].sample()
        print(trainable(config))
        exit()

    result = tune.run(
        trainable,
        # fail_fast=True,
        resources_per_trial={"cpu": num_cpus//4, "gpu": num_gpus/4},
        config=config,
        local_dir=args.outdir,
        num_samples=args.trials,
        keep_checkpoints_num=5,
        checkpoint_score_attr='accuracy',
        # checkpoint_at_end=True,
        #    checkpoint_freq=5, # can't use these with functional api checkpointing
        scheduler=scheduler,
        progress_reporter=reporter,
        name="asha_classify",
        callbacks=[tune.logger.JsonLoggerCallback()],
        resume="AUTO")
    dump(result, args.oundir/'asha_results.joblib')
    best_trial = result.get_best_trial("accuracy", "max", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))
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
        "--debug",
        action="store_true",
        help="run single run without ray",
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
        default="/N/project/phyloML/deep_ils/results/bo_torch_class",
        help="dir containing input files"
    )
    parser.add_argument(
        "--mongodb",
        action="store_true",
        help="use mongo db for parallel search"
    )
    args = parser.parse_args()
    main(args)
