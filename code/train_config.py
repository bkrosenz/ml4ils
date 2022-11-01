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
from typing import Mapping

# import hpsklearn as hps
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data_utils
from joblib import load
from sklearn import impute
from sklearn.metrics import mean_squared_error
from torch import nn
from torch._C import device
from torch.nn.utils import clip_grad
from torch.utils.data import random_split

import models
from pylib import utils as u

# from sqlalchemy.sql.functions import mode
val_batch_size = 256
SEED = 42
CLIPPING = 500


def train(config: dict,
          trainset: data_utils.TensorDataset,
          valset: data_utils.TensorDataset,
          checkpoint_dir: PosixPath = None,
          from_scratch: bool = False,
          save_best: bool = False,
          epochs: int = 100,
          criterion=nn.SmoothL1Loss(reduction='mean'),
          val_criterion=nn.MSELoss(reduction='mean'),
          save_freq: int = 5):
    '''train.  if save_best is True, will overwrite checkpoint based on val loss.
        Patience controls early stopping (independent of save_best).
        Overrides save_freq.'''
    clip_gradient = config.get('clip', CLIPPING)
    patience = config.get('patience', 5)
    net, optimizer = models.load_model(
        config,
        trainset,
        checkpoint_dir=checkpoint_dir if not from_scratch else None,
        with_optimizer=True)

    # train_abs = int(len(trainset) * 0.9)
    # train_subset, val_subset = random_split(
    #     trainset, [train_abs, len(trainset) - train_abs],
    #     generator=torch.Generator().manual_seed(SEED))
    train_batch_size = int(config["batch_size"])
    # WARNING: setting num_workers > 1 causes all networks to learn a CONSTANT PREDICTOR!!!
    trainloader = data_utils.DataLoader(
        trainset,
        batch_size=train_batch_size,
        shuffle=True,
        pin_memory=True,  # saves about 4sec/epoch
        persistent_workers=True,
        num_workers=1)
    valloader = data_utils.DataLoader(
        valset,
        batch_size=val_batch_size,
        shuffle=True,
        persistent_workers=True,
        num_workers=1)
    if checkpoint_dir:
        path = checkpoint_dir / "checkpoint"

    if patience:
        val_losses = np.zeros(patience)
    if save_best:
        best_loss = np.inf
    for epoch in range(1, epochs):  # loop over the dataset multiple times
        net.train()
        running_loss = 0.0
        start_time = time()
        for epoch_steps, data in enumerate(trainloader, 1):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            if args.topology:  # is this necessary?
                labels = labels.long()
            elif args.classify:  # is this necessary?
                labels = labels.float()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs.to(net.device)).squeeze()
            # if outputs.shape[-1] == 1:
            #     outputs = outputs.log().squeeze()  # take log of sigmoid output
            loss = criterion(outputs, labels.to(net.device))
            loss.backward()

            if clip_gradient is not None:
                clip_grad.clip_grad_value_(
                    net.parameters(), clip_gradient)
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if not epoch_steps % 4000:  # print every 3000 mini-batches
                seen = epoch_steps*train_batch_size
                print("[%d, %5d] loss: %.5f" % (epoch, seen,
                                                running_loss / epoch_steps))
        seen = epoch_steps*train_batch_size
        print("[%d, %5d] loss: %.5f, time: %3fs" % (epoch,
                                                    seen,
                                                    running_loss / epoch_steps,
                                                    time()-start_time))
        # Validation loss
        val_loss = 0.0
        net.eval()
        for val_steps, data in enumerate(valloader, 1):
            with torch.no_grad():
                inputs, labels = data
                outputs = net(inputs.to(net.device)).squeeze()
                # must exponentiate labels if they come from trainset split
                loss = val_criterion(outputs, labels.to(net.device))
                val_loss += loss.item()
        val_loss /= val_steps
        print(
            f'Epoch: {epoch}\tVal_Loss: {val_loss :.5f}')

        if save_best and checkpoint_dir and val_loss < best_loss:
            torch.save(
                (net.state_dict(), optimizer.state_dict()), path)
            print('saved checkpoint\n')
            best_loss = val_loss

        if patience:
            if epoch > patience and np.all(val_loss > val_losses):
                print('early stopping')
                break
            val_losses[(epoch-1) % patience] = val_loss
        elif checkpoint_dir and epoch % save_freq:
            torch.save((net.state_dict(), optimizer.state_dict()), path)
            print('saved checkpoint\n')
    if checkpoint_dir and not patience:
        torch.save((net.state_dict(), optimizer.state_dict()), path)
    print("Finished Training")
    return net

# TODO: add method to optimize threshold for classifier


def main(args):
    print("Arguments: ", args)
    cpus = os.sched_getaffinity(0)
    print("cpus:", cpus)
    torch.set_num_threads(len(cpus))
    start = time()
    trainset, valset = u.load_data(
        data_dir=args.data_dir,
        data_files=args.data_files,
        train_test_file=args.outdir / "train_test_split.npz",
        from_scratch=args.overwrite,
        test_size=.1,
        topology=args.topology,
        classify=args.classify,
        rlim=.9 if args.classify else None,
        llim=.4 if args.classify else None,
        # tol=args.classify and .01 or 0,
        dropna=True,
        preprocessor=args.preprocessor,
        random_seed=SEED,
        conditions='ebl<=200',
        log_proba='train',  # ignored if classify or topology is true
    )
    print(f'loaded data in {time()-start} sec\nsize: {trainset}')
    config = load(args.config)
    config['patience'] = args.patience

    if args.topology:
        train_criterion = nn.NLLLoss()

        def val_criterion(x, y):
            return -models.calculate_accuracy(x, y)
    elif args.classify:
        # bce_loss = nn.BCEWithLogitsLoss(
        #     pos_weight=torch.tensor([.5]).to(models.get_device()))  # upweight non-ILS exempla

        # def train_criterion(x, y):
        #     return bce_loss(x+1e-10, y)
        train_criterion = nn.BCELoss()

        def val_criterion(x, y):
            return -models.calculate_f1(x, y, pos=0)
    else:
        # train_criterion = nn.SmoothL1Loss(
        #     reduction='mean')
        # val_criterion = nn.MSELoss(
        #     reduction='mean')
        train_criterion = nn.BCELoss()
        # bce_loss = nn.BCEWithLogitsLoss(
        #     pos_weight=torch.tensor([.5]).to(models.get_device()))  # upweight non-ILS exempla
        # def train_criterion(x, y):
        #     return bce_loss(x+1e-10, y)
        val_criterion = train_criterion
    best_trained_model = train(
        config,
        trainset,
        valset,
        epochs=args.epochs,
        checkpoint_dir=args.outdir,
        from_scratch=args.train_from_scratch,
        criterion=train_criterion,
        val_criterion=val_criterion,
        save_best=True)


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
        #help="binarize y (p>.999 -> 1)",
    )
    parser.add_argument(
        "--classify",
        action="store_true",
        help="predict binary target (p>.999 -> 1)",
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
        help="regenerate train/test splits from db "
    )

    parser.add_argument(
        "--train_from_scratch",
        action="store_true",
        help=" train the model from scratch",
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
