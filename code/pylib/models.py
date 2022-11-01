import torch.nn as nn
from pathlib import Path
import torch
import torch.nn.functional as F

import torch.optim as optim
import torch.utils.data as data_utils


def calculate_f1(log_proba: torch.tensor,
                 target: torch.tensor,
                 pos: int = 0,
                 thresh: float = 2./3):
    """Calculate F1 score.  By default the "positive" class is 0 - ILS."""
    if len(log_proba.shape) > 1:
        preds = log_proba.argmax(1)
    else:
        preds = torch.sigmoid(log_proba) > thresh
    tp = torch.logical_and(preds == pos, target == pos).sum()
    fp = torch.logical_and(preds == pos, target != pos).sum()
    fn = torch.logical_and(preds != pos, target == pos).sum()
    return tp/(tp+(fp+fn).float()/2)


def calculate_mcc(log_proba, target, pos=0):
    preds = log_proba.argmax(1)
    tp = torch.logical_and(preds == pos, target == pos).sum()
    fp = torch.logical_and(preds == pos, target != pos).sum()
    fn = torch.logical_and(preds != pos, target == pos).sum()
    # there may be multiple non-target values
    tn = torch.logical_and(preds != pos, target == preds).sum()
    return (tp*tn-fp*fn) / ((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn).float()/2)


def calculate_accuracy(log_proba, target):
    return torch.mean((log_proba.argmax(1) == target).float())


class Net(nn.Sequential):
    def __init__(self, input_size: int,
                 layer_sizes=[120, 849, 1],
                 dropout: float = 0.1):
        layers = [nn.Linear(input_size, layer_sizes[0]),
                  nn.ReLU()]
        for input_size, output_size in zip(layer_sizes[: -2], layer_sizes[1: -1]):
            layers.extend([nn.Dropout(p=dropout),
                           nn.Linear(input_size, output_size),
                           nn.ReLU()])
        layers.append(nn.Linear(output_size, layer_sizes[-1]))
        # only add sigmoid if output is a probability
        # TODO: need to fix everything to handle logits output. pytorch BCELossWithLogits is unstable.
        if layer_sizes[-1] > 1:
            layers.append(nn.LogSoftmax(dim=1))
        elif layer_sizes[-1] == 1:
            layers.append(nn.Sigmoid())

        super().__init__(*layers)
        device = get_device()
        self.to(device)
        self.device = device


def instantiate_model(dim, layers, clip_value: float = 10) -> Net:
    net = Net(dim, layers, dropout=.05)
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
    print(f'loaded net {net} to {net.device}')
    for p in net.parameters():
        p.register_hook(lambda grad: torch.clamp(
            grad, -clip_value, clip_value))

    return net


def get_device():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    return device  # torch.device(device)


def load_model(config: dict,
               dataset,
               checkpoint_dir: Path = None,
               with_optimizer: bool = False,
               optimizer=optim.Adam):
    """load a network, and optionally an optimizer state"""
    if isinstance(dataset, int):
        n_features = dataset
    else:
        n_features = dataset[0][0].shape[0]
    net = instantiate_model(n_features, config["layers"])
    if with_optimizer:
        # optimizer = optim.SGD(net.parameters(), lr=config["lr"], momentum=0.9)
        opt = optimizer(net.parameters(), lr=config["lr"])

    if checkpoint_dir:
        checkpoint_path = checkpoint_dir / "checkpoint"
        if checkpoint_path.exists():
            model_state, optimizer_state = torch.load(
                checkpoint_path,
                map_location=net.device)
            net.load_state_dict(model_state)
            print('loaded model state')
    if with_optimizer:
        return net, opt
    return net


def test_error(net: Net,
               testset: data_utils.TensorDataset,
               loss=F.mse_loss,
               predict: bool = False):
    '''must call net.eval() first.
    Loss function must be additive.'''
    device = net.device
    testloader = data_utils.DataLoader(
        testset,
        batch_size=256,
        shuffle=False,
        num_workers=1)
    mse = 0.
    total = 0
    preds = []
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            total += labels.size(0)
            if predict:
                preds.append(outputs)
            mse += loss(outputs.squeeze(), labels, reduce='sum')
    mse /= total
    if predict:
        return mse, torch.cat(preds)
    return mse


def df_to_tensor(df):
    """convert a df to tensor to be used in pytorch"""
    device = get_device()
    return torch.from_numpy(df.values).float().to(device)


def corrcoef(x: torch.Tensor, y: torch.Tensor):
    x = x.squeeze()
    y = y.squeeze().to(x.device)
    N = x.shape[0]
    if N != y.shape[0]:
        raise ValueError(f'{x.shape}, {y.shape}')
    mx, my = x.mean(), y.mean()
    return torch.dot(x-mx, y-my)/x.std()/y.std()/N**2
