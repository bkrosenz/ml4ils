"""PyTorch models and metrics used by the ML4ILS workflows."""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils


def calculate_f1(
    log_proba: torch.Tensor,
    target: torch.Tensor,
    pos: int = 0,
    thresh: float = 2.0 / 3,
):
    """Calculate the F1 score for the selected positive class."""
    if len(log_proba.shape) > 1:
        preds = log_proba.argmax(1)
    else:
        preds = torch.sigmoid(log_proba) > thresh
    tp = torch.logical_and(preds == pos, target == pos).sum()
    fp = torch.logical_and(preds == pos, target != pos).sum()
    fn = torch.logical_and(preds != pos, target == pos).sum()
    return tp / (tp + (fp + fn).float() / 2)


def calculate_mcc(log_proba, target, pos=0):
    """Calculate the Matthews correlation coefficient."""
    preds = log_proba.argmax(1)
    tp = torch.logical_and(preds == pos, target == pos).sum()
    fp = torch.logical_and(preds == pos, target != pos).sum()
    fn = torch.logical_and(preds != pos, target == pos).sum()
    tn = torch.logical_and(preds != pos, target == preds).sum()
    return (tp * tn - fp * fn) / (
        (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn).float() / 2
    )


def calculate_accuracy(log_proba, target):
    """Calculate classification accuracy from log probabilities."""
    return torch.mean((log_proba.argmax(1) == target).float())


class Net(nn.Sequential):
    """Feed-forward network used by the training and prediction scripts."""

    def __init__(self, input_size: int, layer_sizes=None, dropout: float = 0.1):
        if layer_sizes is None:
            layer_sizes = [120, 849, 1]
        layers = [nn.Linear(input_size, layer_sizes[0]), nn.ReLU()]
        for input_size, output_size in zip(layer_sizes[:-2], layer_sizes[1:-1]):
            layers.extend(
                [nn.Dropout(p=dropout), nn.Linear(input_size, output_size), nn.ReLU()]
            )
        layers.append(nn.Linear(output_size, layer_sizes[-1]))
        if layer_sizes[-1] > 1:
            layers.append(nn.LogSoftmax(dim=1))
        elif layer_sizes[-1] == 1:
            layers.append(nn.Sigmoid())
        super().__init__(*layers)
        self.device = get_device()
        self.to(self.device)


def instantiate_model(dim, layers, clip_value: float = 10):
    """Build a model and attach gradient clipping hooks."""
    net = Net(dim, layers, dropout=0.05)
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
    print(f"loaded net {net} to {net.device}")
    for parameter in net.parameters():
        parameter.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))
    return net


def get_device():
    """Return the preferred torch device name."""
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def load_model(
    config: dict,
    dataset,
    checkpoint_dir: Path = None,
    with_optimizer: bool = False,
    optimizer=optim.Adam,
):
    """Load a network and optionally its optimizer state."""
    n_features = dataset if isinstance(dataset, int) else dataset[0][0].shape[0]
    net = instantiate_model(n_features, config["layers"])
    if with_optimizer:
        opt = optimizer(net.parameters(), lr=config["lr"])
    if checkpoint_dir:
        checkpoint_path = checkpoint_dir / "checkpoint"
        if checkpoint_path.exists():
            model_state, optimizer_state = torch.load(
                checkpoint_path, map_location=net.device
            )
            net.load_state_dict(model_state)
            print("loaded model state")
    if with_optimizer:
        return net, opt
    return net


def test_error(
    net, testset: data_utils.TensorDataset, loss=F.mse_loss, predict: bool = False
):
    """Evaluate additive loss over a tensor dataset."""
    device = net.device
    testloader = data_utils.DataLoader(
        testset, batch_size=256, shuffle=False, num_workers=1
    )
    mse = 0.0
    total = 0
    preds = []
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            total += labels.size(0)
            if predict:
                preds.append(outputs)
            mse += loss(outputs.squeeze(), labels, reduction="sum")
    mse /= total
    if predict:
        return mse, torch.cat(preds)
    return mse


def df_to_tensor(df):
    """Convert a dataframe to a tensor on the active torch device."""
    return torch.from_numpy(df.values).float().to(get_device())


def corrcoef(x: torch.Tensor, y: torch.Tensor):
    """Calculate the repository's normalized correlation statistic."""
    x = x.squeeze()
    y = y.squeeze().to(x.device)
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"{x.shape}, {y.shape}")
    mx, my = x.mean(), y.mean()
    return torch.dot(x - mx, y - my) / x.std() / y.std() / x.shape[0] ** 2
