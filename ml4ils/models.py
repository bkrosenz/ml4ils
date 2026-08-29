"""PyTorch models and metrics used by the ML4ILS workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

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
) -> torch.Tensor:
    """Compute the F1 score for a positive class, supporting probability and logits inputs."""
    if log_proba.ndim > 1:
        preds = log_proba.argmax(dim=1)
    else:
        preds = torch.sigmoid(log_proba) > thresh
    preds = preds.reshape_as(target)
    tp = torch.logical_and(preds == pos, target == pos).sum().float()
    fp = torch.logical_and(preds == pos, target != pos).sum().float()
    fn = torch.logical_and(preds != pos, target == pos).sum().float()
    denom = tp + (fp + fn) / 2.0
    if denom == 0:
        return torch.tensor(0.0, device=target.device)
    return tp / denom


def calculate_mcc(
    log_proba: torch.Tensor, target: torch.Tensor, pos: int = 0
) -> torch.Tensor:
    """Compute the Matthews correlation coefficient from a confusion matrix."""
    preds = log_proba.argmax(dim=1).reshape_as(target)
    target = target.reshape_as(preds)
    y_pred_pos = preds == pos
    y_true_pos = target == pos
    tp = torch.logical_and(y_pred_pos, y_true_pos).sum().float()
    fp = torch.logical_and(y_pred_pos, ~y_true_pos).sum().float()
    fn = torch.logical_and(~y_pred_pos, y_true_pos).sum().float()
    tn = torch.logical_and(~y_pred_pos, ~y_true_pos).sum().float()
    denominator = torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denominator == 0:
        return torch.tensor(0.0, device=target.device)
    return (tp * tn - fp * fn) / denominator


def calculate_accuracy(log_proba: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Calculate classification accuracy from the predicted class indices."""
    return torch.mean((log_proba.argmax(dim=1) == target).float())


class Net(nn.Sequential):
    """Feed-forward network used by the training and prediction scripts."""

    def __init__(
        self,
        input_size: int,
        layer_sizes: Optional[Iterable[int]] = None,
        dropout: float = 0.1,
    ) -> None:
        """Construct a feed-forward network from input width and per-layer widths."""
        if layer_sizes is None:
            layer_sizes = [120, 849, 1]
        layer_sizes = list(layer_sizes)
        if not layer_sizes:
            raise ValueError("layer_sizes must contain at least one output width")

        layers: list[nn.Module] = []
        current_size = input_size
        for hidden_size in layer_sizes[:-1]:
            layers.extend(
                [
                    nn.Linear(current_size, hidden_size),
                    nn.ReLU(),
                ]
            )
            current_size = hidden_size

        final_size = layer_sizes[-1]
        layers.append(nn.Linear(current_size, final_size))
        if final_size > 1:
            layers.append(nn.LogSoftmax(dim=1))
        elif final_size == 1:
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
    checkpoint_dir: Path | None = None,
    with_optimizer: bool = False,
    optimizer=optim.Adam,
):
    """Load a network and optionally restore optimizer state from a checkpoint."""
    n_features = dataset if isinstance(dataset, int) else dataset[0][0].shape[0]
    net = instantiate_model(n_features, config["layers"])
    opt = None
    if with_optimizer:
        opt = optimizer(net.parameters(), lr=config["lr"])
    if checkpoint_dir:
        checkpoint_path = checkpoint_dir / "checkpoint"
        if checkpoint_path.exists():
            model_state, optimizer_state = torch.load(
                checkpoint_path, map_location=net.device
            )
            net.load_state_dict(model_state)
            if with_optimizer and optimizer_state is not None:
                opt.load_state_dict(optimizer_state)
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
