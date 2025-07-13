import numpy as np
from sympy.printing.pytorch import torch
from torch import nn


def get_loss(loss_name):
    if loss_name == "spearman":
        return spearman_loss
    elif loss_name == "mse":
        return weighted_mse_loss
    elif loss_name == "huber":
        return weighted_huber_loss

    assert False


def spearman_loss(pred, target, factor=0.1):
    import torchsort

    pred = torchsort.soft_rank(pred.transpose(0, 1)).transpose(0, 1)
    target = torchsort.soft_rank(target.transpose(0, 1)).transpose(0, 1)

    eps = 1e-8
    pred = pred - pred.mean()
    pred = pred / (eps + pred.norm())

    num_extra_targets = 0 if len(pred.shape) == 1 else pred.shape[1] - 1

    target = target - target.mean()
    target = target / (eps + target.norm())

    weights = np.insert(
        (1 - factor) * np.ones(num_extra_targets) / num_extra_targets, 0, factor, axis=0
    )
    weights = torch.tensor(weights, device=pred.device)

    return -((pred * target).sum(dim=0) * weights).sum()
    return -(pred * target).sum()


def weighted_generic_loss(criterion, pred, target, factor=0.1):
    target = torch.nan_to_num(target, nan=0.0)
    pred = torch.nan_to_num(pred, nan=0.0)

    num_extra_targets = 0 if len(pred.shape) == 1 else pred.shape[1] - 1

    loss = criterion(pred, target)

    weights = np.insert(
        (1 - factor) * np.ones(num_extra_targets) / num_extra_targets, 0, factor, axis=0
    )

    weights = torch.tensor(weights, device=pred.device)
    weighted_loss = loss * weights
    return weighted_loss.mean()


def weighted_huber_loss(pred, target, factor=0.1):
    criterion = nn.HuberLoss(reduction="none", delta=0.1)
    return weighted_generic_loss(criterion, pred, target, factor)


def weighted_mse_loss(pred, target, factor=0.1):
    criterion = nn.MSELoss(reduction="none")
    return weighted_generic_loss(criterion, pred, target, factor)


def custom_loss(pred, target, factor=0.1):
    return 0.25 * spearman_loss(pred, target, factor) + 0.75 * weighted_mse_loss(
        pred, target, factor
    )
