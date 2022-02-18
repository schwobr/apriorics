import torch
import torch.nn as nn
from .metrics import _flatten, _reduce, dice_score


def get_loss(name: str):
    split_name = name.split("_")
    if split_name[0] == "bce":
        return nn.BCEWithLogitsLoss()
    elif split_name[0] == "focal":
        return FocalLoss()
    elif split_name[0] == "dice":
        return DiceLoss()
    elif split_name[0] == "sum":
        names = split_name[1::2]
        coefs = map(float, split_name[2::2])
        losses_with_coefs = [(get_loss(n), c) for n, c in zip(names, coefs)]
        return SumLosses(*losses_with_coefs)
    else:
        raise ValueError(f"{name} not recognized as a loss function")


def dice_loss(input, target, smooth=1, reduction="mean"):
    input = torch.sigmoid(input)
    dice = dice_score(input, target, smooth=smooth, reduction=reduction)
    return 1 - dice


def focal_loss(input, target, reduction="mean", beta=0.5, gamma=2.0, eps=1e-7):
    target = _flatten(target).to(dtype=input.dtype)
    input = _flatten(input)
    input = torch.sigmoid(input).clamp(eps, 1 - eps)
    focal = -(
        beta * target * (1 - input).pow(gamma) * input.log()
        + (1 - beta) * (1 - target) * input.pow(gamma) * (1 - input).log()
    ).mean(-1)
    return _reduce(focal, reduction=reduction)


class FocalLoss(nn.Module):
    def __init__(self, beta=0.5, gamma=2.0, reduction="mean"):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):
        loss = focal_loss(
            input, target, beta=self.beta, gamma=self.gamma, reduction=self.reduction
        )
        return loss


class DiceLoss(nn.Module):
    def __init__(self, smooth=1, reduction="mean"):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, input, target):
        loss = dice_loss(input, target, smooth=self.smooth, reduction=self.reduction)
        return loss


class SumLosses(nn.Module):
    def __init__(self, *losses_with_coef):
        super().__init__()
        self.losses_with_coef = losses_with_coef

    def forward(self, input, target):
        loss = 0
        for loss_func, coef in self.losses_with_coef:
            loss += coef * loss_func(input, target)
        return loss
