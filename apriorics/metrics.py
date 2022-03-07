from typing import Optional
from torchmetrics import CatMetric
import torch


def _reduce(x: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    r"""
    Optionally reduces input tensor by either averaging or summing its values.

    Args:
        x: input tensor.
        reduction: reduction method, either "mean", "sum" or "none".

    Returns:
        Reduced version of x.
    """
    if reduction == "mean":
        return x.mean()
    elif reduction == "sum":
        return x.sum()
    else:
        return x


def _flatten(x: torch.Tensor) -> torch.Tensor:
    r"""
    Flattens input tensor but keeps first dimension.

    Args:
        x: input tensor of shape (N, ...).

    Returns:
        Flattened version of `x` of shape (N, .).
    """
    return x.view(x.shape[0], -1)


def dice_score(
    input: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1,
    reduction: str = "mean",
) -> torch.Tensor:
    r"""
    Computes dice score (given by :math:`D(p, t) = \frac{2|pt|+s}{|p|+|t|+s}`) between
    predicted input tensor and target ground truth.

    Args:
        input: predicted input tensor of shape (N, ...).
        target: target ground truth tensor of shape (N, ...).
        smooth: smooth value for dice score.
        reduction: reduction method for computed dice scores. Can be one of "mean",
            "sum" or "none".

    Returns:
             Computed dice score, optionally reduced using specified reduction method.
    """
    target = _flatten(target).to(dtype=input.dtype)
    input = _flatten(input)
    inter = (target * input).sum(-1)
    sum = target.sum(-1) + input.sum(-1)
    dice = 2 * (inter + smooth) / (sum + smooth)
    return _reduce(dice, reduction=reduction)


class DiceScore(CatMetric):
    r"""
    `torch.nn.Module` for dice loss (given by
    :math:`D(p, t) = \frac{2|pt|+s}{|p|+|t|+s}`) computation.

    Args:
        smooth: smooth value for dice score.
        reduction: reduction method for computed dice scores. Can be one of "mean",
            "sum" or "none".
    """
    def __init__(self, smooth: float = 1, reduction: str = "mean", **kwargs):
        super().__init__(**kwargs)
        self.smooth = smooth
        self.reduction = reduction

    def update(
        self, input: torch.Tensor, target: torch.Tensor
    ) -> Optional[torch.Tensor]:
        dice = dice_score(input, target, smooth=self.smooth, reduction=self.reduction)
        return super().update(dice)

    def compute(self) -> torch.Tensor:
        dices = super().compute()
        dices = torch.as_tensor(dices)
        return _reduce(dices, reduction=self.reduction)
