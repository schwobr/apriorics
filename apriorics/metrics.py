from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
from pathaia.util.basic import ifnone
from skimage.measure import label
from torchmetrics import CatMetric, Metric, MetricCollection


@torch.jit.unused
def _forward(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Iteratively call forward for each metric.

    Positional arguments (args) will be passed to every metric in the collection, while
    keyword arguments (kwargs) will be filtered based on the signature of the individual
    metric.
    """
    res = {}
    for k, m in self.items():
        out = m(*args, **m._filter_kwargs(**kwargs))
        if isinstance(out, dict):
            res.update(out)
        else:
            res[k] = out
    return res


def _compute(self) -> Dict[str, Any]:
    res = {}
    for k, m in self.items():
        out = m.compute()
        if isinstance(out, dict):
            res.update(out)
        else:
            res[k] = out
    return res


MetricCollection.forward = _forward
MetricCollection.compute = _compute


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


class DetectionSegmentationMetrics(Metric):
    def __init__(
        self,
        iou_thresholds: Optional[Sequence[float]] = None,
        clf_thresholds: Optional[Sequence[float]] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.iou_thresholds = ifnone(iou_thresholds, torch.arange(0, 1, 0.05).tolist())
        self.clf_thresholds = ifnone(clf_thresholds, torch.arange(0, 1, 0.05).tolist())

        self.add_state(
            "tp",
            default=torch.zeros((len(self.clf_thresholds), len(self.iou_thresholds))),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "fp",
            default=torch.zeros((len(self.clf_thresholds), len(self.iou_thresholds))),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "fn",
            default=torch.zeros((len(self.clf_thresholds), len(self.iou_thresholds))),
            dist_reduce_fx="sum",
        )

    def update(self, input: torch.Tensor, target: torch.Tensor):
        for y, y_hat in zip(input, target):
            for i, clf_threshold in enumerate(self.clf_thresholds):
                labels_pred, n_pred = label(
                    (y > clf_threshold).detach().cpu().numpy(), return_num=True
                )
                labels_target, n_target = label(
                    y_hat.bool().cpu().numpy(), return_num=True
                )
                if n_target == 0:
                    self.fp[i] += n_pred
                    continue
                if n_pred == 0:
                    self.fn[i] += n_target
                    continue
                missing = torch.ones(
                    (len(self.iou_thresholds), n_pred, n_target), device=self.tp.device
                )
                for k_pred in range(n_pred):
                    mask_pred = labels_pred == (k_pred + 1)
                    ii, jj = np.nonzero(mask_pred)
                    y0, y1 = ii.min(), ii.max()
                    x0, x1 = jj.min(), jj.max()
                    area_pred = (y1 - y0 + 1) * (x1 - x0 + 1)
                    bbox_pred = np.array([x0, y0, x1, y1])
                    for k_target in range(n_target):
                        mask_target = labels_target == (k_target + 1)
                        ii, jj = np.nonzero(mask_target)
                        y0, y1 = ii.min(), ii.max()
                        x0, x1 = jj.min(), jj.max()
                        area_target = (y1 - y0 + 1) * (x1 - x0 + 1)
                        bbox_target = np.array([x0, y0, x1, y1])
                        bbox_inter = np.concatenate(
                            (
                                np.maximum(bbox_pred[:2], bbox_target[:2]),
                                np.minimum(bbox_pred[2:], bbox_target[2:]),
                            )
                        )
                        x0, y0, x1, y1 = bbox_inter
                        area_inter = (y1 - y0 + 1) * (x1 - x0 + 1)
                        iou = area_inter / (area_pred + area_target - area_inter + 1e-7)
                        missing[iou > self.iou_thresholds, k_pred, k_target] = 0
                self.fp[i] += missing.all(2).sum(-1)
                self.fn[i] += missing.all(1).sum(-1)
                self.tp[i] += (1 - missing).any(1).sum(-1)

    def compute(self) -> Dict[str, torch.Tensor]:
        precisions = self.tp / (self.tp + self.fp + 1e-7)
        recalls = self.tp / (self.tp + self.fn + 1e-7)
        i_25 = self.clf_thresholds.index(0.25)
        i_50 = self.clf_thresholds.index(0.5)
        i_75 = self.clf_thresholds.index(0.75)
        j_25 = self.iou_thresholds.index(0.25)
        j_50 = self.iou_thresholds.index(0.5)
        j_75 = self.iou_thresholds.index(0.75)
        res = {
            "precision_25": precisions[i_50, j_25],
            "precision_50": precisions[i_50, j_50],
            "precision_75": precisions[i_50, j_75],
            "precision_50+": precisions[i_50, j_50:].mean(),
            "recall_25": recalls[i_50, j_25],
            "recall_50": recalls[i_50, j_50],
            "recall_75": recalls[i_50, j_75],
            "recall_50+": recalls[i_50, j_50:].mean(),
            "AUPRC.25": -torch.trapz(
                precisions[
                    i_25,
                ],
                recalls[i_25],
            ),
            "AUPRC.5": -torch.trapz(
                precisions[
                    i_50,
                ],
                recalls[i_50],
            ),
            "AUPRC.75": -torch.trapz(
                precisions[
                    i_75,
                ],
                recalls[i_75],
            ),
        }
        return res
