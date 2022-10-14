from typing import Any, Dict, Optional, Sequence

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
        main_clf_threshold: float = 0.5,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.iou_thresholds = ifnone(
            list(iou_thresholds), torch.arange(0.5, 1, 0.05).tolist()
        )
        self.clf_thresholds = ifnone(
            list(clf_thresholds), torch.arange(0, 1, 0.01).tolist()
        )
        self.main_clf_threshold = main_clf_threshold

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
                missing = torch.ones((len(self.iou_thresholds), n_pred, n_target))
                for k_pred in range(n_pred):
                    for k_target in range(n_target):
                        mask_pred = labels_pred == k_pred
                        mask_target = labels_target == k_target
                        iou = (mask_pred & mask_target).sum() / (
                            (mask_pred | mask_target).sum() + 1e-7
                        )
                        for j, iou_threshold in enumerate(self.iou_thresholds):
                            if iou > iou_threshold:
                                missing[j, k_pred, k_target] = 0
                self.fp[i] = missing.all(2).sum(-1)
                self.fn[i] = missing.all(1).sum(-1)

    def compute(self) -> Dict[str, torch.Tensor]:
        precisions = self.tp / (self.tp + self.fp)
        recalls = self.tp / (self.tp + self.fn)
        i_50 = self.iou_thresholds.index(0.5)
        i_75 = self.iou_thresholds.index(0.75)
        j_main = self.clf_thresholds.index(self.main_clf_threshold)
        res = {
            "precision_50": precisions[i_50, j_main],
            "precision_75": precisions[i_75, j_main],
            "precision_mean": precisions[:, j_main].mean(),
            "recall_50": recalls[i_50, j_main],
            "recall_75": recalls[i_75, j_main],
            "recall_mean": recalls[:, j_main].mean(),
            "AUPRC_50": -torch.trapz(precisions[i_50], recalls[i_50]),
            "AUPRC_75": -torch.trapz(precisions[i_75], recalls[i_75]),
            "AUPRC_mean": -torch.trapz(precisions, recalls).mean(),
        }
        return res
