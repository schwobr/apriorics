from torchmetrics import CatMetric


def _reduce(x, reduction="mean"):
    if reduction == "mean":
        return x.mean()
    elif reduction == "sum":
        return x.sum()
    else:
        return x


def _flatten(x):
    return x.view(x.shape[0], -1)


def dice_score(input, target, smooth=1, reduction="mean"):
    target = _flatten(target).to(dtype=input.dtype)
    input = _flatten(input)
    inter = (target * input).sum(-1)
    sum = target.sum(-1) + input.sum(-1)
    dice = (inter + smooth) / (sum + smooth)
    return _reduce(1 - dice, reduction=reduction)


class DiceScore(CatMetric):
    def __init__(self, smooth=1, reduction="mean", **kwargs):
        super().__init__(**kwargs)
        self.smooth = smooth
        self.reduction = reduction

    def update(self, input, target):
        dice = dice_score(input, target, smooth=self.smooth, reduction=self.reduction)
        super().update(dice)

    def compute(self):
        dices = super().compute()
        return _reduce(dices, reduction=self.reduction)
