import pytorch_lightning as pl
from typing import Optional, Dict, Callable
from torch import Tensor, nn
from torch.optim import Optimizer, AdamW
import torch
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR, ReduceLROnPlateau
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image
from torchmetrics import Metric
from pathaia.util.basic import ifnone
from .model_components.utils import named_leaf_modules


def get_scheduler_func(
    name: str, total_steps: int, lr: float
) -> torch.optim.lr_scheduler._LRScheduler:
    def scheduler_func(opt: Optimizer):
        if name == "one-cycle":
            sched = {
                "scheduler": OneCycleLR(opt, lr, total_steps=total_steps),
                "interval": "step",
            }
        elif name == "cosine-anneal":
            sched = {
                "scheduler": CosineAnnealingLR(opt, total_steps),
                "interval": "step",
            }
        elif name == "reduce-on-plateau":
            sched = {
                "scheduler": ReduceLROnPlateau(opt, patience=3),
                "interval": "epoch",
                "monitor": "val_loss",
            }
        else:
            return None
        return sched

    return scheduler_func


class BasicSegmentationModule(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        loss: nn.Module,
        lr: float,
        wd: float,
        scheduler_func: Optional[Callable] = None,
        metrics: Optional[Dict[str, Metric]] = None,
    ):
        super().__init__()
        self.model = model
        self.loss = loss
        self.lr = lr
        self.wd = wd
        self.scheduler_func = scheduler_func
        self.metrics = ifnone(metrics, [])

    def forward(self, x):
        return torch.sigmoid(self.model(x)).squeeze(1)

    def training_step(self, batch, batch_idx) -> Tensor:
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)

        self.log("train_loss", loss)
        if self.sched is not None:
            self.log("learning_rate", self.sched["scheduler"].get_last_lr()[0])
        return loss

    def validation_step(self, batch, batch_idx, *args, **kwargs) -> Tensor:
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        log_dict = {"val_loss": loss}

        if batch_idx == 0:
            self.log_images(x, y, y_hat)

        for metric_name, metric_func in self.metrics.items():
            log_dict[metric_name] = metric_func(y_hat, y)

        self.log_dict(log_dict)

    def configure_optimizers(self):
        self.opt = AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
        if self.scheduler_func is None:
            self.sched = None
            return self.opt
        else:
            self.sched = self.scheduler_func(self.opt)
            return {"optimizer": self.opt, "lr_scheduler": self.sched}

    def log_images(self, x, y, y_hat):
        sample_imgs = torch.cat((x, y.repeat(1, 3, 1, 1), y_hat.repeat(1, 3, 1, 1)), 0)
        grid = make_grid(sample_imgs, y.shape[0])
        self.logger.experiment.log_image(
            to_pil_image(grid),
            "val_image_sample",
            step=self.current_epoch,
        )

    def freeze_encoder(self):
        for m in named_leaf_modules(self.model):
            if "encoder" in m.name and not isinstance(m, nn.BatchNorm2d):
                for param in m.parameters():
                    param.requires_grad = False
