from typing import List, Optional
import pytorch_lightning as pl

from torch import Tensor, nn
from torch.optim import Optimizer, AdamW
from torch.optim.lr_scheduler import OneCycleLR,CosineAnnealingLR, ReduceLROnPlateau
import torch
from torchmetrics.functional import dice_score, f1_score

import torchvision

def dice_coef(y_true, y_pred, smooth=1):
    intersection = torch.sum(y_true * y_pred, axis=[1,2,3])
    sum = torch.sum(y_true, axis=[1,2,3]) + torch.sum(y_pred, axis=[1,2,3])
    dice = torch.mean((2. * intersection + smooth)/(sum + smooth), axis=0)
    return dice


def jaccard_index(y_true, y_pred, smooth=1):
    intersection =  torch.sum(y_true * y_pred, axis=[1,2,3])
    union = torch.sum(y_true, axis=[1,2,3]) + torch.sum(y_pred, axis=[1,2,3]) - intersection
    return torch.mean((2. * intersection + smooth)/(union + smooth), axis=0)
 



def _get_scheduler(
    opt: Optimizer, name: str, total_steps: int, lr: float
) -> torch.optim.lr_scheduler._LRScheduler:
    if name == "one-cycle":
        sched = {
            "scheduler": OneCycleLR(opt, lr, total_steps=total_steps),
            "interval": "step",
        }
    elif name == "cosine-anneal":
        sched = {"scheduler": CosineAnnealingLR(opt, total_steps), "interval": "step"}
    elif name == "reduce-on-plateau":
        sched = {
            "scheduler": ReduceLROnPlateau(opt, patience=3),
            "interval": "epoch",
            "monitor": "val_loss",
        }
    else:
        return None
    return sched


class BasicSegmentationModule(pl.LightningModule):
    def __init__(self, model: nn.Module, lr:float, wd:float, max_step:int):
        super().__init__()
        self.model = model
        self.lr = lr
        self.wd = wd
        self.max_step = max_step

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx) -> Tensor:
        x, y = batch
        y_hat = self(x)
        loss = torch.tensor(1.) - dice_coef(y_hat, y)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("learning_rate", self.sched["scheduler"].get_last_lr()[0], on_step=True, on_epoch=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx, *args, **kwargs) -> Tensor:
        x, y = batch
        y_hat = self(x)
        dice = 1. - dice_coef(y_hat, y)
        iou = jaccard_index(y_hat, y.int())

        if batch_idx == 0: 
            sample_imgs = torch.cat((x, y.repeat(1,3,1,1), y_hat.repeat(1,3,1,1)),0)
            grid = torchvision.utils.make_grid(sample_imgs, y.shape[0])
            self.logger.experiment.log_image(torchvision.transforms.functional.to_pil_image(grid),'val_image_sample', step=self.current_epoch)

        self.log("val_dice_score", dice, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_IoU", iou, on_step=True, on_epoch=True, prog_bar=True, logger=True, )
        return {"dice_score" : dice, "IoU": iou}        

    def configure_optimizers(self):
        self.opt = AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.wd
        )
        self.sched = _get_scheduler(self.opt, "one-cycle", self.max_step, self.lr)
        return {"optimizer": self.opt, "lr_scheduler": self.sched}

