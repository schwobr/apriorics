import horovod.torch
from argparse import ArgumentParser
from math import ceil
from pathlib import Path
from pytorch_lightning.loggers import CometLogger
from apriorics.model_components.normalization import group_norm
from apriorics.plmodules import get_scheduler_func, BasicSegmentationModule
from apriorics.data import SegmentationDataset
from apriorics.transforms import ToTensor  # , StainAugmentor
from apriorics.metrics import DiceScore
from apriorics.losses import get_loss
from albumentations import RandomRotate90, Flip, Transpose, RandomBrightnessContrast
from pathaia.util.paths import get_files
import pandas as pd
from torchmetrics import JaccardIndex, Precision, Recall, Specificity, Accuracy
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import os
import torch
from torch.utils.data import DataLoader
from timm import create_model


parser = ArgumentParser()
parser.add_argument("--model")
parser.add_argument(
    "--ihc-type",
    choices=[
        "AE1AE3",
        "CD163",
        "CD3CD20",
        "EMD",
        "ERGCaldes",
        "ERGPodo",
        "INI1",
        "P40ColIV",
        "PHH3",
    ],
)
parser.add_argument("--patch-csv-folder", type=Path)
parser.add_argument("--slidefolder", type=Path)
parser.add_argument("--maskfolder", type=Path)
parser.add_argument("--stain-matrices-folder", type=Path)
parser.add_argument("--split-csv", type=Path)
parser.add_argument("--logfolder", type=Path)
parser.add_argument("--gpus", type=int, default=1)
parser.add_argument("--batch-size", type=int, default=8)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--wd", type=float, default=1e-2)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--patch-size", type=int, default=1024)
parser.add_argument("--num-workers", type=int, default=0)
parser.add_argument("--freeze-encoder", action="store_true")
parser.add_argument("--loss", default="bce")
parser.add_argument("--group-norm", action="store_true")
parser.add_argument(
    "--scheduler", choices=["one-cycle", "cosine-anneal", "reduce-on-plateau"]
)
parser.add_argument("--grad-accumulation", type=int, default=1)
parser.add_argument("--resume-version")

if __name__ == "__main__":
    args = parser.parse_args()

    patches_paths = get_files(
        args.patch_csv_folder, extensions=".csv", recurse=False
    ).sorted(key=lambda x: x.stem)
    mask_paths = patches_paths.map(
        lambda x: args.maskfolder / x.with_suffix(".tif").name
    )
    slide_paths = mask_paths.map(
        lambda x: args.slidefolder / x.with_suffix(".svs").name
    )
    stain_matrices_paths = mask_paths.map(
        lambda x: args.stain_matrices_folder / x.with_suffix(".npy").name
    )

    split_df = pd.read_csv(args.split_csv).sort_values("slide")
    train_idxs = (split_df["split"] == "train").values
    val_idxs = ~train_idxs

    transforms = [
        Flip(),
        Transpose(),
        RandomRotate90(),
        RandomBrightnessContrast(),
        ToTensor(),
    ]
    train_ds = SegmentationDataset(
        slide_paths[train_idxs],
        mask_paths[train_idxs],
        patches_paths[train_idxs],
        # stain_matrices_paths[train_idxs],
        # stain_augmentor=StainAugmentor(),
        transforms=transforms,
    )
    val_ds = SegmentationDataset(
        slide_paths[val_idxs],
        mask_paths[val_idxs],
        patches_paths[val_idxs],
        transforms=[ToTensor()],
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=args.num_workers,
        drop_last=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=args.num_workers,
    )

    scheduler_func = get_scheduler_func(
        args.scheduler,
        total_steps=ceil(len(train_dl) / (args.grad_accumulation * args.gpus))
        * args.epochs,
        lr=args.lr,
    )

    model = args.model.split("/")
    if model[0] == "unet":
        encoder_name = model[1]
    else:
        encoder_name = None
    model = create_model(
        model[0],
        encoder_name=encoder_name,
        pretrained=True,
        img_size=args.patch_size,
        num_classes=1,
        norm_layer=group_norm if args.group_norm else torch.nn.BatchNorm2d,
    )

    plmodule = BasicSegmentationModule(
        model,
        loss=get_loss(args.loss),
        lr=args.lr,
        wd=args.wd,
        scheduler_func=scheduler_func,
        metrics=[
            JaccardIndex(2),
            DiceScore(),
            Accuracy(),
            Precision(),
            Recall(),
            Specificity(),
        ],
    )

    if args.freeze_encoder:
        plmodule.freeze_encoder()

    logger = CometLogger(
        api_key=os.environ["COMET_API_KEY"],
        workspace="apriorics",
        save_dir=args.logfolder,
        project_name="apriorics",
        auto_metric_logging=False,
    )

    logger.experiment.add_tag(args.ihc_type)
    logger.log_graph(plmodule)

    ckpt_callback = ModelCheckpoint(
        save_top_k=3,
        monitor=f"val_loss_{args.loss}",
        save_last=True,
        mode="min",
        filename="{epoch}-{val_loss:.3f}",
    )

    trainer = pl.Trainer(
        gpus=1,
        min_epochs=args.epochs,
        max_epochs=args.epochs,
        logger=logger,
        precision=16,
        accumulate_grad_batches=args.grad_accumulation,
        callbacks=[ckpt_callback],
        strategy="horovod",
    )

    if args.resume_version is not None:
        ckpt_path = (
            args.logfolder
            / f"apriorics-ae1ae3/{args.resume_version}/checkpoints/last.ckpt"
        )
        checkpoint = torch.load(ckpt_path)
        missing, unexpected = plmodule.load_state_dict(
            checkpoint["state_dict"], strict=False
        )
    trainer.fit(
        plmodule,
        train_dataloaders=train_dl,
        val_dataloaders=val_dl,
        # ckpt_path=ckpt_path,
    )
