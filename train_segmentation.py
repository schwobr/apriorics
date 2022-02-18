from argparse import ArgumentParser
from pathlib import Path
from torch.utils.data import DataLoader
from apriorics.plmodules import get_scheduler_func, BasicSegmentationModule
from apriorics.data import SegmentationDataset
from apriorics.transforms import StainAugmentor, ToTensor
from apriorics.models import DynamicUnet
from apriorics.metrics import DiceScore
from apriorics.losses import get_loss
from albumentations import RandomRotate90, Flip, Transpose, RandomBrightnessContrast
from pathaia.util.paths import get_files
import pandas as pd
from torchmetrics import JaccardIndex
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CometLogger
import os

parser = ArgumentParser()
parser.add_argument("--encoder")
parser.add_argument("--patch-csv-folder", type=Path)
parser.add_argument("--slidefolder", type=Path)
parser.add_argument("--maskfolder", type=Path)
parser.add_argument("--stain-matrices-folder", type=Path)
parser.add_argument("--split-csv", type=Path)
parser.add_argument("--logfolder", type=Path)
parser.add_argument("--gpu", type=int)
parser.add_argument("--batch-size", type=int, default=8)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--wd", type=float, default=1e-2)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--patch-size", type=int, default=1024)
parser.add_argument("--num-workers", type=int, default=0)
parser.add_argument("--freeze-encoder", action="store_true")
parser.add_argument("--loss", default="bce")
parser.add_argument(
    "--scheduler", choices=["one-cycle", "cosine-anneal", "reduce-on-plateau"]
)


if __name__ == "__main__":
    args = parser.parse_args()

    mask_paths = get_files(args.maskfolder, extensions=".tif", recurse=False).sorted(
        key=lambda x: x.stem
    )
    patches_paths = mask_paths.map(
        lambda x: args.patch_csv_folder / x.with_suffix(".csv").name
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
        stain_matrices_paths[train_idxs],
        stain_augmentor=StainAugmentor(),
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
        drop_last=True
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=args.num_workers,
    )

    scheduler_func = get_scheduler_func(
        args.scheduler, total_steps=len(train_dl) * args.epochs, lr=args.lr
    )
    model = DynamicUnet(
        args.encoder, n_classes=1, input_shape=(3, args.patch_size, args.patch_size)
    )
    plmodule = BasicSegmentationModule(
        model,
        loss=get_loss(args.loss),
        lr=args.lr,
        wd=args.wd,
        scheduler_func=scheduler_func,
        metrics=[JaccardIndex(2), DiceScore()],
    )

    if args.freeze_encoder:
        plmodule.freeze_encoder()

    logger = CometLogger(
        api_key=os.environ["COMET_API_KEY"],
        workspace="schwobr",
        save_dir=args.logfolder,
        project_name="apriorics-ae1ae3",
        auto_metric_logging=False,
    )

    ckpt_callback = ModelCheckpoint(
        save_top_k=3,
        monitor="val_loss",
        save_last=True,
        mode="min",
        filename="{epoch}-{val_loss:.3f}",
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[args.gpu],
        min_epochs=args.epochs,
        max_epochs=args.epochs,
        logger=logger,
        precision=16,
    )
    trainer.fit(plmodule, train_dataloaders=train_dl, val_dataloaders=val_dl)
