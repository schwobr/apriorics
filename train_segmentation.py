import horovod.torch as hvd
from argparse import ArgumentParser
from math import ceil
from pathlib import Path
from pytorch_lightning.loggers import CometLogger
from apriorics.model_components.normalization import group_norm
from apriorics.plmodules import get_scheduler_func, BasicSegmentationModule
from apriorics.data import SegmentationDataset, BalancedRandomSampler
from apriorics.transforms import ToTensor, StainAugmentor
from apriorics.metrics import DiceScore
from apriorics.losses import get_loss
from albumentations import (
    RandomRotate90,
    Flip,
    Transpose,
    RandomBrightnessContrast,
    RandomCrop,
    CenterCrop,
)
from pathaia.util.paths import get_files
import pandas as pd
from torchmetrics import JaccardIndex, Precision, Recall, Specificity, Accuracy
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import os
import torch
from torch.utils.data import DataLoader
from timm import create_model
from pytorch_lightning.utilities.seed import seed_everything

IHCS = [
    "AE1AE3",
    "CD163",
    "CD3CD20",
    "EMD",
    "ERGCaldes",
    "ERGPodo",
    "INI1",
    "P40ColIV",
    "PHH3",
]

parser = ArgumentParser(
    prog=(
        "Train a segmentation model for a specific IHC. To train on multiple gpus, "
        "should be called as `horovodrun -np n_gpus python train_segmentation.py "
        "--horovod`."
    )
)
parser.add_argument(
    "--model",
    help=(
        "Model to use for training. If unet, can be formatted as unet/encoder to "
        "specify a specific encoder. Must be one of unet, med_t, logo, axalunet, gated."
    ),
    required=True,
)
parser.add_argument(
    "--ihc-type",
    choices=IHCS,
    help=f"Name of the IHC to train for. Must be one of {', '.join(IHCS)}.",
    required=True,
)
parser.add_argument(
    "--patch-csv-folder",
    type=Path,
    help="Input folder containing PathAIA csv files.",
    required=True,
)
parser.add_argument(
    "--slidefolder",
    type=Path,
    help="Input folder containing svs slide files.",
    required=True,
)
parser.add_argument(
    "--maskfolder",
    type=Path,
    help="Input folder containing tif mask files.",
    required=True,
)
parser.add_argument(
    "--stain-matrices-folder",
    type=Path,
    help=(
        "Input folder containing npy stain matrices files for stain augmentation. "
        "Optional."
    ),
)
parser.add_argument(
    "--split-csv",
    type=Path,
    help="Input csv file for dataset split containing 2 columns: slide and split.",
    required=True,
)
parser.add_argument(
    "--logfolder",
    type=Path,
    help="Output folder for pytorch lightning log files.",
    required=True,
)
parser.add_argument(
    "--gpu",
    type=int,
    default=0,
    help="GPU index to used when not using horovod. Default 0.",
)
parser.add_argument(
    "--horovod",
    action="store_true",
    help="Specify when using script with horovodrun. Optional.",
)
parser.add_argument(
    "--batch-size",
    type=int,
    default=8,
    help=(
        "Batch size for training. effective batch size is multiplied by the number of"
        " gpus. Default 8."
    ),
)
parser.add_argument(
    "--lr", type=float, default=1e-3, help="Learning rate for training. Default 1e-3."
)
parser.add_argument(
    "--wd",
    type=float,
    default=1e-2,
    help="Weight decay for AdamW optimizer. Default 1e-2.",
)
parser.add_argument(
    "--epochs", type=int, default=10, help="Number of epochs to train on. Default 10."
)
parser.add_argument(
    "--patch-size",
    type=int,
    default=1024,
    help="Size of the patches used foor training. Default 1024.",
)
parser.add_argument(
    "--num-workers",
    type=int,
    default=0,
    help="Number of workers to use for data loading. Default 0 (only main process).",
)
parser.add_argument(
    "--freeze-encoder",
    action="store_true",
    help="Specify to freeze encoder when using unet model. Optional.",
)
parser.add_argument(
    "--loss",
    default="bce",
    help=(
        "Loss function to use for training. Must be one of bce, focal, dice, "
        "sum_loss1_coef1_****. Default bce."
    ),
)
parser.add_argument(
    "--group-norm",
    action="store_true",
    help="Specify to use group norm instead of batch norm in model. Optional.",
)
parser.add_argument(
    "--scheduler",
    choices=["one-cycle", "cosine-anneal", "reduce-on-plateau"],
    help=(
        "Learning rate scheduler to use during training. Must be one of one-cycle, "
        "cosine-anneal, reduce-on-plateau. Optional."
    ),
)
parser.add_argument(
    "--grad-accumulation",
    type=int,
    default=1,
    help="Number of batches to accumulate gradients on. Default 1.",
)
parser.add_argument(
    "--resume-version", help="Version id of a model to load weights from. Optional."
)
parser.add_argument(
    "--seed",
    type=int,
    help=(
        "Specify seed for RNG. Can also be set using PL_GLOBAL_SEED environment "
        "variable. Optional."
    ),
)
parser.add_argument(
    "--augment-stain",
    action="store_true",
    help="Specify to use stain augmentation. Optional.",
)

if __name__ == "__main__":
    args = parser.parse_args()
    if args.horovod:
        hvd.init()

    seed_everything(workers=True)

    patches_paths = get_files(
        args.patch_csv_folder, extensions=".csv", recurse=False
    ).sorted(key=lambda x: x.stem)
    mask_paths = patches_paths.map(
        lambda x: args.maskfolder / x.with_suffix(".tif").name
    )
    slide_paths = mask_paths.map(
        lambda x: args.slidefolder / x.with_suffix(".svs").name
    )

    split_df = pd.read_csv(args.split_csv).sort_values("slide")
    train_idxs = (split_df["split"] == "train").values
    val_idxs = ~train_idxs

    if args.stain_matrices_folder is not None:
        stain_matrices_paths = mask_paths.map(
            lambda x: args.stain_matrices_folder / x.with_suffix(".npy").name
        )
        stain_matrices_paths = stain_matrices_paths[train_idxs]
    else:
        stain_matrices_paths = None

    transforms = [
        RandomCrop(args.patch_size, args.patch_size),
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
        stain_matrices_paths,
        stain_augmentor=StainAugmentor() if args.augment_stain else None,
        transforms=transforms,
    )
    val_ds = SegmentationDataset(
        slide_paths[val_idxs],
        mask_paths[val_idxs],
        patches_paths[val_idxs],
        transforms=[CenterCrop(args.patch_size, args.patch_size), ToTensor()],
    )

    sampler = BalancedRandomSampler(train_ds, p_pos=0.95)
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.num_workers,
        drop_last=True,
        sampler=sampler,
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
        total_steps=ceil(len(train_dl) / (args.grad_accumulation)) * args.epochs,
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

    if not args.horovod or hvd.rank() == 0:
        logger.experiment.add_tag(args.ihc_type)

    ckpt_callback = ModelCheckpoint(
        save_top_k=3,
        monitor=f"val_loss_{args.loss}",
        save_last=True,
        mode="min",
        filename="{epoch}-{val_loss:.3f}",
    )

    trainer = pl.Trainer(
        gpus=1 if args.horovod else [args.gpu],
        min_epochs=args.epochs,
        max_epochs=args.epochs,
        logger=logger,
        precision=16,
        accumulate_grad_batches=args.grad_accumulation,
        callbacks=[ckpt_callback],
        strategy="horovod" if args.horovod else None,
    )

    if args.resume_version is not None:
        ckpt_path = (
            args.logfolder / f"apriorics/{args.resume_version}/checkpoints/last.ckpt"
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
