from argparse import ArgumentParser
from pathlib import Path

from tqdm import tqdm
from apriorics.model_components.normalization import group_norm
from apriorics.plmodules import BasicSegmentationModule
from apriorics.data import TestDataset
from apriorics.polygons import mask_to_polygons_layer
from apriorics.transforms import ToTensor
from albumentations import CenterCrop
from albumentations.augmentations.crops.functional import get_center_crop_coords
from pathaia.util.paths import get_files
import pandas as pd
import torch
from torch.utils.data import DataLoader
from timm import create_model
from pytorch_lightning.utilities.seed import seed_everything
from shapely.affinity import translate
from shapely.ops import unary_union
from shapely.geometry import Polygon, MultiPolygon
import json
import geopandas


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
    "--outfolder", type=Path, help="Output folder for geojsons.", required=True
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
    "--batch-size",
    type=int,
    default=8,
    help=(
        "Batch size for training. effective batch size is multiplied by the number of"
        " gpus. Default 8."
    ),
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
    "--group-norm",
    action="store_true",
    help="Specify to use group norm instead of batch norm in model. Optional.",
)
parser.add_argument(
    "--version", help="Version id of a model to load weights from.", required=True
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
    "--area-threshold",
    type=int,
    default=50,
    help="Minimum area of objects to keep. Default 50.",
)

if __name__ == "__main__":
    args = parser.parse_args()

    seed_everything(workers=True)

    patches_paths = get_files(
        args.patch_csv_folder, extensions=".csv", recurse=False
    ).sorted(key=lambda x: x.stem)
    slide_paths = patches_paths.map(
        lambda x: args.slidefolder / x.with_suffix(".svs").name
    )

    split_df = pd.read_csv(args.split_csv).sort_values("slide")
    train_idxs = (split_df["split"] == "train").values
    val_idxs = ~train_idxs

    model = args.model.split("/")
    if model[0] == "unet":
        encoder_name = model[1]
    else:
        encoder_name = None

    device = torch.device(f"cuda:{args.gpu}")
    model = create_model(
        model[0],
        encoder_name=encoder_name,
        pretrained=True,
        img_size=args.patch_size,
        num_classes=1,
        norm_layer=group_norm if args.group_norm else torch.nn.BatchNorm2d,
    ).eval()
    model.requires_grad_(False)

    model = BasicSegmentationModule(
        model,
        loss=None,
        lr=0,
        wd=0,
    ).to(device)

    ckpt_path = args.logfolder / f"apriorics/{args.version}/checkpoints/last.ckpt"
    checkpoint = torch.load(ckpt_path)
    missing, unexpected = model.load_state_dict(checkpoint["state_dict"], strict=False)

    for slide_path, patches_path in zip(slide_paths[val_idxs], patches_paths[val_idxs]):
        print(slide_path.stem)
        ds = TestDataset(
            slide_path,
            patches_path,
            [CenterCrop(args.patch_size, args.patch_size), ToTensor()],
        )
        dl = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        polygons = []
        for batch_idx, x in tqdm(enumerate(dl), total=len(dl)):
            y = torch.sigmoid(model(x.to(device))).squeeze(1).cpu().numpy() > 0.5
            for k, mask in enumerate(y):
                if not mask.sum():
                    continue
                idx = batch_idx * args.batch_size + k
                patch = ds.patches[idx]
                polygon = mask_to_polygons_layer(mask, angle_th=0, distance_th=0)
                x1, y1, _, _ = get_center_crop_coords(
                    patch.size.y, patch.size.x, args.patch_size, args.patch_size
                )
                polygon = translate(
                    polygon, xoff=patch.position.x + x1, yoff=patch.position.y + y1
                )

                if isinstance(polygon, Polygon) and polygon.area > args.area_threshold:
                    polygons.append(polygon)
                elif isinstance(polygon, MultiPolygon):
                    for pol in polygon.geoms:
                        if pol.area > args.area_threshold:
                            polygons.append(pol)
        polygons = unary_union(polygons)
        if isinstance(polygons, Polygon):
            polygons = MultiPolygon(polygons=[polygons])
        with open(args.outfolder / f"{slide_path.stem}_pred.geojson", "w") as f:
            json.dump(geopandas.GeoSeries(polygons.geoms).__geo_interface__, f)
