import csv
import re
from pathlib import Path
from argparse import ArgumentParser
from pathaia.util.types import Patch, Slide
from pathaia.util.paths import get_files
from pathaia.patches.functional_api import slide_rois_no_image
from pathaia.patches import filter_thumbnail
import numpy as np
from skimage.transform import resize


parser = ArgumentParser(prog="Generates the PathAIA patch CSVs for slides.")
parser.add_argument(
    "--slidefolder",
    type=Path,
    help="Input folder containing slide svs files.",
    required=True,
)
parser.add_argument(
    "--maskfolder",
    type=Path,
    help="Input folder containing mask tif files.",
    required=True,
)
parser.add_argument(
    "--outfolder",
    type=Path,
    help=(
        "Target output folder. Actual output folder will be "
        "outfolder/{patch_size}_{level}/patch_csvs."
    ),
    required=True,
)
parser.add_argument(
    "--recurse",
    "-r",
    action="store_true",
    help="Specify to recurse through slidefolder when looking for svs files. Optional.",
)
parser.add_argument(
    "--patch-size",
    type=int,
    default=1024,
    help="Size of the (square) patches to extract. Default 1024.",
)
parser.add_argument(
    "--file-filter",
    help=(
        "Regex filter input svs files by names. To filter a specific ihc id x, should"
        r' be "^21I\d{6}-\d-\d\d-x_\d{6}". Optional.'
    ),
)
parser.add_argument(
    "--level",
    type=int,
    default=0,
    help="Pyramid level to extract patches on. Default 0.",
)
parser.add_argument(
    "--overlap",
    type=float,
    default=0.3,
    help="Part of the patches that should overlap. Default 0.3.",
)


def get_mask_filter(mask, thumb_size=2000):
    thumb = np.asarray(mask.get_thumbnail((thumb_size, thumb_size)).convert("L")) > 0

    def _filter(x):
        mask = resize(thumb, x.shape[:2])
        return mask

    return _filter


if __name__ == "__main__":
    args = parser.parse_args()
    input_files = get_files(args.slidefolder, extensions=".svs", recurse=args.recurse)
    if args.file_filter is not None:
        filter_regex = re.compile(args.file_filter)
        input_files = filter(
            lambda x: filter_regex.match(x.name) is not None, input_files
        )

    outfolder = args.outfolder / f"{args.patch_size}_{args.level}/patch_csvs"

    interval = -int(args.overlap * args.psize)

    for in_file_path in input_files:
        mask_path = args.maskfolder / in_file_path.relative_to(
            args.slidefolder
        ).with_suffix(".tif")
        if not mask_path.exists():
            continue
        slide = Slide(in_file_path, backend="cucim")
        mask = Slide(mask_path, backend="cucim")

        patches = slide_rois_no_image(
            slide,
            args.level,
            psize=args.patch_size,
            interval=interval,
            slide_filters=[filter_thumbnail],
            thumb_size=2000,
        )

        out_file_path = outfolder / in_file_path.relative_to(
            args.slidefolder
        ).with_suffix(".csv")
        if not out_file_path.parent.exists():
            out_file_path.parent.mkdir(parents=True)

        with open(out_file_path, "w") as out_file:
            writer = csv.DictWriter(out_file, fieldnames=Patch.get_fields() + ["n_pos"])
            writer.writeheader()
            for patch in patches:
                patch_mask = mask.read_region(
                    patch.position, patch.level, patch.size
                ).convert("1")
                row = patch.to_csv_row()
                row["n_pos"] = np.asarray(patch_mask).sum()
                writer.writerow()
