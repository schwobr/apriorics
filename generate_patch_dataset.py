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


parser = ArgumentParser(prog="Generate the patch CSVs for slides")
parser.add_argument("--input-folder", type=Path)
parser.add_argument("--maskfolder", type=Path)
parser.add_argument("--target-folder", type=Path)
parser.add_argument("--recursive", "-r", action="store_true")
parser.add_argument("--patch-size", type=int, default=1024)
parser.add_argument("--file-filter")
parser.add_argument("--level", type=int, default=0)


def get_mask_filter(mask, thumb_size=2000):
    thumb = np.asarray(mask.get_thumbnail((thumb_size, thumb_size)).convert("L")) > 0

    def _filter(x):
        mask = resize(thumb, x.shape[:2])
        return mask

    return _filter


if __name__ == "__main__":
    args = parser.parse_args()
    input_files = get_files(
        args.input_folder, extensions=".svs", recurse=args.recursive
    )
    if args.file_filter is not None:
        filter_regex = re.compile(args.file_filter)
        input_files = filter(
            lambda x: filter_regex.match(x.name) is not None, input_files
        )

    outfolder = args.target_folder / f"{args.patch_size}_{args.level}/patch_csvs"

    for in_file_path in input_files:
        mask_path = args.maskfolder / in_file_path.relative_to(
            args.input_folder
        ).with_suffix(".tif")
        if not mask_path.exists():
            continue
        slide = Slide(in_file_path, backend="cucim")
        mask = Slide(mask_path, backend="cucim")

        patches = slide_rois_no_image(
            slide,
            args.level,
            psize=args.patch_size,
            slide_filters=[filter_thumbnail],  # get_mask_filter(mask, thumb_size=2000)]
            thumb_size=2000,
        )

        out_file_path = outfolder / in_file_path.relative_to(
            args.input_folder
        ).with_suffix(".csv")
        if not out_file_path.parent.exists():
            out_file_path.parent.mkdir(parents=True)

        with open(out_file_path, "w") as out_file:
            writer = csv.DictWriter(out_file, fieldnames=Patch.get_fields())
            writer.writeheader()
            for patch in patches:
                writer.writerow(patch.to_csv_row())
