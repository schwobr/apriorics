from argparse import ArgumentParser
from pathlib import Path
from cucim import CuImage
import numpy as np
import re
from pathaia.util.paths import get_files
from subprocess import run
import cv2
from skimage.morphology import remove_small_objects, remove_small_holes

parser = ArgumentParser()
parser.add_argument("--infolder", type=Path)
parser.add_argument("--file-filter")


if __name__ == "__main__":
    args = parser.parse_args()
    input_files = get_files(args.infolder, extensions=".tif", recurse=False)
    if args.file_filter is not None:
        filter_regex = re.compile(args.file_filter)
        input_files = filter(
            lambda x: filter_regex.match(x.name) is not None, input_files
        )

    for input_file in input_files:
        mask = CuImage(str(input_file))
        if len(mask.metadata["cucim"]["channel_names"]) == 3:
            continue
        print(input_file.stem)
        w, h = mask.size("XY")
        print("Computing mask")
        full_mask = np.zeros((h, w, 3), dtype=np.uint8)
        x, y = 0, 0
        psize = 5000
        delta = 4900
        while y < h:
            while x < w:
                dx = min(psize, w - x)
                dy = min(psize, h - y)
                small_mask = np.asarray(mask.read_region((x, y), (dx, dy), 0))[:, :, 0]
                if small_mask.any():
                    small_mask = remove_small_objects(
                        remove_small_holes(small_mask > 127, area_threshold=10),
                        min_size=10,
                    )
                    full_mask[y : y + dy, x : x + dx] = (
                        small_mask[:, :, None].astype(np.uint8) * 255
                    )
                x += delta
            y += delta
            x = 0
        print("Saving mask")
        cv2.imwrite(str(input_file.with_suffix(".png")), full_mask)
        full_mask = 0
        print("Starting vips")
        vips_cmd = (
            f"vips tiffsave {input_file.with_suffix('.png')} {input_file} "
            "--compression jpeg --Q 100 --tile-width 256 --tile-height 256 --tile "
            "--pyramid"
        )

        run(vips_cmd.split())

        input_file.with_suffix(".png").unlink()
