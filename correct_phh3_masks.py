from argparse import ArgumentParser
from ctypes import c_bool
from multiprocessing import Array, Pool
from pathlib import Path
from subprocess import run

import numpy as np
from pathaia.patches.functional_api import slide_rois_no_image
from pathaia.util.paths import get_files
from pathaia.util.types import Slide
from PIL import Image
from shapely.affinity import translate
from shapely.ops import unary_union
from skimage.morphology import (
    binary_dilation,
    disk,
    remove_small_holes,
    remove_small_objects,
)

from apriorics.masks import remove_large_objects, update_full_mask_mp
from apriorics.polygons import mask_to_polygons_layer

parser = ArgumentParser()
parser.add_argument("--inmaskfolder", type=Path)
parser.add_argument("--slidefolder", type=Path)
parser.add_argument("--outwktfolder", type=Path)
parser.add_argument("--outmaskfolder", type=Path)
parser.add_argument("--num-workers", type=int, default=0)


def get_mask(img):
    mask = np.ones(img.shape[:2], dtype=bool)
    ranges = [(56, 98), (69, 119), (117, 160)]
    for c, r in enumerate(ranges):
        a, b = r
        mask &= (img[..., c] > a) & (img[..., c] < b)
    mask = binary_dilation(remove_small_objects(mask, min_size=50), disk(15))
    return mask


if __name__ == "__main__":
    args = parser.parse_args()

    maskfiles = get_files(args.inmaskfolder, extensions=".tif").filter(
        lambda x: int(x.stem.split("-")[-1].split("_")[0]) == 9
    )
    maskfiles.sort(key=lambda x: x.stem)
    slidefiles = maskfiles.map(lambda x: args.slidefolder / f"{x.stem}.svs")

    for maskfile, slidefile in zip(maskfiles, slidefiles):
        slidename = slidefile.stem
        outfile = args.outmaskfolder / f"{slidefile.stem}.png"
        if outfile.with_suffix(".tif").exists():
            continue
        print(slidename)

        slide_mask = Slide(maskfile, backend="cucim")
        slide = Slide(slidefile, backend="cucim")
        w, h = slide.dimensions

        full_mask = Array(c_bool, h * w)

        def correct_mask(patch):
            mask_reg = np.asarray(
                slide_mask.read_region(patch.position, patch.level, patch.size).convert(
                    "1"
                )
            )
            if not mask_reg.sum():
                return

            slide_reg = np.asarray(
                slide.read_region(patch.position, patch.level, patch.size).convert(
                    "RGB"
                )
            )

            mask = remove_large_objects(
                remove_small_objects(
                    remove_small_holes(
                        ~get_mask(slide_reg) & mask_reg, area_threshold=50
                    ),
                    min_size=50,
                )
            )

            update_full_mask_mp(full_mask, mask, *patch.position, w, h)

            polygons = mask_to_polygons_layer(mask, angle_th=0, distance_th=0)
            polygons = translate(polygons, *patch.position)
            return polygons

        patches = slide_rois_no_image(
            slide, 0, 1000, interval=-100, thumb_size=5000, slide_filters=["full"]
        )
        print("Computing mask...")
        with Pool(processes=args.num_workers) as pool:
            all_polygons = pool.map(correct_mask, patches)
            pool.close()
            pool.join()
        all_polygons = [x for x in all_polygons if x is not None]
        all_polygons = unary_union(all_polygons)

        with open(args.outwktfolder / f"{slidename}.wkt", "w") as f:
            f.write(all_polygons.wkt)

        print("Writing mask...")

        full_mask = np.frombuffer(full_mask.get_obj(), dtype=bool).reshape(h, w)
        full_mask = full_mask.astype(np.uint8) * 255
        Image.fromarray(full_mask).convert("RGB").save(outfile)
        vips_cmd = (
            f"vips tiffsave {outfile} {outfile.with_suffix('.tif')} "
            "--compression jpeg --Q 100 --tile-width 256 --tile-height 256 --tile "
            "--pyramid"
        )
        run(vips_cmd.split())
        outfile.unlink()
