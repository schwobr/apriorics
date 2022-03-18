from argparse import ArgumentParser
import os
from pathlib import Path
from subprocess import run
from pathaia.patches import slide_rois_no_image
from pathaia.util.types import Slide, Coord
from pathaia.util.paths import get_files
from PIL import Image
import numpy as np
from pathaia.util.types import Patch
from shapely.affinity import translate
from shapely.ops import unary_union
import shutil
from apriorics.registration import (
    full_registration,
    get_coord_transform,
)
from apriorics.masks import get_tissue_mask, get_mask_function, update_full_mask_mp
from apriorics.polygons import mask_to_polygons_layer
from ordered_set import OrderedSet
from multiprocessing import Pool, Array
from ctypes import c_bool
from subprocess import PIPE


IHC_MAPPING = {
    13: "AE1AE3",
    14: "CD163",
    15: "CD3CD20",
    16: "EMD",
    17: "ERGCaldesmone",
    18: "ERGPodoplanine",
    19: "INI1",
    20: "P40ColIV",
    21: "PHH3",
}


parser = ArgumentParser(prog="Upload image example")
parser.add_argument("--dab-thr", type=float, default=0.03)
parser.add_argument("--object-min-size", type=int, default=1000)
parser.add_argument(
    "--binary-op", default="closing", choices=["None", "closing", "dilation"]
)
parser.add_argument(
    "--radius", default=10, type=int, help="radius for binary operations"
)
parser.add_argument("--psize", type=int, default=5000)
parser.add_argument("--overlap", type=float, default=0.3)
parser.add_argument("--crop", type=float, default=0.1)
parser.add_argument("--box", type=int, nargs="*")
parser.add_argument("--slidefolder", type=Path)
parser.add_argument("--ihc-id", type=int)
parser.add_argument("--slidefile", type=Path)
parser.add_argument("--tmpfolder", type=Path, default=Path("/tmp/cyto_annotations"))
parser.add_argument("--maskfolder", type=Path)
parser.add_argument("--wktfolder", type=Path)
parser.add_argument("--novips", action="store_true")
parser.add_argument("--num-workers", type=int)
parser.add_argument("-v", "--verbose", action="count")


def get_box_filter(slide, box):
    h = slide.shape[0]
    box[1] = h - box[1]
    box[3] = h - box[3]

    def _filter(thumb):
        dsr = h / thumb.shape[0]
        x0, y0, x1, y1 = map(lambda x: int(x / dsr), box)
        mask = np.zeros(thumb.shape[:2], dtype=bool)
        mask[y0:y1, x0:x1] = True
        return mask

    return _filter


def get_filefilter(slidefile):
    if slidefile is not None:
        with open(slidefile, "r") as f:
            files = set(f.read().rstrip().split("\n"))
    else:
        files = None

    def _filter(items):
        names = OrderedSet(items.map(lambda x: x.stem))
        if files is not None:
            return names.index(names & files)
        else:
            return list(range(len(names)))

    return _filter


def get_register_extract_mask_fn(
    slide_he, slide_ihc, full_mask, crop, mask_function, coord_tfm, args
):
    box = (crop, crop, args.psize - crop, args.psize - crop)

    def register_extract_mask(patch_he):
        patch_ihc = Patch(
            id=patch_he.id,
            slidename="",
            position=coord_tfm(*patch_he.position),
            size=patch_he.size,
            level=patch_he.level,
            size_0=patch_he.size_0,
        )

        restart = True
        iterations = 5000
        count = 0
        maxiter = 4

        while restart and count < maxiter:
            restart = not full_registration(
                slide_he,
                slide_ihc,
                patch_he,
                patch_ihc,
                args.tmpfolder / os.getpid(),
                dab_thr=args.dab_thr,
                object_min_size=args.object_min_size,
                iterations=iterations,
                threads=1,
                stdout=PIPE,
                stderr=PIPE
            )
            if restart:
                break
            else:
                ihc = (
                    Image.open(args.tmpfolder / "ihc_warped.png")
                    .convert("RGB")
                    .crop(box)
                )
                tissue_mask = get_tissue_mask(
                    np.asarray(ihc.convert("L")), whitetol=256
                )
                if tissue_mask.sum() < 0.999 * tissue_mask.size:
                    restart = True
                    iterations *= 2
                    count += 1

        if restart:
            return

        he = Image.open(args.tmpfolder / "he.png")
        he = np.asarray(he.convert("RGB").crop(box))
        mask = mask_function(he, np.asarray(ihc))
        update_full_mask_mp(
            full_mask, mask, *(patch_he.position + crop), *slide_he.dimensions
        )
        polygons = mask_to_polygons_layer(mask)
        x, y = patch_he.position
        moved_polygons = translate(polygons, x + crop, y + crop)
        return moved_polygons

    return register_extract_mask


if __name__ == "__main__":
    args = parser.parse_args()

    if not args.maskfolder.exists():
        args.maskfolder.mkdir()

    if not args.wktfolder.exists():
        args.wktfolder.mkdir()

    if not args.tmpfolder.exists():
        args.tmpfolder.mkdir()
    historeg_path = args.tmpfolder / "historeg"

    crop = int(args.crop * args.psize)

    k = (args.ihc_id - 1) % 12 + 1
    ihc_type = IHC_MAPPING[k + 12]

    hefiles = get_files(args.slidefolder, extensions=[".svs"], recurse=False).filter(
        lambda x: int(x.stem.split("-")[-1].split("_")[0]) == k
    )
    hefiles.sort(key=lambda x: x.stem.split("-")[0])
    ihcfiles = get_files(args.slidefolder, extensions=[".svs"], recurse=False).filter(
        lambda x: int(x.stem.split("-")[-1].split("_")[0]) == k + 12
    )
    ihcfiles.sort(key=lambda x: x.stem.split("-")[0])

    filefilter = get_filefilter(args.slidefile)
    idxs = filefilter(hefiles)
    hefiles = hefiles[idxs]
    ihcfiles = ihcfiles[idxs]

    for hefile, ihcfile in zip(hefiles, ihcfiles):
        if (args.maskfolder / f"{hefile.stem}.tif").exists() or (
            args.maskfolder / f"{hefile.stem}.png"
        ).exists():
            continue

        print(hefile, ihcfile)

        slide_he = Slide(hefile, backend="cucim")
        slide_ihc = Slide(ihcfile, backend="cucim")
        w, h = slide_he.dimensions

        if args.box is not None:
            assert len(args.box) == 4
            slide_filter = get_box_filter(slide_he, args.box)
        else:
            slide_filter = None

        interval = -int(args.overlap * args.psize)

        try:
            coord_tfm = get_coord_transform(slide_he, slide_ihc)
        except IndexError:

            def coord_tfm(x, y):
                return Coord(x, y)

        full_mask = Array(c_bool, [False for _ in range(h * w)])

        with Pool(processes=args.num_workers) as pool:
            register_mask_fn = get_register_extract_mask_fn(
                slide_he,
                slide_ihc,
                full_mask,
                crop,
                get_mask_function(ihc_type),
                coord_tfm,
                args,
            )
            patch_iter = slide_rois_no_image(
                slide_he,
                0,
                (args.psize, args.psize),
                (interval, interval),
                thumb_size=5000,
                slide_filters=["full"],
            )
            all_polygons = pool.map(register_mask_fn, patch_iter)

        all_polygons = filter(lambda x: x is not None, all_polygons)
        all_polygons = unary_union(all_polygons)

        print("Saving full polygons...")

        with (args.wktfolder / f"{hefile.stem}.wkt").open("w") as f:
            f.write(all_polygons.wkt)

        print("Polygons saved.")

        print("Saving mask...")

        maskpath = args.maskfolder / f"{hefile.stem}.png"
        Image.fromarray(full_mask).save(maskpath)
        if not args.novips:
            vips_cmd = (
                f"vips tiffsave {maskpath} {maskpath.with_suffix('.tif')} "
                "--compression jpeg --tile-width 256 --tile-height 256 --tile "
                "--pyramid"
            )

            run(vips_cmd.split())

            maskpath.unlink()

        print("Mask saved.")

    shutil.rmtree(args.tmpfolder)
