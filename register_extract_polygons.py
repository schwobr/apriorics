from argparse import ArgumentParser
from pathlib import Path
from pathaia.patches import slide_rois_no_image
from PIL import Image
import numpy as np
from cucim import CuImage
from pathaia.util.types import Patch
from shapely.affinity import affine_transform
from shapely.ops import unary_union
import shutil
from apriorics.registration import (
    full_registration,
    get_coord_transform,
)
from apriorics.masks import get_tissue_mask, get_mask_function
from apriorics.polygons import mask_to_polygons_layer
from apriorics.cytomine import upload_to_cytomine


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
parser.add_argument("--host", help="The Cytomine host")
parser.add_argument("--public_key", help="The Cytomine public key")
parser.add_argument("--private_key", help="The Cytomine private key")
parser.add_argument(
    "--id_project", type=int, help="The project where to add the uploaded image"
)
parser.add_argument(
    "--term", help="term for the annotations. Must be defined in a cytomine ontology."
)
parser.add_argument("--polygon-type", default="polygon", choices=["polygon", "box"])
parser.add_argument(
    "--slidefolder",
    type=Path,
)
parser.add_argument("--dab-thr", type=float, default=0.085)
parser.add_argument("--object-min-size", type=int, default=1000)
parser.add_argument(
    "--binary-op", default="closing", choices=["None", "closing", "dilation"]
)
parser.add_argument(
    "--radius", default=10, type=int, help="radius for binary operations"
)
parser.add_argument("--psize", type=int, default=10000)
parser.add_argument("--overlap", type=float, default=0.2)
parser.add_argument("--crop", type=float, default=0.05)
parser.add_argument("--box", type=int, nargs="*")
parser.add_argument("--heslide")
parser.add_argument("--ihcslide")
parser.add_argument("--tmpfolder", type=Path, default=Path("/tmp/cyto_annotations"))
parser.add_argument("--outfile", type=Path)
parser.add_argument("--logfile", type=Path)
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


if __name__ == "__main__":
    args = parser.parse_args()

    slide_he = CuImage(str(args.slidefolder / args.heslide))
    slide_ihc = CuImage(str(args.slidefolder / args.ihcslide))
    w, h = slide_he.size("XY")
    ihc_type = IHC_MAPPING[int(args.ihcslide.split("-")[-1].split("_")[0])]

    if args.box is not None:
        assert len(args.box) == 4
        slide_filter = get_box_filter(slide_he, args.box)
    else:
        slide_filter = None

    interval = -int(args.overlap * args.psize)
    if not args.tmpfolder.exists():
        args.tmpfolder.mkdir()
    historeg_path = args.tmpfolder / "historeg"
    crop = int(args.crop * args.psize)
    box = (crop, crop, args.psize - crop, args.psize - crop)

    all_polygons_file = (
        args.outfile.parent / f"{args.outfile.stem}_all{args.outfile.suffix}"
    )

    coord_tfm = get_coord_transform(slide_he, slide_ihc)

    try:
        all_polygons_file.unlink()
    except FileNotFoundError:
        pass

    all_polygons = []

    for patch_he in slide_rois_no_image(
        slide_he,
        0,
        (args.psize, args.psize),
        (interval, interval),
        thumb_size=5000,
        slide_filters=[slide_filter],
    ):
        patch_ihc = Patch(
            id=patch_he.id,
            slidename=args.ihcslide,
            position=patch_he.position,
            size=patch_he.size,
            level=patch_he.level,
            size_0=patch_he.size_0,
        )
        patch_ihc.position = coord_tfm(*patch_ihc.position)
        full_registration(
            slide_he,
            slide_ihc,
            patch_he,
            patch_ihc,
            args.tmpfolder,
            dab_thr=args.dab_thr,
            object_min_size=args.object_min_size,
        )

        print("Computing mask...")
        ihc = Image.open(args.tmpfolder / "ihc_warped.png").convert("RGB").crop(box)
        tissue_mask = get_tissue_mask(np.asarray(ihc.convert("L")), whitetol=255)
        he = Image.open(args.tmpfolder / "he.png")
        he = np.asarray(he.convert("RGB").crop(box))
        mask = get_mask_function(ihc_type)(he, np.asarray(ihc))
        print("Mask done.")

        print("Computing polygons...")
        polygons = mask_to_polygons_layer(mask)
        x, y = patch_he.position
        moved_polygons = affine_transform(
            polygons, [1, 0, 0, -1, x + crop, h - (y + crop)]
        )

        with all_polygons_file.open("a") as f:
            f.write(moved_polygons.wkt)
            f.write("\n")

        all_polygons.append(moved_polygons)
        print("Polygons done.")

    all_polygons = unary_union(all_polygons)

    print("Saving full polygons...")

    with args.outfile.open("w") as f:
        f.write(all_polygons.wkt)

    print("Polygons saved.")
    print("Uploading to cytomine...")

    upload_to_cytomine(
        all_polygons,
        args.heslide,
        args.host,
        args.public_key,
        args.private_key,
        args.id_project,
        term=args.term,
        polygon_type=args.polygon_type,
        object_min_size=args.object_min_size,
    )
    print("Uploading done.")

    shutil.rmtree(args.tmpfolder)
