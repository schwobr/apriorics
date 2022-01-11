from argparse import ArgumentParser
from pathlib import Path
from pathaia.patches import slide_rois_no_image
from PIL import Image
import numpy as np
from cucim import CuImage
from skimage.color import rgb2hed
from shapely.affinity import affine_transform
from shapely.ops import unary_union
import shutil
from skimage.color import hed_from_rgb
from skimage.io import imsave
from apriorics.registration import (
    get_input_images,
    get_tissue_mask,
    has_enough_tissue,
    get_dab_mask,
    equalize_contrasts,
    convert_to_nifti,
    register,
    get_binary_op,
)
from apriorics.polygons import mask_to_polygons_layer
from apriorics.cytomine import upload_polygons_to_cytomine

HED_MAX = (hed_from_rgb * (hed_from_rgb > 0)).sum(0)
HED_MIN = (hed_from_rgb * (hed_from_rgb < 0)).sum(0)

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
    "--binay-op", default="closing", choices=["None", "closing", "dilation"]
)
parser.add_argument(
    "--radius", default=10, type=int, help="radius for binary operations"
)
parser.add_argument("--psize", type=int, default=10000)
parser.add_argument("--overlap", type=float, default=0.2)
parser.add_argument("--crop", type=float, default=0.05)
parser.add_argument("--heslide")
parser.add_argument("--ihcslide")
parser.add_argument("--tmpfolder", type=Path, default=Path("/tmp/cyto_annotations"))
parser.add_argument("--outfile", type=Path)
parser.add_argument("--logfile", type=Path)
parser.add_argument("-v", "--verbose", action="count")


if __name__ == "__main__":
    args = parser.parse_args()

    slide_he = CuImage(str(args.slidefolder / args.heslide))
    slide_ihc = CuImage(str(args.slidefolder / args.ihcslide))
    w, h = slide_he.dimensions

    interval = -int(args.overlap * args.psize)
    if not args.tmpfolder.exists():
        args.tmpfolder.mkdir()
    historeg_path = args.tmpfolder / "historeg"

    crop = int(args.crop * args.psize)
    box = (crop, crop, args.psize - crop, args.psize - crop)

    all_polygons_file = (
        args.outfile.parent / f"{args.outfile.stem}_all{args.outfile.suffix}"
    )

    try:
        all_polygons_file.unlink()
    except FileNotFoundError:
        pass

    all_polygons = []

    for patch in slide_rois_no_image(
        slide_he,
        0,
        (args.psize, args.psize),
        (interval, interval),
        thumb_size=5000,
        slide_filters=["full"],
    ):
        print(patch.position)
        he_H_path = args.tmpfolder / "he_H.png"
        ihc_H_path = args.tmpfolder / "ihc_H.png"
        he_path = args.tmpfolder / "he.png"
        ihc_path = args.tmpfolder / "ihc.png"
        reg_path = args.tmpfolder / "ihc_warped.png"

        he, he_G, he_H = get_input_images(slide_he, patch, HED_MIN[0], HED_MAX[0])
        ihc, ihc_G, ihc_H = get_input_images(slide_ihc, patch, HED_MIN[0], HED_MAX[0])

        if not has_enough_tissue(he_G, ihc_G):
            print("Patch doesn't contain enough tissue, skipping.")
            continue

        mask = get_dab_mask(
            ihc, dab_thr=args.dab_thr, object_min_size=args.object_min_size
        )

        if not mask.sum():
            print("Mask would be empty, skipping.")
            continue

        he_H, ihc_H = equalize_contrasts(he_H, ihc_H, he_G, ihc_G)

        imsave(he_H_path, he_H)
        imsave(ihc_H_path, ihc_H)

        imsave(he_path, he)
        imsave(ihc_path, ihc)
        convert_to_nifti(he_path)
        convert_to_nifti(ihc_path)

        print("Starting registration...")

        restart = True
        iterations = 20000
        count = 0
        maxiter = 3
        tfm_path = historeg_path / "ihc_H_registered_to_he_H/metrics/full_resolution"

        while restart and count < maxiter:
            restart = False
            register(
                args.tmpfolder,
                he_H_path,
                ihc_H_path,
                he_path.with_suffix(".nii.gz"),
                ihc_path.with_suffix(".nii.gz"),
                reg_path.with_suffix(".nii.gz"),
                iterations=iterations,
            )

            print("Registration done...")
            print("Computing mask...")

            ihc = Image.open(reg_path).crop(box)
            tissue_mask = get_tissue_mask(np.array(ihc.convert("L")), whitetol=255)
            if tissue_mask.sum() < 0.99 * tissue_mask.size:
                restart = True
                iterations *= 2
                count += 1
                print("Affine registration failed, restarting...")

        if restart:
            print("Registration failed, skipping...")
            continue

        ihc = np.array(ihc)
        he = Image.open(he_path)
        he = rgb2hed(np.array(he.convert("RGB").crop(box)))[:, :, 2]
        mask = get_dab_mask(
            ihc,
            dab_thr=args.dab_thr,
            object_min_size=args.object_min_size,
            tissue_mask=tissue_mask,
            he_img=he,
            binary_op=get_binary_op(args.binary_op),
            r=args.radius,
        )

        print("Mask done.")
        print("Computing polygons...")

        polygons = mask_to_polygons_layer(mask)
        x, y = patch.position
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

    upload_polygons_to_cytomine(
        all_polygons,
        args.heslide,
        args.host,
        args.public_key,
        args.private_key,
        args.id_project,
        term=args.term,
        polygon_type=args.polygon_type,
        object_min_size=args.object_min_size
    )
    print("Uploading done.")

    shutil.rmtree(args.tmpfolder)
