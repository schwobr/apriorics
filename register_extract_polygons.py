from argparse import ArgumentParser
from cytomine import Cytomine
from cytomine.models import (
    Annotation,
    AnnotationCollection,
    AnnotationTerm,
    TermCollection,
    ImageInstanceCollection,
)
from cytomine.models.collection import CollectionPartialUploadException
from pathlib import Path
from pathaia.patches import slide_rois_no_image
from PIL import Image
import numpy as np
from cucim import CuImage
from skimage.morphology import (
    binary_closing,
    binary_dilation,
)
from skimage.color import rgb2hed
import shapely
from shapely.affinity import affine_transform
from shapely.ops import unary_union
from rasterio import features
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
    register
)

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


def get_reduced_coords(coords, angle_th, distance_th):
    vector_rep = np.diff(coords, axis=0)
    angle_th_rad = np.deg2rad(angle_th)
    points_removed = [0]
    while len(points_removed):
        points_removed = list()
        for i in range(len(vector_rep) - 1):
            if len(coords) - len(points_removed) == 3:
                break
            v01 = vector_rep[i]
            v12 = vector_rep[i + 1]
            d01 = np.linalg.norm(v01)
            d12 = np.linalg.norm(v12)
            if d01 < distance_th and d12 < distance_th:
                points_removed.append(i + 1)
                vector_rep[i + 1] = coords[i + 2] - coords[i]
                continue
            angle = np.arccos(np.dot(v01, v12) / (d01 * d12))
            if angle < angle_th_rad:
                points_removed.append(i + 1)
                vector_rep[i + 1] = coords[i + 2] - coords[i]
        coords = np.delete(coords, points_removed, axis=0)
        vector_rep = np.diff(coords, axis=0)
    return coords.astype(int)


def reduce_polygon(polygon, angle_th=0, distance_th=0):
    ext_poly_coords = get_reduced_coords(
        np.array(polygon.exterior.coords[:]), angle_th, distance_th
    )
    interior_coords = [
        get_reduced_coords(np.array(interior.coords[:]), angle_th, distance_th)
        for interior in polygon.interiors
    ]
    return shapely.geometry.Polygon(ext_poly_coords, interior_coords)


def get_binary_op(op_name):
    if op_name == "closing":
        return binary_closing
    elif op_name == "dilation":
        return binary_dilation
    else:
        return None


def mask_to_polygons_layer(mask):
    all_polygons = []
    for shape, _ in features.shapes(mask.astype(np.int16), mask=(mask > 0)):
        all_polygons.append(
            reduce_polygon(shapely.geometry.shape(shape), angle_th=2, distance_th=3)
        )

    all_polygons = shapely.geometry.MultiPolygon(all_polygons)
    if not all_polygons.is_valid:
        all_polygons = all_polygons.buffer(0)
        # Sometimes buffer() converts a simple Multipolygon to just a Polygon,
        # need to keep it a Multi throughout
        if all_polygons.type == "Polygon":
            all_polygons = shapely.geometry.MultiPolygon([all_polygons])
    return all_polygons


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
            tissue_mask = get_tissue_mask(np.array(ihc.convert('L')), whitetol=255)
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

    with Cytomine(
        host=args.host, public_key=args.public_key, private_key=args.private_key
    ) as cytomine:
        images = ImageInstanceCollection().fetch_with_filter("project", args.id_project)
        images = filter(lambda x: args.he_slide in x.filename, images)

        try:
            id_image = next(images).id
        except StopIteration:
            print(
                f"Slide {args.he_slide} was not found in cytomine, please upload it. Stopping program..."
            )
            raise RuntimeError

        terms = TermCollection().fetch_with_filter("project", args.id_project)
        terms = filter(lambda x: x.name == args.term, terms)

        try:
            id_term = next(terms).id
        except StopIteration:
            print(
                f"Term {args.term} was not found on cytomine, please upload it. Resuming with not term..."
            )
            id_term = None

        annotations = AnnotationCollection()
        for polygon in all_polygons:
            if polygon.area < args.object_min_size:
                continue
            if args.polygon_type == "box":
                bbox = shapely.geometry.box(*polygon.bounds)
                location = bbox.wkt
            else:
                location = polygon.wkt
            annotations.append(
                Annotation(
                    location=location,
                    id_image=id_image,
                    id_project=args.id_project,
                )
            )
        try:
            results = annotations.save()
        except CollectionPartialUploadException as e:
            print(e)
        for _, (_, message) in results:
            ids = map(int, message.split()[1].split(","))
            for id in ids:
                AnnotationTerm(id_annotation=id, id_term=id_term).save()
    print("Uploading done.")

    shutil.rmtree(args.tmpfolder)
