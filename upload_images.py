from apriorics.cytomine import upload_image_to_cytomine, get_uploaded_images
from argparse import ArgumentParser
from pathaia.util.paths import get_files


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


parser = ArgumentParser()
parser.add_argument("--infolder")
parser.add_argument("--host")
parser.add_argument("--public_key")
parser.add_argument("--private_key")
parser.add_argument("--id_project", type=int)
parser.add_argument("--ihc-type")
parser.add_argument("--recurse", action="store_true")


if __name__ == "__main__":
    args = parser.parse_args()

    filepaths = get_files(args.infolder, extensions=[".svs"], recurse=args.recurse)

    if args.ihc_type is not None:

        def _filter(x):
            k = int(x.stem.split("-")[-1].split("_")[0])
            k = (k - 1) % 12 + 13
            return IHC_MAPPING[k] == args.ihc_type

        filepaths = filepaths.filter(_filter)

    uploaded_images = get_uploaded_images(
        args.host, args.public_key, args.private_key, args.id_project
    )

    for filepath in filepaths:
        if filepath.name not in uploaded_images:
            upload_image_to_cytomine(
                filepath, args.host, args.public_key, args.private_key, args.id_project
            )
