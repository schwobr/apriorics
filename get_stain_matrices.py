from argparse import ArgumentParser
from csv import DictReader
from pathlib import Path

import numpy as np
from pathaia.util.paths import get_files
from pathaia.util.types import Patch, Slide
from staintools.stain_extraction.vahadane_stain_extractor import VahadaneStainExtractor

parser = ArgumentParser(
    prog=(
        "Compute Vahadane stain matrices as npy files for input slides using PathAIA "
        "patch csvs."
    )
)
parser.add_argument(
    "--patch-csv-folder",
    type=Path,
    help="Input folder containing PathAIA patch csvs.",
    required=True,
)
parser.add_argument(
    "--slidefolder",
    type=Path,
    help="Input folder containing svs slide files.",
    required=True,
)
parser.add_argument(
    "--recurse",
    action="store_true",
    help="Specify to recurse through slidefolder when looking for svs files. Optional.",
)
parser.add_argument(
    "--outfolder", type=Path, help="Target output folder.", required=True
)

if __name__ == "__main__":
    args = parser.parse_args()

    if not args.outfolder.exists():
        args.outfolder.mkdir()

    csv_paths = get_files(
        args.patch_csv_folder, extensions=".csv", recurse=args.recurse
    )

    for csv_path in csv_paths:
        print(csv_path.stem)
        rel_path = csv_path.relative_to(args.patch_csv_folder)
        slide = Slide(args.slidefolder / rel_path.with_suffix(".svs"), backend="cucim")
        stain_matrices = []

        with csv_path.open("r") as f:
            reader = DictReader(f)
            for row in reader:
                patch = Patch.from_csv_row(row)
                img = slide.read_region(patch.position, patch.level, patch.size)
                img = np.asarray(img.convert("RGB"))
                stain_matrix = VahadaneStainExtractor.get_stain_matrix(img)
                stain_matrices.append(stain_matrix)
            stain_matrix = np.median(stain_matrices, axis=0)
            np.save(args.outfolder / rel_path.with_suffix(".npy"), stain_matrix)
