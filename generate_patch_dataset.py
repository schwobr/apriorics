import csv
import re
from pathlib import Path
from argparse import ArgumentParser
from pathaia.util.types import Patch, Slide
from pathaia.util.paths import get_files
from pathaia.patches.functional_api import slide_rois_no_image
from pathaia.patches import filter_thumbnail

parser = ArgumentParser(prog="Generate the patch CSVs for slides")
parser.add_argument("script_path", type=Path)
parser.add_argument("input_folder", type=Path)
parser.add_argument("target_folder", type=Path)
parser.add_argument("--recursive", "-r", action="store_true")
parser.add_argument("--patch-size", type=int, default=1024)
parser.add_argument("--file-filter", type=str)

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

    for in_file_path in input_files:
        slide = Slide(in_file_path, backend="cucim")
        patches = slide_rois_no_image(
            slide, 0, psize=args.patch_size, slide_filters=[filter_thumbnail]
        )

        out_file_path = args.output_folder / in_file_path.relative_to(
            args.input_folder
        ).with_suffix(".csv")
        with open(out_file_path, "w") as out_file:
            writer = csv.DictWriter(out_file, fieldnames=Patch.get_fields())
            writer.writeheader()
            for patch in patches:
                writer.writerow(patch.to_csv_row())
