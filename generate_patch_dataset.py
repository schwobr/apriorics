import sys
import csv
import re

from argparse import ArgumentParser
from typing import List
from pathaia.util.types import *
from pathaia.util.paths import get_files
from pathaia.patches.functional_api import slide_rois_no_image
from pathaia.patches import filter_thumbnail

parser = ArgumentParser(prog="Generate the patch CSVs for slides")
parser.add_argument("script_path", type=str )
parser.add_argument("input_folder", type=str)
parser.add_argument("target_folder", type=str )
parser.add_argument("--recursive", "-r", action="store_true")
parser.add_argument("--patch-size", type=int, default=1024)
parser.add_argument("--file-filter", type=str)
namespace = vars(parser.parse_args(sys.argv))

if __name__ == "__main__":
    input_folder = namespace["input_folder"]
    output_folder = namespace["target_folder"]
    input_files = get_files(input_folder, extensions=".svs", recurse= namespace["recursive"]) 
    if namespace["file_filter"]:
        filter_regex = re.compile(namespace["file_filter"])
        input_files = filter(lambda x: filter_regex.match(x.name) is not None, input_files)

    patch_size = namespace["patch_size"]

    for in_file_path in input_files:
        slide = Slide(in_file_path, backend="cucim")
        patches = slide_rois_no_image(slide, 0, psize=(patch_size, patch_size), slide_filters=[filter_thumbnail])

        out_file_path = Path(output_folder)/in_file_path.relative_to(input_folder).with_suffix(".csv")
        with open(out_file_path, "w") as out_file:
            writer = csv.DictWriter(out_file, fieldnames=Patch.get_fields())
            writer.writeheader()
            for patch in patches:
                writer.writerow(patch.to_csv_row())