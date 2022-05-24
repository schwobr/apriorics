from argparse import ArgumentParser
from pathlib import Path
from subprocess import run

from pathaia.util.paths import get_files

parser = ArgumentParser(
    prog=(
        "Converts all png in a folder to pyramidal tiled tif. Silently deletes png "
        "files afterwards."
    )
)
parser.add_argument(
    "--infolder",
    type=Path,
    help=(
        "Input folder containing png files. Tif files will be writtent in the same "
        "folder."
    ),
)
parser.add_argument(
    "--recurse",
    action="store_true",
    help="Specify to recurse through slidefolder when looking for svs files. Optional.",
)


if __name__ == "__main__":
    args = parser.parse_args()
    files = get_files(args.infolder, extensions=".png", recurse=args.recurse)

    for pngfile in files:
        print(pngfile.stem)
        vips_cmd = (
            f"vips tiffsave {pngfile} {pngfile.with_suffix('.tif')} "
            "--compression jpeg --tile-width 256 --tile-height 256 --tile "
            "--pyramid"
        )

        run(vips_cmd.split())

        pngfile.unlink()
