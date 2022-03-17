from argparse import ArgumentParser
from pathlib import Path
from pathaia.util.paths import get_files
from subprocess import run


parser = ArgumentParser()
parser.add_argument("--infolder", type=Path)
parser.add_argument("--recurse", action="store_true")


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
