from apriorics.polygons import hovernet_to_wkt
from pathaia.util.paths import get_files
from argparse import ArgumentParser
from pathlib import Path

parser = ArgumentParser()
parser.add_argument("--jsonfolder", type=Path)
parser.add_argument("--wktfolder", type=Path)


if __name__ == "__main__":
    args = parser.parse_args()

    jsonfiles = get_files(args.jsonfolder, extensions=[".json"], recurse=False)

    if not args.wktfolder.exists():
        args.wktfolder.mkdir()

    for jsonfile in jsonfiles:
        wktfile = args.wktfolder/f"{jsonfile.stem}.wkt"
        hovernet_to_wkt(jsonfile, wktfile)
