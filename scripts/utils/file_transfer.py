import json
import shutil
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from apriorics.data_utils import get_info_from_filename
from pathaia.util.paths import get_files

parser = ArgumentParser()
parser.add_argument("--remote-path", type=Path, required=True)
parser.add_argument("--remote-rel-path", type=Path, required=True)
parser.add_argument("--data-path", type=Path, required=True)
parser.add_argument("--rel-path", type=Path, required=True)
parser.add_argument("--extension", default=".svs")
parser.add_argument("--import-ihc", action="store_true")
parser.add_argument("--clean-previous", action="store_true")
parser.add_argument("--ihc-type", required=True)
parser.add_argument("--mapping-file", type=Path, required=True)
parser.add_argument("--recurse", action="store_true")
parser.add_argument("--out-csv", type=Path)


if __name__ == "__main__":
    args = parser.parse_known_args()[0]

    remote_path = args.remote_path / args.remote_rel_path
    local_path = args.data_path / args.rel_path
    if not local_path.exists():
        local_path.mkdir()

    with open(args.mapping_file, "r") as f:
        ihc_mapping = json.load(f)

    assert (
        args.ihc_type in ihc_mapping["HE"].keys()
    ), f"ihc-type must be one of {', '.join(ihc_mapping['HE'].keys())}"

    if args.clean_previous:
        for ihc in ihc_mapping["HE"].keys():
            if ihc != args.ihc_type and (local_path / ihc).exists():
                shutil.rmtree(local_path / ihc)

    files = get_files(remote_path, extensions=args.extension, recurse=args.recurse)

    data = {"id": [], "path": []}
    for file in files:
        info = get_info_from_filename(file.stem, ihc_mapping)
        if not (
            info["ihc_type"] == args.ihc_type
            and (info["slide_type"] == "HE" or args.import_ihc)
        ):
            continue

        outfolder = local_path / info["ihc_type"] / info["slide_type"]
        if not outfolder.exists():
            outfolder.mkdir(parents=True)

        outfile = outfolder / f"{info['block']}{file.suffix}"
        data["id"].append(file.stem)
        data["path"].append(str(outfile))
        if not outfile.exists():
            print(outfile)
            shutil.copyfile(file, outfile)

    if args.out_csv is not None:
        df = pd.DataFrame(data)
        df.to_csv(args.out_csv, index=False)
