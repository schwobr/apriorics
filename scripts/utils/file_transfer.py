import json
import os
import shutil
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from pathaia.util.paths import get_files

from apriorics.data_utils import get_info_from_filename

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
parser.add_argument("--num-workers", type=int, default=os.cpu_count())


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

    def transfer_file(file):
        try:
            info = get_info_from_filename(file.stem, ihc_mapping)
        except ValueError:
            info = {
                "ihc_type": file.parts[-3],
                "slide_type": file.parts[-2],
                "block": file.stem,
            }
        if not (
            info["ihc_type"] == args.ihc_type
            and (info["slide_type"] == "HE" or args.import_ihc)
        ):
            return

        outfolder = local_path / info["ihc_type"] / info["slide_type"]
        if not outfolder.exists():
            outfolder.mkdir(parents=True)

        outfile = outfolder / f"{info['block']}{file.suffix}"
        if not outfile.exists():
            print(outfile)
            shutil.copyfile(file, outfile)
        return file.stem, str(outfile)

    data = {"id": [], "path": []}
    with ThreadPoolExecutor(max_workers=args.num_workers) as pool:
        for res in pool.map(transfer_file, files):
            if res is not None:
                data["id"].append(res[0])
                data["path"].append(res[1])

    if args.out_csv is not None:
        df = pd.DataFrame(data)
        df.to_csv(args.out_csv, index=False)
