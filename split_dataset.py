from argparse import ArgumentParser
import os
from pathlib import Path
from pathaia.util.paths import get_files
import numpy as np
from numpy.random import default_rng
import pandas as pd
import re


parser = ArgumentParser(prog="Splits a dataset between train and validation slides. ")
parser.add_argument(
    "--patch-csv-folder",
    type=Path,
    help="Input folder containing patch csv files.",
    required=True,
)
parser.add_argument(
    "--out-csv",
    type=Path,
    help="Output csv containing 2 columns: slide and split.",
    required=True,
)
parser.add_argument(
    "--existing-csv",
    type=Path,
    help=(
        "Existing incomplete split csv. If specified, only missing data will be "
        "split. Optional"
    ),
)
parser.add_argument(
    "--recurse",
    action="store_true",
    help="Specify to recurse through slidefolder when looking for svs files. Optional.",
)
parser.add_argument(
    "--train-ratio",
    type=float,
    default=0.8,
    help="Part of the dataset to use for training. Default 0.8.",
)
parser.add_argument(
    "--file-filter",
    help=(
        "Regex filter input svs files by names. To filter a specific ihc id x, should"
        r' be "^21I\d{6}-\d-\d\d-x_\d{6}". Optional.'
    ),
)
parser.add_argument(
    "--seed",
    type=int,
    help=(
        "Specify seed for RNG. Can also be set using PL_GLOBAL_SEED environment "
        "variable. Optional."
    ),
)

if __name__ == "__main__":
    args = parser.parse_args()
    maskfiles = get_files(
        args.patch_csv_folder, extensions=".csv", recurse=args.recurse
    )

    if args.file_filter is not None:
        filter_regex = re.compile(args.file_filter)
        maskfiles = maskfiles.filter(lambda x: filter_regex.match(x.name) is not None)

    n = len(maskfiles)
    n_train = int(args.train_ratio * n)

    if args.existing_csv is not None:
        ex_df = pd.read_csv(args.existing_csv)
        n_train -= (ex_df["split"] == "train").sum()
        maskfiles = maskfiles.filter(lambda x: x.stem not in ex_df["slide"].values)
        n = len(maskfiles)

    if args.seed is None:
        args.seed = os.environ.get("PL_GLOBAL_SEED")
        if args.seed is not None:
            args.seed = int(args.seed)

    rng = default_rng(args.seed)
    train_idxs = rng.choice(np.arange(n), size=n_train, replace=False)
    splits = np.full(n, "valid")
    splits[train_idxs] = "train"

    df = pd.DataFrame({"slide": maskfiles.map(lambda x: x.stem), "split": splits})
    if args.existing_csv is not None:
        df = ex_df.concat(df, ignore_index=True)
    df.to_csv(args.out_csv, index=False)
