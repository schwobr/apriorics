from argparse import ArgumentParser
from pathlib import Path
from pathaia.util.paths import get_files
import numpy as np
from numpy.random import default_rng
import pandas as pd


parser = ArgumentParser()
parser.add_argument("--maskfolder", type=Path)
parser.add_argument("--out-csv", type=Path)
parser.add_argument("--recurse", action="store_true")
parser.add_argument("--train-ratio", type=float, default=0.8)

if __name__ == "__main__":
    args = parser.parse_args()
    maskfiles = get_files(args.maskfolder, extensions=".tif", recurse=args.recurse)
    n = len(maskfiles)
    n_train = int(args.train_ratio * n)

    rng = default_rng()
    train_idxs = rng.choice(np.arange(n), size=n_train, replace=False)
    splits = np.full(n, "valid")
    splits[train_idxs] = "train"

    df = pd.DataFrame({"slide": maskfiles.map(lambda x: x.stem), "split": splits})
    df.to_csv(args.out_csv, index=False)
