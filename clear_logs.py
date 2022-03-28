from argparse import ArgumentParser
import comet_ml
from pathlib import Path
import shutil


parser = ArgumentParser(prog="Clears logs from experiments that are archived on comet.")
parser.add_argument("--logfolder", type=Path, required=True)


if __name__ == "__main__":
    args = parser.parse_args()

    api = comet_ml.API()
    experiments = [
        exp.key for exp in api.get_experiments("apriorics", project_name="apriorics")
    ]

    for exp_folder in (args.logfolder / "apriorics").iterdir():
        if exp_folder.name not in experiments:
            shutil.rmtree(exp_folder)
