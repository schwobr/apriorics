import argparse
import json
import sqlite3

from apriorics.data_utils import create_database, get_info_from_filename
from ordered_set import OrderedSet
from pathaia.util.paths import get_files

parser = argparse.ArgumentParser()
parser.add_argument("registry")
parser.add_argument("--data-path", required=True)
parser.add_argument("--recurse", action="store_true")
parser.add_argument("--extension", default=".svs")
parser.add_argument("--mapping-file", required=True)


if __name__ == "__main__":
    args = parser.parse_known_args()[0]
    create_database(args.registry, "slides")

    with sqlite3.connect(args.registry) as con:
        db_ids = OrderedSet(
            map(lambda x: x[0], con.execute("SELECT id FROM slides").fetchall())
        )

        files = get_files(
            args.data_path, extensions=args.extension, recurse=args.recurse
        )
        file_ids = OrderedSet(map(lambda x: x.stem, files))

        to_add = file_ids - db_ids

        with open(args.mapping_file, "r") as f:
            mapping = json.load(f)

        for id in sorted(to_add):
            entries = get_info_from_filename(id, mapping)
            entries["id"] = id
            idx = file_ids.index(id)
            entries["path"] = str(files[idx].relative_to(args.data_path))
            cmd = """
            INSERT INTO slides (id, block, slide_type, ihc_type, path)
            VALUES(:id, :block, :slide_type, :ihc_type, :path)
            """
            con.execute(cmd, entries)
