import socket
from argparse import ArgumentParser

import yaml

parser = ArgumentParser()
parser.add_argument("params_file")
parser.add_argument("server_params_file")


if __name__ == "__main__":
    args = parser.parse_known_args()[0]

    with open(args.params_file, "r") as f:
        params = yaml.safe_load(f)

    with open(args.server_params_file, "r") as f:
        server_params = yaml.safe_load(f)

    server = socket.gethostname().lower()
    params["base"]["server"] = server
    params["base"].update(server_params[server])

    with open(args.params_file, "w") as f:
        yaml.safe_dump(params, f)
