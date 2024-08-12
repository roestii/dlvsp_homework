import sys
import os
import yaml
import argparse

from src import train

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    if not os.path.exists(args.config):
        sys.exit("The provided config file does not exist.")

    with open(args.config, "r") as y_file:
        args = yaml.load(y_file, Loader=yaml.FullLoader)
        train.main(args)

if __name__ == "__main__":
    main()
