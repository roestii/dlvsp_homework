import sys
import os
import yaml

from src import meta_val_ijepa

def main():
    if len(sys.argv) < 2 or sys.argv[1] != "--config":
        sys.exit("Please pass in a config.")

    config_path = sys.argv[2] 
    if not os.path.exists(config_path):
        sys.exit("The provided config file does not exist.")

    with open(config_path, "r") as y_file:
        args = yaml.load(y_file, Loader=yaml.FullLoader)
        meta_val_ijepa.main(args)

if __name__ == "__main__":
    main()
