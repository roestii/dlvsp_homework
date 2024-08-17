import torch
import numpy as np

import sys
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()
    if not os.path.isdir(args.input):
        sys.exit(f"{args.input} not found.")

    if not os.path.isdir(args.output):
        sys.exit(f"{args.output} not found.")

    max = 0
    for _, _, files in os.walk(args.input):
        for file in files:
            i = file.find("_")
            num = int(file[i + 1:-4])
            if num > max:
                max = num

    for i in range(0, max + 1):
        emb_path = os.path.join(args.input, f"embedding_{i}.pth")
        cl_path = os.path.join(args.input, f"class_{i}.pth")
        embs = torch.load(emb_path, map_location=torch.device("cpu"))
        cl = torch.load(cl_path, map_location=torch.device("cpu"))

        assert(len(embs) == len(cl))
        for k in range(len(cl)):
            c = cl[k].item()
            e = embs[k]

            p = os.path.join(args.output, str(c))
            if not os.path.exists(p):
                os.mkdir(p)

            fpath = os.path.join(p, f"{i}_{k}.npy")
            np.save(fpath, e.numpy())

        del embs
        del cl

if __name__ == "__main__":
    main()
