import argparse
import os

import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def visualize(emb_path, include_n, plot_path):
    embeddings = []
    class_labels = []

    for entry in os.listdir(emb_path):
        p = os.path.join(emb_path, entry)
        if not os.path.isdir(p):
            continue

        print(f"files for {p}")

        for i, fname in enumerate(os.listdir(p)):
            if i >= include_n:
                break

            fpath = os.path.join(p, fname)
            emb = np.load(fpath).reshape(1, -1)[0]
            embeddings.append(emb)
            class_labels.append(int(entry))

    embeddings = np.array(embeddings)
    class_labels = np.array(class_labels)
    transformed = TSNE(
        n_components=2, 
        perplexity=3,
        n_iter=5000
    ).fit_transform(embeddings)

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(
        transformed[:, 0], 
        transformed[:, 1], 
        c=class_labels, 
        cmap="tab20"
    )
    plt.savefig(plot_path)
    plt.close()
    # plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb-path", required=True)
    parser.add_argument("--include-n", type=int, required=True)
    parser.add_argument("--plot-path", required=True)

    args = parser.parse_args()
    visualize(args.emb_path, args.include_n, args.plot_path)

if __name__ == "__main__": 
    main()
