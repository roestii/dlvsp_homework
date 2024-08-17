import argparse
import sys
import json
import os

from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from src.datasets.embeddings import KFold
from src.utils.logging import AverageMeter

def map_classifier(args):
    match args.base_learner:
        case "logistic_regression":
            classifier = LogisticRegression()
        case "k_nearest_neighbor":
            classifier = KNeighborsClassifier(n_neighbors=args.include_n)
        case "svm":
            classifier = SVC()
        case _: 
            sys.exit("this cannot happen")
    return classifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb-path", type=str, required=True)
    parser.add_argument("--include-n", type=int, required=True)
    parser.add_argument(
        "--base-learner", 
        type=str,  
        required=True, 
        choices=["logistic_regression", "k_nearest_neighbor", "svm"]
    )
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--output", type=str, default="results/")
    parser.add_argument("--test-size", type=int, required=True)

    args = parser.parse_args()

    kfold = KFold(
        args.emb_path, 
        args.include_n, 
        args.k, 
        args.test_size
    )
    val_it = iter(kfold)
    avg_acc = AverageMeter()
    avg_recall = AverageMeter()
    avg_precision = AverageMeter()

    for i, (train, test) in enumerate(val_it):
        print(f"kfold epoch: {i}")
        classifier = map_classifier(args) 
        classifier.fit(train.x, train.y) 
        y_pred = classifier.predict(test.x)

        acc = accuracy_score(test.y, y_pred)
        recall = recall_score(test.y, y_pred, average="micro")
        precision = precision_score(test.y, y_pred, average="micro")

        avg_acc.update(acc)
        avg_recall.update(recall)
        avg_precision.update(precision)

    res = {
        "avg_accuracy": avg_acc.avg, 
        "avg_recall": avg_recall.avg,
        "avg_precision": avg_precision.avg
    }

    model_kind = "base" if "base" in args.emb_path else "ijepa"
    fname = f"{model_kind}_{args.base_learner}_{args.include_n}shot.json"
    fpath = os.path.join(args.output, fname)
    with open(fpath, "w") as out_file:
        json.dump(res, out_file)

if __name__ == "__main__": 
    main()
