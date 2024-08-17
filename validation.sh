#!/bin/bash

# Base configuration: 10 fold cross validation using a test size of 100 embeddings per class
# Parameters: 
# 	n train embeddings per class: 5 shot, 10 shot
#	base learner: logistic regression, k nearest neighbor, support vector machine
#	embedding model: resnet-18 (baseline, embeddings_base_unpacked), ijepa (embeddings_unpacked)
# NOTE: Embeddings were precomputed and saved in the corresponding folders

# Baseline validation
# Logistic regression:
echo "Starting baseline logistic regression validation, 1 shot"
python3 baselearner_main.py --emb-path datasets/embeddings_base_unpacked/val --include-n 1 --k 10 --test-size 100 --base-learner logistic_regression

echo "Starting baseline logistic regression validation, 5 shot"
python3 baselearner_main.py --emb-path datasets/embeddings_base_unpacked/val --include-n 5 --k 10 --test-size 100 --base-learner logistic_regression

echo "Starting baseline logistic regression validation, 10 shot"
python3 baselearner_main.py --emb-path datasets/embeddings_base_unpacked/val --include-n 10 --k 10 --test-size 100 --base-learner logistic_regression

# K nearest neighbor:
echo "Starting baseline k nearest neighbor validation, 1 shot"
python3 baselearner_main.py --emb-path datasets/embeddings_base_unpacked/val --include-n 1 --k 10 --test-size 100 --base-learner k_nearest_neighbor 

echo "Starting baseline k nearest neighbor validation, 5 shot"
python3 baselearner_main.py --emb-path datasets/embeddings_base_unpacked/val --include-n 5 --k 10 --test-size 100 --base-learner k_nearest_neighbor

echo "Starting baseline k nearest neighbor validation, 10 shot"
python3 baselearner_main.py --emb-path datasets/embeddings_base_unpacked/val --include-n 10 --k 10 --test-size 100 --base-learner k_nearest_neighbor

# svm:
echo "Starting baseline svm validation, 1 shot"
python3 baselearner_main.py --emb-path datasets/embeddings_base_unpacked/val --include-n 1 --k 10 --test-size 100 --base-learner svm 

echo "Starting baseline svm validation, 5 shot"
python3 baselearner_main.py --emb-path datasets/embeddings_base_unpacked/val --include-n 5 --k 10 --test-size 100 --base-learner svm

echo "Starting baseline svm validation, 10 shot"
python3 baselearner_main.py --emb-path datasets/embeddings_base_unpacked/val --include-n 10 --k 10 --test-size 100 --base-learner svm

# IJEPA validation 
# Logistic regression:
echo "Starting ijepa logistic regression validation, 1 shot"
python3 baselearner_main.py --emb-path datasets/embeddings_unpacked/val --include-n 1 --k 10 --test-size 100 --base-learner logistic_regression

echo "Starting ijepa logistic regression validation, 5 shot"
python3 baselearner_main.py --emb-path datasets/embeddings_unpacked/val --include-n 5 --k 10 --test-size 100 --base-learner logistic_regression

echo "Starting ijepa logistic regression validation, 10 shot"
python3 baselearner_main.py --emb-path datasets/embeddings_unpacked/val --include-n 10 --k 10 --test-size 100 --base-learner logistic_regression

# K nearest neighbor:
echo "Starting ijepa k nearest neighbor validation, 1 shot"
python3 baselearner_main.py --emb-path datasets/embeddings_unpacked/val --include-n 1 --k 10 --test-size 100 --base-learner k_nearest_neighbor 

echo "Starting ijepa k nearest neighbor validation, 5 shot"
python3 baselearner_main.py --emb-path datasets/embeddings_unpacked/val --include-n 5 --k 10 --test-size 100 --base-learner k_nearest_neighbor

echo "Starting ijepa k nearest neighbor validation, 10 shot"
python3 baselearner_main.py --emb-path datasets/embeddings_unpacked/val --include-n 10 --k 10 --test-size 100 --base-learner k_nearest_neighbor

# svm:
echo "Starting ijepa svm validation, 1 shot"
python3 baselearner_main.py --emb-path datasets/embeddings_unpacked/val --include-n 1 --k 10 --test-size 100 --base-learner svm 

echo "Starting ijepa svm validation, 5 shot"
python3 baselearner_main.py --emb-path datasets/embeddings_unpacked/val --include-n 5 --k 10 --test-size 100 --base-learner svm

echo "Starting ijepa svm validation, 10 shot"
python3 baselearner_main.py --emb-path datasets/embeddings_unpacked/val --include-n 10 --k 10 --test-size 100 --base-learner svm
