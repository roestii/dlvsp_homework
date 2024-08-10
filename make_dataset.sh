#!/bin/bash

cd datasets

curl -L http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz > food-101.tar.gz
tar xf food-101.tar.gz
cd food-101/images

mkdir ../train
mkdir ../val
mkdir ../test

N_TRAIN=$(($(ls | wc -l) * 7 / 10))
N_VAL=$((($(ls | wc -l) - $N_TRAIN) / 2))
N_TEST=$(($(ls | wc -l) - $N_TRAIN - $N_VAL))

ls | head -n $N_TRAIN | xargs -I {} mv -t ../train {}
echo "copied the files to train"
ls | head -n $N_VAL | xargs -I {} mv -t ../val {}
echo "copied the files to val"
ls | head -n $N_TEST | xargs -I {} mv -t ../test {}
echo "copied the files to test"

cd ..
ls | grep -v -e train -e val -e test | xargs rm -rf
