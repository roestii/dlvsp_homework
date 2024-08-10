#!/bin/bash

echo loading latest imagenet checkpoint
curl -L https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.14-300e.pth.tar > checkpoints/IN1K-vit.h.14-300e.pth.tar

echo loading dataset
chmod +x make_dataset.sh
./make_dataset.sh

cd ..
pip3 install torch torchvision torchaudio
