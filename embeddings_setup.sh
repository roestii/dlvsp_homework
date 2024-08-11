#!/bin/bash

echo loading dataset
chmod +x make_dataset.sh
./make_dataset.sh

pip3 install torch torchvision torchaudio
