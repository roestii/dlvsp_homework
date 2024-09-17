# Steps to reproduce
**NOTE**:   Some directories have to be created prior to executing the commands. Otherwise, errors might be thrown.
            The fine-tuning of the I-JEPA was performed on a google colab instance with an A100 40GB GPU. Running 
            the fine-tuning or inference locally may not be suitable. The google colab notebooks can be found 
            in the `colab_notebooks` folder. Please contact me if you are interested in the fine-tuned weights or 
            the precomputed embeddings from both the I-JEPA and baseline model.

### Fine-tuning the I-JEPA model
```
chmod +x ./make_dataset.sh
python3 fine_tune.py --config configs/f101_vith14_ep300.yaml
```
Save the latest model under `checkpoints/f101_finetuned_ijepa_ep310.pth.tar`.

### Precompute embeddings using both the fine-tuned I-JEPA and the baseline model
```
python3 make_embeddings_main.py --config configs/f101_finetuned_ep310.yaml
python3 make_embeddings_main.py --config configs/make_embeddings_base.yaml
```
 
### Unpack embedding files
```
python3 unpack_embeddings.py --input embeddings/val --output embeddings_unpacked/val
```

### Visualize embeddings
```
python3 visualize_embeddings.py --emb-path embeddings_unpacked/val --include-n 200 --plot-path plots/plot.png
```

### Train and evaluate the base learner
```
python3 baselearner_main.py --emb-path embeddings_unpacked/val --include-n 5 --k 10 --test-size 100 --base-learner logistic_regression --output test_results
```
