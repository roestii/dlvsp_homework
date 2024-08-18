# Steps to reproduce
**NOTE**: Some directories have to be created prior to executing the commands. Otherwise, errors might be thrown.

### Fine-tuning the I-JEPA model
```
chmod +x ./make_dataset.sh
python3 fine_tune.py --config configs/f101_vith14_ep300.yaml
```
Save the latest model under `checkpoints/f101_finetuned_ijepa_ep310.pth.tar`.

### Precompute embeddings using the fine-tuned I-JEPA
```
python3 make_embeddings_main.py --config configs/f101_finetuned_ep310.yaml
```
 
### Unpack embedding files
```
python3 unpack_embeddings.py --input embeddings/val --output embeddings_unpacked/val
```

### Visualize embeddings
```
python3 visualize_embeddings.py --emb-path datasets/embeddings_unpacked/val --include-n 200 --plot-path plots/plot.png
```

### Train and evaluate the base learner
```
python3 baselearner_main.py --emb-path datasets/embeddings_unpacked/val --include-n 5 --k 10 --test-size 100 --base-learner logistic_regression --output test_results
```
