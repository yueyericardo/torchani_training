# Torchani-training

### Create Dataset
`datapack.py` gives an example how to create a h5 dataset file (dataset.h5), which could be read by torchani
```
python datapack.py
```

### Training
`training.py` gives a miminal example how to load dataset, and train with torchani
You could check more detail at torchani documentation page: [Train Your Own Neural Network Potential — TorchANI 2.1.1 documentation](https://aiqm.github.io/torchani/examples/nnp_training.html)
This script will use GPU by default if GPU is available. you can also specify cpu by using `--device cpu`

Usage Example:
```bash
python training.py --num_epochs 10 --batch_size 100 dataset.h5
```
Output: Because this dataset is fake datapoints, it quickly went to overfit, you may need to adjust learning rate.
```
=> loading dataset...
=> loading dataset.h5, total molecules: 2
2/2  [==============================] - 0.2s
=> loading dataset.h5, total molecules: 2
2/2  [==============================] - 0.1s
=> start training
Epoch: 1/10
320/320 [========] - 4s 13ms/step - train_rmse: 3.2684 - val_rmse_kcal: 0.0877
Epoch: 2/10
320/320 [========] - 4s 14ms/step - train_rmse: 1.8745 - val_rmse_kcal: 3.9669
Epoch: 3/10
320/320 [========] - 4s 13ms/step - train_rmse: 2.5425 - val_rmse_kcal: 0.3750
Epoch: 4/10
320/320 [========] - 4s 13ms/step - train_rmse: 1.2013 - val_rmse_kcal: 3.0925
...
```

### For ensemble traning
Usage Example:
```bash
python training-ensemble.py --ensembles_size 8 --ensemble_index 0 --seed 12345 dataset.h5
```
