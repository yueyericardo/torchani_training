# Torchani-training

## Create Dataset
[datapack.py](https://github.com/yueyericardo/torchani_training/blob/master/datapack.py) gives an example on how to create a h5 dataset file (dataset.h5), which could be read by torchani.
```
python datapack.py  # to generate such dataset.h5
```

## Training
[training.py](https://github.com/yueyericardo/torchani_training/blob/master/training.py) gives a miminal example on how to load dataset, and train with torchani.  
Check more detail at torchani documentation page: [Train Your Own Neural Network Potential — TorchANI 2.1.1 documentation](https://aiqm.github.io/torchani/examples/nnp_training.html)  
If GPU is available, this script will use GPU by default, you can also specify CPU by using `--device cpu`

Usage Example:
```bash
python training.py --num_epochs 10 --batch_size 100 dataset.h5
```
Output: Because this dataset's datapints are fake, it quickly went to overfit, learning rate may need to be adjusted in this case.
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

## Ensemble traning
Usage Example:
```bash
python training-ensemble.py --ensembles_size 8 --ensemble_index 0 --seed 12345 dataset.h5
```

## AEV calculation
[aev_calc.ipynb](https://github.com/yueyericardo/torchani_training/blob/master/aev_calc.ipynb)
walks through some ways on how to calculate aevs for single molecule, multiple molecules, or ase molecules. (Because of padding issue with different length molecules.)
