import torch
import torchani
import time
import timeit
import argparse
import pkbar
from torchani.units import hartree2kcalmol


if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--ensembles_size',
                        help='Number of ensembles',
                        default=4, type=int)
    parser.add_argument('-i', '--ensemble_index',
                        help='Index of current ensemble (zero-indexed)',
                        default=0, type=int)
    parser.add_argument('-s', '--seed',
                        help='Seed for reproducible shuffling',
                        default=12345, type=int)
    parser.add_argument('-b', '--batch_size',
                        help='Number of conformations of each batch',
                        default=2560, type=int)
    parser = parser.parse_args()

    # for reproducible shuffling
    torch.manual_seed(parser.seed)

    shuffled_indices = torch.randperm(22)
    shuffled_indices_ensembles = torch.chunk(shuffled_indices, parser.ensembles_size)
    training_ensemble_indices = [i for i in range(parser.ensembles_size) if i != parser.ensemble_index]
    training_indices = [d for i, d in enumerate(shuffled_indices_ensembles) if i != parser.ensemble_index]
    training_indices = torch.cat(training_indices)
    validation_indices = shuffled_indices_ensembles[parser.ensemble_index]
    print(shuffled_indices)
    print(shuffled_indices_ensembles)
    print('Validation dataset ensemble index: {}'.format(parser.ensemble_index))
    print('Training dataset ensemble index: {}'.format(training_ensemble_indices))
    print('Validation dataset index: {}'.format(validation_indices))
    print('Training dataset index: {}'.format(training_indices))
    print()
