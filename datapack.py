# In this example, it creates a dataset.h5 file, which has 30,000 CH4 molecules and 10,000 H2O molecules
import torchani
import numpy as np

# path to created h5 file,
# in mode='w', file will be overwrite if exsist, be carefull!
# https://docs.h5py.org/en/2.3/high/file.html for other mode
dpack = torchani.data._pyanitools.datapacker('./dataset.h5', mode='w')

# CH4, fake data
species = ['C', 'H', 'H', 'H', 'H']
coordinates = np.array([[[0.03192167, 0.00638559, 0.01301679],  # coordinates shape: (1, 5, 3)
                         [-0.83140486, 0.39370209, -0.26395324],
                         [-0.66518241, -0.84461308, 0.20759389],
                         [0.45554739, 0.54289633, 0.81170881],
                         [0.66091919, -0.16799635, -0.91037834]]
                        ])
energies = np.array([-40.45, ])                                 # energies shape: (1)
# repeat to make more fake data
coordinates = np.repeat(coordinates, 30000, axis=0)             # coordinates shape: (30000, 5, 3)
energies = np.repeat(energies, 30000, axis=0)                   # energies shape: (30000)
# store CH4 data
dpack.store_data('CH4', coordinates=coordinates, energies=energies, species=species)

# H2O, fake data
species = ['O', 'H', 'H']
coordinates = np.array([[[0.03192167, 0.00638559, 0.01301679],  # coordinates shape: (1, 3, 3)
                         [-0.83140486, 0.39370209, -0.26395324],
                         [-0.66518241, -0.84461308, 0.20759389]],
                        ])
energies = np.array([-76.0, ])                                  # energies shape: (1)
# repeat to make more fake data
coordinates = np.repeat(coordinates, 10000, axis=0)             # coordinates shape: (10000, 3, 3)
energies = np.repeat(energies, 10000, axis=0)                   # energies shape: (10000)
# store H2O data
dpack.store_data('H2O', coordinates=coordinates, energies=energies, species=species)

# ArV, fake data, example of other elements
species = ['Ar', 'V']
coordinates = np.array([[[0.03192167, 0.00638559, 0.01301679],  # coordinates shape: (1, 2, 3)
                         [-0.83140486, 0.39370209, -0.26395324]]
                        ])
energies = np.array([-10.0, ])                                  # energies shape: (1)
# repeat to make more fake data
coordinates = np.repeat(coordinates, 10000, axis=0)             # coordinates shape: (10000, 2, 3)
energies = np.repeat(energies, 10000, axis=0)                   # energies shape: (10000)
# store H2O data
dpack.store_data('ArV', coordinates=coordinates, energies=energies, species=species)

dpack.cleanup()
