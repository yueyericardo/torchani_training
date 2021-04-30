import torch
import torchani
import argparse
import pkbar
import numpy as np
from torchani.units import hartree2kcalmol

H_network = torch.nn.Sequential(
    torch.nn.Linear(384, 160),
    torch.nn.CELU(0.1),
    torch.nn.Linear(160, 128),
    torch.nn.CELU(0.1),
    torch.nn.Linear(128, 96),
    torch.nn.CELU(0.1),
    torch.nn.Linear(96, 1)
)

C_network = torch.nn.Sequential(
    torch.nn.Linear(384, 144),
    torch.nn.CELU(0.1),
    torch.nn.Linear(144, 112),
    torch.nn.CELU(0.1),
    torch.nn.Linear(112, 96),
    torch.nn.CELU(0.1),
    torch.nn.Linear(96, 1)
)

N_network = torch.nn.Sequential(
    torch.nn.Linear(384, 128),
    torch.nn.CELU(0.1),
    torch.nn.Linear(128, 112),
    torch.nn.CELU(0.1),
    torch.nn.Linear(112, 96),
    torch.nn.CELU(0.1),
    torch.nn.Linear(96, 1)
)

O_network = torch.nn.Sequential(
    torch.nn.Linear(384, 128),
    torch.nn.CELU(0.1),
    torch.nn.Linear(128, 112),
    torch.nn.CELU(0.1),
    torch.nn.Linear(112, 96),
    torch.nn.CELU(0.1),
    torch.nn.Linear(96, 1)
)

if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path',
                        help='Path of the dataset, can a hdf5 file \
                            or a directory containing hdf5 files')
    parser.add_argument('-d', '--device',
                        help='Device of modules and tensors',
                        default=('cuda' if torch.cuda.is_available() else 'cpu'))
    parser.add_argument('-b', '--batch_size',
                        help='Number of conformations of each batch',
                        default=2560, type=int)
    parser.add_argument('-n', '--num_epochs',
                        help='epochs',
                        default=3, type=int)
    parser.add_argument('-s', '--seed',
                        help='Seed for reproducible shuffling',
                        default=12345, type=int)
    parser = parser.parse_args()
    # for reproducible shuffling
    torch.manual_seed(parser.seed)

    # Variables for radial and angular equations
    # https://pubs.acs.org/na101/home/literatum/publisher/achs/journals/content/jcisd8/2020/jcisd8.2020.60.issue-7/acs.jcim.0c00451/20200722/images/large/ci0c00451_0001.jpeg
    Rcr = 5.2000e+00  # Rc for radial
    Rca = 3.5000e+00  # Rc for angular
    EtaR = torch.tensor([1.6000000e+01], device=parser.device)  # Eta for radial
    ShfR = torch.tensor([9.0000000e-01, 1.1687500e+00, 1.4375000e+00, 1.7062500e+00, 1.9750000e+00, 2.2437500e+00, 2.5125000e+00, 2.7812500e+00, 3.0500000e+00, 3.3187500e+00, 3.5875000e+00, 3.8562500e+00, 4.1250000e+00, 4.3937500e+00, 4.6625000e+00, 4.9312500e+00], device=parser.device)  # Shift for radial
    Zeta = torch.tensor([3.2000000e+01], device=parser.device)  # Zeta for angular
    ShfZ = torch.tensor([1.9634954e-01, 5.8904862e-01, 9.8174770e-01, 1.3744468e+00, 1.7671459e+00, 2.1598449e+00, 2.5525440e+00, 2.9452431e+00], device=parser.device)  # theta_s for angular
    EtaA = torch.tensor([8.0000000e+00], device=parser.device)  # Eta for angular
    ShfA = torch.tensor([9.0000000e-01, 1.5500000e+00, 2.2000000e+00, 2.8500000e+00], device=parser.device)  # R_s for angular
    num_species = 4
    species_order = ['H', 'C', 'N', 'O']
    aev_computer = torchani.AEVComputer(Rcr, Rca, EtaR, ShfR, EtaA, Zeta, ShfA, ShfZ, num_species)
    lr = 0.000001 #learning rate

    nn = torchani.ANIModel([H_network, C_network, N_network, O_network])
    model = torch.nn.Sequential(aev_computer, nn).to(parser.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse = torch.nn.MSELoss(reduction='none')

    print('=> loading dataset...')
    energy_shifter = torchani.utils.EnergyShifter(None)
    training_dataset, validation_dataset = torchani.data.load(parser.dataset_path).subtract_self_energies(energy_shifter, species_order).species_to_indices(species_order).shuffle().split(0.8, None)
    training_dataset = training_dataset.collate(parser.batch_size).cache()
    validation_dataset = validation_dataset.collate(parser.batch_size).cache()

    print('=> start training')

    for epoch in range(0, parser.num_epochs):
        # training
        model.train()
        print('Epoch: %d/%d' % (epoch + 1, parser.num_epochs))
        progbar = pkbar.Kbar(target=len(training_dataset), width=8)

        for i, properties in enumerate(training_dataset):
            species = properties['species'].to(parser.device)
            coordinates = properties['coordinates'].to(parser.device).float()
            true_energies = properties['energies'].to(parser.device).float()
            num_atoms = (species >= 0).sum(dim=1, dtype=true_energies.dtype)
            _, predicted_energies = model((species, coordinates))
            loss = (mse(predicted_energies, true_energies) / num_atoms.sqrt()).mean()
            rmse = hartree2kcalmol((mse(predicted_energies, true_energies)).mean()).detach().cpu().numpy()
            loss.backward()
            optimizer.step()

            progbar.update(i, values=[("train_rmse", rmse)])

        # validation
        model.eval()
        val_rmse_kcal = []

        with torch.no_grad():
            for i, properties in enumerate(validation_dataset):
                species = properties['species'].to(parser.device)
                coordinates = properties['coordinates'].to(parser.device).float()
                true_energies = properties['energies'].to(parser.device).float()
                num_atoms = (species >= 0).sum(dim=1, dtype=true_energies.dtype)
                _, predicted_energies = model((species, coordinates))
                loss = (mse(predicted_energies, true_energies) / num_atoms.sqrt()).mean()
                rmse = hartree2kcalmol((mse(predicted_energies, true_energies)).mean()).detach().cpu().numpy()

                val_rmse_kcal.append(rmse)
            val_rmse_kcal_mean = np.mean(val_rmse_kcal)
            progbar.add(1, values=[("val_rmse_kcal", val_rmse_kcal_mean)])
