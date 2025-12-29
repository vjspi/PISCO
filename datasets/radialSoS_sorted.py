from torch.utils.data import Dataset
import torch
import numpy as np
import h5py
from utils.mri import downsample_csm, scale_traj
from datasets.radial_base import RadialDatasetBase

class RadialSoSDataset(RadialDatasetBase, Dataset):
    def __init__(self, config):
        """
        Custom PyTorch Dataset for handling radial SoS MRI data.

        Parameters:
            config (dict): Configuration dictionary containing necessary parameters.
        """
        ## Initialize base class (includes data loading and pre-processing)
        super().__init__(config)

        if self.config["echo"] is not None:
            self.create_sample_points()

        # initialize increment
        self.increment = 1  # 100 percent as default (all points are sampled)

    def create_sample_points(self):
        ### move coil dimension to output
        nDim = self.config["coord_dim"]
        kcoords = np.zeros((self.n_spokes, self.n_fe, nDim))  # (nav, spokes, FE, 3) -> nav, ky, (contr, slices, spokes, FE, coils, 5) -> contr, kx, ky, nc, nav
        klatent = np.zeros((self.n_spokes, self.n_fe, 1))
        kspoke = np.zeros((self.n_spokes, self.n_fe, 1))

        kcoords[..., 1] = np.reshape(self.traj[..., 0], (1, self.n_spokes, self.n_fe))  # ky
        kcoords[..., 2] = np.reshape(self.traj[..., 1], (1, self.n_spokes, self.n_fe))  # kx
        kcoords[..., 0] = np.reshape(self.self_nav, (1, self.n_spokes, 1))
        klatent[..., 0] = np.reshape(self.self_nav, (1, self.n_spokes, 1))
        kspoke[..., 0] = np.reshape(np.linspace(0, 1, self.n_spokes), (self.n_spokes, 1))

        ### put coils to output
        assert self.kdata.shape[0] == self.n_coils        # coils x spokes x FE
        kdata = np.transpose(self.kdata, (1,2,0))    # spokes x FE x coils


        ### sort data from center to outer edge if calib region is required
        # ToDo: Clean condition
        kcoords = np.reshape(kcoords.astype(np.float32), (-1, nDim))
        klatent = np.reshape(klatent.astype(np.float32), (-1, 1))
        kdatapoints = np.reshape(kdata.astype(np.complex64), (-1, self.n_coils)) # (nav*spokes*FE, 1)
        kspoke = np.reshape(kspoke.astype(np.float32), (-1, 1))
        dist_to_center = np.sqrt(kcoords[..., 1] ** 2 + kcoords[..., 2] ** 2)

        if "remove_center" in self.config["dataset"] and self.config["dataset"]["remove_center"]:
            idx_center = np.isclose(dist_to_center, 0, rtol=1e-6)
            kcoords = kcoords[~idx_center]
            klatent = klatent[~idx_center]
            kdatapoints = kdatapoints[~idx_center] # (nav*spokes*FE, 1)
            kspoke = kspoke[~idx_center]
            assert len(kcoords) == (len(dist_to_center) - np.count_nonzero(idx_center))

        if "patch_schedule" in self.config and self.config["patch_schedule"]["calib_region"] < 1:
            dist_to_center = np.sqrt(kcoords[..., 1] ** 2 + kcoords[..., 2] ** 2)
            idx = np.argsort(dist_to_center)
            kcoords = kcoords[idx]
            klatent = klatent[idx]
            kdatapoints = kdatapoints[idx] # (nav*spokes*FE, 1)
            kspoke = kspoke[idx]

        self.kcoords = kcoords
        self.klatent = klatent
        self.kdatapoints = kdatapoints
        self.kspoke = kspoke

        self.kcoords_flat = torch.from_numpy(self.kcoords)
        self.klatent_flat = torch.from_numpy(self.klatent)
        self.kdata_flat = torch.from_numpy(self.kdatapoints)
        self.kspoke_flat = torch.from_numpy(self.kspoke)
        self.n_kpoints = self.kcoords.shape[0]

    def __len__(self):
        """
        Return the number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return self.n_kpoints

    def __getitem__(self, index):
        """
        Get a sample from the dataset at the specified index.

        Parameters:
            index (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing 'coords', 'latent', 'targets', and 'kspoke' tensors for the sample.
        """
        
        # In case the pool of samples is reduced (i.e. increment <1): Map the index to another the reduced range
        # mapped_index = index projected to desired increment region (e.g. with desired increment of 0.4 and 100 kpoints, index 60 corresponds to 20)
        no_samples = int(np.floor(self.increment * self.n_kpoints))
        index = index % no_samples      # maps all data points to specified range

        # point wise sampling
        sample = {
            'coords': self.kcoords_flat[index],
            'latent': self.klatent_flat[index],
            'targets': self.kdata_flat[index],
            'kspoke': self.kspoke_flat[index]
        }
        return sample
