from abc import abstractmethod

import matplotlib.pyplot as plt
import numpy as np

import utils.mri
from utils.mri import scale_traj, map_to_hysteresis, scale_nav
from datasets.utils_abdominal import load_abdominal_data
from datasets.utils_cardiac import load_cardiac_data
from datasets.utils_knee import load_knee_data


class RadialDatasetBase(object):
    def __init__(self, config):
        """
        Custom PyTorch Dataset for loading and processing radial data

        Parameters:
            config (dict): Configuration dictionary containing necessary parameters.
        """

        ### Default Settings
        self.config = config
        self.config["data_type"] = self.config["data_type"] if "data_type" in self.config else "abdominal"
        self.config["dataset"]["nav_drift"] = self.config["dataset"]["nav_drift"] if "nav_drift" in self.config["dataset"] else False
        self.config["dataset"]["hysteresis"] = self.config["dataset"]["hysteresis"] if "hysteresis" in self.config["dataset"] else False
        self.nav_min = self.config["dataset"]["nav_min"] if "nav_min" in self.config["dataset"] else 0
        self.nav_max = self.config["dataset"]["nav_max"] if "nav_max" in self.config["dataset"] else 1.

        ### Load data depending on data_type
        if self.config["data_type"] == "cardiac_cine":
            data_dict = load_cardiac_data(config)
        elif self.config["data_type"] == "abdominal_sos":
            data_dict = load_abdominal_data(config)
        elif self.config["data_type"] == "knee":
            data_dict = load_knee_data(config)
        else:
            raise Exception("Data type {} not supported".format(self.config["data_type"]))

        ### Assign the variables to instance variables
        self.config.update(data_dict["config"])
        for key, value in data_dict.items():
            if key != "config":
                setattr(self, key, value)

        ### Dimensions
        self.im_size = self.csm.shape
        self.n_fe = self.config["fe_steps"]
        self.n_slices = self.config["n_slices"]
        self.n_coils = self.im_size[0]
        self.n_echo = self.kdata.shape[0]
        if "n_coils" in config:
            assert self.n_coils == self.config["nc_recon"]
        if "n_echo" in config:
            assert self.n_echo == self.config["n_echo"]

        ### reshape data
        self.kdata = self.kdata.reshape(self.n_echo, self.n_coils, self.n_slices, -1, self.n_fe)  # (ech, coils, slices, spokes, FE)
        self.traj = self.traj.reshape(self.n_echo, -1, self.n_fe, 2)
        print("Trajectory shape:", self.traj.shape)
        self.n_spokes = self.kdata.shape[-2]  # imsize is coils * x * y

        ### Data processing
        # if "acc_factor" in self.config["dataset"]:
        #     if self.config["data_type"] == "abdominal":
        #         self.retrospectiveAcceleration(acc_factor=self.config["dataset"]["acc_factor"])
        if "fe_crop" in self.config["dataset"] and self.config["dataset"]["fe_crop"] < 1:
            self.traj_orig = self.traj.copy()
            self.traj, self.kdata, self.n_fe = utils.mri.fe_crop(self.traj, self.kdata, fe_crop=self.config["dataset"]["fe_crop"])
        if self.config["dataset"]["nav_drift"]:     ## Preprocess navigator
            self.removeNavigatorDrift(plot=False)
        if self.config["dataset"]["hysteresis"]:     ## account for hysteresis effect in navigator
            self.mapHysteresis()

        self.scaleNavigator()
        self.traj = scale_traj(self.traj, max_value=1.0) ### Scale trajectory to -1 to 1

        ### Consistency checks
        print("Original image shape", self.im_size)
        if "coil_select" in config and self.n_coils != len(config["coil_select"]):
            print("Careful: coil selection does not match sensitivity maps")
        assert self.kdata.shape[-2] == self.traj.shape[-3]

        ### Data Selection (Reduce slice and echo)
        self.ref_all = self.ref  # assumes x,y,z,dyn, ech
        self.select_slice()
        if self.config["echo"] is not None:
            self.select_echo()

        ### Normalize data
        self.kdata /= np.max(np.abs(self.kdata))  # ToDO: move normalization, consider coils
        if "scale_coils" in self.config["dataset"] and self.config["dataset"]["scale_coils"]:
            self.coil_factors = np.max(np.abs(self.kdata), axis=(1, 2))
            self.kdata /= self.coil_factors[:, None, None]

    def select_slice(self):
        slice = self.config['slice']
        self.kdata = self.kdata[:, :, slice, :, :]  # minimize data
        # self.ref_all = self.ref     # assumes x,y,z,dyn, ech
        if self.ref is not None and self.ref.shape[2] > 1: # only if multiple slices in ref
            self.ref = self.ref[:, :, [slice], :, :]

    def select_echo(self):
        echo = self.config['echo'] if "echo" in self.config else 0
        self.n_echo = 1

        self.kdata = self.kdata[echo, :, :, :]  # minimize data
        self.traj = self.traj[echo, :]
        self.weights = self.weights[echo, :, :] if self.weights is not None else None

        # self.ref_all = self.ref     # assumes x,y,z,dyn, ech
        self.ref = self.ref[..., [echo]] if self.ref is not None else None

    def select_data(self):
        slice = self.config['slice']
        echo = self.config['echo'] if "echo" in self.config else 0
        self.n_echo = 1

        self.kdata = self.kdata[echo, :, slice, :, :]  # minimize data
        self.traj = self.traj[echo, :]
        self.weights = self.weights[echo, :, :] if self.weights is not None else None

        self.ref_all = self.ref     # assumes x,y,z,dyn, ech
        if self.ref is not None and self.ref.shape[2] > 1: # only if multiple slices in ref
            self.ref = self.ref[:, :, [slice], :, :]
        self.ref = self.ref[:, :, :, :, [echo]] if self.ref is not None else None

    def removeNavigatorDrift(self, plot=True):
        nav_orig = self.self_nav.copy()
        coefficients = np.polyfit(np.arange(len(nav_orig)), nav_orig, 1)
        fitted_curve = np.polyval(coefficients, np.arange(len(nav_orig)))
        self.self_nav = self.self_nav - fitted_curve
        if plot:
            plt.figure(figsize=(10,3))
            plt.plot(nav_orig, '-g', label="Original navigator")
            plt.plot(self.self_nav, 'k',label="Shifted navigator")
            plt.plot(fitted_curve, 'r')
            plt.legend()
            plt.title("Navigator signal")
            plt.show()

    def scaleNavigator(self):

        self.self_nav = scale_nav(self.self_nav, self.nav_min, self.nav_max)
        if self.map2ms is not None:
            self.map2ms = scale_nav(self.map2ms, self.nav_min, self.nav_max)

    def retrospectiveAcceleration(self, acc_factor):
        nspoke_acc = int(self.kdata.shape[-2] / acc_factor)
        self.kdata = self.kdata[:, :, :, :nspoke_acc, :]
        self.traj = self.traj[:, :nspoke_acc, :, :]
        self.weights = self.weights[:, :nspoke_acc, :] if self.weights is not None else None
        self.self_nav = self.self_nav[:nspoke_acc]
        self.acc = acc_factor # ToDo: MArk all preprocessing
        self.n_spokes = self.kdata.shape[-2] # overwrite spokes

    @abstractmethod
    def create_sample_points(self) -> None:
        """Create the sample pairs of coordinates and signal values"""
        pass
    @abstractmethod
    def __len__(self):
        """Return the number of samples in the dataset."""
        pass
    @abstractmethod
    def __getitem__(self, index):
        """Get a sample from the dataset at the specified index."""
        pass