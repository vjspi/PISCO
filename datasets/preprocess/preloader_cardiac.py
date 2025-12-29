import os
import matplotlib.pyplot as plt
import scipy.io as sio
import torch
import numpy as np
import medutils.visualization as vis

import utils.mri
from utils.mri import generateRadialTrajectory, generateRadialTrajectory_v2, fillHermitianConjugate, center_crop
from utils.basic import torch2numpy, numpy2torch
from pathlib import Path
from datasets.utils_cardiac import find_subject_file, get_subject_params
class DataLoader:
    def __init__(self, config):
        """
        Custom DataLoader depending on structure of raw data
        Parameters:
            config (dict): Configuration dictionary containing necessary parameters.
        """
        self.config = config
        self.device = self.config["gpu"]
        ## Default
        self.config["slice"] = 0

        self.base_path = os.path.join(Path.home(), self.config['data_root'])
        self.raw_path = os.path.join(self.base_path, "raw")

    def load_raw_data(self):


        # Find Data set and load data
        filename = find_subject_file(self.raw_path, self.config["subject_name"])
        filepath = f"{self.raw_path}/{filename}"
        self.filename = filename
        config_file = get_subject_params(filename)
        self.config.update(config_file)

        ## Load data
        # data comes in shape [slice, phases, nFE (zero-padded), nSpokes, nCoils, 2 (Re/Imag)]
        data = sio.loadmat(filepath)
        kdata = np.array(data["data"][...,0] + 1j * data["data"][...,1], dtype=np.complex64)
        kdata = kdata.transpose(0,1,3,2,4) # slices, phases, spokes, nFE, coils
        kdata = kdata.reshape(self.config["n_slices"], -1, self.config["fe_steps"], self.config["nc_recon"]) # slices, phases * spokes, nFE, coils
        kdata = kdata.transpose(3,0,1,2)    # coils, slices, phases*spokes, nFE
        kdata = kdata[None,...]             # add echo dimension
        # kdata = kdata / np.max(np.abs(kdata))

        ### Fill unsymmetric k-space due to Partial Fourier acquisition
        # start_idx, end_idx = find_nonzero_complex_indices(kdata[0,0,0,0,:])
        kdata_partialFourier = kdata.copy()
        kdata = fillHermitianConjugate(kdata_partialFourier)

        ## remove zerofilling
        mask = kdata == 0j
        kdata_crop = kdata[~mask].reshape(*kdata.shape[:-1], -1).copy()
        kdata_crop = kdata_crop[..., :-1] # drop last value to center cropped k-space (e.g. start with 1, have middle index at 0 and end at (1-delta_fe)
        kdata = kdata_crop.copy()
        self.config["fe_steps"] = kdata_crop.shape[-1]

        # create navigator
        self_nav = np.arange(0,self.config["phases"])[:,None].repeat(self.config["n_spokes"], 1)
        self_nav = self_nav.reshape(-1)  # phases * spokes

        # create trajectory
        # traj = generateRadialTrajectory_v2(Nread=self.config["fe_steps"], Nspokes=self.config["n_spokes"])   # 2, nFE, spokes
        # traj = traj.transpose(2,1,0)
        traj = generateRadialTrajectory(self.config["fe_steps"], self.config["n_spokes"], kmax=1.0)   # 2, nFE, spokes
        traj = traj[...,None,:].repeat(self.config["phases"], -2)       # 2, nFE, phases, spokes
        traj = traj.reshape(2, self.config["fe_steps"], -1)             # 2, nFE, phases * spokes
        traj = traj.transpose(2,1,0)                                    # phases * spokes, nFE, 2
        traj = traj[None,...]
        # traj[...,0] = -1. * traj[...,0]                               # Flip trajectory

        print("FE steps after hermitian fill and zerocropping: {}".format(self.config["fe_steps"]))

        # Weights
        weights = None

        ### Compute sensmaps with espirit on raw data
        self.orig_size = [int(self.config["fe_steps"]), int(self.config["fe_steps"])]
        self.im_size = [self.config["nx"], self.config["ny"]]


        csm_path = self.base_path + "/smaps_oversampled/" + self.filename
        os.makedirs(os.path.dirname(csm_path), exist_ok=True)
        if os.path.exists(csm_path + ".npz"):
            csm = np.load(csm_path + ".npz")["csm"]
        else:
            from utils.espirit import espirit_from_radial_wrapper
            csm, img_cart = espirit_from_radial_wrapper(kdata, traj = traj, im_size=self.orig_size,
                                                        device=self.device, debug=False,
                                                        ncalib=None, k=6, r=24, t=0.01, c=0.95)
            csm = csm.permute(3,0,1,2) # reformat to c, x, y, z
            # debug
            img_cc = utils.mri.coilcombine(img_cart[0,:,0,:,:], csm[..., 0], coil_dim=0)
            vis.imshow(vis.contrastStretching(vis.plot_array(torch2numpy(img_cc))), title="NUFFT: Coil images - Mag")
            vis.imshow(vis.contrastStretching(vis.plot_array(np.angle(torch2numpy(img_cc)))), title="NUFFT: Coil images - Phase")
            plt.show()
            csm = torch2numpy(csm)
            ### Adjust coil sensitivity to desired image size to save memory
            # csm = csm.transpose(0,3,1,2) # bring x and y to last dimension
            # csm = center_crop(csm, shape=[self.config["nx"], self.config["ny"]])
            # csm = csm.transpose(0,2,3,1) # bring z back to last dimension: c, x, y, z
            # Save sensemaps
            # np.savez(csm_path + ".eps", csm=csm)
            np.savez(csm_path, csm=csm)

        vis.imshow(vis.contrastStretching(vis.plot_array(torch2numpy(csm[..., 0]))), title="NUFFT:coil sensitivity maps - Mag")
        vis.imshow(vis.plot_array(np.angle(torch2numpy(csm[..., 0]))), title="NUFFT:coil sensitivity maps - Phase")
        plt.show()
        vis.imsave(vis.contrastStretching(vis.plot_array(torch2numpy(csm[..., 0]))), csm_path + "mag" + ".png")
        vis.imsave(vis.plot_array(np.angle(torch2numpy(csm[..., 0]))), csm_path + "phase" + ".png")

        ref = None

        self.kdata = kdata      # echo, coils, slices, phases*spokes, nFE
        self.weights = weights
        self.ref = ref
        self.csm = csm          # c, x, y, z
        self.traj = traj        # echo, slices, phases*spokes, nFE, 2
        self.self_nav = self_nav    # phases*spokes
        # return kdata, traj, self_nav, ref, csm, weights

    def compute_reference(self, reference_path):

        os.makedirs(reference_path, exist_ok=True)
        save_file_path = os.path.join(reference_path, self.filename + ".npz")
        save_file_png = os.path.join(reference_path, self.filename + ".png")
        if os.path.exists(save_file_path):
            self.ref = np.load(save_file_path)["ref"]
        else:
            kdata = self.kdata.copy().reshape(-1, self.config["nc_recon"], self.config["n_slices"], self.config["phases"],
                                              self.config["n_spokes"] *  self.config["fe_steps"]) # echos, coils, slices, phases, spokes*nFE
            traj = self.traj.copy().reshape(-1, self.config["phases"], self.config["n_spokes"] * self.config["fe_steps"], 2) # echos, phases, spokes*nFE, 2
            traj = traj.transpose(0, 1, 3, 2)
            csm = self.csm.copy() # c, x, y, z
            csm = csm.transpose(3,0,1,2) # to z, c, x, y
            n_echos = kdata.shape[0]
            img = torch.zeros(n_echos, self.config["phases"], self.config["n_slices"],
                              *self.im_size, dtype=torch.complex64, device=self.device)

            # echo and phase dim could be collapsed for faster processing
            for ech in range(n_echos):
                for p in range(self.config["phases"]):
                    for sl in range(self.config["n_slices"]):
                        from utils.mri import mriRadialAdjointOp
                        img_coils =  mriRadialAdjointOp(kdata[[ech], :, sl, p, ...], shape=self.orig_size,
                                 traj=traj[[ech],p, ...], dcf="calc", csm=None, osf=2, device=self.device)
                        # separated to centercrop smaps to im_size if needed (not possible in mriRadialAdjointOp)
                        assert csm.shape[-1] >= self.im_size[-1]
                        img[ech, p, sl, :, :] = utils.mri.coilcombine(img_coils[sl,...],
                                                                      numpy2torch(csm[sl,...], device=self.device),
                                                                      im_size=self.im_size,
                                                                      coil_dim=0)

            self.ref = torch2numpy(img)
            np.savez(save_file_path, ref=self.ref)

        vis.imshow(vis.plot_array(self.ref[0,:,0,...]), title="Reference recon - mag")
        vis.imshow(vis.plot_array(np.angle(self.ref[0,:,0,...])), title="Reference recon - phase")
        plt.show()
        vis.imsave(vis.plot_array(self.ref[0,:,0,...]), save_file_png)

    def save_processed_data(self, processed_path):
        '''
        returns
        kdata: numpy array [ech, c, z, nFE*nPE]  - echo, coils, slices, phases*spokes, nFE
        traj: numpy array [ech, nFE*nPE, 2]
        self_nav: [nPE]
        csm: numpy array [c,x,y,z]
        ref: [x, y, slices, ms, ech*dyn]
        '''

        # Save processed data
        os.makedirs(processed_path, exist_ok=True)
        save_file_path = os.path.join(processed_path,self.filename + ".npz")
        np.savez(save_file_path,
                 kdata = self.kdata,
                 weights = self.weights,
                 self_nav = self.self_nav,
                 ref = self.ref,
                 csm = self.csm,
                 traj = self.traj,
                 acc_factor = self.config["acc_factor"])

        print(f"Processed data saved to: {save_file_path}")

    def retrospectiveUndersampling_spoke(self, spokes=14):

        acc_factor = self.config["n_spokes"] / spokes
        self.retrospectiveUndersampling(acc_factor=acc_factor)

    def retrospectiveUndersamplingPseudoGoldenAngle(self, spoke_number, pseudo_angle=111.25):

        kdata = self.kdata.copy()
        traj = self.traj.copy()
        self_nav = self.self_nav.copy()

        # number of spokes
        nspokes_full_phase = self.config["n_spokes"]
        # nspokes_full_all = nspokes_full_phase * self.config["phases"]
        nspokes_acc_phase = spoke_number
        nspokes_acc_all = nspokes_acc_phase * self.config["phases"]


        pseudo_angle_radian = pseudo_angle * np.pi / 180
        phis_pseudogolden = np.array([((pseudo_angle_radian * n) % np.pi) for n in range(nspokes_acc_all)])
        phis_pseudogolden = phis_pseudogolden.reshape(self.config["phases"], nspokes_acc_phase)

        # angles for full sampling
        # dphi_full =  (np.mod(nspokes_full_phase, 2) + 1) * np.pi * 1 / nspokes_full_phase
        dphis_full = np.array([(np.mod(nspokes_full_phase, 2) + 1) * np.pi * n / nspokes_full_phase for n in range(nspokes_full_phase)])

        # angles for accelerated sampling
        # dphi_acc = (np.mod(nspokes_acc_phase, 2) + 1) * np.pi * 1 / nspokes_acc_phase
        # dphis_acc = np.array([(np.mod(nspokes_acc_phase, 2) + 1) * np.pi * n / nspokes_acc_phase for n in range(nspokes_acc_phase)])

        # reshape kspace data for each phase
        kdata = kdata.reshape(-1, self.config["nc_recon"], self.config["n_slices"],
                              self.config["phases"], self.config["n_spokes"], self.config["fe_steps"])
        kdata_acc = np.zeros_like(kdata[..., :nspokes_acc_phase, :])
        traj = traj.reshape(-1, self.config["phases"], self.config["n_spokes"], self.config["fe_steps"], 2)
        traj_acc = np.zeros_like(traj[..., :nspokes_acc_phase, :, :])
        self_nav = self_nav.reshape(self.config["phases"], self.config["n_spokes"])
        self_nav_acc = np.zeros_like(self_nav[..., :nspokes_acc_phase])

        for p in range(self.config["phases"]):
            # dphi_offset = (np.pi / nspokes_acc_phase)  * (p / n_tw) # see publication: (pi * p) / (T_s * T_tw)
            # dphis = dphis_acc + dphi_offset
            dphis_idx = np.array([np.argmin(np.abs(dphis_full - phi)) for phi in phis_pseudogolden[p, :]]) # find index of closest value in dphis_full
            kdata_acc[..., p, :,:] = np.take(kdata[..., p, :,:], dphis_idx, axis=-2)
            traj_acc[:,  p,...] = np.take(traj[:,  p,...], dphis_idx, axis=-3)
            self_nav_acc[p,...] = np.take(self_nav[p,...], dphis_idx, axis=-1)

        from utils.vis import plot_trajectory
        plot_trajectory(traj_acc, one_plot=True)

        self.config["n_spokes"] = nspokes_acc_phase
        self.kdata = kdata_acc.reshape(-1, self.config["nc_recon"], self.config["n_slices"], self.config["phases"] * self.config["n_spokes"], self.config["fe_steps"])
        self.traj = traj_acc.reshape(-1, self.config["phases"] * self.config["n_spokes"], self.config["fe_steps"], 2)
        self.self_nav = self_nav_acc.reshape(self.config["phases"] * self.config["n_spokes"])

        print("Retrospective Undersampling for {} spokes per phased with pseudoangle {}°".format(spoke_number, pseudo_angle))


    def retrospectiveUndersampling(self, acc_factor=1.0, n_tw=None):

        kdata = self.kdata.copy()
        traj = self.traj.copy()
        self_nav = self.self_nav.copy()

        # number of spokes
        n_tw = self.config["phases"] if n_tw is None else n_tw      # n time window, here defined as complete cycle, in publication 7
        nspokes_full_phase = self.config["n_spokes"]
        # nspokes_full_all = nspokes_full_phase * self.config["phases"]
        nspokes_acc_phase = int(nspokes_full_phase/acc_factor)
        # nspokes_acc_all = nspokes_acc_phase * self.config["phases"]

        # angles for full sampling
        dphis_full = np.array([(np.mod(nspokes_full_phase, 2) + 1) * np.pi * n / nspokes_full_phase for n in range(nspokes_full_phase)])

        # angles for accelerated sampling
        dphis_acc = np.array([(np.mod(nspokes_acc_phase, 2) + 1) * np.pi * n / nspokes_acc_phase for n in range(nspokes_acc_phase)])

        # reshape kspace data for each phase
        kdata = kdata.reshape(-1, self.config["nc_recon"], self.config["n_slices"],
                              self.config["phases"], self.config["n_spokes"], self.config["fe_steps"])
        kdata_acc = np.zeros_like(kdata[..., :nspokes_acc_phase, :])
        traj = traj.reshape(-1, self.config["phases"], self.config["n_spokes"], self.config["fe_steps"], 2)
        traj_acc = np.zeros_like(traj[..., :nspokes_acc_phase, :, :])
        self_nav = self_nav.reshape(self.config["phases"], self.config["n_spokes"])
        self_nav_acc = np.zeros_like(self_nav[..., :nspokes_acc_phase])

        for p in range(self.config["phases"]):
            dphi_offset = (np.pi / nspokes_acc_phase)  * (p / n_tw) # see publication: (pi * p) / (T_s * T_tw)
            dphis = dphis_acc + dphi_offset
            dphis_idx = np.array([np.argmin(np.abs(dphis_full - phi)) for phi in dphis]) # find index of closest value in dphis_full
            kdata_acc[..., p, :,:] = np.take(kdata[..., p, :,:], dphis_idx, axis=-2)
            traj_acc[:,  p,...] = np.take(traj[:,  p,...], dphis_idx, axis=-3)
            self_nav_acc[p,...] = np.take(self_nav[p,...], dphis_idx, axis=-1)

        from utils.vis import plot_trajectory
        plot_trajectory(traj_acc, one_plot=True)

        self.config["n_spokes"] = nspokes_acc_phase
        self.kdata = kdata_acc.reshape(-1, self.config["nc_recon"], self.config["n_slices"], self.config["phases"] * self.config["n_spokes"], self.config["fe_steps"])
        self.traj = traj_acc.reshape(-1, self.config["phases"] * self.config["n_spokes"], self.config["fe_steps"], 2)
        self.self_nav = self_nav_acc.reshape(self.config["phases"] * self.config["n_spokes"])

        # kdata_acc = np.take(kdata, dphis_idx, axis=-2)
        print("Retrospective Undersampling for factor {} done".format(self.config["acc_factor"]))

def find_nonzero_complex_indices(arr):
    first_nonzero_idx = None
    last_nonzero_idx = None

    # Find index of first non-zero complex number
    for idx, value in enumerate(arr):
        if value != 0j:
            first_nonzero_idx = idx
            break

    # Find index of last non-zero complex number
    for idx in range(len(arr) - 1, -1, -1):
        if arr[idx] != 0j:
            last_nonzero_idx = idx
            break

    return first_nonzero_idx, last_nonzero_idx