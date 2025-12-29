import torch
import random

from utils.loss import *
from utils import kernel_coords
## Functions for debugging purposes (in case uncommented):
from utils.vis import plot_histogram_2d, plot_coords_scatter, plot_coords_scatter2D
from utils.basic import torch2numpy

class PiscoLoss(torch.nn.Module):
    """
    PISCO Loss for self-supervised regularization of k-space
    1. Initialize the loss class with all relevant parameters (e.g. patching, weight solving, loss type)
    2. Create pairs of target and patch coordinates (either with the provided sample_patches function or manually or with a new defined function)
    3. Sample the signal values for the target and patch coordinates, receiving [output, output_patches] from [coords, coords_patch]
    4. Run forward function to compute the PISCO loss based on the inputted [output, output_patches, coords, coords_patch]
    """

    def __init__(self, config):
        '''
        :param config: dictionary containing all relevant parameters for the loss
        '''
        super().__init__()

        ################################### General Settings ########################################
        self.config = config
        self.device = self.config["gpu"]
        assert "kreg" in config
        assert "model" in config
        self.complex_handling = config["kreg"]["complex_handling"] if "complex_handling" in config["kreg"] else "mag_phase"
        assert self.complex_handling == "mag_phase" or self.complex_handling == "img_real"

        ## Get some params on dataset critical for later patching
        self.out_dim = self.config["out_dim"]       # how many output values (usually n_coils)
        self.coord_dim = self.config["coord_dim"]   # how many input coordinates (e.g. 3 for x/y/t)
        self.nxy = self.config["fe_steps"]          # number of target sample points (used to calculate kernel distance in xy)

        ### Set params for optimization (if optimized separately)
        self.optim = config["kreg"]["optimizer"]["name"] if "optimizer" in config["kreg"] else "Adam"
        self.optim_config = {key: float(value) if isinstance(value, (int, float, str)) else value
                             for key, value in config["kreg"]["optimizer"]["params"].items()}
        self.optim_scheduler = config["kreg"]["optimizer"]["scheduler"] if "scheduler" in config["kreg"]["optimizer"] else None

        ################################### PISCO Design Choices ########################################
        ### 1. Set params for patching (relevant if the sample_patches function is used) ###
        self.patch_origin = self.config["model"]["patch"]["origin"] if "origin" in self.config["model"]["patch"] else "radial"  # determines the target origins
        self.patch_type = self.config["model"]["patch"]["type"] if "type" in self.config["model"]["patch"] else "cart"          # determines the geometry/kernel how the patches are sampled
        self.patch_dist = self.config["model"]["patch"]["dist"] if "dist" in self.config["model"]["patch"] else 1.0             # distance between individual patch points
        self.patch_size = self.config["model"]["patch"]["size"] if "size" in self.config["model"]["patch"] else [3,3]           # size of the patch
        self.patch_tconst = self.config["model"]["patch"]["tconst"] if "tconst" in self.config["model"]["patch"] else False     # if the time dimension is considered in the patch
        self.patch_tdelta = self.config["model"]["patch"]["tdelta"] if "tdelta" in self.config["model"]["patch"] else False     # if the time dimension is considered in the patch, how large the temporal delta per subset may be
        self.exclude_center = self.config["model"]["patch"]["exclude_center"] if "exclude_center" in self.config["model"]["patch"] else 5 # exclude center of k-space for patching to avoid high magnitudes

        ### 2. Set parameters for weight vector (W) solving ###
        self.overdet_factor = config["kreg"]["overdetermination"] if "overdetermination" in config["kreg"] else 1.0             # number of samples per subset, if f_od > 1 more patches than variables are included (f_od * N_w = N_m)
        self.min_sets = config["kreg"]["min_sets"] if "min_sets" in config["kreg"] else 10                                      # minimum number of subsets (N_s) sampled for consistency measure computation
        self.max_sets = config["kreg"]["max_sets"] if "max_sets" in config["kreg"] else 20                                      # maximum number of subsets (N_s) sampled for consistency measure computation
        self.reg_type = config["kreg"]["reg_type"] if "reg_type" in config["kreg"] else None                                    # regularization type "reg" for weight solving of |PW - T| + lamda * reg(|W|)
        self.reg_alpha = float(config["kreg"]["reg_alpha"]) if "reg_alpha" in config["kreg"] else 0.00                          # regularization parameter "alpha" for weight solving of |PW - T| + alpha * reg(|W|)
        self.sort_sets = config["kreg"]["sort_sets"] if "sort_sets" in config["kreg"] else None                                 # sort the subsets along temporal dimension (relevant if self.patch_tdelta > 0 or tconst is False)
        self.sort_freqs = config["kreg"]["sort_freqs"] if "sort_freqs" in config["kreg"] else False                             # sort the patches along frequency dimension before sorting into subsets
                                                                                                                                # (relevant for equal magnitude distribution, e.g. center patches with high magnitudes together
                                                                                                                                # and outside patches with low magnitudes together)

        ### 3. Design choices for self-supervised PISCO loss calculation ###
        self.loss_type = config["kreg"]["loss_type"] if "loss_type" in config["kreg"] else "residual"                           # consistency measure of the weight vectors
        self.loss_norm = config["kreg"]["loss_norm"] if "reg_alpha" in config["kreg"] else "L1"                                 # reduction of all loss values
        self.reg_lamda = float(config["kreg"]["reg_lamda"]) if 'reg_lamda' in config["kreg"] else 0.01                          # regularization weight of pisco loss

        # Define out dimensions (assuming all coils are sampled for all patches)
        self.out_dim_patch = self.out_dim
        self.out_dim_target =  self.out_dim

        # Compute function for number of possible linear combinations
        self.num_sets_func = lambda x: int(x // self.n_samples_per_set) \
            if int(x // self.n_samples_per_set) < self.max_sets else self.max_sets          # function to compute the number of subsets (N_s) possible to sample from x samples

        print("Loss for k-REG used: {}".format(self.loss_type))
        print("Complex numbers split to: {}".format(self.complex_handling))                                                     # solely setting for visualization purpose
        print("k-Reg weight solving regularized by {} with weight {}".format(self.reg_type, self.reg_alpha))

    def forward(self, output, output_patch, coords, coords_patch, reduce=True):
        '''
        Forward function to compute the PISCO loss based on the inputted pairs of target and patch values (coordinate origin for sorting etc. required as well)
        :param output: stacked target values [T] for the weight solving                     [batch, out_dim]
        :param output_patch: stacked patch values [P] for the weight solving                [batch, n_neighbors, out_dim]
        :param coords: stacked target coordinates (e.g. origin of the target values T)      [batch, coord_dim]
        :param coords_patch: stacked patch coordinates (e.g. origin of the patch values P)  [batch, n_neighbors, coord_dim]
        '''

        self.n_neighbors = output_patch.shape[1]                                            # number of neighbors sampled within patch (N_n)
        self.n_weights = self.n_neighbors * self.out_dim_patch * self.out_dim_target        # number of weights to be estimated (N_w = N_n * out_dim, out_dim) - out_dim is normally the number of coils N_c
        self.n_samples_per_set = int(self.n_weights * self.overdet_factor)                  # number of samples (N_m) within one subset S which is used to solve for kernel weights W_s

        # calculate the samples needed/used
        num_samples = output.shape[0]                                                       # number of samples given within the batch in this forward pass
        self.num_sets = self.num_sets_func(num_samples)                                     # number of subsets (N_s) that can be sampled from this batch
        num_used_samples = int(self.num_sets * self.n_samples_per_set)                      # number of samples actually used for the weight solving (N_m * N_s) (to exclude incomplete sets)

        # Reshaping
        output = output.unsqueeze(-1)                                           # [batch, out_dim_targ, 1]
        output_patch = output_patch.reshape(num_samples, -1).unsqueeze(-1)      # [batch, patch * out_dim_patch, 1]
        coords_patch = coords_patch.reshape(num_samples, -1, self.coord_dim)    # [batch, n_neighbors, coords_dim]

        # Crop target and patch samples to the required amount of sampled
        T = output[:num_used_samples, :, :]
        C_T = coords[:num_used_samples, :]
        C_P = coords_patch[:num_used_samples, :]
        P = output_patch[:num_used_samples, :, :]

        # Sort target and patch samples along on temporal dimension
        if self.sort_sets == "temp":
            _, temp_sort_idx = torch.sort(C_T[:,0])
            C_T = C_T[temp_sort_idx]
            C_P = C_P[temp_sort_idx]
            T = T[temp_sort_idx]
            P = P[temp_sort_idx]

        # Sort target and patch samples along frequency dimension (low to high freqency)
        if self.sort_freqs:
            d_T = torch.sqrt(C_T[..., 1] ** 2 + C_T[..., 2] ** 2)
            _, spatial_sort_idx = torch.sort(d_T)
            C_T = C_T[spatial_sort_idx]
            C_P = C_P[spatial_sort_idx]
            T = T[spatial_sort_idx]
            P = P[spatial_sort_idx]

        ## Debugging
        # C_T = C_T.reshape(self.num_sets, self.n_samples_per_set, self.coord_dim)  # here num_patches = numweightsS, W, W - s set of W weight comcinations
        # C_P = C_P.reshape(self.num_sets, self.n_samples_per_set, self.n_neighbors, self.coord_dim)  # here num_patches = numweightsS, W, W - s set of W weight comcinations
        # max_deltat = torch.max(C_T[:, -1, 0] - C_T[:, 0, 0]).item()
        # max_deltap = torch.max(C_P[:, -1, 0, 0] - C_P[:, 0, 0, 0]).item()
        # plot_coords_scatter(data=C_T[:20,...].clone().detach().cpu(), data_patch=C_P[:20,...].clone().detach().cpu())
        # plot_coords_scatter2D(data=C_T[:20,...,1:].clone().detach().cpu(), data_patch=None)
        # d_T = torch.sqrt(C_T[..., 1] ** 2 + C_T[..., 2] ** 2)
        # plot_histogram_2d(torch2numpy(d_T), bins=30)

        T = T.reshape(self.num_sets, self.n_samples_per_set, self.out_dim_target)
        P = P.reshape(self.num_sets, self.n_samples_per_set, self.n_neighbors*self.out_dim_patch) # here num_patches = numweightsS, W, W - s set of W weight comcinations


        ###################### Start Weight Solving ######################
        ## to solve for weight vectors like in GRAPPA we need:
        #   P  : (n_samples_per_set, n_neighbors x n_coils) or (N_m, N_n x N_c)
        #   T  : (n_samples_per_set, n_coils) or (N_m, N_c)
        #   -> W : (n_neighbors x n_coils, n_coils) or (N_n x N_c, N_c)
        #
        # PW = T
        # P^H P W = P^H T
        # W = ( P^H P)^-1 * PH T
        # W = ( P^H P + lamda * I)^-1 * PH T (with regularization)
        #
        # T and P come as [num_sets, n_samples_per_set, nc] and [num_sets, n_samples_per_set, nnxnc] respectively (num_sets is in the batch dimension)
        PhP = P.mH @ P   # [n_neighbors*n_coils, n_samples_per_set] * [n_samples_per_set, n_neighbors*n_coils]
        PhT = P.mH @ T   # [n_neighbors*n_coils, n_samples_per_set] * [n_samples_per_set, n_coils]

        # Weight/LES solving is done with torch.linalg.lstsq since using pinv (which uses SVD) has some issues with complex numbers
        if self.reg_type == "Tikhonov":
            lamda0 = self.reg_alpha * torch.linalg.matrix_norm(PhP) / (PhP.shape[-1])   # frobenius norm, calculated for each batch to weigh regularization
            I = torch.eye(PhP.shape[-1]).to(PhP)
            W = torch.linalg.lstsq(PhP + lamda0.view(-1,1,1) * I, PhT).solution         # add regularization to each batch individually
        else:
            W = torch.linalg.lstsq(PhP, PhT).solution

        ## Debug for weight solving
        # import medutils.visualization as vis
        # vis.imshow(vis.plot_array(torch.abs(W[0, ...].reshape(self.n_neighbors, self.out_dim_patch, self.out_dim_target)).detach().cpu().numpy()))
        # plt.show()

        if self.loss_type in ["L1_dist"]:
            # Compute complex L1 distance between all weight vectors, e.g. sum(real(W_i - W_j) + imag(W_i - W_j)) for all i,j in S
            W_stack = torch.concat([W.real, W.imag], axis=-1)
            W_error = torch.cdist(W_stack, W_stack, p=1) / W_stack.shape[-1]    # normalize by number of weights
        elif self.loss_type in ["residual"]:
            # Compute estimated target points T_est based on solved weights, e.g. PW for all subsets S
            T_est = P @ W
            # Compute residual of estimated weights, e.g. |T_est - T| = |PW - T| for all subsets S
            # Note: torch.linalg.norm computes fro by default, with dim=-1 specified it computes the srt(x_i^2) of each row, so in the end 2-norm of each weight vector
            # torch.linalg.norm, if ord=1 with dim=-1 specified, it computes the abs(x_i) of each row, so in the end 1-norm of each weight vector
            W_error = torch.linalg.norm(T.view(*T.shape[:1], -1) - T_est.view(*T_est.shape[:1], -1), dim=-1)  # error for (num_sets(T), num_sets(W), weights)
        elif self.loss_type in ["residual_all"]:
            # P W != T
            # Compute estimated target points T_est based on solved weights with each W_s for all subsets, e.g. PW_s for all subsets S
            T_est = torch.einsum('ijk,lkm->iljm', P, W)  # Shape: (num_sets (P), num_sets (W), n_samples_per_set, out_dim_target)
            # Expand T to match T_reg dimensions for broadcasting
            T_expanded = T.unsqueeze(1).expand(-1, T_est.shape[1], -1, -1)  # Shape: (num_sets (T), num_sets (W), n_samples_per_set, out_dim_target)
            W_error = torch.linalg.norm(T_expanded.reshape(*T_expanded.shape[:2] , -1) - T_est.reshape(*T_est.shape[:2], -1),
                                        ord=1, dim = -1)# error for (num_sets(T), num_sets(W), weights)

        # Calculate losses on solved weight sets
        W = W.reshape(self.num_sets, -1)                        # [num_sets, n_weights]
        W_magphase = torch.stack([W.abs(), W.angle()], axis=-1) # stack magnitude and phase

        ##### Debug
        # plt.show()
        # plt.figure(figsize=(20, 5))
        # plt.plot(np.angle(W.reshape(self.num_sets, -1).T.detach().cpu().numpy()), alpha=0.5, color="b")
        # plt.plot(np.mean(np.angle(W.reshape(self.num_sets, -1).detach().cpu().numpy()), axis=0), alpha=1.0, linewidth=1,
        #          color="k")

        #### Compute loss norm ####
        error = W_error
        if self.loss_norm in ["L1","L1_dist"]:
            error = l1_loss_from_difference(error)
        elif self.loss_norm in ["L2","L2_dist"]:
            error = l2_loss_from_difference(error)
        elif self.loss_norm in ["huber"]:
            error = huber_loss_from_difference(error)
        elif self.loss_norm in ["frobenius"]:
            error = torch.linalg.matrix_norm(error, ord="fro", dim=(-2, -1))
        else:
            AssertionError("Loss norm {} not supported".format(self.loss_norm))

        if reduce:
            return torch.mean(error),  [W_magphase]
        else:
            return error, [W_magphase]


    def sample_patches(self, sample):
        '''
        Sample patches based on predefined patching parameters and the inputted sample point
        :param sample: dictionary containing the sampled coordinates within the batch ("coords")
        :return: dictionary containing the sampled patches ("coords_patch") and the target coordinates ("coords_target")
        '''

        ############################# 1. Sample coords_target #############################
        # First, patch_origin defines if coords_target are sampled from the
        # inputted coords (sample["coords"] or from a predefined grid (as for "cart")
        if self.patch_origin in ["cart"]:
            # Create cartesian coordinates to sample coords_target from
            np_dtype = np.dtype(str(sample["coords"].dtype).split('.')[1])
            coords = np.stack(np.meshgrid(np.linspace(-1, 1 - (2 / self.nxy), self.nxy, dtype=np_dtype),
                                          np.linspace(-1, 1 - (2 / self.nxy), self.nxy, dtype=np_dtype),
                                          indexing='ij'), -1)
            coords_xy = coords.reshape(-1, 2)
            coords_t = np.zeros_like(coords_xy[:, 0]).reshape(-1, 1)
            coords = np.concatenate([coords_t, coords_xy], axis=-1)

            if self.min_sets > 0:
                num_weights = self.patch_size[0] * (self.patch_size[1] - 1) * self.out_dim_patch * self.out_dim_target
                num_patches = int(np.ceil(self.min_sets * num_weights * self.overdet_factor))
            else:
                num_patches = self.config["batch_size"]

            # get random origin coordinates based on batch
            indices = np.random.choice(len(coords), size=num_patches, replace=False)
            origin_coord = coords[indices]
        elif self.patch_origin in ["radial"]:
            # Sample coords_target from given radial input coordinates
            sample["coords"] = sample["coords"].reshape(-1, self.coord_dim)
            origin_coord = sample["coords"].detach().cpu().clone().numpy()


        ############################# 2. Sample coords_patch #############################
        # patch_type defines the geometry of the patches sampled around coords_target to optain coords_patch
        if self.patch_type in ["cart", "cartX", "cartY", "cartXY"]:
            dx = 2.0 / self.nxy
            dy = 2.0 / self.nxy
            # dt could also be defined in future work
            if self.patch_type == "cart":
                coord_neighbors = kernel_coords.create_cartesian_kernel(origin_coord,
                                                                    kernel_size=self.patch_size,
                                                                    patch_dist=self.patch_dist,
                                                                    delta_dist=[dx,dy])
            elif self.patch_type == "cartX":
                coord_neighbors = kernel_coords.create_cartesian_kernel_XY(origin_coord,
                                                                    kernel_size=self.patch_size,
                                                                    patch_dist=self.patch_dist,
                                                                    delta_dist=[dx,dy], exclude_center="x")
            elif self.patch_type == "cartY":
                coord_neighbors = kernel_coords.create_cartesian_kernel_XY(origin_coord,
                                                                    kernel_size=self.patch_size,
                                                                    patch_dist=self.patch_dist,
                                                                    delta_dist=[dx,dy], exclude_center="y")
            elif self.patch_type == "cartXY":
                if not hasattr(self, 'patch_flag') or not self.patch_flag:
                    self.patch_flag = "x"
                if self.patch_flag == "x":
                    coord_neighbors = kernel_coords.create_cartesian_kernel_XY(origin_coord,
                                                                    kernel_size=self.patch_size,
                                                                    patch_dist=self.patch_dist,
                                                                    delta_dist=[dx,dy], exclude_center="x")
                    self.patch_flag = "y"
                elif self.patch_flag == "y":
                    coord_neighbors = kernel_coords.create_cartesian_kernel_XY(origin_coord,
                                                                    kernel_size=self.patch_size,
                                                                    patch_dist=self.patch_dist,
                                                                    delta_dist=[dx,dy], exclude_center="y")
                    self.patch_flag = "x"

        elif self.patch_type == "radial_full":
            # for cartesian fully sampled: nspokes >= (pi/2) * nFE
            # phi = (2*pi) / nspokes =  pi / (nFE * ( pi/2)) = 2 / nFE     # only pi, because all spokes until there fill the other half as well
            # spoke_FS = int(np.ceil(np.pi * 0.5 * self.config['fe_steps']))
            delta_phi = 2.0 / self.nxy   # in radian
            delta_fe = 2.0 / self.nxy

            origin_coord = sample["coords"].detach().cpu().clone().numpy()
            coord_neighbors = kernel_coords.create_radial_kernel(origin_coord,
                                                                 kernel_size=self.patch_size,
                                                                 patch_dist=self.patch_dist,
                                                                 delta_dist_rad=[delta_fe,delta_phi],
                                                                 half = False)
            dx = delta_fe  # for later filtering of valid coords

        elif self.patch_type == "radial_equi_full":
            # for cartesian fully sampled: nspokes >= (pi/2) * nFE
            # phie = (2*pi) / nspokes =  2*pi / (nFE * ( pi/2)) = 4 / nFE
            # spoke_FS = int(np.ceil(np.pi * 0.5 * self.config['fe_steps']))
            delta_phi = 2.0 / self.nxy   # in radian
            delta_fe = 2.0 / self.nxy

            origin_coord = sample["coords"].detach().cpu().clone().numpy()
            coord_neighbors = kernel_coords.create_radial_equidistant_kernel(origin_coord,
                                                                 kernel_size=self.patch_size,
                                                                 patch_dist=self.patch_dist,
                                                                 delta_dist_rad=[delta_fe,delta_phi],
                                                                 half = False)
            dx = delta_fe # for later filtering of valid coords
        else:
            AssertionError("Patch Type unknown")

        ## Debug
        # import matplotlib.pyplot as plt
        # plt.scatter(coord_neighbors[[1], :, 1], coord_neighbors[[1], :, 2])
        # plt.scatter(origin_coord[[1], 1], origin_coord[[1], 2])
        # plt.show()

        ##################### Filter out undesired coordinates (not at edge and not from center) ####################
        edge_coords_idx = kernel_coords.get_edge_coords(origin_coord, coord_neighbors, dx)
        if self.exclude_center is not None:
            center_coords_idx = kernel_coords.get_center_coord(origin_coord, coord_neighbors, dx*self.exclude_center)
        else:
            center_coords_idx = np.zeros_like(edge_coords_idx)
        valid_coords_idx = np.logical_not(edge_coords_idx | center_coords_idx)

        sample["valid_coords_idx"] = valid_coords_idx
        sample["n_valid_coords"] = len(np.argwhere(valid_coords_idx == True))
        origin_coord = origin_coord[sample["valid_coords_idx"], :]
        coord_neighbors = coord_neighbors[sample["valid_coords_idx"], :, :]

        ##################### Patching for temporal dimension ####################
        # Set all temporal values to the same value
        if self.patch_tconst:
            # set temporal value to same value for each subset
            origin_coord[:,0] = sample["coords"][0,0].clone().detach().cpu().numpy()
            coord_neighbors[:,:,0] = sample["coords"][0,0].clone().detach().cpu().numpy() # coord_neighbors[0, : ,0]

        # Optional: if temporal delta is set, sample random time point within this temporal delta
        if self.patch_tdelta:
            # limit temporal values to a certain range delta-t
            range = self.config["dataset"]["nav_max"] - self.config["dataset"]["nav_min"]
            tdelta = self.config["model"]["patch"]["tdelta"] * range
            t = origin_coord[0,0] # sample random time point as center for deltat
            if t > self.config["dataset"]["nav_max"] - tdelta/2:
                t = self.config["dataset"]["nav_max"] - tdelta/2
            elif t < self.config["dataset"]["nav_min"] + tdelta/2:
                t = self.config["dataset"]["nav_min"] + tdelta/2
            else:
                t = t
            nt = origin_coord[:,0].shape
            origin_coord[:,0] = np.random.uniform(t - tdelta/2, t + tdelta/2, nt)
            coord_neighbors[:,:,0] = coord_neighbors[0, : ,0]

        sample['coords_patch'] = torch.from_numpy(coord_neighbors).to(self.device)
        sample['coords_target'] = torch.from_numpy(origin_coord).to(self.device)
        self.n_neighbors = sample['coords_patch'].shape[1]

        return sample

