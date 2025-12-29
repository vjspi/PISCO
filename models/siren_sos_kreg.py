import torch
import torch.nn as nn
import numpy as np
import os

import losses.hdr
from utils.mri import coilcombine, ifft2c_mri
from .base import NIKBase
from models.base_sos import NIKSoSBase
from utils.basic import import_module
from losses.hdr import HDRLoss_FF
from losses.pisco import PiscoLoss
from utils import kernel_coords

class NIKSiren(NIKSoSBase):
    def __init__(self, config) -> None:
        super().__init__(config)

        self.create_network()
        self.to(self.device)

    def create_network(self):
        # overwrite outdim with number of channels
        self.config["out_dim"] = self.config["nc"]
        out_dim = self.config["out_dim"]
        coord_dim = self.config["coord_dim"]

        if "params" in self.config["model"]:
            self.network_kdata = Siren(coord_dim, out_dim,
                                       **self.config['model']['params']).to(self.device)
        else:
            feature_dim = self.config["feature_dim"]
            num_layers = self.config["num_layers"]
            omega = self.config["omega"]
            self.network_kdata = Siren(coord_dim, out_dim, feature_dim, num_layers,
                                      omega_0=omega).to(self.device)

        if "encoding" in self.config:
            self.network_kdata.generate_B(self.config["encoding"])
        else:
            self.network_kdata.generate_B(out_dim)

    def init_expsummary(self, resume=False):
        """
        Initialize the visualization tools.
        Should be called in init_train after the initialization of self.exp_id.
        """
        if self.config['exp_summary'] == 'wandb':
            import wandb
            self.exp_summary = wandb.init(
                project=self.config['wandb_project'],
                name=self.exp_id,
                config=self.config,
                group=self.group_id,
                entity=self.config['wandb_entity'],
                resume=resume
            )

    def init_train(self, resume = False):
        """Initialize the network for training.
        Should be called before training.
        It does the following things:
            1. set the network to train mode
            2. create the optimizer to self.optimizer
            3. create the model save directory
            4. initialize the visualization tools
        If you want to add more things, you can override this function.
        """
        self.network_kdata.train()
        self.create_criterion()
        self.create_optimizer()
        if "kreg" in self.config:
            self.create_regularizer()
        self.load_names()
        if 'log' in self.config and self.config["log"]:
            self.init_expsummary(resume=resume)
            if not os.path.exists(self.model_save_path):
                os.makedirs(self.model_save_path)

    def create_criterion(self):
        """Create the loss function."""
        self.criterion = HDRLoss_FF(self.config)
        # self.criterion = AdaptiveHDRLoss(self.config)
    def create_optimizer(self):
        """Create the optimizer."""
        config_optim = {key: float(value) if isinstance(value, (int, float, str)) else value
                        for key, value in self.config["optimizer"]["params"].items()}

        # self.optimizer = torch.optim.Adam([self.parameters(), self.network.parameters()], lr=self.config['lr'])
        if self.config["optimizer"]["module"] == "Adam":
            self.optimizer = torch.optim.Adam(self.parameters(), **config_optim)
        elif self.config["optimizer"]["module"] == "AdamW":
            self.optimizer = torch.optim.AdamW(self.parameters(),  **config_optim)
        elif self.config["optimizer"]["module"] == "SGD":
            self.optimizer = torch.optim.SGD(self.parameters(),  **config_optim)
        else:
            AssertionError("No valib optimizer specified, using Adam")
            # self.optimizer = torch.optim.Adam(self.parameters(), **config_optim)

    def create_regularizer(self):
        self.criterion_reg = PiscoLoss(self.config)

        ## Create regularization optimizer
        if self.criterion_reg.optim == "SGD":
            self.optimizer_reg = torch.optim.SGD(self.parameters(),  **self.criterion_reg.optim_config)
        elif self.criterion_reg.optim == "Adam":
            self.optimizer_reg = torch.optim.Adam(self.parameters(),  **self.criterion_reg.optim_config)
        elif self.criterion_reg.optim == "AdamW":
            self.optimizer_reg = torch.optim.AdamW(self.parameters(),  **self.criterion_reg.optim_config)
        elif self.criterion_reg.optim == "RMSProp":
            self.optimizer_reg = torch.optim.RMSprop(self.parameters(),  **self.criterion_reg.optim_config)
        else:
            AssertionError("Optimizer for criterion_reg not defined")

        if self.criterion_reg.optim_scheduler in ["half1pk", "half1Pk", "half1PK"]:
            lr_lambda = lambda epoch: 0.5 ** (epoch // 1000)
            self.lr_scheduler_reg = torch.optim.lr_scheduler.LambdaLR(self.optimizer_reg, lr_lambda=lr_lambda)
        elif self.criterion_reg.optim_scheduler in ["half2pk", "half2Pk", "half2PK"]:
            lr_lambda = lambda epoch: 0.5 ** (epoch // 2000)
            self.lr_scheduler_reg = torch.optim.lr_scheduler.LambdaLR(self.optimizer_reg, lr_lambda=lr_lambda)
        elif self.criterion_reg.optim_scheduler in ["lr_increase_500"]:
            # target_lr = self.criterion_reg.optim_config["lr"]
            lr_lambda = lambda epoch: (epoch / 500) if epoch < 500 else 1
            self.lr_scheduler_reg = torch.optim.lr_scheduler.LambdaLR(self.optimizer_reg, lr_lambda=lr_lambda)
        elif self.criterion_reg.optim_scheduler in ["lr_increase_linear_fast"]:
            # target_lr = self.criterion_reg.optim_config["lr"]
            lr_lambda = lambda epoch: (epoch / 500) if epoch < epoch else 1
            self.lr_scheduler_reg = torch.optim.lr_scheduler.LambdaLR(self.optimizer_reg, lr_lambda=lr_lambda)
        elif self.criterion_reg.optim_scheduler in ["lr_increase"]:
            target_lr = self.criterion_reg.optim_config["lr"]
            lr_lambda = lambda epoch: (target_lr/10) * (epoch // 100) if epoch < 1000 else target_lr
            self.lr_scheduler_reg = torch.optim.lr_scheduler.LambdaLR(self.optimizer_reg, lr_lambda=lr_lambda)
        elif self.criterion_reg.optim_scheduler in ["cyclic", "cyclic2000", "cyclic5000"]:
            base_lr = self.criterion_reg.optim_config["lr"] * 1e-1 # Set your base learning rate
            max_lr = self.criterion_reg.optim_config["lr"]  # Set your maximum learning rate
            if self.criterion_reg.optim_config["lr"] in ["cyclic2000"]:
                step_size_up = 2000
                step_size_down = 2000
            elif self.criterion_reg.optim_config["lr"] in ["cyclic5000"]:
                step_size_up = 5000
                step_size_down = 5000
            else:
                step_size_up = 1000
                step_size_down = 1000  # Number of training iterations in the decreasing half of a cycle (optional)
            self.lr_scheduler_reg = torch.optim.lr_scheduler.CyclicLR(self.optimizer_reg, base_lr=base_lr,
                                                                      max_lr=max_lr, step_size_up=step_size_up,
                                                                      step_size_down=step_size_down, cycle_momentum=False)

        else:
            print("No LR scheduler for regularization")

    def init_test(self):
        """Initialize the network for testing.
        Should be called before testing.
        It does the following things:f
            1. set the network to eval mode
            2. load the network parameters from the weight file path
        If you want to add more things, you can override this function.
        """
        self.weight_path = self.config['weight_path']

        self.load_network()

        self.network_kdata.eval()

        exp_id = self.weight_path.split('/')[-2]
        epoch_id = self.weight_path.split('/')[-1].split('.')[0]

        # setup model save dir
        results_save_dir = os.path.join(self.group_id, self.exp_id)
        if "results_root" in self.config:
            results_save_dir = "".join([self.config["results_root"],'/', results_save_dir])

        if not os.path.exists(results_save_dir):
            os.makedirs(results_save_dir)

        self.result_save_path

    def pre_process(self, coords):
        """
        Preprocess the input coordinates.
        """

        # inputs['coords'] = inputs['coords'].to(self.device)     # required for loss calculation
        coords = coords.to(self.device)     # required for loss calculation

        # features = self.network_kdata.pre_process(inputs['coords_patch'])

        if "encoding" in self.config:
            if self.config["encoding"]["type"] == "spatial":
                factor = 2 * torch.pi if "pi_factor" in self.config["encoding"] and self.config["encoding"]["pi_factor"] else 1.
                features = torch.cat([torch.sin(factor * coords @ self.network_kdata.B),
                                      torch.cos(factor * coords @ self.network_kdata.B)], dim=-1)
            elif self.config["encoding"]["type"] == "spatial+temporal":
                factor = 2 * torch.pi if "pi_factor" in self.config["encoding"] and self.config["encoding"][
                    "pi_factor"] else 1.
                features = torch.cat([torch.sin(factor * coords[:, 1:] @ self.network_kdata.B_spat),
                                      torch.cos(factor * coords[:, 1:] @ self.network_kdata.B_spat),
                                      torch.sin(factor * coords[:, :1] @ self.network_kdata.B_temp),
                                      torch.cos(factor * coords[:, :1] @ self.network_kdata.B_temp)],
                                     dim=-1)
            elif self.config["encoding"]["type"] == "STIFF":
                factor = 2 * torch.pi if "pi_factor" in self.config["encoding"] and self.config["encoding"][
                    "pi_factor"] else 1.
                features = torch.cat([torch.cos(factor * coords[:, 1:] @ self.network_kdata.B_spat),
                                      torch.sin(factor * coords[:, 1:] @ self.network_kdata.B_spat),
                                      torch.cos(factor * coords[:, 1:] @ self.network_kdata.B_temp) * torch.cos(factor * coords[:, :1]),
                                      torch.cos(factor * coords[:, 1:] @ self.network_kdata.B_temp) * torch.sin(factor * coords[:, :1]),
                                      torch.sin(factor * coords[:, 1:] @ self.network_kdata.B_temp) * torch.cos(factor * coords[:, :1]),
                                      torch.sin(factor * coords[:, 1:] @ self.network_kdata.B_temp) * torch.sin(factor * coords[:, :1])
                                      ], dim=-1)
            elif self.config["encoding"]["type"] == "STIFF+t":
                factor = 2 * torch.pi if "pi_factor" in self.config["encoding"] and self.config["encoding"][
                    "pi_factor"] else 1.
                features = torch.cat([torch.cos(factor * coords[:, 1:] @ self.network_kdata.B_spat),
                                      torch.sin(factor * coords[:, 1:] @ self.network_kdata.B_spat),
                                      torch.cos(factor * coords[:, 1:] @ self.network_kdata.B_dyn) * torch.cos(factor * coords[:, :1]),
                                      torch.cos(factor * coords[:, 1:] @ self.network_kdata.B_dyn) * torch.sin(factor * coords[:, :1]),
                                      torch.sin(factor * coords[:, 1:] @ self.network_kdata.B_dyn) * torch.cos(factor * coords[:, :1]),
                                      torch.sin(factor * coords[:, 1:] @ self.network_kdata.B_dyn) * torch.sin(factor * coords[:, :1]),
                                      torch.sin(factor * coords[:, :1] @ self.network_kdata.B_temp),
                                      torch.cos(factor * coords[:, :1] @ self.network_kdata.B_temp),
                                      ], dim=-1)

        else:
            features = torch.cat([torch.sin(coords @ self.network_kdata.B),
                                  torch.cos(coords @ self.network_kdata.B)], dim=-1)

        return features

    def post_process(self, output):
        """
        Convert the real output to a complex-valued output.
        The first half of the output is the real part, and the second half is the imaginary part.
        """
        output_complex = torch.complex(output[..., 0:self.config["out_dim"]].clone(),
                                       output[..., self.config["out_dim"]:].clone())
        return output_complex

    def train_batch(self, sample):

        self.network_kdata.train()
        torch.autograd.set_detect_anomaly(True)
        self.optimizer.zero_grad()
        if "kreg" in self.config:
            self.optimizer_reg.zero_grad()

        sample['coords'], sample['targets'] = sample['coords'].to(self.device), sample['targets'].to(self.device)
        output = self.predict(sample['coords'])  ## only NIK

        with torch.no_grad():       ## sample patches
            sample = self.criterion_reg.sample_patches(sample)
            # sample = self.sample_patches(sample)
            coords_P, coords_T = sample["coords_patch"], sample['coords_target']

        if "kreg" in self.config:
            if self.config["kreg"]["optim_type"] in ["joint", "joint_noBack", "joint_backP", "joint_backT"]:
                # Predict based on pre-defined backprop
                if self.config["kreg"]["optim_type"] == "joint":
                    output_P, output_T = self.predict(coords_P), self.predict(coords_T)
                elif self.config["kreg"]["optim_type"] == "joint_backT":
                    output_T = self.predict(coords_T)
                    with torch.no_grad():
                        output_P = self.predict(coords_P)
                elif self.config["kreg"]["optim_type"] == "joint_backP":
                    output_P = self.predict(coords_P)
                    with torch.no_grad():
                        output_T = self.predict(coords_T)
                elif self.config["kreg"]["optim_type"] == "joint_noBack":
                    with torch.no_grad():
                        output_P, output_T = self.predict(coords_P), self.predict(coords_T)

                loss_dc, reg = self.criterion(output, sample['targets'], sample['coords'])
                loss_reg, W_reg = self.criterion_reg(output_T, output_P, coords_T, coords_P)
                loss = loss_dc + self.criterion_reg.reg_lamda * loss_reg
                loss.backward()
                self.optimizer.step()

            elif self.config["kreg"]["optim_type"] in ["noreg", "onlyreg"]:
                loss_dc, reg = self.criterion(output, sample['targets'], sample['coords'])
                if self.config["kreg"]["optim_type"] == "onlyreg":
                    output_P, output_T = self.predict(coords_P), self.predict(coords_T)
                    loss_reg, W_reg = self.criterion_reg(output_T, output_P, coords_T, coords_P)
                    loss = loss_reg
                else:
                    with torch.no_grad():
                        output_P, output_T = self.predict(coords_P), self.predict(coords_T)
                        loss_reg, W_reg = self.criterion_reg(output_T, output_P, coords_T, coords_P)
                    loss = loss_dc
                loss.backward()
                self.optimizer.step()

            elif self.config["kreg"]["optim_type"] in ["separate_ZG", "separate_backP_ZG", "separate_backT_ZG"]:
                loss_dc, reg = self.criterion(output, sample['targets'], sample['coords'])
                loss_dc.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                # self.optimizer_reg.zero_grad()

                if self.config["kreg"]["optim_type"] == "separate_ZG":
                    output_P, output_T = self.predict(coords_P), self.predict(coords_T)
                elif self.config["kreg"]["optim_type"] == "separate_backT_ZG":
                    output_T = self.predict(coords_T)
                    with torch.no_grad():
                        output_P = self.predict(coords_P)
                elif self.config["kreg"]["optim_type"] == "separate_backP_ZG":
                    output_P = self.predict(coords_P)
                    with torch.no_grad():
                        output_T = self.predict(coords_T)

                loss_reg, W_reg = self.criterion_reg(output_T, output_P, coords_T, coords_P)
                loss_reg_weighted = self.criterion_reg.reg_lamda * loss_reg
                loss_reg_weighted.backward()
                self.optimizer.step()
                loss = loss_reg_weighted + loss_dc

            elif self.config["kreg"]["optim_type"] in ["separate", "separate_backP", "separate_backT"]:
                loss_dc, reg = self.criterion(output, sample['targets'], sample['coords'])
                loss_dc.backward()
                self.optimizer.step()

                if self.config["kreg"]["optim_type"] == "separate":
                    output_P, output_T = self.predict(coords_P), self.predict(coords_T)
                elif self.config["kreg"]["optim_type"] == "separate_backT":
                    output_T = self.predict(coords_T)
                    with torch.no_grad():
                        output_P = self.predict(coords_P)
                elif self.config["kreg"]["optim_type"] == "separate_backP":
                    output_P = self.predict(coords_P)
                    with torch.no_grad():
                        output_T = self.predict(coords_T)

                loss_reg, W_reg = self.criterion_reg(output_T, output_P, coords_T, coords_P)
                loss_reg_weighted = self.criterion_reg.reg_lamda * loss_reg
                loss_reg_weighted.backward()
                self.optimizer_reg.step()
                loss = loss_reg_weighted + loss_dc
            else:
                AssertionError("{} is no valid training strategy".format(self.config["kreg"]["optim_type"]))

            if hasattr(self, 'lr_scheduler') and self.lr_scheduler is not None:
                self.lr_scheduler.step()
            if hasattr(self, 'lr_scheduler_reg') and self.lr_scheduler_reg is not None:
                self.lr_scheduler_reg.step()

            return loss, [loss_dc, loss_reg], W_reg

        else:
            loss_dc, reg = self.criterion(output, sample['targets'], sample['coords'])
            loss_dc.backward()
            self.optimizer.step()

            return loss_dc, [loss_dc, loss_dc], 0


    def train_batch_dc(self, sample):

        self.network_kdata.train()
        torch.autograd.set_detect_anomaly(True)
        self.optimizer.zero_grad()
        self.optimizer_reg.zero_grad()

        sample['coords'], sample['targets'] = sample['coords'].to(self.device), sample['targets'].to(self.device)
        output = self.predict(sample['coords'])  ## only NIK
        loss_dc, reg = self.criterion(output, sample['targets'], sample['coords'])
        loss_dc.backward()
        self.optimizer.step()

        return loss_dc

    # def train_batch_reg(self, sample):
    #
    #     self.optimizer_reg.zero_grad()
    #
    #     sample['coords'], sample['targets'] = sample['coords'].to(self.device), sample['targets'].to(self.device)
    #
    #     with torch.no_grad():       ## sample patches
    #         sample = self.sample_patches(sample)
    #         coords_P, coords_T = sample["coords_patch"], sample['coords_target']
    #
    #     if self.config["kreg"]["optim_type"] in ["separate", "separate_backP", "separate_backT"]:
    #         if self.config["kreg"]["optim_type"] == "separate":
    #             output_P, output_T = self.predict(coords_P), self.predict(coords_T)
    #         elif self.config["kreg"]["optim_type"] == "separate_backT":
    #             output_T = self.predict(coords_T)
    #             with torch.no_grad():
    #                 output_P = self.predict(coords_P)
    #         elif self.config["kreg"]["optim_type"] == "separate_backP":
    #             output_P = self.predict(coords_P)
    #             with torch.no_grad():
    #                 output_T = self.predict(coords_T)
    #     else:
    #         AssertionError("optim type not known")
    #
    #     loss_reg, W_reg = self.criterion_reg(output_T, output_P, coords_T, coords_P)
    #     loss_reg_weighted = self.criterion_reg.reg_lamda * loss_reg
    #     loss_reg_weighted.backward()
    #     self.optimizer_reg.step()
    #     return loss_reg, W_reg

    def predict(self, coords):
        if coords.ndim == 3:
            nsamples = coords.shape[0]
            coords_flat = coords.reshape(-1, self.config["coord_dim"])
            features = self.pre_process(coords_flat)
        else:
            features = self.pre_process(coords)
        output = self.forward(features)
        output = self.post_process(output)
        if coords.ndim == 3:
            output = output.reshape(nsamples, -1, output.shape[-1])
        return output

    def set_inference_coord(self, nx=None, ny=None, nt = None, edges=True, ts = None):

        nav_min = self.config["dataset"]["nav_min"] if "nav_min" in self.config["dataset"] else -1.
        nav_max = self.config["dataset"]["nav_max"] if "nav_max" in self.config["dataset"] else 1.
        nx = self.config["nx"] if nx is None else nx
        ny = self.config["ny"] if ny is None else ny

        if ts is None:
            nt = self.dataset.config["nnav"] if nt is None else nt
        else:
            nt = len(ts)

        if self.config["coord_dim"] == 4:
            ksamples = [self.config["nc"], nx, ny, nt]  # kc, kx, ky, nav
            kc = torch.linspace(-1, 1, ksamples[0])
            kxs = torch.linspace(-1, 1 - 2 / ksamples[1], ksamples[1])
            kys = torch.linspace(-1, 1 - 2 / ksamples[2], ksamples[2])
            # knav = torch.linspace(-1, 1, ksamples[3])
            if edges:
                self.knav = torch.linspace(nav_min, nav_max, ksamples[3])
            else:
                self.knav = torch.linspace(nav_min + 1 / ksamples[3], nav_max - 1 / ksamples[3], ksamples[3])

            grid_coords = torch.stack(torch.meshgrid(kc, kxs, kys, self.knav, indexing='ij'), -1)  # nt, nx, ny, nc, 4

        elif self.config["coord_dim"] == 3:
            ksamples = [nt, nx, ny]  # kx, ky, nav
            # kc = torch.linspace(-1, 1, ksamples[0])
            kxs = torch.linspace(-1, 1 - 2 / ksamples[1], ksamples[1])
            kys = torch.linspace(-1, 1 - 2 / ksamples[2], ksamples[2])
            # knav = torch.linspace(-1, 1, ksamples[0])
            if edges:
                self.knav = torch.linspace(nav_min, nav_max, ksamples[0])
            else:
                self.knav = torch.linspace(nav_min + 1 / ksamples[0], nav_max - 1 / ksamples[0], ksamples[0])

            grid_coords = torch.stack(torch.meshgrid(self.knav, kys, kxs, indexing='ij'), -1)  # nav, nx, ny,3
            # dist_to_center = torch.sqrt(grid_coords[..., 1] ** 2 + grid_coords[..., 2] ** 2)
            # dist_to_center = dist_to_center.unsqueeze(1).expand(-1, self.NIKmodel.config["nc"], -1, -1)

        if ts is not None:
            assert grid_coords[...,0].shape[0] == len(ts)
            grid_coords[...,0] = torch.tensor(ts[:, None,None]) # add x and y dimension - from ts to [ts, 1, 1]

        self.ksamples = ksamples
        self.grid_coords = grid_coords

    def test_batch(self, input=None, input_dim=None):
        """
        Test the network with a cartesian grid.
        if sample is not None, it will return image combined with coil sensitivity.
        """
        self.network_kdata.eval()
        self.conv_patch.eval() if hasattr(self, 'conv_patch') else None

        with torch.no_grad():
            nc = self.config['nc']  # len(self.config['coil_select'])  # nc = self.config['nc']
            if input is None:
                if input_dim is not None:  # if dim given, use these - otherwise default from config
                    assert len(input_dim) == self.config["coord_dim"]
                    nnav, nx, ny = input_dim
                else:
                    nx = self.config['nx']
                    ny = self.config['ny']
                    nnav = self.config['nnav']

                nav_min = float(self.config["dataset"]["nav_min"]) if "nav_min" in self.config["dataset"] else -1.
                nav_max = float(self.config["dataset"]["nav_max"]) if "nav_max" in self.config["dataset"] else -1.
                delta_nav = (nav_max - nav_min) / nnav

                # coordinates: kx, ky, nav
                kxs = torch.linspace(-1, 1 - 2 / nx, nx)
                kys = torch.linspace(-1, 1 - 2 / ny, ny)
                if self.config["data_type"] in ["cardiac_cine"]:
                    knav = torch.linspace(nav_min, nav_max, nnav) if nnav > 1 else torch.tensor(nav_min)
                else:
                    knav = torch.linspace(nav_min + delta_nav / nnav, nav_max - delta_nav / nnav,
                                  nnav) if nnav > 1 else torch.tensor(nav_min)

                grid_coords = torch.stack(torch.meshgrid(knav, kys, kxs, indexing='ij'), -1)  # nav, nx, ny,3

            else:
                nnav, nx, ny = input.shape[:3]
                grid_coords = input
                # grid_coords = grid_coords.to(self.device)

            dist_to_center = torch.sqrt(grid_coords[..., 1] ** 2 + grid_coords[..., 2] ** 2)
            dist_to_center = dist_to_center.unsqueeze(1).expand(-1, nc, -1, -1)  # nt, nc, nx, ny

            nDim = grid_coords.shape[-1]
            contr_split = 1

            contr_split_num = np.ceil(grid_coords.shape[0] / contr_split).astype(int) # split t for memory saving
            kpred_list = []
            for t_batch in range(contr_split_num):
                grid_coords_batch = grid_coords[t_batch * contr_split:(t_batch + 1) * contr_split]
                grid_coords_batch = grid_coords_batch.reshape(-1, nDim).requires_grad_(False)
                # get prediction
                sample = {'coords': grid_coords_batch}
                features = self.pre_process(sample["coords"])   # encode time differently?
                kpred = self.forward(features)
                kpred = self.post_process(kpred)
                kpred_list.append(kpred)
            kpred = torch.concat(kpred_list, 0)

            # if input_dim is not None:  # reshape if default or input_dim given (but not if coords)
            # TODO: clearning this part of code
            kpred = kpred.reshape(nnav, ny, nx, nc).permute(0,3,1,2) # nt, nc, ny, nx
            k_outer = 1 #
            kpred[dist_to_center >= k_outer] = 0

            if hasattr(self, "coil_factors") and self.coil_factors is not None:
                # rescale coil based on maximum intensity
                kpred = kpred * torch.tensor(self.coil_factors.reshape(1,-1,1,1)).to(kpred)

            return kpred

    def forward(self, input_coords):
        x = self.network_kdata(input_coords)
        return x


"""
The following code is a demo of mlp with sine activation function.
We suggest to only use the mlp model class to do the very specific 
mlp task: takes a feature vector and outputs a vector. The encoding 
and post-process of the input coordinates and output should be done 
outside of the mlp model (e.g. in the prepocess and postprocess 
function in your NIK model class).
"""

class Siren(nn.Module):
    def __init__(self, coord_dim, out_dim, hidden_features, num_layers, omega_0=30, exp_out=True, norm=None) -> None:
        super().__init__()

        self.coord_dim = coord_dim
        self.hidden_features = hidden_features


        self.net = [SineLayer(hidden_features, hidden_features, is_first=True, omega_0=omega_0)]
        for i in range(num_layers - 1):
            self.net.append(SineLayer(hidden_features, hidden_features, is_first=False, omega_0=omega_0, norm=norm))

        final_linear = nn.Linear(hidden_features, out_dim * 2)
        with torch.no_grad():
            # Initialize the weights without in-place operation
            final_linear.weight.data.uniform_(-np.sqrt(6 / hidden_features) / omega_0,
                                              np.sqrt(6 / hidden_features) / omega_0)
        self.net.append(final_linear)
        self.net = nn.Sequential(*self.net)

        torch.manual_seed(0)

    def generate_B(self, config=None):

        if config is None:
            config["type"] = "spatial"
            config["sigma"] = 1.

        if config["type"] == "spatial":
            B = torch.randn((self.coord_dim, self.hidden_features // 2), dtype=torch.float32) * config["sigma"]
            self.register_buffer('B', B)
        elif config["type"] == "spatial+temporal":
            spat_feat_num = int(config["spat_feat_perc"] * self.hidden_features)
            B_spat = torch.randn((self.coord_dim-1, (spat_feat_num) // 2 ),
                                 dtype=torch.float32) * config["sigma"]
            B_temp = torch.randn((1, (self.hidden_features - spat_feat_num) // 2), dtype=torch.float32) * config["sigma"]
            self.register_buffer('B_spat', B_spat)
            self.register_buffer('B_temp',B_temp)
        elif config["type"] == "STIFF":
            # static and dynamic component for k-space (both temporal & spatial apply to coordinates and t is multiplied in addition)
            spat_feat_num = int(config["spat_feat_perc"] * self.hidden_features)
            if "B_init" in config and config["B_init"] == "uniform":
                B_spat = (torch.rand((self.coord_dim-1, (spat_feat_num) // 2 ),
                                     dtype=torch.float32) - 0.5) * config["sigma"]
                B_temp = (torch.rand((self.coord_dim-1, (self.hidden_features - spat_feat_num) // 4),
                                    dtype=torch.float32) - 0.5) * config["sigma"]
            else:
                B_spat = torch.randn((self.coord_dim-1, (spat_feat_num) // 2 ),
                                     dtype=torch.float32) * config["sigma"]
                B_temp = torch.randn((self.coord_dim-1, (self.hidden_features - spat_feat_num) // 4), dtype=torch.float32) * config["sigma"]
            self.register_buffer('B_spat', B_spat)
            self.register_buffer('B_temp',B_temp)
        elif config["type"] == "STIFF+t":
            # static and dynamic component for k-space (both temporal & spatial apply to coordinates and t is multiplied in addition)
            spat_feat_num = int(config["spat_feat_perc"] * self.hidden_features)
            dyn_feat_num = int((self.hidden_features - 32 - spat_feat_num))
            B_spat = torch.randn((self.coord_dim-1, (spat_feat_num) // 2 ),
                                 dtype=torch.float32) * config["sigma"]
            B_dyn = torch.randn((self.coord_dim-1, dyn_feat_num // 4 ), dtype=torch.float32) * config["sigma"]
            B_temp = torch.randn((1, (self.hidden_features - spat_feat_num - dyn_feat_num) // 2), dtype=torch.float32) * config["sigma"]
            self.register_buffer('B_spat', B_spat)
            self.register_buffer('B_dyn',B_dyn)
            self.register_buffer('B_temp',B_temp)

    def forward(self, features):
        x = self.net(features)
        return x


class SineLayer(nn.Module):
    """Linear layer with sine activation. Adapted from Siren repo"""

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30, norm=None):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.use_norm = True if norm is not None else False
        if self.use_norm:
            if norm == "batch":
                self.norm = nn.BatchNorm1d(out_features)


        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                            1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                            np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        out = self.linear(input)
        if self.use_norm:
            out = self.norm(out)
        return torch.sin(self.omega_0 * out)
