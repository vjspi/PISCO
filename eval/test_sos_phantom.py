import torch
import argparse
import numpy as np
from pathlib import Path
import os
import medutils
import matplotlib.pyplot as plt
import json

import utils.vis
import utils.eval
from utils.basic import parse_config, import_module, torch2numpy
from reference import run_xdgrasp

from utils.vis import k2img
import medutils.visualization as vis


class TestNIKPhantom():
    def __init__(self, path):

        self.recon = {}
        self.model_path = path
        self.results_path = os.path.join(self.model_path,
                                    "rec_test")
        os.makedirs(self.results_path) if not os.path.exists(self.results_path) else None

        self.weights_path = os.path.join(self.model_path,
                                    "model_checkpoints")

        # enable Double precision
        torch.set_default_dtype(torch.float32)

        ## Load model
        for i in os.listdir(self.weights_path):
            if os.path.isfile(os.path.join(self.weights_path, i)) and 'config' in i:
                self.config = parse_config(os.path.join(self.weights_path, i))

        ## identify training home directory
        self.config['data_root'] = os.path.join(Path.home(), self.config["data_root"][self.config["results_root"].find("workspace"):])  # need to cut home directory from training
        self.config["results_root"] = os.path.join(Path.home(), self.config["results_root"][self.config["results_root"].find("workspace"):])  # need to cut home directory from training
        self.config['gpu'] = 0

    def set_model(self, model = "best_model"):
        ## Select model weights
        # model = "_e400"
        self.model = model
        self.config["weight_path"] = os.path.join(self.weights_path, model) if model == "best_network" \
            else os.path.join(self.weights_path, "_e{}".format(model))
        ## Load information about model
        if model == "best_network":
            with open(self.config["weight_path"] + '.txt', 'r') as f:
                best_model_info = dict([line.strip().split(':', 1) for line in f])

    def load_model(self):
        model_class = import_module(self.config["model"]["module"], self.config["model"]["name"])
        self.NIKmodel = model_class(self.config)
        self.NIKmodel.load_names()
        self.NIKmodel.init_test()

    def load_data(self):
        # self.config['filename'] = self.config['filename'] if "filename" in self.config else "test_reconSoS.npz"
        # file_name = f"{self.config['data_root']}S{self.config['subject_name']}/{self.config['filename']}"
        # self.data = np.load(file_name)
        dataset_class = import_module(self.config["dataset"]["module"], self.config["dataset"]["name"])
        self.dataset = dataset_class(self.config)
        self.config['nc'] = self.dataset.n_coils
        self.config['ne'] = self.dataset.n_echo if hasattr(self.dataset, "n_echo") else None
        self.config['nx'] = self.dataset.csm.shape[1]  # overwrite, since relevant in case data was downsampled
        self.config['ny'] = self.dataset.csm.shape[2]
        self.config['n_spokes'] = self.dataset.n_spokes
        # self.csm = self.data["smaps"][..., self.config["slice"]]
        # self.recon["gt"] = self.data["ref"][..., self.config["slice"],:, :] if "ref" in self.data else None

    def set_inference_coord_with_kt(self, nx =300, ny=300, kt = [0], edges=True):
        if self.NIKmodel.config["coord_dim"] == 4:
            ksamples = [self.NIKmodel.config["nc"], nx, ny, len(kt)]  # kc, kx, ky, nav
            kc = torch.linspace(-1, 1, ksamples[0])
            kxs = torch.linspace(-1, 1 - 2 / ksamples[1], ksamples[1])
            kys = torch.linspace(-1, 1 - 2 / ksamples[2], ksamples[2])
            self.knav = kt
            grid_coords = torch.stack(torch.meshgrid(kc, kxs, kys, self.knav, indexing='ij'), -1)  # nt, nx, ny, nc, 4
        elif self.NIKmodel.config["coord_dim"] == 3:
            ksamples = [len(kt), nx, ny]  # kx, ky, nav
            # kc = torch.linspace(-1, 1, ksamples[0])
            kxs = torch.linspace(-1, 1 - 2 / ksamples[1], ksamples[1])
            kys = torch.linspace(-1, 1 - 2 / ksamples[2], ksamples[2])
            self.knav = kt
            grid_coords = torch.stack(torch.meshgrid(self.knav, kys, kxs, indexing='ij'), -1)

    def set_inference_coord(self, nx=None, ny=None, nt = None, edges=True, ts = None):

        nav_min = self.config["dataset"]["nav_min"] if "nav_min" in self.config["dataset"] else -1.
        nav_max = self.config["dataset"]["nav_max"] if "nav_max" in self.config["dataset"] else 1.
        nx = self.config["nx"] if nx is None else nx
        ny = self.config["ny"] if ny is None else ny

        if ts is None:
            nt = self.dataset.config["nnav"] if nt is None else nt
        else:
            nt = len(ts)

        if self.NIKmodel.config["coord_dim"] == 4:
            ksamples = [self.NIKmodel.config["nc"], nx, ny, nt]  # kc, kx, ky, nav
            kc = torch.linspace(-1, 1, ksamples[0])
            kxs = torch.linspace(-1, 1 - 2 / ksamples[1], ksamples[1])
            kys = torch.linspace(-1, 1 - 2 / ksamples[2], ksamples[2])
            # knav = torch.linspace(-1, 1, ksamples[3])
            if edges:
                self.knav = torch.linspace(nav_min, nav_max, ksamples[3])
            else:
                self.knav = torch.linspace(nav_min + 1 / ksamples[3], nav_max - 1 / ksamples[3], ksamples[3])

            grid_coords = torch.stack(torch.meshgrid(kc, kxs, kys, self.knav, indexing='ij'), -1)  # nt, nx, ny, nc, 4

        elif self.NIKmodel.config["coord_dim"] == 3:
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

    def run_inference(self, espirit=False):
        # Run model
        kpred = self.NIKmodel.test_batch(input=self.grid_coords) # nt, nc, nx, ny if coord_dim=3
        # dist_to_center = torch.sqrt(self.grid_coords[..., 1] ** 2 + self.grid_coords[..., 2] ** 2)
        assert len(self.ksamples) == self.NIKmodel.config["coord_dim"]

        if self.NIKmodel.config["coord_dim"] == 4:
            kpred = kpred.reshape(*self.ksamples)
        elif self.NIKmodel.config["coord_dim"] == 3:
            if "echo_pos" in self.NIKmodel.config["dataset"]:
                if self.NIKmodel.config["dataset"]["echo_pos"] == "in":
                    kpred = kpred.reshape(self.config["ne"],*self.ksamples, self.config["nc"])  # echo, time, x, y, coil
                    kpred = kpred.permute(1, 0, 4, 2, 3)        # time, echo, coil, x, y
                elif self.NIKmodel.config["dataset"]["echo_pos"] == "out":
                    kpred = kpred.reshape(*self.ksamples, self.config["nc"], self.config["ne"])
                    kpred = kpred.permute(0, 4, 3, 1, 2)        # time, echo, coil, x, y
                else:
                    AssertionError
            else:
                kpred = kpred[:, None, ...] # add echo dim -> [nt, 1, nc, nx, ny]

        assert len(kpred.shape) == 5 and kpred.shape[2] == self.NIKmodel.config["nc"] # check if coil dimension was added
        assert kpred.shape[1] == self.NIKmodel.config["ne"]

        if espirit:
            from utils.espirit import espirit_torch_slicewise_wrapper
            kspace_espirit = kpred[[0], 0, ...].permute(2, 3, 0, 1)
            smaps = espirit_torch_slicewise_wrapper(kspace_espirit, device=self.NIKmodel.device)
            smaps = smaps.permute(3, 2, 0, 1)
            vis.imshow(vis.plot_array(np.abs(torch2numpy(smaps[:, 0, ...]))), title="Espirit Maps - Mag")
            vis.imshow(vis.plot_array(np.angle(torch2numpy(smaps[:, 0, ...]))), title="Espirit Maps - Phase")
            plt.show()

        else:
            if len(self.dataset.csm.shape) == 4:
                csm = self.dataset.csm[..., self.NIKmodel.config["slice"]]
            elif len(self.dataset.csm.shape) == 3:
                csm = self.dataset.csm

        self.model_result = []
        for ech in range(self.NIKmodel.config["ne"]):
            if len(self.dataset.csm.shape) == 3:
                self.model_result.append(k2img(kpred[:,ech,...], csm = csm))
            elif len(self.dataset.csm.shape) == 4:
                self.model_result.append(k2img(kpred[:,ech,...], csm=csm))

        self.recon["model"] = np.stack([item["combined_img"] for item in self.model_result], axis = 1) # stack over echo axis
        # self.recon["kpred"] = kpred

    def run_xdgrasp(self, config_xdgrasp):
        # create xd grasp comparison
        self.config["xdgrasp"] = config_xdgrasp
        nMS = self.config["xdgrasp"]["nMS"]
        name = "xdgrasp" + str(nMS)

        if len(self.dataset.csm.shape) == 3:      # z slice already extracted
            self.recon[name] = run_xdgrasp(self.dataset.kdata[None, ...], self.dataset.traj[None, ...],
                                       self.dataset.self_nav, self.dataset.csm[..., :],
                                       self.dataset.weights, config=self.config["xdgrasp"],
                                       ref=self.dataset.ref).cpu().numpy()
            self.recon[name] = self.recon[name].transpose(0, 2, 1, 3, 4) # flip dyn and echo
        elif len(self.dataset.csm.shape) == 4:    # if z slice not extracted yet
            slice = self.NIKmodel.config["slice"]
            self.recon[name] = run_xdgrasp(self.dataset.kdata[None, ...], self.dataset.traj[None, ...],
                                       self.dataset.self_nav, self.dataset.csm[..., slice],
                                       self.dataset.weights, config=self.config["xdgrasp"],
                                       ref=self.dataset.ref).cpu().numpy()
            self.recon[name] = self.recon[name].transpose(0, 2, 1, 3, 4)  # flip dyn and echo

        if nMS == 4:
            recon_hyst = np.vstack((self.recon[name], np.flip(self.recon[name], axis=0)))   # 8 MS: 1,2,3,4,4,3,2,1
            repeat_counts = np.array([13,12,13,12,13,12,13,12])
            self.recon_hyst[name] = np.concatenate([np.repeat(recon_hyst[i:i+1], repeat_counts[i], axis=0)
                                                    for i in range(len(repeat_counts))], axis=0)
        else:
            print("Hysteresis is applied for {} nMS".format(nMS))
            self.recon_hyst[name] = np.vstack((self.recon[name], np.flip(self.recon[name], axis=0)))
        # else:
        #     AssertionError("Need to define expansion process for this number of motion states")

    def create_comparison_with_hystereses(self, nMS):
        self.recon_hyst = {}
        self.recon_hyst["ref"] = self.recon["ref"]
        self.recon_hyst["model"] = np.vstack((self.recon["model"], np.flip(self.recon["model"], axis=0)))

    def save_plots(self, recon_dict, keys = None, ms=0, path=None):

        dict = getattr(self, recon_dict, None)
        if dict["ref"].shape[0] == dict["model"].shape[0]:
            f"Careful! Shape mismatch: ref has shape {dict['ref'].shape[0]} while model has shape {dict['model'].shape[0]}"

        for key, array in dict.items():
            if keys is not None:
                if key in keys:
                    pass
                else:
                    continue

            if isinstance(array, torch.Tensor):  # Check if array is a PyTorch tensor
                if array.device.type != 'cpu':
                    array = array.cpu()
                array_mag_ms = np.abs(array[ms,...].squeeze().numpy())
            else:
                array_mag_ms = np.abs(array[ms,...])
            medutils.visualization.imsave(array_mag_ms,
                                          filepath=path + '/{}_kt{}_sub{}_sl{}_{}.png'.format(recon_dict,
                                                                                              ms,
                                                                                              self.config['subject_name'],
                                                                                              self.config['slice'],
                                                                                              key))

    def save_gifs(self, recon_dict, value_dict=None, path=None,
                  intensity_factor = 1.5, total_duration = 5000):
        dict = getattr(self, recon_dict, None)
        dict_v = getattr(self, value_dict, None) if value_dict is not None else None
        for key, array in dict.items():
            ## images
            if isinstance(array, torch.Tensor):  # Check if array is a PyTorch tensor
                if array.device.type != 'cpu':
                    array = array.cpu()
                array_mag = np.abs(array.squeeze().numpy())
            elif array is not None:
                array_mag = np.abs(array)
            else:
                continue
            ## values to string
            if value_dict is not None:
                default_eval_values = ["ssim", "psnr", "nrmseAbs"]     # define values to extract
                values_text_list = []
                if key in dict_v:
                    values_list = dict_v[key]
                    for entry in values_list:       # create text string of values to print for each echo -> saved in list
                        temp_list = []
                        for i, metric in enumerate(default_eval_values):
                            temp_val=round(entry[metric], 2)
                            temp_list.append(f"{temp_val:.2f}")
                        temp_text = "/".join(temp_list)
                        values_text_list.append(temp_text)
                else:
                    values_text_list= None
            else:
                values_text_list = None

            knav = np.linspace(0, total_duration / 1000, array_mag.shape[0])

            if len(array_mag.shape) < 5: # expand echo dimension
                array_mag = array_mag[:, None,...]

            for ech in range(array_mag.shape[1]):
                values_text = values_text_list[ech] if values_text_list is not None else None
                utils.vis.save_gif(array_mag[:,ech,...].squeeze(),
                                   str=values_text,
                                   numbers_array=np.around(knav, decimals=2),
                                   total_duration= total_duration,
                                   intensity_factor = intensity_factor,
                                    filename=path +
                                             '/{}_dyn{}_sub{}_sl{}_ech{}_{}.gif'.format(recon_dict, len(self.knav),
                                                                                  self.config['subject_name'],
                                                                                  self.config['slice'], ech, key))

    def bias_corr(self, recon_dict):
        recon = getattr(self, recon_dict, None)
        recon_corr = {}
        recon_corr["ref"] = recon["ref"]
        if recon["ref"].shape[0] == recon["model"].shape[0]:
            f"Careful! Shape mismatch: ref has shape {recon['ref'].shape[0]} while model has shape {recon['model'].shape[0]}"
        for key, array in recon.items():
            if key != "ref":
                pass
            else:
                continue
            recon_corr[key] = np.stack([utils.eval.bias_corr(recon[key][i, ...],
                                                        (recon["ref"][0, ...])) for i in
                                                        range(recon[key].shape[0])])

        setattr(self, recon_dict + "_corr", recon_corr)

    def compute_eval(self, recon_dict, out_dict="eval_dict", path=None):
        recon = getattr(self, recon_dict, None)
        eval_dict = dict()
        if recon["ref"].shape[0] == recon["model"].shape[0]:
            f"Careful! Shape mismatch: ref has shape {recon['ref'].shape[0]} while model has shape {recon['model'].shape[0]}"
        for key, array in recon.items():
            if key == "ref":
                continue
            print(key, ":")

            eval_dict[key] = []
            if len(recon[key].shape) == 4:
                eval_dict[key].append(utils.eval.get_eval_metrics(recon[key], recon["ref"], axes=(1,2), mean = True))
            else:
                for ech in range(recon[key].shape[1]):
                    eval_dict[key].append(utils.eval.get_eval_metrics(recon[key][:,ech,...], recon["ref"][:,ech,...],
                                                                      axes=(1,2), mean = True))

        setattr(self, out_dict, eval_dict)

        def float32_serializer(obj):
            if isinstance(obj, np.float32):
                return round(float(obj), 3) # Convert float32 to standard float for JSON serialization
            raise TypeError(f"Type {type(obj)} not serializable")

        with open(path + '/eval_{}_{}.txt'.format(recon_dict, self.model), 'w') as f:
            json.dump(eval_dict, f, indent=4, default=float32_serializer)

    def plot_eval_slicewise(self, recon_dict, path):
        recon = getattr(self, recon_dict, None)
        eval_dict_i = dict()
        if recon["ref"].shape[0] == recon["model"].shape[0]:
            f"Careful! Shape mismatch: ref has shape {recon['ref'].shape[0]} while model has shape {recon['model'].shape[0]}"
        for key, array in recon.items():
            if key == "ref":
                continue

            eval_dict_i[key] = []
            if len(recon[key].shape) == 4:
                eval_dict_i[key].append(utils.eval.get_eval_metrics(recon[key], recon["ref"], axes=(1,2), mean=False))
            else:
                for ech in range(recon[key].shape[1]):
                    eval_dict_i[key].append(utils.eval.get_eval_metrics(recon[key][:, ech, ...], recon["ref"][:,ech,...],
                                                                        axes=(1,2), mean=False))

        num_subentries = len(next(iter(eval_dict_i.values()))[0])

        # Create subplots
        fig, axes = plt.subplots(num_subentries, 1, figsize=(10, 6))
        handles, labels = [], []
        # Iterate over each subplot and plot each entry's data for the same subentry
        for i, (ax, subentry_key) in enumerate(zip(axes, next(iter(eval_dict_i.values()))[0].keys())):
            for entry_key, sub_list in eval_dict_i.items():
                for ech, sub_dict in enumerate(sub_list):
                    line, = ax.plot(sub_dict[subentry_key], label = entry_key + "ech{}".format(ech))
            if i==0:
                ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            ax.set_title(f"{subentry_key}")
        plt.tight_layout()
        fig.savefig(path + "/eval_metric_i_{}.png".format(recon_dict))
        fig.savefig(path + "/eval_metric_i_{}.eps".format(recon_dict))

    def load_reference(self):
        if self.dataset.ref is not None: # expexted format x,y,z,t
            # ref format: x,y,z,ms,e - target: ms, e, z, x, y
            self.recon["ref"] = np.moveaxis(self.dataset.ref,[-2, -1, -3],[0, 1, 2]) # move ms and echo dimension to front
        if len(self.dataset.ref.shape) == 4:      # if only one echo
            # ref format: x,y,z,ms - target: ms, e, z, x, y
            self.recon["ref"] = np.moveaxis(self.dataset.ref, [-1, -2], [0, 1]) # move ms and echo dimension to front
            self.recon["ref"] = self.recon["ref"][:, None, ...]

    def run_model(self, model_checkpoint="best_model", nt=None):
        self.set_model(model_checkpoint)
        self.load_model()
        self.load_data()
        if hasattr(self.dataset, "map2ms") and self.dataset.map2ms is not None: # load specific nav points for comparison
            self.set_inference_coord(nt=nt, ts=self.dataset.map2ms)
        else:
            self.set_inference_coord(nt=nt)
        self.run_inference()

    def run_model_espirit(self, nx, ny, nt=None, model_checkpoint="best_model"):
        self.set_model(model_checkpoint)
        self.load_model()
        self.load_data()
        self.set_inference_coord(nx=nx, ny=ny,nt=nt)
        self.run_inference(espirit=True)

    def merge_individual_echos(self, path_list, model_checkpoint="best_model", nt=None):
        temp_objects = []
        for echo_path in path_list:
            object = TestNIKPhantom(echo_path)
            object.run_model(model_checkpoint, nt=nt)
            object.load_reference()
            temp_objects.append(object)
        self.recon["model"] = np.concatenate([obj.recon["model"] for obj in temp_objects], axis=1)
        self.recon["ref"] = np.concatenate([obj.recon["ref"] for obj in temp_objects], axis=1)
        self.dataset.ref = np.concatenate([obj.dataset.ref for obj in temp_objects], axis=-1)
        self.dataset.kdata = np.concatenate([obj.dataset.kdata for obj in temp_objects], axis=0)
        self.dataset.kdata_echo = np.concatenate([obj.dataset.kdata_echo for obj in temp_objects], axis=0)
        self.dataset.traj = np.concatenate([obj.dataset.traj for obj in temp_objects], axis=0)
        self.dataset.weights = np.concatenate([obj.dataset.weights for obj in temp_objects], axis=0)
        self.dataset.n_echo = len(temp_objects)
        self.knav = temp_objects[0].knav

def plot_normalized(array):
    fig, ax = plt.subplots(array.shape[0], 1, figsize = (array.shape[0]*10, 10))

    for idx, i in enumerate(zip(range(array.shape[0]), ax.flatten())):
        img = medutils.visualization.normalize(array[idx,...])
        l = np.percentile(img, 99)
        ax[idx].imshow(img, cmap='gray', interpolation='nearest', vmin=0, vmax=l)
        ax[idx].axis('off')

    plt.tight_layout()

    plt.show()


def main(path):

    if isinstance(path, str):
        testObject = TestNIKPhantom(path)
        nMS = testObject.dataset["config"]["nnav"]
        testObject.run_model("best_model", nt=nMS)
        if "phantom" in str(testObject.config["subject_name"]):
            testObject.load_reference()

    elif isinstance(path, list):
        testObject = TestNIKPhantom(path[0]) # use first echo path as destimation and for data loading
        testObject.load_data()
        testObject.set_model() ## just for dummy
        testObject.load_model()
        testObject.merge_individual_echos(path)
    else:
        AssertionError("Invalid input format of path - either string or list of strings expected")

    testObject.create_comparison_with_hystereses(nMS = nMS)

    ## Reference
    config_xdgrasp = testObject.config["xdgrasp_solver"]
    config_xdgrasp["beta"] = (0.1,1.0)
    config_xdgrasp["lambda"] = 0.1
    # config_xdgrasp["nMS"] = nMS
    # testObject.run_xdgrasp(config_xdgrasp)
    config_xdgrasp["nMS"] = 4
    testObject.run_xdgrasp(config_xdgrasp)

    if "phantom" not in str(testObject.config["subject_name"]):
        testObject.recon_hyst["ref"] = testObject.recon_hyst["xdgrasp4"]
        testObject.recon["ref"] = testObject.recon["xdgrasp4"]

    testObject.bias_corr(recon_dict="recon_hyst")
    testObject.compute_eval(recon_dict="recon_hyst_corr", out_dict="eval_dict", path=testObject.results_path)

    # testObject.save_plots(recon_dict="recon_hyst_corr", ms=0, path=testObject.results_path)
    testObject.save_gifs(recon_dict="recon_hyst_corr", value_dict="eval_dict",
                         intensity_factor = 1.5, path=testObject.results_path)

    testObject.plot_eval_slicewise(recon_dict="recon_hyst_corr", path =testObject.results_path)


    ## water fat separation
    if testObject.recon["model"].shape[1] != 2:
        print("Array does not contain dual-echo,not doing WFS")
        return
    testObject.run_wfs(recon_dict="recon_hyst")
    testObject.bias_corr(recon_dict="water_dict")
    testObject.bias_corr(recon_dict="fat_dict")
    testObject.compute_eval(recon_dict="water_dict_corr", out_dict="eval_water", path=testObject.results_path)
    testObject.compute_eval(recon_dict="fat_dict_corr", out_dict="eval_fat", path=testObject.results_path)
    testObject.save_wfs_gifs(water_dict="water_dict_corr", fat_dict="fat_dict_corr", fieldmap_dict="fieldmap_dict",
                             water_eval="eval_water", fat_eval="eval_fat", path=testObject.results_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--path', type=str, default='configs/config_abdominal.yml')
    parser.add_argument('-g', '--gpu', type=int, default=0)
    parser.add_argument('-n', '--nt', type=int, default=20)
    args = parser.parse_args()
    torch.cuda.set_device(args.gpu)
    main(args.path)





