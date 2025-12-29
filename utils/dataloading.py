import torch
import numpy as np
import argparse
import os
from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from utils.basic import parse_config, import_module, torch2numpy, numpy2torch
from utils.mri import mriRadialAdjointOp, mriRadialForwardOp, coilcombine, ifft2c_mri

def load_data(subject):
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='configs/pisco/exp_config_kreg_knee.yml')
    parser.add_argument('-g', '--gpu', type=int, default=0)
    parser.add_argument('-r', '--r_acc', type=int, default=None)
    # parser.add_argument('-sub', '--subject', type=str, default='phantom1FS')
    parser.add_argument('-s', '--slice', type=int, default=40)
    parser.add_argument('-d', '--debug', type=bool, default=False)

    args = parser.parse_args()
    args.subject = subject
    # enable Double precision
    torch.set_default_dtype(torch.float32)
    # parse config
    project_path = os.path.join(Path.home(), "workspace/AbdoNIK")
    config = parse_config(os.path.join(project_path, args.config))
    if "cardiac" in args.subject:
        data_config_path = os.path.join(project_path, "configs/subjects/cardiac.yml")
        config["subject_name"] = int(args.subject.split("cardiac")[1])
    elif "abdominal" in args.subject:
        data_config_path = os.path.join(project_path, "configs/subjects/abdominal.yml")
        config["subject_name"] = args.subject.split("abdominal")[1]
    elif "knee" in args.subject:
        data_config_path = os.path.join(project_path, "configs/subjects/knee.yml")
        config["subject_name"] = "11_knee"
    else:
        data_config_path = os.path.join(project_path, "configs/subjects/", args.subject + ".yml")

    if args.subject is not None:  # get general subject info # ToDo: modularize
        data_config = parse_config(data_config_path)
        config.update(data_config)
    if args.slice is not None:
        config['slice'] = args.slice
        # config['subject_name'] = args.subject
    config['data_root'] = os.path.join(Path.home(), config["data_root"], config["data_type"])
    config["results_root"] = os.path.join(Path.home(), config["results_root"], config["data_type"])
    config["coord_dim"] = 3
    # config['slice_name'] = slice_name
    config['gpu'] = args.gpu
    torch.cuda.set_device(args.gpu)
    # config['exp_summary'] = args.log

    dataset_class = import_module(config["dataset"]["module"], config["dataset"]["name"])
    dataset = dataset_class(config)
    # dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True,
    #                         num_workers=config['num_workers'], drop_last=True)
    config['nc'] = dataset.n_coils
    config['ne'] = dataset.n_echo if hasattr(dataset, "n_echo") else None
    config['nx'] = dataset.csm.shape[1]  # overwrite, since relevant in case data was downsampled
    config['ny'] = dataset.csm.shape[2]
    config['n_spokes'] = dataset.n_spokes
    config["out_dim"] = 3

    # create model
    # model_class = import_module(config["model"]["module"], config["model"]["name"])
    # NIKmodel = model_class(config)
    # NIKmodel.init_train()

    # return img, kspace_radial, kspace_cart, traj_radial, traj_cart, smaps, config
    return dataset, config

def create_cartesian_dataset(dataset, config):

    smaps = dataset.csm[..., config["slice"]]
    nCoils = smaps.shape[0]

    im_size = dataset.csm.shape[1:3]

    if config["data_type"] == "cardiac_cine":
        traj_tbkn = dataset.traj.reshape(config["nnav"], -1, dataset.n_fe, 2)
        kdata_tbkn = dataset.kdata.reshape(dataset.kdata.shape[0], traj_tbkn.shape[0], traj_tbkn.shape[1],
                                           dataset.n_fe)
        traj_tbkn = traj_tbkn.transpose(0,3,1,2).reshape(config["nnav"], 2, -1)
        kdata_tbkn = kdata_tbkn.transpose(1, 0, 2, 3).reshape(config["nnav"], dataset.n_coils, -1)# now nnav, n_coils, n_spokes, n_fe

        img = mriRadialAdjointOp(kdata_tbkn, traj=traj_tbkn, shape=im_size,
                                 dcf="calc", device=config["gpu"],
                                 csm=smaps)

    else:
        traj_tbkn = dataset.traj.reshape(-1, 2).transpose(1, 0)[None, ...]
        kdata_tbkn = dataset.kdata.reshape(dataset.kdata.shape[0], -1)[None, ...]

        img = mriRadialAdjointOp(kdata_tbkn, traj=traj_tbkn, shape=im_size,
                                 dcf="calc", device=config["gpu"],
                                 csm=smaps)
        # img = torch2numpy(img.squeeze())

    kspace_radial = kdata_tbkn

    ## create cartesian k-space
    nx, ny= config["nx"], config["ny"]
    coords = np.stack(np.meshgrid(np.linspace(-1, 1 - (2 / nx), nx, dtype=np.float64),
                       np.linspace(-1, 1 - (2 / ny), ny, dtype=np.float64),
                       indexing='ij'), -1)
    coords = coords.reshape(1, -1, 2).transpose(0,2,1)
    dataset.kspace_cart = mriRadialForwardOp(img, traj=coords, dcf="calc", csm=smaps, osf=2, shape=[nx,ny])
    dataset.kspace_cart = dataset.kspace_cart.reshape(-1, nCoils, nx, ny)

    ## create cartesian k-space
    # kspace_cart = mriInterpAdjointOP(kspace_radial, traj=traj_tbkn, dcf="calc", shape=im_size, osf=1)
    ## for sanity_check:
    img_cart = ifft2c_mri(dataset.kspace_cart)
    img_cart = coilcombine(img_cart, csm=smaps)
    plt.imshow(img_cart[0, 0, ...].abs().detach().cpu().numpy(), cmap="gray")
    plt.show()

    dataset.traj_cart = coords.reshape(2, nx, ny)
    traj_radial = traj_tbkn.reshape(2, dataset.n_spokes, dataset.n_fe)

    return dataset, config

