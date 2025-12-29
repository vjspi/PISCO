import glob
import os
import time
import csv
import torch
import argparse
import numpy as np
import random
from pathlib import Path
import os
import imageio
import yaml
import io
import medutils
import json

import wandb
import utils.mri
from utils import basic
from utils import eval
from torch.utils.data import DataLoader
from utils.eval import log_recon_to_wandb, log_quant_metrics, log_difference_images
from utils.vis import angle2color, k2img, alpha2img
from utils.eval import bias_corr, get_eval_metrics


def main():
    # parse args and get config
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='configs/config_abdominal.yml')
    parser.add_argument('-g', '--gpu', type=int, default=0)
    parser.add_argument('-r', '--r_acc', type=float, default=None)
    parser.add_argument('-sub', '--subject', type=str, default=None)
    parser.add_argument('-s', '--slice', type=int, default=None)
    parser.add_argument('-log', '--log', type=int, choices=[0, 1], default=1)
    parser.add_argument('-e', '--encoding', type=str, choices=["spatial", "STIFF", "STIFF+t", "spatial+temporal"],
                        default=None)
    parser.add_argument('-feat', '--hidden_features', type=int, default=None)
    parser.add_argument('-nav', '--nav_range', type=str, default=None)
    parser.add_argument('-ep', '--epochs', type=int, default=None)
    parser.add_argument('-seed', '--seed', type=int, default=0)
    parser.add_argument('-sig', '--sigma', type=float, default=None)
    parser.add_argument('-bs', '--batch_size', type=int, default=None)
    parser.add_argument('-om', '--omega_0', type=int, default=None)
    parser.add_argument('-lr', '--lr', type=float, default=None)

    args = parser.parse_args()

    # enable Double precision
    # set gpu and random seed
    torch.set_default_dtype(torch.float32)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    num_cuda_devices = torch.cuda.device_count()
    if num_cuda_devices > 0:
        print("Found", num_cuda_devices, "CUDA device(s) available.")
        for i in range(num_cuda_devices):
            print("CUDA Device", i, ":", torch.cuda.get_device_name(i))
    else:
        print("No CUDA devices found.")

    # parse config
    config = basic.parse_config(args.config)
    if "cardiac" in args.subject:
        data_config_path = os.path.join(os.getcwd(), "configs/subjects/cardiac.yml")
        config["subject_name"] = int(args.subject.split("cardiac")[1])
    elif "abdominal" in args.subject:
        data_config_path = os.path.join(os.getcwd(), "configs/subjects/abdominal.yml")
        config["subject_name"] = args.subject.split("abdominal")[1]
    elif "knee" in args.subject:
        data_config_path = os.path.join(os.getcwd(), "configs/subjects/knee.yml")
        config["subject_name"] = "11_knee"
    else:
        data_config_path = os.path.join(os.getcwd(), "configs/subjects/", args.subject + ".yml")

    if args.subject is not None:  # get general subject info # ToDo: modularize
        data_config = basic.parse_config(data_config_path)
        config.update(data_config)

    if config["data_type"] == "abdominal_phantom_nohist":
        config['data_root'] = os.path.join(Path.home(), config["data_root"], "abdominal_phantom")
    else:
        config['data_root'] = os.path.join(Path.home(), config["data_root"], config["data_type"])
    config["results_root"] = os.path.join(Path.home(), config["results_root"], config["wandb_project"],
                                          config["data_type"])
    config['gpu'] = args.gpu
    torch.cuda.set_device(args.gpu)
    config['log'] = bool(args.log)
    # config['exp_summary'] = args.log

    # config['slice_name'] = slice_name
    # config['gpu'] = args.gpu
    # config['exp_summary'] = args.log

    ### optional from command line (otherwise in config)
    config["slice"] = args.slice if args.slice is not None else config["slice"]
    config["num_steps"] = args.epochs if args.epochs is not None else config["num_steps"]
    config["batch_size"] = args.batch_size if args.batch_size is not None else config["batch_size"]
    config["encoding"]["sigma"] = args.sigma if args.sigma is not None else config["encoding"]["sigma"]
    config["encoding"]["type"] = args.encoding if args.encoding is not None else config["encoding"]["type"]
    config["model"]["params"]["hidden_features"] = args.hidden_features if args.hidden_features is not None else \
    config["model"]["params"]["hidden_features"]
    config["model"]["params"]["omega_0"] = args.omega_0 if args.omega_0 is not None else config["model"]["params"][
        "omega_0"]
    config["optimizer"]["params"]["lr"] = args.lr if args.lr is not None else config["optimizer"]["params"]["lr"]


    # Dataset settings
    if args.r_acc is not None:
        config['dataset']['acc_factor'] = int(args.r_acc) if args.r_acc == int(args.r_acc) else args.r_acc
    if args.nav_range is not None:
        config["dataset"]["nav_min"], config["dataset"]["nav_max"] = \
            tuple(map(float, args.nav_range.split(',')))
    config["dataset"]["hysteresis"] = config["dataset"]["hysteresis"] if "hysteresis" in config["dataset"] else False

    ### regularization settings

    # create dataset
    dataset_class = basic.import_module(config["dataset"]["module"], config["dataset"]["name"])
    dataset = dataset_class(config)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'], drop_last=True)
    config['nc'] = dataset.n_coils
    config['ne'] = dataset.n_echo if hasattr(dataset, "n_echo") else None
    # config['nx'] = dataset.csm.shape[1]     # overwrite, since relevant in case data was downsampled
    # config['ny'] = dataset.csm.shape[2]
    config['n_spokes'] = dataset.n_spokes

    # create model
    model_class = basic.import_module(config["model"]["module"], config["model"]["name"])
    NIKmodel = model_class(config)

    NIKmodel.init_train() # Do init train before loading model!
    NIKmodel.coil_factors = dataset.coil_factors if hasattr(dataset, "coil_factors") else None
    params = basic.count_parameters(NIKmodel.network_kdata)
    print("Network contains {} trainable parameters.".format(params))

    loaded_epoch = 0

    ### Load pretrained MLP
    if "pretrained" in NIKmodel.config["model"] and NIKmodel.config["model"]["pretrained"] is not None:
        start_epoch = None
        group_name = (config["model"]["pretrained"]["pretrain_group"]
                      + "*_S" + str(config["subject_name"]))
        exp_name = (config["model"]["pretrained"]["pretrain_exp"]
                    + "*_omega" + str(config['model']['params']['omega_0'])
                    + "*_sigma" +  str(config['encoding']['sigma'])
                    + f"*_{config['encoding']['type']}"
                    + f"{config['model']['params']['hidden_features']}_"
                    + "*_hdr" + str(config['hdr_ff_factor'])
                    + "_slice" + str(config["slice"])
                    + "_R" + str(config['dataset']['acc_factor']))
        if isinstance(NIKmodel.config["model"]["pretrained"]["epoch"], int):
            model_name = "_e{}".format(NIKmodel.config["model"]["pretrained"]["epoch"])
            start_epoch = NIKmodel.config["model"]["pretrained"]["epoch"] + 1
        else:
            model_name = NIKmodel.config["model"]["pretrained"]["epoch"]
        group_path = basic.find_subfolder(config["results_root"], group_name)
        exp_path = basic.find_subfolder(group_path, exp_name)
        # find latest model
        run_id = os.listdir(exp_path)[0]
        NIKmodel.config['weight_path'] = os.path.join(exp_path, run_id, "model_checkpoints", model_name)
        pretrained_model_path = NIKmodel.config['weight_path']
        pretrained_dict = torch.load(pretrained_model_path, map_location=NIKmodel.device)

        ## load optimizer settings # ToDO: ensure that correct states are updated?
        # optim_dict = NIKmodel.optimizer.state_dict()
        # pretrained_optim_dict = {k: v for k, v in pretrained_dict["optimizer_state_dict"].items() if k in optim_dict}
        # NIKmodel.optimizer.load_state_dict(pretrained_optim_dict)
        # optim_dict.update(pretrained_optim_dict)
        # NIKmodel.optimizer.load_state_dict(optim_dict)

        # load B for features
        model_dict = NIKmodel.network_kdata.state_dict()
        pretrained_model_dict = {k: v for k, v in pretrained_dict["model_state_dict"].items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_model_dict)
        NIKmodel.network_kdata.load_state_dict(model_dict)

        # load SirenNet weights
        model_dict = NIKmodel.network_kdata.siren_net.state_dict()
        pretrained_dict = torch.load(pretrained_model_path, map_location=NIKmodel.device)
        pretrained_dict = {k: v for k, v in pretrained_dict["model_state_dict"].items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        NIKmodel.network_kdata.siren_net.load_state_dict(model_dict)

        ## update kernel weights if existant
        if "diff_loss" in NIKmodel.config["model"]["params"] and NIKmodel.config["model"]["params"]["diff_loss"]:
            if "pretrained_kernel" in NIKmodel.config["patch_schedule"]:
                pretrained_with_kernel_dict = torch.load(os.path.join(config["results_root"],
                                                                      NIKmodel.config["patch_schedule"]["pretrained_kernel"],
                                                                      "best_model"),
                                                                      map_location=NIKmodel.device)
                NIKmodel.network_kdata.load_state_dict(pretrained_with_kernel_dict)
                loaded_epoch = 20

    # save config with all changed params for later evaluation
    with io.open(NIKmodel.model_save_path + '/config.yml', 'w', encoding='utf8') as outfile:
        yaml.dump(config, outfile, default_flow_style=False, allow_unicode=True)

    epoch_list, duration_list, loss_list, loss_reg_list, loss_dc_list = [],[],[],[], []
    loss_model = 1e10
    ## save model info
    from torchinfo import summary
    summary_text = summary(NIKmodel.network_kdata, (1, NIKmodel.config["model"]["params"]["hidden_features"]))
    with open(NIKmodel.model_save_path + "/model_summary.txt", "w") as text_file:
        text_file.write("Number of samples: " + str(len(dataset)))
        text_file.write(str(summary_text))

    # start_time = time.time()
    #% Train model
    generator = iter(dataloader)

    for epoch in range(loaded_epoch, config['num_steps']):
        ## Predefine that only one batch in each epoch
        n_iter = 1
        loss_epoch, loss_dc_epoch, loss_reg_epoch = 0,0,0
        W_reg = []

        #### SET KERNEL SCHEDULE for Informed Correction (ICo):
        if 'patch_schedule' in config:
            if config['patch_schedule']['calib_timing'] == "None":
                pass
            elif config['patch_schedule']['calib_timing'] == "jump":
                if epoch <= config['patch_schedule']['freeze_epoch']:

                    ## calibrate kernel
                    dataloader.dataset.increment = config['patch_schedule']['calib_region']

                    ## freeze MLP and train CONV in calib region
                    for param in NIKmodel.network_kdata.siren_net.parameters():
                        param.requires_grad = False

                    if hasattr(NIKmodel.network_kdata, 'conv_patch'):
                        for param in NIKmodel.network_kdata.conv_patch.parameters():
                            param.requires_grad = True
                    if hasattr(NIKmodel.network_kdata, 'conv_patch_t'):
                        for param in NIKmodel.network_kdata.conv_patch_t.parameters():
                            param.requires_grad = True
                        for param in NIKmodel.network_kdata.conv_patch_x.parameters():
                            param.requires_grad = True

                    loss_epoch = 0
                    for i in range(n_iter):
                        try:
                            sample = next(generator)
                        except StopIteration:
                            generator = iter(
                                dataloader)  # restart the generator if the previous generator is exhausted.
                            sample = next(generator)
                        # kcoord, kv = sample['coords'], sample['target']
                        loss = NIKmodel.train_batch(sample)
                        print(f"Epoch: {epoch}, Iter: {i}, Loss: {loss}")
                        loss_epoch += loss

                else:

                    ## apply kernel
                    # Load complete region and freeze CONV
                    dataloader.dataset.increment = 1.0
                    for param in NIKmodel.network_kdata.siren_net.parameters():
                        param.requires_grad = True
                    for param in NIKmodel.network_kdata.conv_patch.parameters():
                        param.requires_grad = False

                    loss_epoch = 0
                    for i in range(n_iter):
                        try:
                            sample = next(generator)
                        except StopIteration:
                            generator = iter(
                                dataloader)  # restart the generator if the previous generator is exhausted.
                            sample = next(generator)
                        # kcoord, kv = sample['coords'], sample['target']
                        if "diff_loss" in NIKmodel.config["model"]["params"] and NIKmodel.config["model"]["params"]["diff_loss"]:
                            loss = NIKmodel.train_batch(sample, diff_loss = True)
                        else:
                            loss = NIKmodel.train_batch(sample)
                        print(f"Epoch: {epoch}, Iter: {i}, Loss: {loss}")
                        loss_epoch += loss


        log_dict = {
            'epoch': epoch,
            'loss': loss_epoch / len(dataloader)}

        loss_list.append(loss_epoch.detach().item())
        epoch_list.append(epoch)
        # log test reconstruction for each step

        if 'log_test' in config and config["log_test"]: # and 'log' in config and config["log"]:
            if epoch == start_epoch or epoch % 500 == 0 or epoch == config["num_steps"]-1:

                config["nt"] = config["nnav"] if (np.min(dataset.self_nav) != np.max(dataset.self_nav) and
                                                  config["dataset"]["nav_min"] != config["dataset"]["nav_max"]) else 1

                if dataset.map2ms is None:
                    NIKmodel.set_inference_coord(nx=config["nx"], ny=config["ny"], nt=config["nt"])
                else:  # use ms2nav
                    NIKmodel.set_inference_coord(nx=config["nx"], ny=config["ny"], ts=dataset.map2ms)

                kpred_all = NIKmodel.test_batch(input=NIKmodel.grid_coords, conv=True) # if conv False, then only SirenNet (no convolution)ü3
                kpred_all = kpred_all.unsqueeze(1) if len(kpred_all.shape) == 4 else kpred_all

                csm = utils.mri.center_crop(dataset.csm[..., NIKmodel.config["slice"]],
                                            [NIKmodel.config["final_shape"], NIKmodel.config["final_shape"]])
                vis_img, temp_dict = log_recon_to_wandb(kpred_all, csm = csm,
                                                        multi_coil=False, log_xt=True,
                                                        cycle_duration=float(NIKmodel.config["cycle_duration"]))
                log_dict.update(temp_dict)

                ## Quantitative results
                if NIKmodel.config["data_type"] in ["knee", "abdominal_phantom", "abdominal_phantom_nohist", "cardiac_cine"]:
                # if "phantom" in str(NIKmodel.config["subject_name"]):
                    ref = dataset.ref.transpose(3, 4, 2, 0, 1)
                    ref = utils.mri.center_crop(ref, [NIKmodel.config["final_shape"], NIKmodel.config["final_shape"]])
                    img = vis_img["combined_img"]

                    if img.shape[-1] != ref.shape[-1]:
                        img = utils.mri.center_crop(img, ref.shape[-2:])

                    if NIKmodel.config["data_type"] in ["abdominal_phantom", "abdominal_phantom_nohist"] and not NIKmodel.config["dataset"]["hysteresis"]:
                        img = eval.create_hystereses(img, dim_axis=0)

                    img, ref = np.abs(img), np.abs(ref)  # get magnitude images
                    img = eval.postprocess(img, ref=ref)
                    ref = eval.postprocess(ref, ref=ref)

                    eval_dict = log_quant_metrics(img, ref)
                    diff_dict = log_difference_images(img, ref, mag_factor=5)
                    log_dict.update(eval_dict)
                    log_dict.update(diff_dict)


                # log progress
                NIKmodel.exp_summary_log(log_dict)

            # save checkpoints
            if loss_model > loss_epoch:
                l = loss_epoch / n_iter
                NIKmodel.save_network("best_network", epoch, l.detach().item())
                loss_model = loss_epoch
                # save best images

            if epoch % 500 == 0 or epoch == 200 or epoch == config["num_steps"]-1:
                l = loss_epoch / n_iter
                NIKmodel.save_network("_e{}".format(epoch), epoch, l.detach().item())


    ### After training loop completes
    csv_file_path = NIKmodel.result_save_path + "training_logs.csv"
    data_to_write = zip(loss_list, loss_dc_list, loss_reg_list)
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Duration', 'Loss', 'Loss_dc', 'Loss_reg'])
        writer.writerows(data_to_write)

    ### Testing
    import test_pisco
    overwrite = True
    t=10
    nt=NIKmodel.config["nt"]
    # Create prediction of model
    test_pisco.main(os.path.dirname(NIKmodel.model_save_path), nt=nt, t=t, overwrite=overwrite, device=NIKmodel.device)

    ### save best model
    best_model_info = NIKmodel.load_best_network()
    kpred = NIKmodel.test_batch()
    vis_img = k2img(kpred, csm = dataset.csm, scale = False)
    vis_img_comp = k2img(kpred, csm = dataset.csm, scale = True)
    results_path = os.path.join(os.path.dirname(NIKmodel.model_save_path), 'results')
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    np.savez(results_path + '/{}_best_model.npz'.format(vis_img["combined_mag"].shape),
             img=vis_img["combined_mag"],
             img_c = vis_img["combined_img"])

    ## compute eval metrics
    # ToDo: requires path to reference
    # ref_path = os.path.join(Path.home(),
    #                         "workspace_results/shared/grasprecon/S{}/recons/sl{}".format(config['subject_name'],
    #                                                                                      config['slice']))
    # ref = glob.glob(ref_path + "/*R1*.npz", recursive=True)[0]
    # ref_recon = np.load(ref)["ref"].squeeze()
    #
    # # correct bias
    # pred_corr = np.stack([bias_corr(vis_img["combined_img"][ms,0,...], ref_recon[ms,...])
    #                       for ms in range(ref_recon.shape[0])])
    # # get eval metrics
    # best_model_info.update({"ref_path": ref})
    # best_model_info.update(get_eval_metrics(pred_corr[0, ...], ref_recon[0, ...]))

    if config['exp_summary'] == 'wandb':
        wandb.log(best_model_info)

    with open(results_path + '_best_model.txt', 'w') as f:
        json.dump(best_model_info,f)



if __name__ == '__main__':
    main()