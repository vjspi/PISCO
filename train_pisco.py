import os
import time
import csv
import torch
import argparse
import numpy as np
import random
from pathlib import Path
import os
import yaml
import io
import wandb
import medutils
from torch.utils.data import DataLoader

import utils.mri
from utils import basic
from utils.vis import log_kspace_weights
from utils import eval
from utils.eval import log_recon_to_wandb, log_quant_metrics, log_difference_images


def main(config_input=None, config_sweep=None):

    if config_input is not None:
        config = config_input
    else:

        ########################## Parse arguments and config ##########################
        ### Parse arguments
        parser = argparse.ArgumentParser()
        parser.add_argument('-c', '--config', type=str, default='configs/config_abdominal.yml')
        parser.add_argument('-g', '--gpu', type=int, default=0)
        parser.add_argument('-r', '--r_acc', type=float, default=None)
        parser.add_argument('-sub', '--subject', type=str, default=None)
        parser.add_argument('-s', '--slice', type=int, default=None)
        parser.add_argument('-log', '--log', type=int, choices=[0, 1], default=1)
        parser.add_argument('-e', '--encoding', type=str, choices=["spatial", "STIFF", "STIFF+t", "spatial+temporal"], default=None)
        parser.add_argument('-feat', '--hidden_features', type=int, default=None)
        parser.add_argument('-nav', '--nav_range', type=str, default=None)
        parser.add_argument('-ep', '--epochs', type=int, default=None)
        parser.add_argument('-a', '--alpha', type=float, default=None)
        parser.add_argument('-l', '--lamda', type=float, default=None)
        parser.add_argument('-od', '--overdetermination', type=float, default=None)
        parser.add_argument('-seed', '--seed', type=int, default=0)
        parser.add_argument('-sig', '--sigma', type=float, default=None)
        parser.add_argument('-bs', '--batch_size', type=int, default=None)
        parser.add_argument('-om', '--omega_0', type=int, default=None)
        parser.add_argument('-td', '--tdelta', type=float, default=None)
        parser.add_argument('-ks', '--kernel_size', type=int, default=None)

        parser.add_argument('-pd', '--path_data', type=str, default=None)
        parser.add_argument('-pr', '--path_results', type=str, default=None)
        # parser.add_argument('-s', '--seed', type=int, default=0)
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

        ### Parse config
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

        if args.subject is not None:    # get general subject info # ToDo: modularize
            data_config = basic.parse_config(data_config_path)
            config.update(data_config)

        ### optional from command line (otherwise in config)
        # Dataset settings
        config["slice"] = args.slice if args.slice is not None else config["slice"]
        if args.r_acc is not None:
            config['dataset']['acc_factor'] = int(args.r_acc) if args.r_acc == int(args.r_acc) else args.r_acc
        if args.nav_range is not None:
            config["dataset"]["nav_min"], config["dataset"]["nav_max"] = \
                tuple(map(float, args.nav_range.split(',')))
        config["dataset"]["hysteresis"] = config["dataset"]["hysteresis"] if "hysteresis" in config["dataset"] else False
        # Training settings
        config["num_steps"] = args.epochs if args.epochs is not None else config["num_steps"]
        config["batch_size"] = args.batch_size if args.batch_size is not None else config["batch_size"]
        # Model settings including coordinate encoding
        config["encoding"]["sigma"] = args.sigma if args.sigma is not None else config["encoding"]["sigma"]
        config["encoding"]["type"] = args.encoding if args.encoding is not None else config["encoding"]["type"]
        config["model"]["params"]["hidden_features"] = args.hidden_features if args.hidden_features is not None else config["model"]["params"]["hidden_features"]
        config["model"]["params"]["omega_0"] = args.omega_0 if args.omega_0 is not None else config["model"]["params"]["omega_0"]

        ### Set paths ###
        config['data_root'] = os.path.join(Path.home(), config["data_root"], config["data_type"]) if args.path_data is None else args.path_data
        config["results_root"] = os.path.join(Path.home(), config["results_root"], config["wandb_project"], config["data_type"]) if args.path_results is None else args.path_results
        config['gpu'] = args.gpu
        torch.cuda.set_device(args.gpu)
        config['log'] = bool(args.log)

        ### PISCO Regularization settings
        # Settings for coordinate sampling (how patches are sampled) for PISCO loss
        config["model"]["patch"]["tdelta"] = args.tdelta if args.tdelta is not None else config["model"]["patch"]["tdelta"]
        config["model"]["patch"]["size"] = [args.kernel_size, args.kernel_size] if args.kernel_size is not None else config["model"]["patch"]["size"]
        # Overwrite params if set in command line:
        if args.lamda is not None:
            config["kreg"]["reg_lamda"] = args.lamda
        if args.alpha is not None:
            config["kreg"]["reg_alpha"] = float(args.alpha)
        if args.overdetermination is not None:
            config["kreg"]["overdetermination"] = args.overdetermination

        ## Create paths to pretrained NIK  to retrieve preconditioned model weights
        if "kreg" in config:
            config["exp_name"] = (f"{config['exp_name']}_"
                                  f"omega{config['model']['params']['omega_0']}_"
                                  f"sigma{config['encoding']['sigma']}_"
                                  f"{config['encoding']['type']}"
                                  f"{config['model']['params']['hidden_features']}_"
                                  f"nav{config['dataset']['nav_min']}to{config['dataset']['nav_max']}")
            if config["kreg"]["optim_type"] != "noreg":
                config["exp_name"] = (f"{config['exp_name']}_"
                                  f"td{config['model']['patch']['tdelta']}_"
                                  f"od{config['kreg']['overdetermination']}_"
                                  f"lamda{config['kreg']['reg_lamda']}_"
                                  f"alpha{config['kreg']['reg_alpha']}_")
        else:
            config["exp_name"] = (f"{config['exp_name']}_"
                                  f"omega{config['model']['params']['omega_0']}_"
                                  f"sigma{config['encoding']['sigma']}_"
                                  f"{config['encoding']['type']}"
                                  f"{config['model']['params']['hidden_features']}_"
                                  f"nav{config['dataset']['nav_min']}to{config['dataset']['nav_max']}")


    ######################## Create dataset ########################
    dataset_class = basic.import_module(config["dataset"]["module"], config["dataset"]["name"])
    dataset = dataset_class(config)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'], drop_last=True)
    config['nc'] = dataset.n_coils
    config['ne'] = dataset.n_echo if hasattr(dataset, "n_echo") else None
    config['n_spokes'] = dataset.n_spokes

    ######################## Create model ########################
    model_class = basic.import_module(config["model"]["module"], config["model"]["name"])
    NIKmodel = model_class(config)

    if config_input is not None:
        NIKmodel.model_save_path = config_input["model_save_path"]
        NIKmodel.results_save_path = config_input["results_save_path"]

    NIKmodel.init_train(resume=False) # Do init train before loading model!
    NIKmodel.coil_factors = dataset.coil_factors if hasattr(dataset, "coil_factors") else None
    params = basic.count_parameters(NIKmodel.network_kdata)
    print("Network contains {} trainable parameters.".format(params))

    ######################## Load pretrained model if required ########################
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

        NIKmodel.load_network()
        if start_epoch is None:
            start_epoch = torch.load(NIKmodel.config['weight_path'],
                                 map_location=NIKmodel.device)["epoch"] + 1
        print("Continuing model training from epoch {}".format(start_epoch))
    else:
        start_epoch = 0

    # set log settings
    if config['exp_summary'] == 'wandb':
        # log params
        params_to_log = []
        for idx, (name, param) in enumerate(NIKmodel.named_parameters()):
            if "weight" in name:
                params_to_log.append(name)

    ######################## Save model config and params for reference ########################
    NIKmodel.config["model_save_path"] = NIKmodel.model_save_path
    NIKmodel.config["result_save_path"] = NIKmodel.result_save_path
    NIKmodel.config["weight_path"] = NIKmodel.weight_path
    if os.path.exists(NIKmodel.model_save_path):
        with io.open(NIKmodel.model_save_path + '/config.yml', 'w', encoding='utf8') as outfile:
            yaml.dump(config, outfile, default_flow_style=False, allow_unicode=True)

    ## save model info
    from torchinfo import summary
    summary_text = summary(NIKmodel.network_kdata, (1, NIKmodel.config["model"]["params"]["hidden_features"]))
    with open(NIKmodel.model_save_path + "/model_summary.txt", "w") as text_file:
        text_file.write("Number of samples: " + str(len(dataset)))
        text_file.write(str(summary_text))

    ######################## Train model ########################
    epoch_list, duration_list, loss_list, loss_reg_list, loss_dc_list = [], [], [], [], []
    loss_model = 1e10
    loss_reg_flag = True
    start_time = time.time()
    generator = iter(dataloader)

    for epoch in range(start_epoch, config['num_steps']):
        loss_epoch, loss_dc_epoch, loss_reg_epoch = 0,0,0
        W_reg = []

        ## Predefine that only one batch is sampled in each epoch
        n_iter = 1

        for i in range(n_iter):
            try:
                sample = next(generator)
            except StopIteration:
                generator = iter(dataloader) # restart the generator if the previous generator is exhausted.
                sample = next(generator)
            loss, [loss_dc, loss_reg], W_reg_i = NIKmodel.train_batch(sample)
            W_reg.append(W_reg_i)
            print(f"Epoch: {epoch}, Iter: {i}, Loss: {loss}, Loss_kreg: {loss_reg}")
            loss_epoch += loss
            loss_dc_epoch += loss_dc
            loss_reg_epoch += loss_reg

        log_dict = {
            'epoch': epoch,
            'loss': loss_epoch.item() / n_iter,
            'loss_dc': loss_dc_epoch.item() / n_iter,
            'loss_reg': loss_reg_epoch.item() / n_iter,
            'lr_dc': NIKmodel.optimizer.param_groups[0]['lr'],
        }

        if "kreg" in NIKmodel.config:
            log_dict['lr_reg'] = NIKmodel.optimizer_reg.param_groups[0]['lr']

        ########################  Log training duration and losses ########################
        loss_list.append(loss_epoch.detach().item())
        loss_reg_list.append(loss_reg_epoch.detach().item())
        loss_dc_list.append(loss_dc_epoch.detach().item())
        epoch_list.append(epoch)
        duration = time.time() - start_time
        duration_list.append(duration)

        ######################## Log test reconstruction to wandb ########################
        if 'log_test' in config and config["log_test"]: # and 'log' in config and config["log"]:
            if epoch == start_epoch or epoch % 500 == 0 or epoch == config["num_steps"]-1:
                ### Log PISCO weights
                if W_reg[0] is not None and W_reg[0] != 0:
                    W_reg = [torch.stack([W_reg[i][idx] for i in range(len(W_reg))], dim=0) for idx in range(len(W_reg[0]))]
                    W_dict = log_kspace_weights(W_reg[0].detach(), title="W", vmin=0, vmax=1.0)
                    log_dict.update(W_dict)

                ### Log reconstructions
                # kpred is of size c*t*y*x - y and x are defined by the size of csm (see above)
                config["nt"] = config["nnav"] if (np.min(dataset.self_nav) != np.max(dataset.self_nav)
                                                  and config["dataset"]["nav_min"] != config["dataset"]["nav_max"]) else 1

                NIKmodel.set_inference_coord(nx=config["nx"], ny=config["ny"], nt=config["nt"])
                kpred_all = NIKmodel.test_batch(input = NIKmodel.grid_coords)
                kpred_all = kpred_all.unsqueeze(1) if len(kpred_all.shape) == 4 else kpred_all  # add echo dimension

                csm = utils.mri.center_crop(dataset.csm[..., NIKmodel.config["slice"]],
                                            [NIKmodel.config["final_shape"], NIKmodel.config["final_shape"]])
                vis_img, temp_dict = log_recon_to_wandb(kpred_all, csm = csm,
                                                        multi_coil=False, log_xt=True,
                                                        cycle_duration=float(NIKmodel.config["cycle_duration"]))
                log_dict.update(temp_dict)

                ### Log quantitative reconstruction results if reference is available
                if NIKmodel.config["data_type"] in ["knee", "cardiac_cine"]:
                    ref = dataset.ref.transpose(3, 4, 2, 0, 1)
                    ref = utils.mri.center_crop(ref, [NIKmodel.config["final_shape"], NIKmodel.config["final_shape"]])
                    img = vis_img["combined_img"]

                    if img.shape[-1] != ref.shape[-1]:
                        img = utils.mri.center_crop(img, ref.shape[-2:])

                    img, ref = np.abs(img), np.abs(ref)  # get magnitude images
                    img = eval.postprocess(img, ref=ref)
                    ref = eval.postprocess(ref, ref=ref)

                    eval_dict = log_quant_metrics(img, ref)
                    diff_dict = log_difference_images(img, ref, mag_factor=5)
                    log_dict.update(eval_dict)
                    log_dict.update(diff_dict)

            # log progress
            NIKmodel.exp_summary_log(log_dict, step=epoch)

        # save checkpoints
        if loss_model > loss_epoch:
            l = loss_epoch / n_iter
            NIKmodel.save_network("best_network", epoch, l.detach().item())
            loss_model = loss_epoch
            # save best images

        if epoch % 500 == 0 or epoch == 200 or epoch == config["num_steps"]-1:
            l = loss_epoch / n_iter
            NIKmodel.save_network("_e{}".format(epoch), epoch, l.detach().item())

    ######################## After training loop completes ########################
    ### Save results
    csv_file_path = NIKmodel.result_save_path + "training_logs.csv"
    data_to_write = zip(epoch_list, duration_list, loss_list, loss_dc_list, loss_reg_list)
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Duration', 'Loss', 'Loss_dc', 'Loss_reg'])
        writer.writerows(data_to_write)

    ### run reconstructions and evaluation to save results
    import test_pisco
    overwrite = True
    t=10
    nt=NIKmodel.config["nt"]
    # Create prediction of model
    test_pisco.main(os.path.dirname(NIKmodel.model_save_path), nt=nt, t=t, overwrite=overwrite, device=NIKmodel.device)

if __name__ == '__main__':
    main()