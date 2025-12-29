import numpy as np
from utils.basic import torch2numpy
import os
from pathlib import Path
def load_knee_data(config):
    """
    Load knee data from preprocessed npz file
    :param config: configuration dictionary
    :return: data dictionary with kdata, traj, self_nav, ref, csm, weights
    """

    ### Define paths
    ref_file_name = "grasp_reference_{}_1MS_R1.npz".format(config["slice"])
    np_file_name = config['filename'] if "filename" in config else "test_reconSoS.npz"

    ref_file_name = os.path.join(Path.home(), "workspace_results/grasprecon",
                            config["data_type"], "S{}".format(config["subject_name"]),
                            "slice{}".format(config["slice"]), ref_file_name)
    file_name = f"{config['data_root']}/S{config['subject_name']}/{np_file_name}"

    data = np.load(file_name)

    kdata = data["kspace"]  # expected input [ech, c, z, nFE*nPE]
    traj = data["traj"]

    if "nav_gt" in data:  # Navigator
        self_nav = data["nav_gt"]
        if self_nav.ndim == 2:
            self_nav = self_nav[:, config["slice"]]  # expected input [nPE, z]
    elif "pcaCurve" in data:
        self_nav = data["pcaCurve"]  # expected input [nPE]
    else:
        AssertionError("No navigator signal")

    # Reference
    try:
        ref = np.load(ref_file_name, allow_pickle=True)["grasp"][:,0,...]   # expected input from grasp: [ech, t1, t2, sl, x, y ]
        ref = ref.transpose(3,4,2,1,0)# required [x, y, slices, ms, ech*dyn]
    except:
        ref = None

    # Sensmaps
    if "smaps" in data:
        csm = data["smaps"][...]  # load all slices sensitivity map,
    else:
        csm = data["csm"][...]  # expected input [c,x,y,z]

    # reshape data to separate fe and pe
    config["n_spokes"] = self_nav.shape[0]
    kdata = kdata.reshape(*kdata.shape[:-1], config["n_spokes"], -1)
    traj = traj.reshape(traj.shape[0], config["n_spokes"], -1, 2)
    config["n_echo"], config["nc_recon"], config["n_slices"], _, config["fe_steps"] = kdata.shape

    # Weights
    try:
        weights = data["weights"]  # expected input [x, y, slices, ms, ech*dyn]
        weights = weights.reshape(weights.shape[0], -1,
                                  config["fe_steps"]) if weights is not None else None
    except:
        weights = None

    ## Retrospectively downsample the dataset
    # Todo: Analyze the trajectory distribution
    if "acc_factor" in config["dataset"]:
        ## Retrospective Acceleration
        acc_factor = config["dataset"]["acc_factor"]
        nspoke_acc = int(kdata.shape[-2] / acc_factor)
        kdata = kdata[:, :, :, :nspoke_acc, :]
        traj = traj[:, :nspoke_acc, :, :]
        weights = weights[:, :nspoke_acc, :] if weights is not None else None
        self_nav = self_nav[:nspoke_acc]
        n_spokes = kdata.shape[-2]  # overwrite spokes
        config["n_spokes"] = n_spokes

    data_dict = {}
    data_dict["config"] = config
    data_dict["kdata"] = kdata
    data_dict["traj"] = traj
    data_dict["self_nav"] = self_nav
    data_dict["ref"] = ref
    data_dict["csm"] = csm
    data_dict["weights"] = weights
    data_dict["map2ms"] = map2ms if "map2ms" in locals() else None

    return data_dict

