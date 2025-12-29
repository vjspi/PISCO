import numpy as np
import os
import re
import torch
from pathlib import Path

from utils.mri import fillHermitianConjugate, center_crop

def load_cardiac_data(config):
    '''
    Load cardiac data from preprocessed data
    :param config: configuration dictionary
    :return: data dictionary with kdata, traj, self_nav, ref, csm, weights
    '''

    ### Define paths
    base_path = os.path.join(Path.home(), config["data_root"],
                                 "processed_pseudogolden_accfactor{}".format(config["dataset"]["acc_factor"]))
    ref_path = os.path.join(Path.home(), "workspace_results/grasprecon",
                            config["data_type"], "S{}".format(config["subject_name"]),
                            "slice{}".format(config["slice"]))
    ref_file = "grasp_reference_{}_25MS_R1.npz".format(config["slice"])

    if not os.path.exists(base_path):
        AssertionError("Preprocessed data {} does not exist".format(base_path))
    filename = find_subject_file(base_path, config["subject_name"])
    if not os.path.exists(os.path.join(base_path, filename)):
        AssertionError("Preprocessed data does not exist for subject {}".format(filename))


    ### load data
    data = np.load(os.path.join(base_path, filename + ".npz"))
    kdata = data["kdata"]
    traj = data["traj"]
    csm = data["csm"] if "csm" in data else None # c,x,y,z
    self_nav = data["self_nav"]
    try:
        ref = np.load(os.path.join(ref_path, ref_file))["grasp"]
        ref = ref[:, 0,...] # xdgrasp returns batch, t1, t2, z, y -> remove t1 dim to get [ech, dyn, z, x, y]
    except:
        ref = None

    if "final_shape" in config:
        if ref is not None:
            ref = center_crop(ref, shape=[config["final_shape"], config["final_shape"]])
            ref = ref.transpose(3, 4, 2, 1, 0)  # required: x, y, z, dyn, ech

    try:
        weights = data["weights"]
    except:
        weights = None

    ### update params
    config["n_echo"], config["nc_recon"], config["n_slices"], config["n_spokes"], config["fe_steps"] = kdata.shape
    config["slice"] = 0 # since only one slice in dataset

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


def find_subject_file(path, subject):
    # Find Data set and load data
    subject = str(subject).zfill(2)
    filenames = os.listdir(path)
    filename = [filename for filename in filenames if f"P{subject}_" in filename]
    assert len(filename) == 1
    filename = filename[0]
    filename = os.path.splitext(filename)[0] # remove ".mat"
    return filename

def get_subject_params(filename):
    pattern = r'_(\D+)(\d+)'
    matches = re.findall(pattern, filename)
    config = {}
    # Assign numbers to variables with custom names
    for letter, number in matches:

        if letter == "P":
            config["subject"] = int(number)
        elif letter == "slc":
            config["n_slices"] = int(number)
        elif letter == "phs":
            config["phases"] = int(number)
        elif letter == "cha":
            config["nc_recon"] = int(number)
        elif letter == "cols":
            config["fe_steps"] = int(number)
        elif letter == "lins":
            config["n_spokes"] = int(number)
    return config


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