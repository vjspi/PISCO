import numpy as np
import os
import yaml
import io
import sys
import matplotlib.pyplot as plt

# Add parent directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), '..')))

from datasets.preprocess.preloader_cardiac import DataLoader
from pathlib import Path
from datasets.utils_cardiac import load_cardiac_data, find_subject_file

def main():

    config = {
        "gpu": "cuda:0",  # GPU device if needed
        "spoke_number": 2, # R1: 196 spokes, R15: 14 spokes, R26: 8 spokes, R52: 4 spokes
        "subject_names": [
            "1", "2", "3", "4", "5",
            "6", "7", "8", "9", "10",
            "11", "12", "13", "14", "15",
            "16", "17", "18", "19", "20",
            "21", "22", "23", "24", "25",
            "26", "27", "28", "29", "30",
        ],
        "data_root": "workspace_data/cardiac_cine",
        "fe_crop": 0.5,
        "oversample": 2,
        "nx": 208,
        "ny": 208,
        "n_tw": 7
    }

    raw_name = "raw"
    if "acc_factor" not in config:
        config["acc_factor"] = np.int32(np.round(config["nx"] / config["spoke_number"]))
    processed_name = "processed_pseudogolden_accfactor{}".format(config["acc_factor"])
    reference_name = "ref"

    base_path = os.path.join(Path.home(),config["data_root"])
    raw_path = os.path.join(base_path,raw_name)
    processed_path = os.path.join(base_path,processed_name)
    reference_path = os.path.join(base_path,reference_name)

    # Create a folder for processed data
    os.makedirs(processed_path, exist_ok=True)

    # Process each file in the folder
    filenames = []
    if config["subject_names"] is not None:
        for sub in config["subject_names"]:
            filenames.append(find_subject_file(raw_path, sub))
    else:
        filenames = [os.path.splitext(filename)[0] for filename in os.listdir(raw_path)]
        config["subject_names"] = [int(filename.split("P")[1].split("_")[0]) for filename in filenames]

    for sub, filename in zip(config["subject_names"],filenames):

        config['subject_name'] = sub
        dataloader = DataLoader(config)
        dataloader.load_raw_data()
        if config["acc_factor"] > 1:
            dataloader.retrospectiveUndersamplingPseudoGoldenAngle(spoke_number=config["spoke_number"])
        dataloader.compute_reference(reference_path)
        dataloader.save_processed_data(processed_path)

    with io.open(processed_path + '/config.yml', 'w', encoding='utf8') as outfile:
        yaml.dump(config, outfile, default_flow_style=False, allow_unicode=True)

    plt.close()

if __name__ == "__main__":
    main()