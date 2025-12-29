import torch
import numpy as np
import argparse
import os
from eval.test_sos_subject import TestNIKSubject

def main(path, nt, t, model="best_network", overwrite=True, device=0):

    testObject = TestNIKSubject(path)
    savename = os.path.join(testObject.results_path, "recon_{}".format(model))

    if os.path.exists(savename + ".npz") and not overwrite:
        print("Not overwriting {}".format(savename))
        return

    testObject.config["gpu"] = device
    n_fe = int(testObject.config["fe_steps"])

    testObject.run_model(nt=nt, model_checkpoint=model)
    testObject.recon["model_smaps"] = testObject.recon["model"]

    if str(testObject.config["data_type"]) in ["cardiac_cine", "knee"]:
        testObject.load_reference()

    # save best model prediction
    img = {}
    img["recon"] = testObject.recon["model"]
    print("Saving data into {}.npz ....".format(savename))
    np.savez(savename + '.npz',  **img    )
    print("Saved {}.npz!".format(savename))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, default='configs/config_abdominal.yml')
    parser.add_argument('-g', '--gpu', type=int, default=0)
    parser.add_argument('-n', '--nt', type=int, default=None)
    parser.add_argument('-t', '--timepoint', type=int, default=8)
    args = parser.parse_args()
    torch.cuda.set_device(args.gpu)
    main(path=args.path, nt=args.nt, t=args.timepoint, device=args.gpu)

