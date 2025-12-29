import torch
import argparse
import numpy as np

from eval.test_sos_phantom import TestNIKPhantom

class TestNIKSubject(TestNIKPhantom):
    def __init(self, path):
        print("initialized")
    def dummy(self):
        print("test")

    def merge_individual_echos(self, path_list, model_checkpoint="best_model", nt=50):

        temp_objects = []
        for echo_path in path_list:
            object = TestNIKPhantom(echo_path)
            object.run_model(model_checkpoint, nt=nt)
            object.load_reference()
            temp_objects.append(object)
        self.recon["model"] = np.concatenate([obj.recon["model"] for obj in temp_objects], axis=1)
        # self.recon["ref"] = np.concatenate([obj.recon["ref"] for obj in temp_objects], axis=1)
        # self.dataset.ref = np.concatenate([obj.dataset.ref for obj in temp_objects], axis=-1)
        self.dataset.kdata = np.concatenate([obj.dataset.kdata for obj in temp_objects], axis=0)
        self.dataset.kdata_echo = np.concatenate([obj.dataset.kdata_echo for obj in temp_objects], axis=0)
        self.dataset.traj = np.concatenate([obj.dataset.traj for obj in temp_objects], axis=0)
        self.dataset.weights = np.concatenate([obj.dataset.weights for obj in temp_objects], axis=0)
        self.dataset.n_echo = len(temp_objects)
        self.knav = temp_objects[0].knav

def main(path):

    nMS = 50

    if isinstance(path, str):
        testObject = TestNIKSubject(path)
        testObject.run_model("best_model", nt=nMS)
        if "phantom" in str(testObject.config["subject_name"]):
            testObject.load_reference()

    elif isinstance(path, list):
        testObject = TestNIKSubject(path[0]) # use first echo path as destimation and for data loading
        testObject.load_data()
        testObject.set_model() ## just for dummy
        testObject.load_model()
        testObject.merge_individual_echos(path)
    else:
        AssertionError("Invalid input format of path - either string or list of strings expected")

    testObject.recon["ref"] = None  # ToDo: Identify???
    testObject.dataset.ref = None
    testObject.create_comparison_with_hystereses(nMS = nMS)

    ## Reference
    config_xdgrasp = testObject.config["xdgrasp"]
    config_xdgrasp["beta"] = (0.1,1.0)
    config_xdgrasp["lambda"] = 0.1
    # config_xdgrasp["nMS"] = nMS
    # testObject.run_xdgrasp(config_xdgrasp)
    config_xdgrasp["nMS"] = 4
    testObject.run_xdgrasp(config_xdgrasp)


    testObject.save_gifs(recon_dict="recon_hyst", value_dict=None,
                         intensity_factor = 1.5, path=testObject.results_path)

    testObject.run_wfs(recon_dict="recon_hyst")
    testObject.save_wfs_gifs(water_dict="water_dict", fat_dict="fat_dict", fieldmap_dict="fieldmap_dict",
                             water_eval=None, fat_eval=None, path=testObject.results_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--path', type=str, default='configs/config_abdominal.yml')
    parser.add_argument('-g', '--gpu', type=int, default=0)
    parser.add_argument('-n', '--nt', type=int, default=20)
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)


    main(args.path)





