
import os

try:
    from ulab import numpy as np
except:
    import numpy as np

try:
    from ulab import scipy as spy
except:
    import scipy as spy

from bocpdms import bocpdms
from cpd_detector import cpd_detector

import accuracy

class bocpdms_detector(cpd_detector):
    def __init__(self, version=0, prior_a=0.01, prior_b=0.01, intensity=10):
        self.prior_a = prior_a
        self.prior_b = prior_b
        self.intensity = intensity
        

    def reinit(self):
        pass

    def detect(self, data):
        cps = bocpdms.detect(data, self.prior_a, self.prior_b, self.intensity)
        return cps

    def set_params(self, params_path, dataset_name):
        # format:dataset,alpha,beta,kappa,mu,f1,cover
        with open(params_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                if line.split(",")[0] == dataset_name:
                    self.prior_a = float(line.split(",")[1])
                    self.prior_b = float(line.split(",")[2])
                    self.intensity = float(line.split(",")[3])
                    break

def self_test():
    csv_folder = "../datasets/csv"
    params_path = "../params/params_bocpdms_best.csv"
    for filename in os.listdir(csv_folder):
        if filename.endswith(".csv"):
            data = np.loadtxt(os.path.join(csv_folder, filename), delimiter=",")
            if len(data.shape) == 1:
                # wrap to 2D array
                data = np.expand_dims(data, axis=1)
            print("Testing", filename)
            detector = bocpdms_detector()
            dataset_name = filename.split(".")[0]
            detector.set_params(params_path, dataset_name)
            cps = detector.detect(data)
            print("cps:", cps)
            print("accuracy:", accuracy.scores(cps, dataset_name, data.shape[0]))

if __name__ == "__main__":
    self_test()