import math
import gc
import os

from cpd_detector import cpd_detector

try:
    from ulab import numpy as np
except:
    import numpy as np


class cusum_detector(cpd_detector):

    def __init__(self, version=0):
        self._t_warmup = 30
        self._p_limit = 0.01
        self.statring = True
        self.current_t = 0

        self._reset()

    def reinit(self):
        self._reset()

    def set_params(self, params_path, dataset_name):

        # strip the dataset name in case it has a path
        file_name = dataset_name.split("/")[-1].split(".")[0]

        # get params from a csv file
        print(f"Reading parameters from {params_path}")
        open_file = open(params_path, "r")
        params = []
        # read line by line, not enough ram to read all at once

        file_name = file_name.split(".")[0]
        for line in open_file.readlines():
            line = line.strip().split(",")
            if line[0] == file_name:
                params = line[1:]

        if len(params) == 0:
            print(f"Params not found for {file_name}")
            return -1

        self.t_warmup = float(params[0])
        self.p_limit = float(params[1])
        gc.collect()

    def detect(self, data):
        changepoints = []
        for i in range(len(data)):
            prob, is_changepoint = self.predict_next(data[i])
            if is_changepoint:
                changepoints.append(i)
        return changepoints

    def predict_next(self, y):
        self._update_data(y)

        if self.current_t == self._t_warmup:
            self._init_params()

        if self.current_t >= self._t_warmup:
            prob, is_changepoint = self._check_for_changepoint()
            if is_changepoint:
                self._reset()

            return (1 - prob), is_changepoint

        else:
            return 0, False

    def _reset(self):
        self.current_t = 0

        self.current_obs = []

        self.current_mean = 0
        self.current_std = 0.1

    def _update_data(self, y):
        self.current_t += 1
        # convert ndarray y to list
        list_y = y.tolist()
        self.current_obs.append(list_y)

    def _init_params(self):
        self.current_mean = np.mean(np.array(self.current_obs))
        self.current_std = np.std(np.array(self.current_obs))

    def _check_for_changepoint(self):
        standardized_sum = np.sum(np.array(self.current_obs) - self.current_mean) / (
            self.current_std * self.current_t**0.5
        )
        prob = float(self._get_prob(standardized_sum))

        return prob, prob < self._p_limit

    def _get_prob(self, y):

        p = 0.5 * (1 + math.erf(abs(y) / math.sqrt(2)))
        prob = 2 * (1 - p)

        return prob


def test_single(data_path, params_path):
    print(f"Testing {data_path}")

    data = np.loadtxt(data_path, delimiter=",")

    detector = cusum_detector()
    detector.set_params(params_path, data_path)
    cpds = detector.detect(data)
    print(f"Change points: {cpds}")


def test_all():
    import platform

    platform = platform.platform()
    if "Linux" in platform:
        csv_folder = "../datasets/csv"
        params_path = "../params/params_cusum_best.csv"
    else:
        csv_folder = "./csv"
        params_path = "./params/params_cusum_best.csv"
    # simply test all the distance measures

    files = os.listdir(csv_folder)

    for file in files:
        file_path = csv_folder + "/" + file
        test_single(file_path, params_path)


def main():
    test_all()


if __name__ == "__main__":
    main()
