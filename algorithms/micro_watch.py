import gc
import math
import os
import time

try:
    from ulab import numpy as np
except ImportError:
    import numpy as np

import distance
from cpd_detector import cpd_detector


distance_measures = [
    distance.acc,
    distance.add_chisq,
    distance.bhattacharyya,
    distance.braycurtis,
    distance.canberra,
    distance.chebyshev,
    distance.chebyshev_min,
    distance.clark,
    distance.czekanowski,
    distance.divergence,
    distance.euclidean,
    distance.google,
    distance.gower,
    distance.hellinger,
    distance.jeffreys,
    distance.jensenshannon_divergence,
    distance.jensen_difference,
    distance.k_divergence,
    distance.kl_divergence,
    distance.kulczynski,
    distance.lorentzian,
    distance.manhattan,
    distance.matusita,
    distance.max_symmetric_chisq,
    distance.minkowski,
    distance.motyka,
    distance.neyman_chisq,
    distance.nonintersection,
    distance.pearson_chisq,
    distance.penroseshape,
    distance.soergel,
    distance.squared_chisq,
    distance.squaredchord,
    distance.taneja,
    distance.tanimoto,
    distance.topsoe,
    distance.vicis_symmetric_chisq,
    distance.vicis_wave_hedges,
    distance.wave_hedges,
]
# Note: Cosine distance does not work, dice distance does not work, jaccard distance does not work
# kumarjohnson distance does not work, maryland bridge distance does not work, squared euclidean does not work
# correlation_pearson does not work for most of the datasets

DISTANCE_COUNT = len(distance_measures)


def distance_function_mean(sample, dist_mean, dist_metric):
    dist_mean_list = np.array([dist_mean] * sample.shape[0])
    distance = dist_metric(sample, dist_mean_list)
    return distance


def iterate_batches(points, batch_size):
    samples_number = math.ceil(points.shape[0] / batch_size)
    for sample_id in range(0, samples_number):
        sample = points[sample_id * batch_size : (sample_id + 1) * batch_size]
        yield sample


class micro_watch(cpd_detector):
    def __init__(self, version=0):
        self.threshold_ratio = 0.51
        self.max_dist_size = 72
        self.new_dist_buffer_size = 32
        self.batch_size = 3
        self.is_creating_new_dist = True
        self.dist = None
        self.locations = []
        self.metric = distance_measures[version]
        self.version = version
        self.sum = None

    def reinit(self):
        self.is_creating_new_dist = True
        self.dist = None
        self.locations = []
        self.sum = None

    def set_params(self, params_path, dataset_name):
        # strip the dataset name in case it has a path
        file_name = dataset_name.split("/")[-1].split(".")[0]

        distance_index = self.version
        print(f"Reading parameters from {params_path}")
        open_file = open(params_path, "r")
        # read line by line, not enough ram to read all at once

        params = []
        for line in open_file.readlines():
            # line format: file_name,distance_index,batch_size,threshold,max_dist_size,new_dist_buffer_size
            line = line.strip().split(",")
            if line[0] == file_name and int(line[1]) == distance_index:
                # strip the file name and distance index
                params = line[2:]
                open_file.close()

        if len(params) == 0:
            print(
                f"Parameters not found for {file_name} and distance index {distance_index}"
            )
            return
        batch_size, threshold, max_dist_size, new_dist_buffer_size = params

        self.threshold_ratio = float(threshold)
        self.max_dist_size = int(max_dist_size)
        self.new_dist_buffer_size = int(new_dist_buffer_size)
        self.batch_size = int(batch_size)

    def detect(self, data):
        data = data.reshape((data.shape[0], -1))
        data_width = data.shape[1]
        max_size = max(self.max_dist_size, self.new_dist_buffer_size) + self.batch_size
        if max_size > data.shape[0]:
            max_size = data.shape[0]
        if self.sum is None:
            self.sum = np.zeros(data_width)
        if self.dist is None:
            self.dist = np.zeros((max_size, data_width))
        dist_len = 0
        for batch_id, batch in enumerate(iterate_batches(data, self.batch_size)):
            if self.is_creating_new_dist:
                batch_size = batch.shape[0]
                self.dist[dist_len : dist_len + batch_size] = batch
                dist_len += batch_size
                self.sum += np.sum(batch, axis=0)
                if dist_len >= self.new_dist_buffer_size:
                    self.is_creating_new_dist = False
                    dist_array = self.dist[:dist_len, ::]
                    dist_mean = self.sum / dist_len
                    max_dist = 0
                    for s in iterate_batches(dist_array, self.batch_size):
                        cur_dist = distance_function_mean(s, dist_mean, self.metric)
                        if cur_dist > max_dist:
                            max_dist = cur_dist
                    self.threshold = max_dist * self.threshold_ratio
            else:
                dist_mean = self.sum / dist_len
                value = distance_function_mean(batch, dist_mean, self.metric)

                if value > self.threshold:
                    self.locations.append(batch_id * self.batch_size)
                    self.dist = np.zeros((max_size, data_width))
                    self.is_creating_new_dist = True
                    dist_len = 0
                    self.sum = np.zeros(data_width)
                if dist_len < self.max_dist_size:
                    self.dist[dist_len : dist_len + len(batch)] = batch
                    dist_len += len(batch)
                    self.sum += np.sum(batch, axis=0)
                    dist_array = self.dist[:dist_len, ::]
                    dist_mean = self.sum / dist_len
                    max_dist = 0
                    for s in iterate_batches(dist_array, self.batch_size):
                        cur_dist = distance_function_mean(s, dist_mean, self.metric)
                        if cur_dist > max_dist:
                            max_dist = cur_dist
                    self.threshold = max_dist * self.threshold_ratio

        return self.locations


def test_single(data_path, distance_index, params_path):
    print(f"Testing {data_path} with distance index {distance_index}")

    data = np.loadtxt(data_path, delimiter=",")

    detector = micro_watch(distance_index)
    detector.set_params(params_path, data_path)
    cpds = detector.detect(data)
    print(f"Change points: {cpds}")


def test_all():
    import platform

    platform = platform.platform()
    if "Linux" in platform:
        csv_folder = "../datasets/csv"
        params_path = "../params/params_watch_best.csv"
    else:
        csv_folder = "./csv"
        params_path = "./params/params_watch_best.csv"

    # simply test all the distance measures
    files = os.listdir(csv_folder)

    for index in range(DISTANCE_COUNT):
        print(f"Testing distance index {index}")
        for file in files:
            file_path = csv_folder + "/" + file
            test_single(file_path, index, params_path)


def main():
    test_all()


if __name__ == "__main__":
    main()
