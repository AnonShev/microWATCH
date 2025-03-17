import gc
from math import floor, sqrt
import os

try:
    from ulab import numpy as np
except ImportError:
    import numpy as np

from cpd_detector import cpd_detector
import accuracy


def diagonal_sum(matrix):
    return np.trace(matrix)


def pdist(X, metric="euclidean"):
    # implementation of pairwise distance calculation, like scipy.spatial.distance.pdist
    length = X.shape[0]
    result_dims = length * (length - 1) // 2
    result = np.zeros(result_dims)
    index = 0
    for i in range(length):
        for j in range(i + 1, length):
            a = X[i]
            b = X[j]
            distance = 0
            if metric == "euclidean":
                distance = np.sqrt(np.sum((a - b) ** 2))
            elif metric == "sqeuclidean":
                distance = np.sum((a - b) ** 2)
            elif metric == "manhattan":
                distance = np.sum(np.abs(a - b))
            else:
                raise ValueError("Unknown metric")
            result[index] = distance
            index += 1
    return np.array(result)


def squareform(X):
    # implementation of squareform, pairwise distance matrix to condensed distance matrix
    # no vice versa, because it is not needed
    s = X.shape[0]
    d = round(sqrt(2 * s + 0.25) + 0.5)

    if d * (d - 1) != s * 2:
        raise ValueError(
            "Incompatible vector size. It must be a binomial "
            "coefficient n choose 2 for some integer n >= 2."
        )
    gc.collect()
    M = np.zeros((d, d))
    # fill the matrix
    p_count = 0
    for i in range(d):
        for j in range(i + 1, d):
            M[i, j] = X[p_count]
            M[j, i] = X[p_count]
            p_count += 1
    return M


class pelt_detector(cpd_detector):

    def __init__(
        self,
        pen=10,
        min_size=2,
        jump=1,
    ):
        self.cost = CostRbf()
        self.pen = pen
        self.min_size = min_size
        self.jump = jump
        self.n_samples = None

    def reinit(self):
        gc.collect()
        self.cost = CostRbf()

    def set_params(self, params_path, dataset_name):

        file_name = dataset_name.split("/")[-1].split(".")[0]

        print(f"Reading parameters from {params_path}")
        open_file = open(params_path, "r")
        params = []

        file_name = file_name.split(".")[0]
        for line in open_file.readlines():
            line = line.strip().split(",")
            if line[0] == file_name:
                params = line[1:]

        if len(params) == 0:
            print(f"Params not found for {file_name}")
            return -1

        self.pen = float(params[0])
        self.min_size = int(params[1])
        self.jump = int(params[2])

    def _seg(self, pen):
        partitions = dict()  # this dict will be recursively filled
        partitions[0] = {(0, 0): 0}
        admissible = []

        # Recursion
        ind = [k for k in range(0, self.n_samples, self.jump) if k >= self.min_size]
        ind += [self.n_samples]
        for bkp in ind:
            # adding a point to the admissible set from the previous loop.
            new_adm_pt = floor((bkp - self.min_size) / self.jump)
            new_adm_pt *= self.jump
            admissible.append(new_adm_pt)

            subproblems = list()
            for t in admissible:
                # left partition
                try:
                    tmp_partition = partitions[t].copy()
                except KeyError:  # no partition of 0:t exists
                    continue
                # we update with the right partition
                tmp_partition.update({(t, bkp): self.cost.error(t, bkp) + pen})
                subproblems.append(tmp_partition)

            # finding the optimal partition
            partitions[bkp] = min(subproblems, key=lambda d: sum(d.values()))
            # trimming the admissible set
            admissible = [
                t
                for t, partition in zip(admissible, subproblems)
                if sum(partition.values()) <= sum(partitions[bkp].values()) + pen
            ]

        best_partition = partitions[self.n_samples]
        del best_partition[(0, 0)]
        return best_partition

    def fit(self, signal) -> "pelt_detector":
        self.cost.fit(signal)

        if len(signal.shape) == 1:
            (n_samples,) = signal.shape
        else:
            n_samples, _ = signal.shape
        self.n_samples = n_samples
        return self

    def predict(self, pen):

        partition = self._seg(pen)
        bkps = sorted(e for s, e in partition.keys())
        return bkps

    def detect(self, data):
        self.fit(data)
        return self.predict(self.pen)


class CostRbf:
    model = "rbf"

    def __init__(self, gamma=None):
        """Initialize the object."""
        self.min_size = 1
        self.gamma = gamma
        self._gram = None

    @property
    def gram(self):
        if self._gram is None:
            K = pdist(self.signal, metric="sqeuclidean")
            if self.gamma is None:
                self.gamma = 1.0
                # median heuristics
                K_median = np.median(K)
                if K_median != 0:
                    # K /= K_median
                    self.gamma = 1 / K_median
            K *= self.gamma
            K = np.clip(K, 1e-2, 1e2)  # clipping to avoid exponential under/overflow
            self._gram = np.exp(squareform(-K))
        return self._gram

    def fit(self, signal) -> "CostRbf":
        if len(signal.shape) == 1:
            self.signal = signal.reshape(-1, 1)
        else:
            self.signal = signal

        if self.gamma is None:
            self.gram

        return self

    def error(self, start, end) -> float:
        if end - start < self.min_size:
            raise NotEnoughPoints
        sub_gram = self.gram[start:end, start:end]
        val = diagonal_sum(sub_gram)
        val -= np.sum(sub_gram) / (end - start)
        sub_gram = None
        return val


class NotEnoughPoints(Exception):
    """Raise this exception when there is not enough point to calculate a cost
    function."""


def test_single(data_path, params_path):
    print(f"Testing {data_path}")

    data = np.loadtxt(data_path, delimiter=",")

    detector = pelt_detector()
    detector.set_params(params_path, data_path)
    cpds = detector.detect(data)
    print(f"Change points: {cpds}")
    return cpds, data.shape[0]


def get_file_size(file_path):
    data = np.loadtxt(file_path, delimiter=",")
    return data.size


def test_all():
    import platform

    platform = platform.platform()
    if "Linux" in platform:
        csv_folder = "../datasets/csv"
        params_path = "../params/params_pelt_best.csv"
    else:
        csv_folder = "./csv"
        params_path = "./params/params_pelt_best.csv"

    files = os.listdir(csv_folder)

    # sort by size
    files.sort(key=lambda x: get_file_size(csv_folder + "/" + x))

    f1_cover_sum = 0
    f1_sum = 0
    for file in files:
        file_path = csv_folder + "/" + file
        cps, nobs = test_single(file_path, params_path)
        dataset_name = file.split(".")[0]
        f1, cover = accuracy.scores(cps, dataset_name=dataset_name, n_obs=nobs)
        print(f"F1: {f1}, Cover: {cover}")
        f1_cover_sum += f1 + cover
        f1_sum += f1

    print(f"Total F1 + Cover: {f1_cover_sum}")
    print(f"Total F1: {f1_sum}")


def main():
    gc.enable()
    test_all()


if __name__ == "__main__":
    main()
