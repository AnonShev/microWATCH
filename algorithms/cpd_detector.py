
try:
    from ulab import numpy as np
except ImportError:
    import numpy as np

# abstract change point detector, no implementation, interface only
class cpd_detector():

    def __init__(self, version=0):
        # algorithm_version is the index of the algorithm in a list of different algorithms
        # it is used for the microwatch algorithm to get the distance metric
        raise NotImplementedError("This is an interface, do not instantiate it")

    def detect(self, data):
        raise NotImplementedError("This is an interface, do not instantiate it")

    def set_params(self, params_path, dataset_name):
        raise NotImplementedError("This is an interface, do not instantiate it")
    

    def reinit(self):
        # reinitialize the detector
        raise NotImplementedError("This is an interface, do not instantiate it")


    