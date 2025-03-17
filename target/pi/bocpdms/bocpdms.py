import copy
from itertools import product
import json
import os
import concurrent.futures
import numpy as np
from bocpdms.BVARNIG import BVARNIG
from bocpdms.CpModel import CpModel
import accuracy
from bocpdms.detector import Detector

bocpd_intensities = [10, 50, 100, 200]
bocpd_prior_a = [0.01, 0.1, 1.0, 10, 100]
bocpd_prior_b = [0.01, 0.1, 1.0, 10, 100]


threshold = 100


def load_dataset(filename):
    """Load a CPDBench dataset"""
    with open(filename, "r") as fp:
        data = json.load(fp)

    if data["time"]["index"] != list(range(0, data["n_obs"])):
        raise NotImplementedError(
            "Time series with non-consecutive time axis are not yet supported."
        )

    mat = np.zeros((data["n_obs"], data["n_dim"]))
    for j, series in enumerate(data["series"]):
        mat[:, j] = series["raw"]

    # We normalize to avoid numerical errors.
    mat = (mat - np.nanmean(mat, axis=0)) / np.sqrt(np.nanvar(mat, axis=0, ddof=1))

    return data, mat


def run_bocpdms(mat, params):
    """Set up and run BOCPDMS"""

    AR_models = []
    for lag in range(params["lower_AR"], params["upper_AR"] + 1):
        AR_models.append(
            BVARNIG(
                prior_a=params["prior_a"],
                prior_b=params["prior_b"],
                S1=params["S1"],
                S2=params["S2"],
                prior_mean_scale=params["prior_mean_scale"],
                prior_var_scale=params["prior_var_scale"],
                intercept_grouping=params["intercept_grouping"],
                nbh_sequence=[0] * lag,
                restriction_sequence=[0] * lag,
                hyperparameter_optimization="online",
            )
        )

    cp_model = CpModel(params["intensity"])

    model_universe = np.array(AR_models)
    model_prior = np.array([1 / len(AR_models) for m in AR_models])

    detector = Detector(
        data=mat,
        model_universe=model_universe,
        model_prior=model_prior,
        cp_model=cp_model,
        S1=params["S1"],
        S2=params["S2"],
        T=mat.shape[0],
        store_rl=True,
        store_mrl=True,
        trim_type="keep_K",
        threshold=params["threshold"],
        save_performance_indicators=True,
        generalized_bayes_rld="kullback_leibler",
        # alpha_param_learning="individual",  # not sure if used
        # alpha_param=0.01,  # not sure if used
        # alpha_param_opt_t=30,  # not sure if used
        # alpha_rld_learning=True,  # not sure if used
        loss_der_rld_learning="squared_loss",
        loss_param_learning="squared_loss",
    )
    detector.run()

    return detector


def detect(data, intensity, prior_a, prior_b):
    
    # data, mat = load_dataset(file_name)
    # print(mat)
    
    # normalize the data to get mat
    mat = data
    if len(mat.shape) == 1:
        # wrap to 2D array
        mat = np.expand_dims(mat, axis=1)
    mat = (mat - np.nanmean(mat, axis=0)) / np.sqrt(np.nanvar(mat, axis=0, ddof=1))
    

    # setting S1 as dimensionality follows the 30portfolio_ICML18.py script.
    defaults = {
        "S1": mat.shape[1],
        "S2": 1,
        "intercept_grouping": None,
        "prior_mean_scale": 0,  # data is standardized
        "prior_var_scale": 1,  # data is standardized
    }

    # pick the lag lengths based on the paragraph below the proof of Theorem 1,
    # using C = 1, as in ``30portfolio_ICML18.py``.
    T = mat.shape[0]
    Lmin = 1
    Lmax = int(pow(T / np.log(T), 0.25) + 1)
    defaults["lower_AR"] = Lmin
    defaults["upper_AR"] = Lmax

    args = {}
    args["intensity"] = intensity
    args["prior_a"] = prior_a
    args["prior_b"] = prior_b
    args["threshold"] = threshold

    # join two dictionaries: defaults and args
    parameters = copy.deepcopy(defaults)
    parameters.update(args)
    print(parameters)

    detector = run_bocpdms(mat, parameters)
    # According to the Nile unit test, the MAP change points are in
    # detector.CPs[-2], with time indices in the first of the two-element
    # vectors.
    locations = [x[0] for x in detector.CPs[-2]]

    # Based on the fact that time_range in plot_raw_TS of the EvaluationTool
    # starts from 1 and the fact that CP_loc that same function is ensured to
    # be in time_range, we assert that the change point locations are 1-based.
    # We want 0-based, so subtract 1 from each point.
    locations = [loc - 1 for loc in locations]

    # convert to Python ints
    locations = [int(loc) for loc in locations]
    return locations