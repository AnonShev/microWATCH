"""Distance measures to compare two probability density functions."""

try:
    from ulab import numpy as np
except ImportError:
    import numpy as np


# EPSILON = np.finfo(float).eps
EPSILON = (
    7.0 / 3 - 4.0 / 3 - 1
)  # no definition for np.finfo(float).eps in ulab, manually calculated


def isnan(num):
    return (num - num) != 0


def nansum(array):
    # no implementation for np.nansum in ulab, manually implement this
    return np.sum(np.where(isnan(array), 0, array))


def abs(array):
    # abs does not work in ulab, manually implement this
    return np.where(array < 0, -array, array)


def power(array, power):
    # no implementation for np.power in ulab, manually implement this
    return array**power


def acc(u, v):
    return (manhattan(u, v) + chebyshev(u, v)) / 2


def add_chisq(u, v):
    uvmult = u * v
    return np.sum(np.where(uvmult != 0, ((u - v) ** 2 * (u + v)) / uvmult, 0))


def bhattacharyya(u, v):
    return -np.log(np.sum(np.sqrt(u * v)))


def braycurtis(u, v):
    return np.sum(abs(u - v)) / np.sum(abs(u + v))


def canberra(u, v):
    return nansum(abs(u - v) / (abs(u) + abs(v)))


def chebyshev(u, v):
    return np.max(abs(u - v))


def chebyshev_min(u, v):
    return np.min(abs(u - v))


def clark(u, v):
    return np.sqrt(nansum(power(abs(u - v) / (u + v), 2)))


def czekanowski(u, v):
    return np.sum(abs(u - v)) / np.sum(u + v)


def divergence(u, v):
    return 2 * nansum(power(u - v, 2) / power(u + v, 2))


def euclidean(u, v):
    return np.linalg.norm(u - v)


def google(u, v):
    x = float(np.sum(u))
    y = float(np.sum(v))
    summin = float(np.sum(np.minimum(u, v)))
    return (max([x, y]) - summin) / ((x + y) - min([x, y]))


def gower(u, v):
    return np.sum(abs(u - v)) / u.size


def hellinger(u, v):
    return np.sqrt(2 * np.sum((np.sqrt(u) - np.sqrt(v)) ** 2))


def jeffreys(u, v, epsilon=EPSILON):
    u = np.where(u == 0, epsilon, u)
    v = np.where(v == 0, epsilon, v)
    return np.sum((u - v) * np.log(u / v))


def jensenshannon_divergence(u, v, epsilon=EPSILON):
    u = np.where(u == 0, epsilon, u)
    v = np.where(v == 0, epsilon, v)
    dl = u * np.log(2 * u / (u + v))
    dr = v * np.log(2 * v / (u + v))
    return (np.sum(dl) + np.sum(dr)) / 2


def jensen_difference(u, v, epsilon=EPSILON):
    u = np.where(u == 0, epsilon, u)
    v = np.where(v == 0, epsilon, v)
    el1 = (u * np.log(u) + v * np.log(v)) / 2
    el2 = (u + v) / 2
    return np.sum(el1 - el2 * np.log(el2))


def k_divergence(u, v, epsilon=EPSILON):
    u = np.where(u == 0, epsilon, u)
    v = np.where(v == 0, epsilon, v)
    return np.sum(u * np.log(2 * u / (u + v)))


def kl_divergence(u, v, epsilon=EPSILON):
    u = np.where(u == 0, epsilon, u)
    v = np.where(v == 0, epsilon, v)
    return np.sum(u * np.log(u / v))


def kulczynski(u, v):
    return np.sum(abs(u - v)) / np.sum(np.minimum(u, v))


def lorentzian(u, v):
    return np.sum(np.log(abs(u - v) + 1))


def manhattan(u, v):
    return np.sum(abs(u - v))


def marylandbridge(u, v):
    uvdot = np.dot(u, v)
    return 1 - (uvdot / np.dot(u, u) + uvdot / np.dot(v, v)) / 2


def matusita(u, v):
    return np.sqrt(np.sum((np.sqrt(u) - np.sqrt(v)) ** 2))


def max_symmetric_chisq(u, v):
    return max(neyman_chisq(u, v), pearson_chisq(u, v))


def minkowski(u, v, p=2):
    return np.linalg.norm(u - v)


def motyka(u, v):
    return np.sum(np.maximum(u, v)) / np.sum(u + v)


def neyman_chisq(u, v):
    return np.sum(np.where(u != 0, (u - v) ** 2 / u, 0))


def nonintersection(u, v):
    return 1 - np.sum(np.minimum(u, v))


def pearson_chisq(u, v):
    return np.sum(np.where(v != 0, (u - v) ** 2 / v, 0))


def penroseshape(u, v):
    umu = np.mean(u)
    vmu = np.mean(v)
    return np.sqrt(np.sum(((u - umu) - (v - vmu)) ** 2))


def soergel(u, v):
    return np.sum(abs(u - v)) / np.sum(np.maximum(u, v))


def squared_chisq(u, v):
    uvsum = u + v
    return np.sum(np.where(uvsum != 0, (u - v) ** 2 / uvsum, 0))


def squaredchord(u, v):
    return np.sum((np.sqrt(u) - np.sqrt(v)) ** 2)


def taneja(u, v, epsilon=EPSILON):
    u = np.where(u == 0, epsilon, u)
    v = np.where(v == 0, epsilon, v)
    uvsum = u + v
    return np.sum((uvsum / 2) * np.log(uvsum / (2 * np.sqrt(u * v))))


def tanimoto(u, v):
    usum = np.sum(u)
    vsum = np.sum(v)
    minsum = np.sum(np.minimum(u, v))
    return (usum + vsum - 2 * minsum) / (usum + vsum - minsum)


def topsoe(u, v, epsilon=EPSILON):
    u = np.where(u == 0, epsilon, u)
    v = np.where(v == 0, epsilon, v)
    dl = u * np.log(2 * u / (u + v))
    dr = v * np.log(2 * v / (u + v))
    return np.sum(dl + dr)


def vicis_symmetric_chisq(u, v):
    u_v = (u - v) ** 2
    uvmin = np.minimum(u, v) ** 2
    return np.sum(np.where(uvmin != 0, u_v / uvmin, 0))


def vicis_wave_hedges(u, v):
    u_v = abs(u - v)
    uvmin = np.minimum(u, v)
    return np.sum(np.where(uvmin != 0, u_v / uvmin, 0))


def wave_hedges(u, v):
    u_v = abs(u - v)
    uvmax = np.maximum(u, v)
    return np.sum(np.where(((u_v != 0) & (uvmax != 0)), u_v / uvmax, 0))
