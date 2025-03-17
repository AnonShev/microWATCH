import json

# from metrics import covering
# from metrics import f_measure

# message 1 for accuracy
# message 2 for accuracy


def true_positives(T, X, margin=5):
    # make a copy so we don't affect the caller
    X = set(list(X))
    TP = set()
    for tau in T:
        close = [(abs(tau - x), x) for x in X if abs(tau - x) <= margin]
        close.sort()
        if not close:
            continue
        dist, xstar = close[0]
        TP.add(tau)
        X.remove(xstar)
    return TP


def f_measure(annotations, predictions, margin=5, alpha=0.5, return_PR=False):
    # ensure 0 is in all the sets
    Tks = {k + 1: set(annotations[uid]) for k, uid in enumerate(annotations)}
    for Tk in Tks.values():
        Tk.add(0)

    X = set(predictions)
    X.add(0)

    Tstar = set()
    for Tk in Tks.values():
        for tau in Tk:
            Tstar.add(tau)

    K = len(Tks)

    P = len(true_positives(Tstar, X, margin=margin)) / len(X)

    TPk = {k: true_positives(Tks[k], X, margin=margin) for k in Tks}
    R = 1 / K * sum(len(TPk[k]) / len(Tks[k]) for k in Tks)

    F = P * R / (alpha * R + (1 - alpha) * P)
    if return_PR:
        return F, P, R
    return F


def overlap(A, B):
    return len(A.intersection(B)) / len(A.union(B))


def partition_from_cps(locations, n_obs):
    T = n_obs
    partition = []
    current = set()

    all_cps = iter(sorted(set(locations)))

    try:
        cp = next(all_cps)
    # handle the stop iteration error
    except StopIteration:
        cp = None

    for i in range(T):
        if i == cp:
            if current:
                partition.append(current)
            current = set()
            try:
                cp = next(all_cps)
            except StopIteration:
                cp = None
        current.add(i)
    partition.append(current)
    return partition


def cover_single(S, Sprime):
    T = sum(map(len, Sprime))
    assert T == sum(map(len, S))
    C = 0
    for R in S:
        C += len(R) * max(overlap(R, Rprime) for Rprime in Sprime)
    C /= T
    return C


def covering(annotations, predictions, n_obs):
    Ak = {
        k + 1: partition_from_cps(annotations[uid], n_obs)
        for k, uid in enumerate(annotations)
    }
    pX = partition_from_cps(predictions, n_obs)

    Cs = [cover_single(Ak[k], pX) for k in Ak]
    return sum(Cs) / len(Cs)


def load_annotations(filename, dataset):
    with open(filename, "r") as fp:
        data = json.load(fp)
    return data[dataset]


def clean_cps(locations, n_obs):
    # Filter change points to ensure they fall within the valid range [1, n_obs - 2]
    valid_cps = set([cp for cp in locations if 1 <= cp < n_obs - 1])
    # Return the sorted list of valid change points
    return sorted(valid_cps)


def scores(locations, dataset_name, n_obs, annotations_file="../../annotations.json"):
    # Initialize the attribute if it doesn't exist
    annotations = load_annotations(annotations_file, dataset_name)

    locations = clean_cps(locations, n_obs)
    f1, precision, recall = f_measure(annotations, locations, return_PR=True)
    cover = covering(annotations, locations, n_obs)
    return f1, cover
