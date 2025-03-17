import os

try:
    from ulab import numpy as np
except:
    import numpy as np

try:
    from ulab import scipy as spy
except:
    import scipy as spy

from cpd_detector import cpd_detector

import accuracy

gammaln = spy.special.gammaln


def gamma(z):
    return spy.special.gamma(z)


def hazard_function(r):
    lam = 250
    return 1 / lam * np.ones(r.shape)


def online_changepoint_detection(data, hazard_function, log_likelihood_class):

    maxes = np.zeros(len(data) + 1)

    R = np.zeros((len(data) + 1, len(data) + 1))
    R[0, 0] = 1

    for t, x in enumerate(data):

        predprobs = log_likelihood_class.pdf(x)

        H = hazard_function(np.array(range(t + 1)))

        R[1 : t + 2, t + 1] = R[0 : t + 1, t] * predprobs * (1 - H)

        R[0, t + 1] = np.sum(R[0 : t + 1, t] * predprobs * H)

        R[:, t + 1] = R[:, t + 1] / np.sum(R[:, t + 1])

        log_likelihood_class.update_theta(x, t=t)

        maxes[t] = np.argmax(R[:, t])

    Nw = 5
    cp_probs = R[Nw, Nw:-1]

    cps = []
    for i in range(1, len(cp_probs)):
        if cp_probs[i] > 0.1:
            cps.append(i - Nw)

    return cps


def t_pdf(x, df, loc=0, scale=1):

    z = (x - loc) / scale

    numerator = gamma((df + 1) / 2.0)
    df_sqrt = np.sqrt(df * np.pi)
    df_gamma = gamma(df / 2.0)
    denominator = df_sqrt * df_gamma

    coefficient = numerator / denominator

    # (1 + z^2/df)^(-(df+1)/2)
    exponent = -(df + 1) / 2.0
    pdf_val_standard = coefficient * (1 + (z**2 / df)) ** exponent

    # loc-scale变换后的 PDF: 对标准 PDF 再除以 scale
    return pdf_val_standard / scale


class StudentT:
    def __init__(
        self, alpha: float = 0.1, beta: float = 0.1, kappa: float = 1, mu: float = 0
    ):

        self.alpha0 = self.alpha = np.array([alpha])
        self.beta0 = self.beta = np.array([beta])
        self.kappa0 = self.kappa = np.array([kappa])
        self.mu0 = self.mu = np.array([mu])

    def pdf(self, data: np.array):
        return t_pdf(
            x=data,
            df=2 * self.alpha,
            loc=self.mu,
            scale=np.sqrt(self.beta * (self.kappa + 1) / (self.alpha * self.kappa)),
        )

    def update_theta(self, data: np.array, **kwargs):

        muT0 = np.concatenate(
            (self.mu0, (self.kappa * self.mu + data) / (self.kappa + 1))
        )
        kappaT0 = np.concatenate((self.kappa0, self.kappa + 1.0))
        alphaT0 = np.concatenate((self.alpha0, self.alpha + 0.5))
        betaT0 = np.concatenate(
            (
                self.beta0,
                self.beta
                + (self.kappa * (data - self.mu) ** 2) / (2.0 * (self.kappa + 1.0)),
            )
        )

        self.mu = muT0
        self.kappa = kappaT0
        self.alpha = alphaT0
        self.beta = betaT0


def isnan(num):
    return (num - num) != 0


def logaddexp(x1, x2):
    # impelment logaddexp
    return np.log(np.exp(x1) + np.exp(x2))


def var(x):
    # implement variance without using numpy.var
    return np.mean((x - np.mean(x)) ** 2)


def simple_logsumexp(x):
    x_max = np.max(x)
    return x_max + np.log(np.sum(np.exp(x - x_max)))


def const_prior(t, p: float = 0.25):
    return np.log(p)


class BOCPD(cpd_detector):
    def __init__(self, version=0, alpha=0.1, beta=0.01, kappa=1, mu=0):
        self.version = version
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.mu = mu

    def reinit(self):
        pass

    def detect(self, data):
        cps = online_changepoint_detection(
            data,
            hazard_function,
            StudentT(alpha=self.alpha, beta=self.beta, kappa=self.kappa, mu=self.mu),
        )
        return cps

    def set_params(self, params_path, dataset_name):
        # format:dataset,alpha,beta,kappa,mu,f1,cover
        with open(params_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                if line.split(",")[0] == dataset_name:
                    self.alpha = float(line.split(",")[1])
                    self.beta = float(line.split(",")[2])
                    self.kappa = float(line.split(",")[3])
                    self.mu = float(line.split(",")[4])
                    break


def main():
    test()


def test():
    # for all univariate datasets, run the BOCPD algorithm
    dataset_path = "../datasets/csv"

    for file_name in os.listdir(dataset_path):
        data = np.loadtxt(f"{dataset_path}/{file_name}", delimiter=",")
        if len(data.shape) > 1 and data.shape[1] > 1:
            continue

        print(f"Running BOCPD on {file_name}")
        bocpd = BOCPD()
        # set parameters for the dataset
        dataset_name = file_name.split(".")[0]
        bocpd.set_params("../params/params_bocpd_best.csv", dataset_name)

        cps = bocpd.detect(data)
        f1, cover = accuracy.scores(cps, dataset_name, len(data))
        print(f"F1 score: {f1}, Covering: {cover}")
        print(f"Change points detected: {cps}")


if __name__ == "__main__":
    main()
