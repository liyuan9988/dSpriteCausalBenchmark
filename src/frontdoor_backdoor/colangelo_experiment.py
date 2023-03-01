import numpy as np
from numpy.random import default_rng
from scipy.stats import norm

from ..frontdoor_backdoor.data_class import BackDoorTrainDataSet, ATETestDataSet


def generate_test_colangelo() -> ATETestDataSet:
    """
    Returns
    -------
    test_data : ATETestDataSet
        Uniformly sampled from price. time and emotion is averaged to get structural.
    """
    treatment = np.linspace(0.0, 1.0, 11)
    structural = 1.2 * treatment + treatment * treatment

    return ATETestDataSet(treatment=treatment[:, np.newaxis],
                          structural=structural[:, np.newaxis])


def generate_colangelo_ate(data_size: int,
                           rand_seed: int = 42) -> BackDoorTrainDataSet:
    """
    Generate the data in Double Debiased Machine Learning Nonparametric Inference with Continuous Treatments
    [Colangelo and Lee, 2020]

    Parameters
    ----------
    data_size : int
        size of data
    rand_seed : int
        random seed


    Returns
    -------
    train_data : ATETrainDataSet
    """

    rng = default_rng(seed=rand_seed)
    backdoor_dim = 100
    backdoor_cov = np.eye(backdoor_dim)
    backdoor_cov += np.diag([0.5] * (backdoor_dim - 1), 1)
    backdoor_cov += np.diag([0.5] * (backdoor_dim - 1), -1)
    backdoor = rng.multivariate_normal(np.zeros(backdoor_dim), backdoor_cov,
                                       size=data_size)  # shape of (data_size, backdoor_dim)

    theta = np.array([1.0 / ((i + 1) ** 2) for i in range(backdoor_dim)])
    treatment = norm.cdf(backdoor.dot(theta) * 3) + 0.75 * rng.standard_normal(size=data_size)
    outcome = 1.2 * treatment + 1.2 * backdoor.dot(theta) + treatment ** 2 + treatment * backdoor[:, 0]
    outcome += rng.standard_normal(size=data_size)

    train = BackDoorTrainDataSet(backdoor=backdoor,
                                 treatment=treatment[:, np.newaxis],
                                 outcome=outcome[:, np.newaxis])

    test = generate_test_colangelo()

    return train, test
