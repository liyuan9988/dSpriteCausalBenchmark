import numpy as np
from numpy.random import default_rng

from ..proxy_causal.data_class import PVTrainDataSet, PVTestDataSet


def generate_kpv_experiment_pv(n_sample: int, seed=42, **kwargs):
    global A_scaler
    rng = default_rng(seed=seed)
    U2 = rng.uniform(-1, 2, n_sample)
    U1 = rng.uniform(0, 1, n_sample)
    U1 = U1 - (np.logical_and(U2 > 0, U2 < 1))
    W1 = U1 + rng.uniform(-1, 1, n_sample)
    W2 = U2 + rng.normal(0, 1, n_sample) * 3
    Z1 = U1 + rng.normal(0, 1, n_sample) * 3
    Z2 = U2 + rng.uniform(-1, 1, n_sample)
    A = U2 + rng.normal(0, 1, n_sample) * 0.05
    Y = U2 * np.cos(2 * (A + 0.3 * U1 + 0.2))
    train_data = PVTrainDataSet(outcome=Y[:, np.newaxis],
                                treatment=A[:, np.newaxis],
                                treatment_proxy=np.c_[Z1, Z2],
                                outcome_proxy=np.c_[W1, W2])
    test_data = generate_test_kpv_experiment()
    return train_data, test_data


def get_structure(A: float):
    n_sample = 100
    rng = default_rng(seed=42)
    U2 = rng.uniform(-1, 2, n_sample)
    U1 = rng.uniform(0, 1, n_sample)
    U1 = U1 - (np.logical_and(U2 > 0, U2 < 1))
    Y = U2 * np.cos(2 * (A + 0.3 * U1 + 0.2))
    return np.mean(Y)


def generate_test_kpv_experiment():
    test_a = np.linspace(-1.0, 2.0, 20)
    structure = np.array([get_structure(a) for a in test_a])
    return PVTestDataSet(structural=structure[:, np.newaxis],
                         treatment=test_a[:, np.newaxis])
