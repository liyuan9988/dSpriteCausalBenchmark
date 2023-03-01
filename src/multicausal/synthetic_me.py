from typing import Tuple
import numpy as np
from numpy.random import default_rng

from src.multicausal.data_class import METrainDataSet, METestDataSet


def generate_train_synthetic_me(data_size: int, rand_seed: int = 42) -> Tuple[METrainDataSet, METestDataSet]:
    rng = np.random.default_rng(rand_seed)
    backdoor = rng.uniform(-1.5, 1.5, size=data_size)
    u = rng.uniform(-2, 2, size=data_size)
    v = rng.uniform(-2, 2, size=data_size)
    w = rng.uniform(-2, 2, size=data_size)
    treatment = 0.3 * backdoor + w
    mediate = 0.3 * treatment + 0.3 * backdoor + v
    outcome = 0.3 * treatment + 0.3 * mediate + 0.5 * treatment * mediate + 0.3 * backdoor + 0.25 * treatment ** 3 + u

    train_data = METrainDataSet(treatment=treatment[:, np.newaxis],
                                backdoor=backdoor[:, np.newaxis],
                                mediate=mediate[:, np.newaxis],
                                outcome=outcome[:, np.newaxis])

    test_data = generate_test_synthetic_me()
    return train_data, test_data

def generate_test_synthetic_me():
    xv, yv = np.meshgrid(np.linspace(-1.5, 1.5, 9), np.linspace(-1.5, 1.5, 9))
    treatment = xv.ravel()
    new_treatment = yv.ravel()
    structural = 0.3 * new_treatment + 0.09 * treatment + 0.15 * treatment * new_treatment + 0.25 * new_treatment ** 3
    return METestDataSet(treatment=treatment[:, np.newaxis],
                         new_treatment=new_treatment[:, np.newaxis],
                         structural=structural[:, np.newaxis])
