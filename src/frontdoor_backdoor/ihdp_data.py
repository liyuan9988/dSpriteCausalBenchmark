from filelock import FileLock
import numpy as np
import pathlib

from src.frontdoor_backdoor.data_class import BackDoorTrainDataSet, ATETestDataSet

DATA_PATH = pathlib.Path(__file__).resolve().parent.parent.parent.joinpath("data/")


def generate_ihdp_ate(rand_seed: int = 42) -> (BackDoorTrainDataSet, ATETestDataSet):
    idx = (rand_seed % 1000) + 1
    with FileLock("./data.lock"):
        data = np.genfromtxt(DATA_PATH.joinpath(f"IHDP/sim_data/ihdp_{idx}.csv"), delimiter=" ")
    t, y, y_cf = data[:, 0][:, None], data[:, 1][:, None], data[:, 2][:, None]
    mu_0, mu_1, x = data[:, 3][:, None], data[:, 4][:, None], data[:, 5:]
    mean_y = np.mean(y)

    train_data = BackDoorTrainDataSet(backdoor=x,
                                      outcome=y - mean_y,
                                      treatment=t)
    test_data = ATETestDataSet(treatment=np.array([[0], [1]]),
                               structural=np.array([[np.mean(mu_0 - mean_y)], [np.mean(mu_1 - mean_y)]]))

    return train_data, test_data
