from filelock import FileLock
import numpy as np

import pathlib
from typing import Literal

from ..proxy_causal.data_class import PVTrainDataSet, PVTestDataSet

DATA_PATH = pathlib.Path(__file__).resolve().parent.parent.parent.joinpath("data/")


def generate_deaner_experiment_pv(id: Literal["IM", "IR"], seed=42, **kwargs):
    with FileLock("./data.lock"):
        seed = (seed % 10) * 100
        if seed == 0:
            seed = 1000

        test_data_raw = np.load(DATA_PATH.joinpath(f'sim_1d_no_x/do_A_edu_{id}_80_seed100.npz'))
        train_data_raw = np.load(DATA_PATH.joinpath(f'sim_1d_no_x/main_edu_{id}_80_seed{seed}.npz'))

    train_data = PVTrainDataSet(outcome=train_data_raw["train_y"],
                                treatment=train_data_raw["train_a"],
                                treatment_proxy=train_data_raw["train_z"],
                                outcome_proxy=train_data_raw["train_w"])

    test_data = PVTestDataSet(structural=test_data_raw['gt_EY_do_A'][:, np.newaxis],
                              treatment=test_data_raw['do_A'])

    return train_data, test_data
