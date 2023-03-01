from filelock import FileLock
import numpy as np
import pathlib
import pandas as pd
from typing import Literal
from src.frontdoor_backdoor.data_class import BackDoorTrainDataSet

DATA_PATH = pathlib.Path(__file__).resolve().parent.parent.parent.parent.joinpath("data/")


def filter_jobcorp(data, subsets: Literal["all", "d-filter", "dm-filter"]):
    if subsets == "all":
        sub = data
    elif subsets == "d-filter":
        sub = data.loc[data["d"] >= 40, :]
    elif subsets == "dm-filter":
        sub = data.loc[data["m"] > 0, :]
        sub = sub.loc[sub["d"] >= 40, :]
    else:
        raise ValueError()
    return sub


def generate_jobcorp_ate(subsets: Literal["all", "d-filter", "dm-filter"]) -> BackDoorTrainDataSet:
    with FileLock(DATA_PATH.joinpath("./data.lock")):
        data = pd.read_csv(DATA_PATH.joinpath("job_corps/JCdata.csv"), sep=" ")

    sub = filter_jobcorp(data, subsets)
    outcome = sub["m"].to_numpy()
    treatment = sub["d"].to_numpy()
    backdoor = sub.iloc[:, 3:].to_numpy()
    return BackDoorTrainDataSet(backdoor=backdoor,
                                outcome=outcome[:, np.newaxis],
                                treatment=treatment[:, np.newaxis])
