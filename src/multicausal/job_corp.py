from filelock import FileLock
import numpy as np
import pathlib
import pandas as pd
from typing import Literal, NamedTuple, Optional

from src.multicausal.data_class import METrainDataSet, SATETrainDataSet

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


def generate_jobcorp_me(subsets: Literal["all", "d-filter", "dm-filter"]) -> METrainDataSet:
    with FileLock("./data.lock"):
        data = pd.read_csv(DATA_PATH.joinpath("job_corps/JCdata.csv"), sep=" ")

    sub = filter_jobcorp(data, subsets)
    outcome = sub["y"].to_numpy()
    treatment = sub["d"].to_numpy()
    mediate = sub["m"].to_numpy()
    backdoor = sub.iloc[:, 3:].to_numpy()

    return METrainDataSet(outcome=outcome[:, np.newaxis],
                          treatment=treatment[:, np.newaxis],
                          mediate=mediate[:, np.newaxis],
                          backdoor=backdoor)



def generate_jobcorp_sate():
    with FileLock("./data.lock"):
        X1 = pd.read_csv(DATA_PATH.joinpath("multi_stage_corp/X1.csv"), sep=",").to_numpy()
        X2 = pd.read_csv(DATA_PATH.joinpath("multi_stage_corp/X2.csv"), sep=",").to_numpy()
        D1 = pd.read_csv(DATA_PATH.joinpath("multi_stage_corp/D1.csv"), sep=",").to_numpy()
        D2 = pd.read_csv(DATA_PATH.joinpath("multi_stage_corp/D2.csv"), sep=",").to_numpy()
        Y = pd.read_csv(DATA_PATH.joinpath("multi_stage_corp/Y1.csv"), sep=",").to_numpy()

    flg = (D1[:, 0] + D2[:, 0]) > 40
    flg = np.logical_and(flg, Y[:, 0] > 0)
    return SATETrainDataSet(backdoor_1st=X1[flg],
                            backdoor_2nd=X2[flg],
                            treatment_1st=D1[flg],
                            treatment_2nd=D2[flg],
                            outcome=Y[flg])
