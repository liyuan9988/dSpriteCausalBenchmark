from typing import NamedTuple, Optional
import numpy as np


class BackDoorTrainDataSet(NamedTuple):
    treatment: np.ndarray
    backdoor: np.ndarray
    outcome: np.ndarray


class FrontDoorTrainDataSet(NamedTuple):
    treatment: np.ndarray
    frontdoor: np.ndarray
    outcome: np.ndarray


class ATETestDataSet(NamedTuple):
    treatment: np.ndarray
    structural: np.ndarray


class ATTTestDataSet(NamedTuple):
    treatment_cf: np.ndarray  # counterfactual treament
    treatment_ac: np.ndarray  # actural treament
    target: np.ndarray
