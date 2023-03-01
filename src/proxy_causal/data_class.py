from typing import NamedTuple, Optional
import numpy as np

class PVTrainDataSet(NamedTuple):
    treatment: np.ndarray
    treatment_proxy: np.ndarray
    outcome_proxy: np.ndarray
    outcome: np.ndarray


class PVTestDataSet(NamedTuple):
    treatment: np.ndarray
    structural: Optional[np.ndarray]

