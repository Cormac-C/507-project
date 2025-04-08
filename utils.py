import numpy as np
from numpy import ndarray, linalg as LA
from typing import Tuple, Literal


def normalize_data(data: ndarray) -> ndarray:
    for column in range(data.shape[1]):
        mean = np.nanmean(data[:, column])
        std = np.nanstd(data[:, column])
        data[:, column] = (data[:, column] - mean) / std
    return data


def clipped_norm(values: Tuple[float]) -> float:
    """
    Clip the values to be above 0 return the norm.
    Args:
        values (tuple[float]): Values to be clipped and normalized
    Returns:
        float: Clipped norm
    """
    clipped_values = np.clip(values, 0, None)
    return LA.norm(clipped_values)
