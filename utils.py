import numpy as np
from numpy import ndarray


def normalize_data(data: ndarray) -> ndarray:
    for column in range(data.shape[1]):
        mean = np.nanmean(data[:, column])
        std = np.nanstd(data[:, column])
        data[:, column] = (data[:, column] - mean) / std
    return data
