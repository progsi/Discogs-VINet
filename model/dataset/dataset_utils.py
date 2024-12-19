import numpy as np


def mean_downsample_cqt(cqt: np.ndarray, mean_window_length: int) -> np.ndarray:
    """Downsamples the CQT by taking the mean of every `mean_window_length` frames without
    overlapping. Adapted from https://github.com/yzspku/TPPNet/blob/master/data/gencqt.py

    Parameters:
    -----------
        cqt: np.ndarray, shape=(T,F), CQT to downsample
        mean_window_length: int, number of frames to average together

    Returns:
    --------
        new_cqt: np.ndarray, shape=(T//mean_window_length,F), downsampled CQT
    """
    # get dimensions and get new T
    cqt_T, cqt_F = cqt.shape
    new_T = cqt_T // mean_window_length
    
    # downsample
    reshaped_cqt = cqt[:new_T * mean_window_length].reshape(new_T, mean_window_length, cqt_F)
    new_cqt = reshaped_cqt.mean(axis=1)
    return new_cqt
