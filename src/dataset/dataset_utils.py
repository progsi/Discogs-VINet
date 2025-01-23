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


def normalize_cqt(cqt: np.ndarray) -> np.ndarray:
    """_summary_
    Args:
        cqt (np.ndarray): cqtspectrogram
    Returns:
        np.ndarray: normalized cqt spectrogram
    """
    cqt /= (
                np.max(cqt) + 1e-6
            )
    return cqt
    
def upscale_cqt_values(cqt: np.ndarray) -> np.ndarray:
    """This is done in CoverHunter to obtain higher values in the CQT spectrogram.
    See: https://github.com/Liu-Feng-deeplearning/CoverHunter/blob/main/src/cqt.py
    Args:
        cqt (np.ndarray): _description_
    """
    cqt = np.maximum(cqt, 1e-10)
    ref_value = np.max(cqt)
    cqt = 20 * np.log10(cqt) - 20 * np.log10(ref_value)
    return cqt

def normalize_cqt(cqt: np.ndarray) -> np.ndarray:
    """Normalize the CQT spectrogram by dividing by the maximum value.
    Args:
        cqt (np.ndarray): cqt spectrogram
    Returns:
        np.ndarray: normalized cqt spectrogram
    """
    cqt /= (
                np.max(cqt) + 1e-6
            )
    return cqt
