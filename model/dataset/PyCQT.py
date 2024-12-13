import numpy as np
import scipy

def compute_kernel(sf: int, octave_res: int, fmin: int, fmax: int) -> np.array:
    """_summary_
    Args:
        sf (int): sampling frequency in in Hz
        octave_res (int): octave resolution
        fmin (int): minimum frequency in in Hz
        fmax (int): maximum frequency in in Hz
    Returns:
        np.array: 
    """
    qf = 1 / (pow(2, 1 / octave_res) - 1)
    K = round(octave_res * np.log2(fmax / fmin))
    fft_len = int(pow(2, np.ceil(np.log2(qf * sf / fmin))))
    cqt_kernel = np.zeros((K, fft_len), dtype=complex)

    for i in range(K):
        K = fmin * pow(2, i / octave_res)
        win_len = (2 * round(qf * sf / K / 2) + 1)
        temporal_kernel = (np.hamming(win_len) * np.exp(2 * np.pi * 1j * qf * np.arange(-(win_len - 1) / 2, (win_len - 1) / 2 + 1) / win_len) / win_len)
        pad_width = int((fft_len - win_len + 1) / 2)
        cqt_kernel[i, pad_width: pad_width + win_len] = temporal_kernel

    cqt_kernel = np.fft.fft(cqt_kernel, axis=1)
    cqt_kernel[np.absolute(cqt_kernel) < 0.01] = 0
    cqt_kernel = scipy.sparse.csr_matrix(cqt_kernel)
    cqt_kernel = np.conjugate(cqt_kernel) / fft_len
    return cqt_kernel, fft_len
    
def compute_spectogram(y: np.array, sf: int, time_res: int, cqt_kernel: np.array) -> np.array:
    """Compute the CQT spectogram for an audio signal.
    Args:
        y (np.array): audio signal
        sf (int): sampling frequency in Hz
        time_res (int): time resolution
        cqt_kernel (np.array): cqt kernel
    Returns:
        np.array: cqt spectogram
    """
    l = round(sf / time_res)
    n = int(np.floor(len(y) / l))
    K, fft_len = np.shape(cqt_kernel)
    y = np.pad(y, (int(np.ceil((fft_len - l) / 2)), int(np.floor((fft_len - l) / 2))), "constant", constant_values=(0, 0))
    cqt = np.zeros((K, n))
    i = 0
    for j in range(n):
        cqt[:, j] = np.absolute(cqt_kernel * np.fft.fft(y[i: i + fft_len]))
        i = i + l
    return cqt

def compute_cqt(y: np.array, sr: int, hop_size: int, 
                octave_res: int = 12, fmin: int = 32, fmax: int = None) -> np.array:
    """Based on: https://github.com/zafarrafii/Zaf-Python/blob/master/zaf.py
    Args:
        y (np.array): signal
        sr (int): sampling rate
        hop_size (int): 
        octave_res (int, optional): octave resolution. Defaults to 12.
        fmin (int, optional): minimum frequency. Defaults to 32.
        fmax (int, optional): maximum frequency. Defaults to None.
    Returns:
        np.array: cqt feature
    """
    if not fmax:
        fmax = sr // 2

    kernel = compute_kernel(sr, octave_res, fmin, fmax)
    time_res = int(1 / hop_size)
    cqt = compute_spectogram(y, sr, time_res, kernel)
    cqt = cqt + 1e-9
    ref_value = np.max(cqt)
    cqt = 20 * np.log10(cqt) - 20 * np.log10(ref_value)
    return cqt.T
