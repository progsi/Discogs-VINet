from typing import Tuple
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

def fft_mag(cqt: np.ndarray, n: int) -> Tuple[np.ndarray,np.ndarray]:
    """Adopted from: https://github.com/zafarrafii/CQHC-Python/blob/master/cqhc.py
    Args:
        cqt (np.ndarray): CQT spectrogram
        n (int): number of frequency channels
    Returns:
        Tuple[np.ndarray,np.ndarray]: fft, magnitude
    """
    # Compute the Fourier transform of every frame and their magnitude
    fft = np.fft.fft(cqt, 2 * n - 1, axis=0)
    mag = abs(fft)
    return fft, mag

def deconv_spectral(mag: np.ndarray, n: int) -> np.ndarray:
    """Deconv. spectral component.
    Args:
        mag (np.ndarray): magnitude FFT
        n (int): number of frequency channels
    Returns:
        np.ndarray: spectral component
    """
    return np.real(
        np.fft.ifft(mag, axis=0)[0:n, :]
    )
    
def deconv_pitch(fft: np.ndarray, mag: np.ndarray, n: int) -> np.ndarray:
    """Deconv. pitch component.
    Args:
        fft (np.ndarray): 
        mag (np.ndarray): 
        n (int): 
    Returns:
        np.ndarray: 
    """
    return np.real(
        np.fft.ifft(fft / (mag + 1e-16), axis=0)[
            0:n, :])
    
def deconv_cqt(cqt: np.ndarray) -> Tuple[np.ndarray,np.ndarray]:
    """Adopted from: https://github.com/zafarrafii/CQHC-Python/blob/master/cqhc.py
    Deconvolve the constant-Q transform (CQT) spectrogram into a pitch-normalized spectral component and an energy-normalized pitch component.
    Args:
        cqt (np.ndarray): CQT spectrogram
    Returns:
        Tuple[np.ndarray,np.ndarray]: spectral CQT component, pitch CQT component
    """
    # Get the number of frequency channels
    n = np.shape(cqt)[0]
    fft, mag = fft_mag(cqt, n)

    # Derive the spectral component 
    s = deconv_spectral(mag, n)
    # Derive the pitch component 
    p = deconv_pitch(fft, mag, n)
    return s, p

def cqhc(
    cqt: np.ndarray,
    octave_resolution: int =12,
    number_coefficients: int =20,
):
    """Adopted from: https://github.com/zafarrafii/CQHC-Python/blob/master/cqhc.py
    Compute the constant-Q harmonic coefficients (CQHCs).
    Args:
        cqt (np.ndarray): CQT spectrogram
        octave_resolution (int, optional): . Defaults to 12.
        number_coefficients (int, optional): . Defaults to 20.
    Returns:
        np.ndarray: 
    """
    # Derive the spectral component
    n = np.shape(cqt)[0]
    _, mag = fft_mag(cqt, n)

    # Derive the spectral component 
    s = deconv_spectral(mag, n)
    
    # Extract the CQHCs
    js = np.round(
        octave_resolution * np.log2(np.arange(1, number_coefficients + 1))
    ).astype(int)
    s = s[js, :]

    return s
