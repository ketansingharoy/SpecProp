"""
It is a module that contains various functions to estimate spectral contents of a real signal.

The functions are next_power_2, compute_complex_spectrum, compute_spectral_amplitude etc.
"""


import numpy as np
from scipy import signal



def next_power_2(n):
    """Computes and returns the next power of 2.
    
    This function takes an input positive integer. It then computes the next power of 2 closest to the input. Finally, it returns the next power of 2.
    
    Parameter
    ---------
    n: int
        A positive integer.
        
    Returns
    -------
    int
        It is the next power of 2 of the input number.
        
    Examples
    --------
    >>> next_power_2(30)
    32
    """
    return int(2 ** np.ceil(np.log2(n)))


def compute_complex_spectrum(x, fs, nfft=None, taper=5.0):
    """Computes and returns complex amplitude spectrum of a real signal.
    
    If unit of x is 'V' and unit of fs is 'Hz', then unit of output X is 'V'.
    
    Parameters
    ----------
    x: numpy.ndarray of float
        A 1-D array representing a signal.
    fs: int
        Sampling rate.
    nfft: int (optional)
        Number of points for Fourier transformation.
        Default is None.
    taper: float (optional)
        Taper percentage for a tapered cosine window.
        This is applied to (multiplied with) the input signal before computation of spectral contents.
        It ranges in the interval [0.0, 50.0].
        Default is 5.0 (i.e., 5% cosine taper).
        
    Returns
    -------
    f: numpy.ndarray of float
        A 1-D array of frequencies.
    X: numpy.ndarray of complex
        A 1-D array of complex spectral amplitudes.
        
    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([0, 1, 0, 0])
    >>> fs = 1
    >>> f, X = compute_complex_spectrum(x, fs)
    >>> print(f)
    array([0.   0.25 0.5 ])
    >>> print(X)
    array([ 2.+0.j  0.-2.j -2.+0.j])
    """
    
    # length of input signal
    n = len(x)
    
    # sampling interval
    dt = 1 / fs
    
    # taper the signal
    alpha = 2.0 * taper / 100.0
    w = signal.windows.tukey(n, alpha=alpha)
    x = x * w
    
    # number of points for fourier transformation
    if nfft == None:
        nfft = next_power_2(n)
        
    # Form frequency vector
    f = np.fft.rfftfreq(nfft, d=dt)
    
    # Compute complex amplitude spectrum using fourier transformation
    # 2 is multipled, because it is a one-sided spectrum
    X = 2 * np.fft.rfft(x, n=nfft)
    
    return f, X


def compute_spectral_amplitude(x, fs, nfft=None, taper=5.0):
    """Computes and returns spectral amplitude of a real signal.
    
    If unit of x is 'V' and unit of fs is 'Hz', then unit of output A is 'V'.
    
    Parameters
    ----------
    x: numpy.ndarray of float
        A 1-D array representing a signal.
    fs: int
        Sampling rate.
    nfft: int (optional)
        Number of points for Fourier transformation.
        Default is None.
    taper: float (optional)
        Taper percentage for a tapered cosine window.
        This is applied to (multiplied with) the input signal before computation of spectral contents.
        It ranges in the interval [0.0, 50.0].
        Default is 5.0 (i.e., 5% cosine taper).
        
    Returns
    -------
    f: numpy.ndarray of float
        A 1-D array of frequencies.
    A: numpy.ndarray of complex
        A 1-D array of spectral amplitudes.
        
    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([0, 1, 0, 0])
    >>> fs = 1
    >>> f, A = compute_spectral_amplitude(x, fs)
    >>> print(f)
    array([0.   0.25 0.5 ])
    >>> print(A)
    array([0.66666667 0.66666667 0.66666667])
    """
    
    # Compute complex spectrum
    f, X = compute_complex_spectrum(x, fs, nfft=nfft, taper=taper)
    
    # Get nfft
    nfft = len(f)
    
    # Compute spectral amplitude
    A = (1 / nfft) * np.abs(X)
    
    return f, A


def compute_power_spectrum(x, fs, nfft=None, taper=5.0, dB=False):
    """Computes and returns power spectrum of a real signal.
    
    If unit of x is 'V' and unit of fs is 'Hz', then unit of output P is 'V**2' for dB=False and '10*log10(V**2)' for dB=True.
    
    Parameters
    ----------
    x: numpy.ndarray of float
        A 1-D array representing a signal.
    fs: int
        Sampling rate.
    nfft: int (optional)
        Number of points for Fourier transformation.
        Default is None.
    taper: float (optional)
        Taper percentage for a tapered cosine window.
        This is applied to (multiplied with) the input signal before computation of spectral contents.
        It ranges in the interval [0.0, 50.0].
        Default is 5.0 (i.e., 5% cosine taper).
    dB: bool
        If True, returns dB values.
        Default is False
        
    Returns
    -------
    f: numpy.ndarray of float
        A 1-D array of frequencies.
    P: numpy.ndarray of complex
        A 1-D array of power spectrum.
        
    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([0, 1, 0, 0])
    >>> fs = 1
    >>> f, A = compute_power_spectrum(x, fs)
    >>> print(f)
    array([0.   0.25 0.5 ])
    >>> print(A)
    array([0.44444444 0.44444444 0.44444444])
    """
    
    # Compute complex spectrum
    f, X = compute_complex_spectrum(x, fs, nfft=nfft, taper=taper)
    
    # Get nfft
    nfft = len(f)
    
    #--- Compute power spectrum
    P = ((1 / nfft) * np.abs(X)) ** 2
    
    # dB calculation
    if dB:
        P = 10 * np.log10(P)
    
    return f, P


def compute_psd(x, fs, nfft=None, taper=5.0, dB=False):
    """Computes and returns Power Spectral Density (PSD) of a real signal.
    
    It is also called periodogram. If unit of x is 'V' and unit of fs is 'Hz', then unit of output S is 'V**2/Hz' for dB=False and '10*log10(V**2/Hz)' for dB=True.
    
    Parameters
    ----------
    x: numpy.ndarray of float
        A 1-D array representing a signal.
    fs: int
        Sampling rate.
    nfft: int (optional)
        Number of points for Fourier transformation.
        Default is None.
    taper: float (optional)
        Taper percentage for a tapered cosine window.
        This is applied to (multiplied with) the input signal before computation of spectral contents.
        It ranges in the interval [0.0, 50.0].
        Default is 5.0 (i.e., 5% cosine taper).
    dB: bool
        If True, returns dB values.
        Default is False
        
    Returns
    -------
    f: numpy.ndarray of float
        A 1-D array of frequencies.
    S: numpy.ndarray of complex
        A 1-D array of Power Spectral Density (PSD).
        
    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([0, 1, 0, 0])
    >>> fs = 1
    >>> f, A = compute_psd(x, fs)
    >>> print(f)
    array([0.   0.25 0.5 ])
    >>> print(A)
    array([1.33333333 1.33333333 1.33333333])
    """

    # Compute complex spectrum
    f, X = compute_complex_spectrum(x, fs, nfft=nfft, taper=taper)

    # Get nfft
    nfft = len(f)

    # Frequency resolution
    df = fs / nfft

    # Compute Power Spectral Density (PSD)
    S = ((1 / nfft) * np.abs(X)) ** 2 / df
    
    # dB calculation
    if dB:
        S = 10 * np.log10(S)
    
    return f, S

