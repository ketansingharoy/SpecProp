import numpy as np
from scipy import signal



def next_power_2(n):
    return int(2 ** np.ceil(np.log2(n)))


def compute_complex_spectrum(x, fs, nfft=None, taper=5.0):
    
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
    
    # Compute complex spectrum
    f, X = compute_complex_spectrum(x, fs, nfft=nfft, taper=taper)
    
    # Get nfft
    nfft = len(f)
    
    # Compute spectral amplitude
    A = (1 / nfft) * np.abs(X)
    
    return f, A


def compute_power_spectrum(x, fs, nfft=None, taper=5.0, dB=False):
    
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

