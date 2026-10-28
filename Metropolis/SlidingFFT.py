import numpy as np

def sliding_fft(timeseries, window_size, step_size=1):
    """
    Perform a sliding-window Fourier transform (STFT-like) on a 1D time series.

    Parameters
    ----------
    timeseries : array-like
        Input time series (1D numpy array).
    window_size : int
        Size of the sliding window.
    step_size : int, optional (default=1)
        Step size (stride) for sliding the window.

    Returns
    -------
    spectra : list of np.ndarray
        List of FFT spectra for each window.
    freqs : np.ndarray
        Frequencies corresponding to the FFT bins.
    """

    ts = np.asarray(timeseries)
    n = len(ts)

    # frequency axis (same for all windows)
    freqs = np.fft.fftfreq(window_size, d=1)  # assume unit sampling, adjust d if known

    spectra = []
    for start in range(0, n - window_size + 1, step_size):
        window = ts[start:start+window_size]
        fft_vals = np.fft.fft(window)
        spectra.append(fft_vals)

    return spectra, freqs
