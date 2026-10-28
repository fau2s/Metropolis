import numpy as np
def cross_correlations_wiener(signals):
    """
    Compute pairwise cross-correlations using the Wiener–Khinchin theorem.

    Parameters
    ----------
    signals : np.ndarray
        Array of shape (N, L) where N is number of signals and L is length.

    Returns
    -------
    R : np.ndarray
        Array of shape (N, N, 2*L - 1) containing cross-correlations.
        R[i, j, :] is the correlation between signals[i] and signals[j].
    """
    signals = np.asarray(signals)
    N, L = signals.shape

    # Zero-pad to 2L-1 for full correlation
    nfft = 2 * N - 1

    # FFT of all signals
    fft_signals = np.fft.fft(signals, n=nfft, axis=0)

    # Allocate output matrix
    R = np.zeros((L, L, nfft))

    # Compute pairwise cross-correlations
    for i in range(L):
        for j in range(L):
            R[i, j] = np.fft.ifft(fft_signals[:, i] * np.conj(fft_signals[:, j])).real

    return R.real  # cross-correlations are real