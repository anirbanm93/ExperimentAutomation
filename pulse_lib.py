import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray


def gauss_pulse(FWHM: float, sample_rate: int = 1e9, 
                plot: bool = False):
    """
    Generate a Gaussian pulse waveform.

    Parameters
    ----------
    FWHM : float
        Full Width at Half Maximum (FWHM) of the Gaussian pulse in seconds.
    sample_rate : float, optional, default=1e9
        Sampling rate in samples per second (Hz). Determines the resolution of the pulse.
    plot : bool, optional, default=False
        If True, plot the Gaussian pulse automatically.

    Returns
    -------
    pulse_width : float
        Total duration of the generated pulse in seconds (6*STD).
    t : numpy.ndarray
        Array of time points in seconds corresponding to the pulse.
    sig : numpy.ndarray
        Array of normalized Gaussian pulse amplitudes (peak ~ 1).

    Notes
    -----
    - The standard deviation of the Gaussian is computed as STD = FWHM / sqrt(8 * ln(2)).
    - The pulse duration is taken as 6*STD to ensure the pulse decays near zero at the edges.
    - The total number of samples is rounded to the nearest integer.

    Example
    -------
    >>> pulse_width, t, pulse = gauss_pulse(FWHM=5e-9, sample_rate=int(1e9), plot=True)
    """
    STD = FWHM / np.sqrt(8 * np.log(2))
    pulse_width = 6 * STD

    tot_sample = int(round(sample_rate * pulse_width))
    t = np.linspace(0, pulse_width, tot_sample, endpoint=False)
    sig = np.exp(-(t - 0.5 * pulse_width) ** 2 / (2 * STD ** 2))

    if plot:
        plt.figure()
        plt.plot(t * 1e9, sig, marker='o', linestyle='--')
        plt.xlabel("Time (ns)")
        plt.ylabel("Amplitude")
        plt.title(f"Gaussian Pulse: FWHM = {FWHM * 1e9:.2f} ns")
        

    return pulse_width, t, sig


def square_pulse(PW: float, type: str, sample_rate: int = 1e9,
                plot: bool = False):
    """
    Generate a unipolar or bipolar square pulse waveform.

    Parameters
    ----------
    PW : float
        Pulse width in seconds.
    type : str
        Type of square pulse. Must be either "unipolar" or "bipolar".
        - "unipolar" → amplitude remains constant at +1.
        - "bipolar" → amplitude alternates between +1 and -1.
    sample_rate : float, optional, default=1e9
        Sampling rate in samples per second (Hz).
    plot : bool, optional, default=False
        If True, plot the generated pulse automatically.

    Returns
    -------
    t : numpy.ndarray
        Time array corresponding to the generated pulse in seconds.
    sig : numpy.ndarray
        Pulse amplitude array (either unipolar or bipolar).

    Notes
    -----
    - The total number of samples is computed as `sample_rate * PW`.
    - The "bipolar" mode produces a symmetric waveform with equal positive and negative halves.
    - Use high sample rates to generate sharper transitions.

    Example
    -------
    >>> t, sig = square_pulse(PW=10e-9, type="bipolar", sample_rate=int(1e9), plot=True)
    """

    tot_sample = int(round(sample_rate * PW))
    t = np.linspace(0, PW, tot_sample, endpoint=False)

    if type == "unipolar":
        sig = np.ones_like(t)
    if type == "bipolar":
        sig = np.zeros_like(t)
        sig[t < PW / 2] = 1
        sig[t >= PW / 2] = -1

    if plot:
        plt.figure()
        plt.plot(t * 1e9, sig, marker='o', linestyle='--')
        plt.xlabel("Time (ns)")
        plt.ylabel("Amplitude")
        plt.title(f"Square Pulse: Pulse Width = {PW * 1e9:.2f} ns")
        

    return t, sig


def sawtooth_pulse(PW: float, alpha: float, sample_rate: int = 1e9, 
                   plot: bool = False):
    """
    Generate a sawtooth or reverse-sawtooth pulse waveform.

    Parameters
    ----------
    PW : float
        Pulse width in seconds.
    alpha : float
        Fraction of the period at which the waveform peaks (0 < alpha < 1).
        - alpha > 0.5 → normal sawtooth
        - alpha < 0.5 → reverse sawtooth
    sample_rate : float, optional, default=1e9
        Sampling rate in samples per second (Hz).
    plot : bool, optional, default=False
        If True, plot the waveform automatically.

    Returns
    -------
    t : numpy.ndarray
        Time array corresponding to the pulse in seconds.
    sig : numpy.ndarray
        Sawtooth waveform amplitudes (normalized between 0 and 1).

    Notes
    -----
    - The waveform is piecewise linear:
        - 0 → α·PW : linearly increasing segment
        - α·PW → PW : linearly decreasing segment
    - When α > 1, the function exits and prompts a correction message.

    Example
    -------
    >>> t, sig = sawtooth_pulse(PW=10e-9, alpha=0.3, sample_rate=int(1e9), plot=True)
    """

    tot_sample = int(round(sample_rate * PW))
    t = np.linspace(0, PW, tot_sample, endpoint=False)

    sig = np.zeros_like(t)

    if alpha > 1:
        print("0 < alpha < 1. alpha > 0.5 -> sawtooth wave. alpha < 0.5 -> reverse sawtooth.")
        exit()

    idx = t <= alpha * PW  # alpha*t is the distance between start and peak points
    sig[idx] = np.linspace(0, 1, len(t[idx]))

    idx = t > alpha * PW
    sig[idx] = np.flip(np.linspace(0, 1-(1/len(t)), len(t[idx])))

    if plot:
        plt.figure()
        plt.plot(t * 1e9, sig, marker='o', linestyle='--')
        plt.xlabel("Time (ns)")
        plt.ylabel("Amplitude")
        plt.title(f"Sawtooth Pulse: Pulse Width = {PW * 1e9:.2f} ns")
        

    return t, sig


def walsh_code(SH_duration: float,
               N: int, sample_rate: int = 1e9, unipolar: bool = True, plot: bool = False) -> NDArray[np.int8]:
    from scipy.linalg import hadamard

    """
    Generate Walsh codes (Hadamard matrix rows sorted by sequency).

    Parameters
    ----------
    SH_duration : float
        Duration of one sample-hold interval (in seconds).
    N : int
        Number of Walsh codes (rows of the Hadamard matrix). Must be a power of two.
        If not, it is automatically rounded up to the nearest power of two.
    sample_rate : float, optional, default=1e9
        Sampling rate in samples per second (Hz).
    unipolar : bool, optional, default=True
        If True, convert bipolar (+1/-1) Walsh codes to unipolar (1/0).
    plot : bool, optional, default=False
        If True, plot the generated Walsh codes.

    Returns
    -------
    t : numpy.ndarray
        Time array corresponding to the Walsh code signals in seconds.
    W : numpy.ndarray
        Matrix containing the Walsh code waveforms (each row = one code).

    Notes
    -----
    - The Walsh matrix is derived from the Hadamard matrix with rows sorted by sequency.
    - Sequency is the number of zero crossings (sign changes) per row.
    - Codes are repeated over time using the specified `SH_duration`.
    - When `unipolar=True`, mapping is performed as +1→1, -1→0.

    Example
    -------
    >>> t, W = walsh_code(SH_duration=5e-9, N=8, sample_rate=int(1e9), unipolar=True, plot=True)
    """
    if not N % 2 == 0:
        N = 2 ** np.ceil(np.log2(N))
        print(f"The number of bits: {N}")

    H = hadamard(N)

    # Compute sequency = number of sign changes across each row
    # (count positions where consecutive elements differ)
    diffs = (H[:, 1:] != H[:, :-1])
    sequency = diffs.sum(axis=1)

    order = np.argsort(sequency, kind="stable")
    W = H[order]

    if unipolar:
        # Map -1 -> {0} (common mapping for binary codes)
        W = ((W + 1) // 2).astype(np.int8)

    # Digital to analog
    W = np.repeat(W, repeats=int(SH_duration * sample_rate), axis=1)

    t = np.linspace(0, SH_duration * N, W.shape[1], endpoint=False)

    if plot:
        plt.figure()
        for i, sig in enumerate(W):
            plt.plot(t * 1e9, sig + i * 1.2, marker='o', linestyle='--')
        plt.xlabel("Time (ns)")
        plt.ylabel("Amplitude")
        plt.title(f"Walsh Codes: Sample-Hold Duration = {SH_duration * 1e9:.2f} ns")
        

    return t, W


if __name__ == "__main__":

    pw, t, pulse = gauss_pulse(FWHM=10e-9, sample_rate=int(1e9), plot=True)
    print(f"Number of samples: {len(pulse)}")

    t, pulse = square_pulse(PW=10e-9, type="unipolar", sample_rate=int(1e9), plot=True)
    print(f"Number of samples: {len(pulse)}")

    t, pulse = square_pulse(PW=10e-9, type="bipolar", sample_rate=int(1e9), plot=True)
    print(f"Number of samples: {len(pulse)}")

    t, pulse = sawtooth_pulse(PW=100e-9, alpha=0.9, sample_rate=int(1e9), plot=True)
    print(f"Number of samples: {len(pulse)}")

    t, pulse = sawtooth_pulse(PW=100e-9, alpha=0.1, sample_rate=int(1e9), plot=True)
    print(f"Number of samples: {len(pulse)}")

    t, pulse = sawtooth_pulse(PW=100e-9, alpha=0.5, sample_rate=int(1e9), plot=True)
    print(f"Number of samples: {len(pulse)}")

    t, pulse = walsh_code(SH_duration=10e-09, N=7, sample_rate=int(1e9), plot=True)
    print(f"Number of samples: {pulse.shape[1]}")

    plt.show()
